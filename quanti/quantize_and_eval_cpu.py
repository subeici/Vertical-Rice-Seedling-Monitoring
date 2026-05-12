import os
import torch
import warnings

# ================= 魔法补丁：解决 PyTorch 2.6+ UnpicklingError =================
# 强制将全局的 torch.load 的 weights_only 默认值设为 False
_original_load = torch.load


def _safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)


torch.load = _safe_load
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
# ===============================================================================

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm  # 引入进度条

# 导入你的模块
from dataset_general import LeafMultiTaskDataset, get_transforms
from model_general import MultiTaskLeafClassifier as GeneralClassifier
from model_leaf import MultiTaskLeafClassifier as RTDETRClassifier

# ================= 配置区域 =================
QUANTI_DIR = r'D:\Pycharm\Project\mmdet\quanti'
RESULTS_DIR = os.path.join(QUANTI_DIR, 'results')
DATA_ROOT = r'D:\Pycharm\Project\mmdet_dataset'

os.makedirs(RESULTS_DIR, exist_ok=True)

# 映射本地文件名中的关键词到 timm 模型架构
TIMM_MODEL_MAP = {
    'dinov2-base': 'vit_base_patch14_dinov2',
    'dinov2-with-registers-base': 'vit_base_patch14_reg4_dinov2',
    'swinv2_base_window8_256': 'swinv2_base_window8_256',
    'vit_base_patch16_224': 'vit_base_patch16_224',
    'vit_base_patch32_224': 'vit_base_patch32_224',
    'vit_huge_patch14_224': 'vit_huge_patch14_224',
    'vit_large_patch16_224': 'vit_large_patch16_224'
}


# ================= 辅助函数 =================
def get_model_info(filename):
    use_mrffe = '_MRFFE' in filename
    is_rtdetr = 'RT-DETRv4' in filename or 'RTv4' in filename

    img_size = 224
    if '192' in filename:
        img_size = 192
    elif '256' in filename or is_rtdetr:
        img_size = 256

    timm_name = None
    if not is_rtdetr:
        for key, val in TIMM_MODEL_MAP.items():
            if key in filename:
                timm_name = val
                break

    return is_rtdetr, timm_name, use_mrffe, img_size


def extract_valid_preds(outputs, labels, ignore_index=-1):
    valid_mask = (labels != ignore_index)
    valid_labels = labels[valid_mask].cpu().numpy()
    valid_preds = outputs.argmax(dim=1)[valid_mask].cpu().numpy()
    return valid_labels, valid_preds


def evaluate_model(model, dataloader, device='cpu', desc="Evaluating"):
    model.eval()
    model.to(device)

    val_targets = {'age': [], 'yellow': [], 'disease': []}
    val_preds = {'age': [], 'yellow': [], 'disease': []}

    with torch.no_grad():
        # 加入 tqdm 进度条
        for images, labels in tqdm(dataloader, desc=f"  [{desc}]", leave=False, colour='cyan'):
            images = images.to(device)
            lbl_dict = {k: v.to(device) for k, v in labels.items()}

            out_age, out_yellow, out_disease = model(images)

            for task_name, out in zip(['age', 'yellow', 'disease'], [out_age, out_yellow, out_disease]):
                t_lbl, p_cls = extract_valid_preds(out, lbl_dict[task_name])
                val_targets[task_name].extend(t_lbl)
                val_preds[task_name].extend(p_cls)

    metrics = {}
    for task in ['age', 'yellow', 'disease']:
        y_true, y_pred = np.array(val_targets[task]), np.array(val_preds[task])
        acc = (y_true == y_pred).mean() if len(y_true) > 0 else 0
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        metrics[task] = {'acc': acc, 'p': p, 'r': r, 'f1': f1}

    mean_acc = sum(m['acc'] for m in metrics.values()) / 3.0
    metrics['mean_acc'] = mean_acc
    print(
        f"  ✅ [{desc} 完成] Mean Acc: {mean_acc:.2%} | Age F1: {metrics['age']['f1']:.2f} | Yellow F1: {metrics['yellow']['f1']:.2f} | Dis F1: {metrics['disease']['f1']:.2f}")
    return metrics


# ================= 量化函数 =================
def apply_dynamic_quantization(model):
    model.to('cpu')
    model_dq = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    return model_dq


def apply_static_quantization(model, calib_loader):
    model.to('cpu')
    model.eval()

    torch.backends.quantized.engine = 'qnnpack'
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    try:
        model_prepared = torch.quantization.prepare(model)

        # 加入校准进度条
        with torch.no_grad():
            for i, (images, _) in enumerate(
                    tqdm(calib_loader, desc="  [静态量化校准]", total=5, leave=False, colour='green')):
                if i >= 5: break
                model_prepared(images)

        model_sq = torch.quantization.convert(model_prepared)
        return model_sq
    except Exception as e:
        print(f"  ❌ 静态量化失败 (架构可能不兼容): {e}")
        return None


# ================= 可视化函数 =================
def plot_results(results_dict):
    model_names = list(results_dict.keys())
    fp32_acc = [results_dict[m].get('FP32', {}).get('mean_acc', 0) * 100 for m in model_names]
    dq_acc = [results_dict[m].get('DQ', {}).get('mean_acc', 0) * 100 for m in model_names]
    sq_acc = [results_dict[m].get('SQ', {}).get('mean_acc', 0) * 100 for m in model_names]

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width, fp32_acc, width, label='FP32 (Original)', color='#4c72b0')
    rects2 = ax.bar(x, dq_acc, width, label='Dynamic Quant (INT8)', color='#dd8452')
    rects3 = ax.bar(x + width, sq_acc, width, label='Static Quant (INT8)', color='#55a868')

    ax.set_ylabel('Mean Accuracy (%)', fontsize=12)
    ax.set_title('Quantization Impact on Multi-Task Leaf Classification', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=35, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 自动在柱子上标注数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'quantization_comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\n📊 可视化柱状图已保存至: {plot_path}")


# ================= 主流程 =================
def main():
    print("🚀 启动批量量化与测试流水线...\n")

    all_metrics = {}
    csv_rows = []
    loaders_cache = {}

    def get_loaders(img_size):
        if img_size not in loaders_cache:
            val_ds = LeafMultiTaskDataset(DATA_ROOT, split='val', transform=get_transforms('val', img_size))
            test_ds = LeafMultiTaskDataset(DATA_ROOT, split='test', transform=get_transforms('test', img_size))
            val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
            loaders_cache[img_size] = (val_loader, test_loader)
        return loaders_cache[img_size]

    for filename in os.listdir(QUANTI_DIR):
        if not filename.endswith('.pth'):
            continue

        filepath = os.path.join(QUANTI_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        print(f"=========================================")
        print(f"🎯 正在处理模型: {base_name}")

        is_rtdetr, timm_name, use_mrffe, img_size = get_model_info(base_name)

        if not is_rtdetr and timm_name is None:
            print(f"⚠️ 无法解析模型架构，已跳过。")
            continue

        print(f"  -> 构建网络架构 (Image Size: {img_size}, MRFFE: {use_mrffe})")
        if is_rtdetr:
            model = RTDETRClassifier(pretrained_path=filepath, hidden_dim=2048)
        else:
            model = GeneralClassifier(model_type=timm_name, weight_path=filepath, use_mrffe=use_mrffe)

        model.eval()
        all_metrics[base_name] = {}

        val_loader, test_loader = get_loaders(img_size)

        device_fp32 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fp32_metrics = evaluate_model(model, test_loader, device=device_fp32, desc="FP32 基线")
        all_metrics[base_name]['FP32'] = fp32_metrics

        model_dq = apply_dynamic_quantization(model)
        dq_metrics = evaluate_model(model_dq, test_loader, device='cpu', desc="动态量化")
        all_metrics[base_name]['DQ'] = dq_metrics
        torch.save(model_dq.state_dict(), os.path.join(RESULTS_DIR, f"{base_name}_DQ.pth"))

        model_sq = apply_static_quantization(model, val_loader)
        if model_sq is not None:
            sq_metrics = evaluate_model(model_sq, test_loader, device='cpu', desc="静态量化")
            all_metrics[base_name]['SQ'] = sq_metrics
            torch.save(model_sq.state_dict(), os.path.join(RESULTS_DIR, f"{base_name}_SQ.pth"))
        else:
            all_metrics[base_name]['SQ'] = {'mean_acc': 0, 'age': {'f1': 0}, 'yellow': {'f1': 0}, 'disease': {'f1': 0}}

        for quant_type in ['FP32', 'DQ', 'SQ']:
            m = all_metrics[base_name].get(quant_type, {})
            if not m: continue
            csv_rows.append({
                'Model': base_name, 'Quant_Type': quant_type,
                'Mean_Acc': m.get('mean_acc', 0),
                'Age_F1': m.get('age', {}).get('f1', 0),
                'Yellow_F1': m.get('yellow', {}).get('f1', 0),
                'Disease_F1': m.get('disease', {}).get('f1', 0)
            })

    if csv_rows:
        csv_path = os.path.join(RESULTS_DIR, 'quantization_report.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n📝 完整指标评估报告已保存至: {csv_path}")
        plot_results(all_metrics)


if __name__ == '__main__':
    main()