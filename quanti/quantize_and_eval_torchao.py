import os
import glob
import time
import logging
import traceback
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# 导入你真实的 Dataset
from dataset_leaf import LeafMultiTaskDataset, get_transforms

# ==========================================
# 1. 核心映射表与路径配置
# ==========================================
WORK_DIR = r"D:\Pycharm\Project\mmdet\quanti"
RESULTS_DIR = os.path.join(WORK_DIR, "results")
DATA_ROOT = r'D:\Pycharm\Project\mmdet_dataset'
os.makedirs(RESULTS_DIR, exist_ok=True)

TIMM_MODEL_MAP = {
    'dinov2-small': 'vit_small_patch14_dinov2',
    'dinov2-base': 'vit_base_patch14_dinov2',
    'dinov2-large': 'vit_large_patch14_dinov2',
    'dinov2-giant': 'vit_giant_patch14_dinov2',
    'dinov2-with-registers-small': 'vit_small_patch14_reg4_dinov2',
    'dinov2-with-registers-base': 'vit_base_patch14_reg4_dinov2',
    'dinov2-with-registers-large': 'vit_large_patch14_reg4_dinov2',
    'dinov2-with-registers-giant': 'vit_giant_patch14_reg4_dinov2',
    'swinv2_base_window8_256': 'swinv2_base_window8_256',
    'swinv2_large_window12_192': 'swinv2_large_window12_192',
    'swinv2_small_window8_256': 'swinv2_small_window8_256',
    'swinv2_tiny_window8_256': 'swinv2_tiny_window8_256',
    'vit_base_patch16_224': 'vit_base_patch16_224',
    'vit_base_patch32_224': 'vit_base_patch32_224',
    'vit_huge_patch14_224': 'vit_huge_patch14_224',
    'vit_large_patch16_224': 'vit_large_patch16_224'
}


def get_img_size_from_name(folder_name):
    if '192' in folder_name: return 192
    if '256' in folder_name: return 256
    return 224


# ==========================================
# 2. 高可靠双日志系统
# ==========================================
def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(message)s')  # 去掉多余的前缀，让性能表格更干净
    file_path = os.path.join(RESULTS_DIR, log_file)
    file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(file_handler)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


log1_quant = setup_logger('quant_process', 'quantization_process.log')
log2_eval = setup_logger('eval_compare', 'model_performance.log')


# ==========================================
# 3. 三联可视化绘图函数 (新增体积对比)
# ==========================================
def plot_and_save_comparison(model_name, fp32_metrics, int8_metrics, fp32_size, int8_size, save_dir):
    """绘制量化前后的性能与体积对比图并保存"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    width = 0.35

    # --- 子图1：各任务头 F1 分数对比 ---
    labels = ['Age F1', 'Yellow F1', 'Disease F1']
    fp32_scores = [fp32_metrics['age_F1'] * 100, fp32_metrics['yellow_F1'] * 100, fp32_metrics['disease_F1'] * 100]
    int8_scores = [int8_metrics['age_F1'] * 100, int8_metrics['yellow_F1'] * 100, int8_metrics['disease_F1'] * 100]

    x = np.arange(len(labels))
    rects1 = ax1.bar(x - width / 2, fp32_scores, width, label='FP32 Baseline', color='#4C72B0')
    rects2 = ax1.bar(x + width / 2, int8_scores, width, label='INT8 Quantized', color='#DD8452')

    ax1.set_ylabel('F1 Score (%)')
    ax1.set_title('Task Heads F1 Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 110)
    ax1.legend()
    ax1.bar_label(rects1, fmt='%.1f', padding=3)
    ax1.bar_label(rects2, fmt='%.1f', padding=3)

    # --- 子图2：推理速度对比 (FPS) ---
    fps_labels = ['Inference Speed (FPS)']
    fp32_fps = [fp32_metrics['FPS']]
    int8_fps = [int8_metrics['FPS']]

    x_fps = np.arange(len(fps_labels))
    rects_fps1 = ax2.bar(x_fps - width / 2, fp32_fps, width, label='FP32 Baseline', color='#55A868')
    rects_fps2 = ax2.bar(x_fps + width / 2, int8_fps, width, label='INT8 Quantized', color='#C44E52')

    ax2.set_ylabel('Frames Per Second (FPS)')
    ax2.set_title('Inference Speed')
    ax2.set_xticks(x_fps)
    ax2.set_xticklabels(fps_labels)
    ax2.legend()
    ax2.bar_label(rects_fps1, fmt='%.1f', padding=3)
    ax2.bar_label(rects_fps2, fmt='%.1f', padding=3)

    # --- 子图3：模型物理体积对比 (MB) ---
    size_labels = ['Model Size (MB)']
    fp32_sizes = [fp32_size]
    int8_sizes = [int8_size]

    x_size = np.arange(len(size_labels))
    rects_size1 = ax3.bar(x_size - width / 2, fp32_sizes, width, label='FP32 Baseline', color='#8172B3')
    rects_size2 = ax3.bar(x_size + width / 2, int8_sizes, width, label='INT8 Quantized', color='#64B5CD')

    ax3.set_ylabel('File Size (MB)')
    ax3.set_title(f'Model Volume (Compression: {int8_size / fp32_size * 100:.1f}%)')
    ax3.set_xticks(x_size)
    ax3.set_xticklabels(size_labels)
    ax3.legend()
    ax3.bar_label(rects_size1, fmt='%.1f', padding=3)
    ax3.bar_label(rects_size2, fmt='%.1f', padding=3)

    fig.suptitle(f'Comprehensive Quantization Report: {model_name}', fontsize=16)
    fig.tight_layout()

    plot_path = os.path.join(save_dir, f"{model_name}_performance_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path


# ==========================================
# 4. 全指标多任务评估函数 (修复缺失的基本指标)
# ==========================================
def evaluate_model(model, dataloader, device):
    model.eval()
    val_targets = {'age': [], 'yellow': [], 'disease': []}
    val_preds = {'age': [], 'yellow': [], 'disease': []}

    start_time = time.time()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            with torch.amp.autocast('cuda'):
                out_age, out_yellow, out_disease = model(images)

            for task_name, out, lbl_key in zip(['age', 'yellow', 'disease'],
                                               [out_age, out_yellow, out_disease],
                                               ['age', 'yellow', 'disease']):
                lbl = labels[lbl_key].to(device)
                valid_mask = (lbl != -1)
                if valid_mask.sum() == 0: continue

                valid_labels = lbl[valid_mask].cpu().numpy()
                valid_preds_batch = out.argmax(dim=1)[valid_mask].cpu().numpy()

                val_targets[task_name].extend(valid_labels)
                val_preds[task_name].extend(valid_preds_batch)

    end_time = time.time()
    total_samples = sum([len(t) for t in val_targets.values()]) / 3
    fps = total_samples / (end_time - start_time + 1e-9)

    metrics = {"FPS": fps}
    mean_acc = 0.0
    for task in ['age', 'yellow', 'disease']:
        y_true = np.array(val_targets[task])
        y_pred = np.array(val_preds[task])

        acc = (y_true == y_pred).mean() * 100 if len(y_true) > 0 else 0
        # 兼容多分类（如 Age 可能有 3 类），必须使用 average='macro' 否则会报错
        p = precision_score(y_true, y_pred, zero_division=0, average='macro')
        r = recall_score(y_true, y_pred, zero_division=0, average='macro')
        f1 = f1_score(y_true, y_pred, zero_division=0, average='macro')

        metrics[f"{task}_Acc"] = acc
        metrics[f"{task}_P"] = p
        metrics[f"{task}_R"] = r
        metrics[f"{task}_F1"] = f1
        mean_acc += acc

    metrics["Mean Acc"] = mean_acc / 3.0
    return metrics


# ==========================================
# 5. 万能权重加载器
# ==========================================
def load_robust_weights(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'ema' in checkpoint:
        state_dict = checkpoint['ema']['module']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        clean_state_dict[new_key] = v

    model.load_state_dict(clean_state_dict, strict=False)


# ==========================================
# 6. 全自动路由流水线
# ==========================================
def main():
    log1_quant.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 启动 [终极多架构融合] 量化、全指标测试与出图流水线...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pth_files = glob.glob(os.path.join(WORK_DIR, "*.pth"))

    # 过滤掉已经是 INT8 的模型，防止重复量化
    pth_files = [f for f in pth_files if "_INT8.pth" not in f]

    if not pth_files:
        log1_quant.error(f"❌ 在 {WORK_DIR} 目录下没有找到任何待量化的 .pth 文件！")
        return

    for ckpt_path in pth_files:
        model_name_full = os.path.basename(ckpt_path).replace('.pth', '')
        log1_quant.info("\n" + "=" * 70)
        log1_quant.info(f"🎯 [开始处理] 模型: {model_name_full}")

        try:
            # --- 解析与数据准备 ---
            use_mrffe = "MRFFE" in model_name_full
            model_base = model_name_full.replace('_MRFFE', '')
            img_size = get_img_size_from_name(model_base)

            val_transform = get_transforms('val', img_size=img_size)
            val_dataset = LeafMultiTaskDataset(DATA_ROOT, split='test', transform=val_transform)
            val_loader = DataLoader(val_dataset, batch_size=256, num_workers=8, pin_memory=True, shuffle=False)

            # --- 构建与加载模型 ---
            log1_quant.info(f"🔧 自动推断网络架构 (MRFFE: {use_mrffe})...")
            if 'RT-DETR' in model_base:
                try:
                    from model_redetr_m import MultiTaskLeafClassifier
                    model = MultiTaskLeafClassifier(pretrained_path=None, hidden_dim=2048, use_mrffe=use_mrffe)
                except Exception:
                    from model_leaf import MultiTaskLeafClassifier
                    model = MultiTaskLeafClassifier(pretrained_path=None, hidden_dim=2048)
            else:
                from model_general import MultiTaskLeafClassifier
                timm_name = TIMM_MODEL_MAP.get(model_base, model_base)
                model = MultiTaskLeafClassifier(model_type=timm_name, weight_path=None, use_mrffe=use_mrffe)

            model = model.to(device)
            load_robust_weights(model, ckpt_path)

            # 获取原模型物理体积大小 (MB)
            fp32_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

            # --- FP32 基线测试 ---
            log1_quant.info("⏳ [1/2] 开始 FP32 基线真实指标测试...")
            fp32_metrics = evaluate_model(model, val_loader, device)

            # --- INT8 GPU 量化与模型保存 ---
            log1_quant.info("⏳ [2/2] 正在注入 torchao INT8 量化算子...")
            quantize_(model, Int8DynamicActivationInt8WeightConfig())

            quant_save_path = os.path.join(RESULTS_DIR, f"{model_name_full}_INT8.pth")
            torch.save(model.state_dict(), quant_save_path)

            # 获取量化后模型物理体积大小 (MB)
            int8_size_mb = os.path.getsize(quant_save_path) / (1024 * 1024)
            log1_quant.info(f"💾 INT8 模型已保存 | 体积变化: {fp32_size_mb:.1f}MB -> {int8_size_mb:.1f}MB")

            # log1_quant.info("⚡ 正在执行 torch.compile 图融合 (加速推理)...")
            # try:
            #     model = torch.compile(model, mode="max-autotune")
            # except Exception:
            #     pass

            log1_quant.info("⏳ 开始 INT8 量化模型真实指标测试...")
            quant_metrics = evaluate_model(model, val_loader, device)

            # --- 自动绘图与输出详尽日志 ---
            log1_quant.info("📊 正在绘制性能与体积对比可视化三联图...")
            img_path = plot_and_save_comparison(model_name_full, fp32_metrics, quant_metrics, fp32_size_mb,
                                                int8_size_mb, RESULTS_DIR)

            # 打印最详尽的对照表到 eval_compare 日志中
            log2_eval.info("\n" + "━" * 80)
            log2_eval.info(f"🧬 模型架构: {model_name_full} | 🖼️ 图像输入: {img_size}x{img_size}")
            log2_eval.info(
                f"💾 体积对比: 【FP32】 {fp32_size_mb:.2f} MB  ==>  【INT8】 {int8_size_mb:.2f} MB (压缩至 {int8_size_mb / fp32_size_mb * 100:.1f}%)")
            log2_eval.info(
                f"⚡ 速度对比: 【FP32】 {fp32_metrics['FPS']:.1f} FPS  ==>  【INT8】 {quant_metrics['FPS']:.1f} FPS")
            log2_eval.info("-" * 80)
            log2_eval.info("【详细指标评估】          Accuracy | Precision |   Recall  |  F1-Score")

            tasks = ['age', 'yellow', 'disease']
            for task in tasks:
                log2_eval.info(
                    f"  [{task.capitalize():<7}] FP32 基线:   {fp32_metrics[f'{task}_Acc']:6.2f}% |    {fp32_metrics[f'{task}_P']:.4f} |    {fp32_metrics[f'{task}_R']:.4f} |    {fp32_metrics[f'{task}_F1']:.4f}")
                log2_eval.info(
                    f"  [{task.capitalize():<7}] INT8 量化:   {quant_metrics[f'{task}_Acc']:6.2f}% |    {quant_metrics[f'{task}_P']:.4f} |    {quant_metrics[f'{task}_R']:.4f} |    {quant_metrics[f'{task}_F1']:.4f}")
                log2_eval.info("  " + "-" * 65)

            log2_eval.info(
                f"🔥 综合平均 Accuracy: 【FP32】 {fp32_metrics['Mean Acc']:.2f}%  |  【INT8】 {quant_metrics['Mean Acc']:.2f}%")
            log2_eval.info("━" * 80)

            for handler in log2_eval.handlers:
                handler.flush()

        except Exception as e:
            log1_quant.error(f"❌ 模型 {model_name_full} 量化流水线断裂！")
            log1_quant.error(traceback.format_exc())
            log2_eval.error(f"\n模型: {model_name_full} | 状态: 致命错误，已跳过！")
            continue

        finally:
            log1_quant.info("🧹 洗地协议启动：彻底清空 CUDA 缓存，准备承接下一模型...")
            try:
                del model
            except NameError:
                pass
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()