import os
import glob
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig

# 导入你的数据集和预处理逻辑
from dataset_leaf import LeafMultiTaskDataset, get_transforms

# ==========================================
# 1. 路径与基础配置 (完美对齐你的截图)
# ==========================================
WORK_DIR = r"D:\Pycharm\Project\mmdet\quanti"
RESULTS_DIR = os.path.join(WORK_DIR, "results")
CM_OUT_DIR = os.path.join(RESULTS_DIR, "confusion_matrices_test") 
DATA_ROOT = r'D:\Pycharm\Project\mmdet_dataset'

os.makedirs(CM_OUT_DIR, exist_ok=True)

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
# 2. 推理收集函数 (获取真实标签与预测标签)
# ==========================================
def get_predictions(model, dataloader, device):
    """运行模型，收集用于绘制混淆矩阵的预测结果"""
    model.eval()
    all_targets = {'age': [], 'yellow': [], 'disease': []}
    all_preds = {'age': [], 'yellow': [], 'disease': []}
    
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
                valid_preds = out.argmax(dim=1)[valid_mask].cpu().numpy()
                
                all_targets[task_name].extend(valid_labels)
                all_preds[task_name].extend(valid_preds)
                
    return all_targets, all_preds

# ==========================================
# 3. 绘制 2x3 混淆矩阵大图
# ==========================================
def plot_comparative_confusion_matrix(model_name, fp32_targets, fp32_preds, int8_targets, int8_preds, save_dir):
    """绘制 2行(FP32/INT8) x 3列(Age/Yellow/Disease) 的混淆矩阵"""
    tasks = ['age', 'yellow', 'disease']
    task_titles = ['Age Head', 'Yellow Head', 'Disease Head']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f'Confusion Matrix Comparison (Test Set)\nModel: {model_name}', fontsize=18, fontweight='bold', y=0.95)
    
    for col, task in enumerate(tasks):
        y_true_fp32, y_pred_fp32 = np.array(fp32_targets[task]), np.array(fp32_preds[task])
        y_true_int8, y_pred_int8 = np.array(int8_targets[task]), np.array(int8_preds[task])
        
        # 动态获取当前任务的类别数 (例如 Age 可能是 3 类，Disease 是 2 类)
        classes = np.unique(np.concatenate((y_true_fp32, y_pred_fp32, y_true_int8, y_pred_int8)))
        class_names = [f"Class {c}" for c in classes]
        
        acc_fp32 = accuracy_score(y_true_fp32, y_pred_fp32) * 100 if len(y_true_fp32) > 0 else 0
        acc_int8 = accuracy_score(y_true_int8, y_pred_int8) * 100 if len(y_true_int8) > 0 else 0
        
        # --- 绘制第一行 (FP32) ---
        cm_fp32 = confusion_matrix(y_true_fp32, y_pred_fp32, labels=classes)
        sns.heatmap(cm_fp32, annot=True, fmt='d', cmap='Blues', ax=axes[0, col],
                    xticklabels=class_names, yticklabels=class_names, square=True)
        axes[0, col].set_title(f"FP32 Baseline | {task_titles[col]}\nAcc: {acc_fp32:.2f}%", fontsize=12)
        axes[0, col].set_xlabel("Predicted")
        axes[0, col].set_ylabel("Actual")
        
        # --- 绘制第二行 (INT8) ---
        cm_int8 = confusion_matrix(y_true_int8, y_pred_int8, labels=classes)
        # 用橙色系区分量化模型，视觉上一目了然
        sns.heatmap(cm_int8, annot=True, fmt='d', cmap='Oranges', ax=axes[1, col],
                    xticklabels=class_names, yticklabels=class_names, square=True)
        axes[1, col].set_title(f"INT8 Quantized | {task_titles[col]}\nAcc: {acc_int8:.2f}%", fontsize=12)
        axes[1, col].set_xlabel("Predicted")
        axes[1, col].set_ylabel("Actual")

    plt.tight_layout(rect=[0, 0, 1, 0.92]) 
    save_path = os.path.join(save_dir, f"{model_name}_confusion_matrix_test.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎨 混淆矩阵已保存: {save_path}")

# ==========================================
# 4. 万能权重加载器 (核心修复区)
# ==========================================
def load_fp32_weights(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
    elif 'ema' in checkpoint: state_dict = checkpoint['ema']['module']
    elif 'model' in checkpoint: state_dict = checkpoint['model']
    else: state_dict = checkpoint
        
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)

def load_int8_weights(model, int8_ckpt_path):
    """
    【技术细节】如何加载 torchao 保存的 INT8 模型？
    必须先对原生的 FP32 骨架执行 quantize_，使其内部的 Linear 层转换为包含量化张量的结构，
    然后才能读取 _INT8.pth 的 state_dict。
    """
    quantize_(model, Int8DynamicActivationInt8WeightConfig())
    state_dict = torch.load(int8_ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict, strict=False)

# ==========================================
# 5. 主运行流水线
# ==========================================
def main():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 启动 [测试集混淆矩阵] 评估流水线...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 扫描 quanti 目录下的 FP32 模型 (图2)
    fp32_pth_files = [f for f in glob.glob(os.path.join(WORK_DIR, "*.pth")) if "_INT8" not in f]
    
    if not fp32_pth_files:
        print(f"❌ 在 {WORK_DIR} 目录下没有找到任何 FP32 的 .pth 文件！")
        return

    for fp32_ckpt_path in fp32_pth_files:
        model_name_full = os.path.basename(fp32_ckpt_path).replace('.pth', '')
        # 严格匹配 results 目录下的 INT8 模型 (图1)
        int8_ckpt_path = os.path.join(RESULTS_DIR, f"{model_name_full}_INT8.pth")
        
        if not os.path.exists(int8_ckpt_path):
            print(f"⚠️ 跳过 {model_name_full}：在 results 中未找到配对的 INT8 模型。")
            continue
            
        print("\n" + "="*70)
        print(f"🎯 [正在评估] 模型: {model_name_full}")
        
        try:
            # --- 阶段 A：准备真实的 TEST 数据集 ---
            use_mrffe = "MRFFE" in model_name_full
            model_base = model_name_full.replace('_MRFFE', '')
            img_size = get_img_size_from_name(model_base)
            
            val_transform = get_transforms('val', img_size=img_size) 
            # 【核心】强制使用 split='test' 确保数据纯洁度
            test_dataset = LeafMultiTaskDataset(DATA_ROOT, split='test', transform=val_transform)
            test_loader = DataLoader(test_dataset, batch_size=256, num_workers=8, pin_memory=True, shuffle=False)
            
            # --- 阶段 B：构建并评估 FP32 模型 ---
            print("⏳ [1/2] 正在加载并测试 FP32 基线模型...")
            if 'RT-DETR' in model_base:
                try:
                    from model_redetr_m import MultiTaskLeafClassifier
                    model_fp32 = MultiTaskLeafClassifier(pretrained_path=None, hidden_dim=2048, use_mrffe=use_mrffe)
                except Exception:
                    from model_leaf import MultiTaskLeafClassifier
                    model_fp32 = MultiTaskLeafClassifier(pretrained_path=None, hidden_dim=2048)
            else:
                from model_general import MultiTaskLeafClassifier
                timm_name = TIMM_MODEL_MAP.get(model_base, model_base)
                model_fp32 = MultiTaskLeafClassifier(model_type=timm_name, weight_path=None, use_mrffe=use_mrffe)
            
            model_fp32 = model_fp32.to(device)
            load_fp32_weights(model_fp32, fp32_ckpt_path)
            
            fp32_targets, fp32_preds = get_predictions(model_fp32, test_loader, device)
            del model_fp32 # 释放显存
            
            # --- 阶段 C：构建并评估 INT8 量化模型 ---
            print("⏳ [2/2] 正在加载并测试 INT8 量化模型...")
            if 'RT-DETR' in model_base:
                try:
                    from model_redetr_m import MultiTaskLeafClassifier
                    model_int8 = MultiTaskLeafClassifier(pretrained_path=None, hidden_dim=2048, use_mrffe=use_mrffe)
                except Exception:
                    from model_leaf import MultiTaskLeafClassifier
                    model_int8 = MultiTaskLeafClassifier(pretrained_path=None, hidden_dim=2048)
            else:
                from model_general import MultiTaskLeafClassifier
                model_int8 = MultiTaskLeafClassifier(model_type=timm_name, weight_path=None, use_mrffe=use_mrffe)
            
            model_int8 = model_int8.to(device)
            load_int8_weights(model_int8, int8_ckpt_path) # 执行特殊的 INT8 加载逻辑
            
            int8_targets, int8_preds = get_predictions(model_int8, test_loader, device)
            del model_int8 # 释放显存
            
            # --- 阶段 D：绘制混淆矩阵对比图 ---
            plot_comparative_confusion_matrix(model_name_full, fp32_targets, fp32_preds, int8_targets, int8_preds, CM_OUT_DIR)
            
        except Exception as e:
            print(f"❌ 处理 {model_name_full} 时出错: {e}")
            continue
        finally:
            torch.cuda.empty_cache()

    print(f"\n🏆 测试集混淆矩阵生成完毕！请前往 {CM_OUT_DIR} 文件夹查看全部 12 张图片！")

if __name__ == "__main__":
    main()