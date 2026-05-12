import argparse
import cv2
import glob
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from depth_anything_v2.dpt import DepthAnythingV2
from scipy import optimize
from scipy.stats import gaussian_kde

def load_labelme_polygon(json_path):
    """加载 labelme JSON 文件并提取多边形坐标"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    polygons = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            points = np.array(shape['points'], dtype=np.int32)
            polygons.append({
                'points': points,
                'label': shape.get('label', 'unknown')
            })
    
    return polygons

def create_mask_from_polygon(image_shape, polygon_points):
    """根据多边形坐标创建掩码"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], 255)
    return mask

def keep_largest_component(mask):
    """只保留最大的连通组件
    
    Args:
        mask: 二值掩码 (0 或 255)
    
    Returns:
        只包含最大连通组件的掩码
    """
    # 查找所有连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    if num_labels <= 1:
        return mask
    
    # 找到最大的连通组件（排除背景，标签0）
    max_area = 0
    max_label = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = i
    
    # 创建只包含最大连通组件的新掩码
    new_mask = np.zeros_like(mask)
    new_mask[labels == max_label] = 255
    
    return new_mask, max_area

def calculate_iou_and_pa(pred_mask, gt_mask):
    """计算IoU和像素准确率
    
    Args:
        pred_mask: 预测掩码 (二值：0或255)
        gt_mask: 真实掩码/多边形掩码 (二值：0或255)
    
    Returns:
        iou: 交并比
        pa: 像素准确率
        metrics: 详细指标字典
    """
    # 转换为二值（True/False）
    pred_binary = pred_mask > 127
    gt_binary = gt_mask > 127
    
    # 计算基本指标
    tp = np.sum(pred_binary & gt_binary)  # True Positive
    fp = np.sum(pred_binary & ~gt_binary)  # False Positive
    fn = np.sum(~pred_binary & gt_binary)  # False Negative
    tn = np.sum(~pred_binary & ~gt_binary)  # True Negative
    
    # 计算IoU (Intersection over Union)
    intersection = tp
    union = tp + fp + fn
    iou = intersection / union if union > 0 else 0
    
    # 计算像素准确率 (Pixel Accuracy)
    total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
    correct_pixels = tp + tn
    pa = correct_pixels / total_pixels if total_pixels > 0 else 0
    
    # 其他有用的指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 计算mIoU（对于二分类）
    # 类别0（背景）的IoU
    bg_intersection = tn
    bg_union = tn + fp + fn
    bg_iou = bg_intersection / bg_union if bg_union > 0 else 0
    
    # 类别1（前景）的IoU就是上面计算的iou
    fg_iou = iou
    
    # 平均IoU
    miou = (bg_iou + fg_iou) / 2
    
    metrics = {
        'iou': iou,
        'miou': miou,
        'bg_iou': bg_iou,
        'fg_iou': fg_iou,
        'pa': pa,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }
    
    return iou, pa, metrics

def analyze_depth_distribution(depth_map, polygon_mask):
    """分析多边形内外的深度分布"""
    # 多边形内的深度值
    inside_depth = depth_map[polygon_mask > 127]
    
    # 多边形外的深度值
    outside_mask = polygon_mask <= 127
    outside_depth = depth_map[outside_mask]
    
    if len(inside_depth) == 0:
        return None, None, None, None
    
    # 统计信息
    inside_stats = {
        'min': np.min(inside_depth),
        'max': np.max(inside_depth),
        'mean': np.mean(inside_depth),
        'std': np.std(inside_depth),
        'median': np.median(inside_depth),
        'percentiles': {
            1: np.percentile(inside_depth, 1),
            5: np.percentile(inside_depth, 5),
            10: np.percentile(inside_depth, 10),
            25: np.percentile(inside_depth, 25),
            75: np.percentile(inside_depth, 75),
            90: np.percentile(inside_depth, 90),
            95: np.percentile(inside_depth, 95),
            99: np.percentile(inside_depth, 99)
        }
    }
    
    outside_stats = None
    if len(outside_depth) > 0:
        outside_stats = {
            'min': np.min(outside_depth),
            'max': np.max(outside_depth),
            'mean': np.mean(outside_depth),
            'std': np.std(outside_depth),
            'median': np.median(outside_depth),
            'percentiles': {
                1: np.percentile(outside_depth, 1),
                5: np.percentile(outside_depth, 5),
                10: np.percentile(outside_depth, 10),
                25: np.percentile(outside_depth, 25),
                75: np.percentile(outside_depth, 75),
                90: np.percentile(outside_depth, 90),
                95: np.percentile(outside_depth, 95),
                99: np.percentile(outside_depth, 99)
            }
        }
    
    return inside_stats, outside_stats, inside_depth, outside_depth

def calculate_score(depth_min, depth_max, depth_map, polygon_mask, weight_recall=0.7, weight_precision=0.3):
    """计算给定深度范围的得分"""
    # 创建深度掩码
    depth_mask = (depth_map >= depth_min) & (depth_map <= depth_max)
    
    # 计算多边形内的保留率（召回率）
    inside_mask = polygon_mask > 127
    inside_total = np.sum(inside_mask)
    inside_retained = np.sum(depth_mask & inside_mask)
    recall = inside_retained / inside_total if inside_total > 0 else 0
    
    # 计算多边形外的误检率
    outside_mask = polygon_mask <= 127
    outside_total = np.sum(outside_mask)
    outside_detected = np.sum(depth_mask & outside_mask)
    false_positive_rate = outside_detected / outside_total if outside_total > 0 else 0
    
    # 精确率：检测到的区域中，多少是真正在多边形内的
    total_detected = np.sum(depth_mask)
    precision = inside_retained / total_detected if total_detected > 0 else 0
    
    # 综合得分（加权）
    score = weight_recall * recall + weight_precision * precision
    
    return score, recall, precision, false_positive_rate

def optimize_depth_range_smart(depth_map, polygon_mask, inside_stats, outside_stats, 
                                method='grid_search', weight_recall=0.7, weight_precision=0.3):
    """智能优化深度范围"""
    best_depth_min = inside_stats['mean'] - inside_stats['std']
    best_depth_max = inside_stats['mean'] + inside_stats['std']
    best_score = 0
    best_metrics = {}
    
    if method == 'grid_search':
        # 网格搜索最优范围
        min_candidates = [
            inside_stats['percentiles'][1],
            inside_stats['percentiles'][5],
            inside_stats['percentiles'][10],
            inside_stats['mean'] - 2*inside_stats['std'],
            inside_stats['mean'] - 1.5*inside_stats['std'],
            inside_stats['mean'] - inside_stats['std']
        ]
        
        max_candidates = [
            inside_stats['mean'] + inside_stats['std'],
            inside_stats['mean'] + 1.5*inside_stats['std'],
            inside_stats['mean'] + 2*inside_stats['std'],
            inside_stats['percentiles'][90],
            inside_stats['percentiles'][95],
            inside_stats['percentiles'][99]
        ]
        
        for d_min in min_candidates:
            for d_max in max_candidates:
                if d_min < d_max:
                    score, recall, precision, fpr = calculate_score(
                        d_min, d_max, depth_map, polygon_mask, 
                        weight_recall, weight_precision
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_depth_min = d_min
                        best_depth_max = d_max
                        best_metrics = {
                            'recall': recall,
                            'precision': precision,
                            'fpr': fpr,
                            'score': score
                        }
    
    elif method == 'differential':
        # 基于内外分布差异的方法
        if outside_stats:
            inside_mean = inside_stats['mean']
            outside_mean = outside_stats['mean']
            
            if inside_mean > outside_mean:
                best_depth_min = max(inside_stats['percentiles'][10], 
                                    outside_stats['percentiles'][75])
                best_depth_max = inside_stats['percentiles'][95]
            else:
                best_depth_min = inside_stats['percentiles'][5]
                best_depth_max = min(inside_stats['percentiles'][90], 
                                    outside_stats['percentiles'][25])
        else:
            best_depth_min = inside_stats['percentiles'][10]
            best_depth_max = inside_stats['percentiles'][90]
    
    elif method == 'adaptive':
        # 自适应方法
        std_ratio = inside_stats['std'] / inside_stats['mean'] if inside_stats['mean'] != 0 else 1
        
        if std_ratio < 0.15:
            best_depth_min = inside_stats['percentiles'][25]
            best_depth_max = inside_stats['percentiles'][75]
        elif std_ratio < 0.3:
            best_depth_min = inside_stats['percentiles'][10]
            best_depth_max = inside_stats['percentiles'][90]
        else:
            best_depth_min = inside_stats['mean'] - inside_stats['std']
            best_depth_max = inside_stats['mean'] + inside_stats['std']
        
        score, recall, precision, fpr = calculate_score(
            best_depth_min, best_depth_max, depth_map, polygon_mask,
            weight_recall, weight_precision
        )
        best_metrics = {
            'recall': recall,
            'precision': precision,
            'fpr': fpr,
            'score': score
        }
    
    best_depth_min = max(0, best_depth_min)
    best_depth_max = min(255, best_depth_max)
    
    return best_depth_min, best_depth_max, best_metrics

def visualize_optimization_result_enhanced(depth_map, polygon_mask, depth_min, depth_max, 
                                          inside_depth, outside_depth, final_mask, metrics,
                                          save_path=None):
    """增强版可视化，包含IoU和PA指标"""
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 深度分布对比
    ax1 = plt.subplot(3, 3, 1)
    bins = np.linspace(0, 255, 50)
    ax1.hist(inside_depth, bins=bins, alpha=0.5, label='Inside Polygon', 
             color='green', density=True)
    if len(outside_depth) > 0:
        ax1.hist(outside_depth, bins=bins, alpha=0.5, label='Outside Polygon', 
                color='red', density=True)
    ax1.axvline(depth_min, color='blue', linestyle='--', linewidth=2, 
                label=f'Min: {depth_min:.1f}')
    ax1.axvline(depth_max, color='blue', linestyle='--', linewidth=2, 
                label=f'Max: {depth_max:.1f}')
    ax1.set_xlabel('Depth Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Depth Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 原始深度图
    ax2 = plt.subplot(3, 3, 2)
    im2 = ax2.imshow(depth_map, cmap='Spectral_r')
    ax2.set_title('Original Depth Map')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # 3. Ground Truth (多边形掩码)
    ax3 = plt.subplot(3, 3, 3)
    ax3.imshow(polygon_mask, cmap='gray')
    ax3.set_title('Ground Truth (Polygon)')
    
    # 4. 初始深度过滤结果
    ax4 = plt.subplot(3, 3, 4)
    depth_filtered = ((depth_map >= depth_min) & (depth_map <= depth_max)).astype(float)
    ax4.imshow(depth_filtered, cmap='RdYlGn')
    ax4.set_title(f'Initial Depth Filter\n[{depth_min:.1f}, {depth_max:.1f}]')
    
    # 5. 最终掩码（最大主体）
    ax5 = plt.subplot(3, 3, 5)
    ax5.imshow(final_mask, cmap='gray')
    ax5.set_title('Final Mask\n(Largest Component)')
    
    # 6. 混淆矩阵可视化
    ax6 = plt.subplot(3, 3, 6)
    confusion = np.zeros(depth_map.shape + (3,))
    # 真阳性（绿色）
    tp_mask = (polygon_mask > 127) & (final_mask > 127)
    confusion[tp_mask] = [0, 1, 0]
    # 假阳性（红色）
    fp_mask = (polygon_mask <= 127) & (final_mask > 127)
    confusion[fp_mask] = [1, 0, 0]
    # 假阴性（黄色）
    fn_mask = (polygon_mask > 127) & (final_mask <= 127)
    confusion[fn_mask] = [1, 1, 0]
    # 真阴性（黑色）- 默认已是黑色
    
    ax6.imshow(confusion)
    ax6.set_title('Confusion Matrix\n(Green:TP, Red:FP, Yellow:FN)')
    
    # 7. IoU可视化
    ax7 = plt.subplot(3, 3, 7)
    # 创建IoU可视化（交集和并集）
    intersection = (polygon_mask > 127) & (final_mask > 127)
    union = (polygon_mask > 127) | (final_mask > 127)
    iou_vis = np.zeros(depth_map.shape + (3,))
    iou_vis[union] = [0.3, 0.3, 0.3]  # 灰色表示并集
    iou_vis[intersection] = [0, 1, 0]  # 绿色表示交集
    ax7.imshow(iou_vis)
    ax7.set_title(f'IoU Visualization\nIoU = {metrics["iou"]:.3f}')
    
    # 8. 性能指标表格
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    metrics_text = f"""📊 Performance Metrics:
    
IoU (交并比): {metrics['iou']:.3f}
mIoU (平均IoU): {metrics['miou']:.3f}
  - Foreground IoU: {metrics['fg_iou']:.3f}
  - Background IoU: {metrics['bg_iou']:.3f}

PA (像素准确率): {metrics['pa']:.2%}
Precision: {metrics['precision']:.2%}
Recall: {metrics['recall']:.2%}
F1 Score: {metrics['f1']:.3f}

Confusion Matrix:
  TP: {metrics['tp']:,}
  FP: {metrics['fp']:,}
  FN: {metrics['fn']:,}
  TN: {metrics['tn']:,}
"""
    ax8.text(0.05, 0.95, metrics_text, transform=ax8.transAxes, 
             fontsize=10, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 9. 评价等级
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # 根据IoU和PA确定等级
    if metrics['iou'] >= 0.7 and metrics['pa'] >= 0.85:
        grade = "🏆 优秀 (Excellent)"
        grade_color = 'green'
    elif metrics['iou'] >= 0.5 and metrics['pa'] >= 0.75:
        grade = "✅ 良好 (Good)"
        grade_color = 'yellowgreen'
    elif metrics['iou'] >= 0.3 and metrics['pa'] >= 0.65:
        grade = "⚠️ 一般 (Fair)"
        grade_color = 'orange'
    else:
        grade = "❌ 较差 (Poor)"
        grade_color = 'red'
    
    eval_text = f"""🎯 Overall Evaluation:

{grade}

建议:
"""
    if metrics['recall'] < 0.6:
        eval_text += "\n• 召回率较低，考虑放宽深度范围"
    if metrics['precision'] < 0.5:
        eval_text += "\n• 精确率较低，考虑收紧深度范围"
    if metrics['iou'] < 0.5:
        eval_text += "\n• IoU较低，需要调整优化策略"
    if metrics['pa'] < 0.75:
        eval_text += "\n• 像素准确率不足，检查深度阈值"
    
    if metrics['iou'] >= 0.7 and metrics['pa'] >= 0.85:
        eval_text += "\n• 结果优秀，当前参数设置良好！"
    
    ax9.text(0.5, 0.5, eval_text, transform=ax9.transAxes,
             fontsize=11, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor=grade_color, alpha=0.2))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"增强分析图保存: {save_path}")
    
    plt.show()

if __name__ == '__main__':
    # =============================================================================
    # 参数设置
    # =============================================================================
    
    parser = argparse.ArgumentParser(description='Enhanced Depth Optimization with IoU/PA Metrics')
    
    parser.add_argument('--img-path', type=str, required=True, help='Input image path')
    parser.add_argument('--json-path', type=str, required=True, help='LabelMe JSON file path')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth_optimized')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    # 优化相关参数
    parser.add_argument('--optimize-method', type=str, default='grid_search', 
                       choices=['grid_search', 'differential', 'adaptive'])
    parser.add_argument('--weight-recall', type=float, default=0.7)
    parser.add_argument('--weight-precision', type=float, default=0.3)
    
    # 后处理参数
    parser.add_argument('--morph-size', type=int, default=5)
    parser.add_argument('--smooth-edges', action='store_true')
    parser.add_argument('--keep-largest', action='store_true', default=True,
                       help='Only keep the largest connected component (default: True)')
    parser.add_argument('--min-area', type=int, default=500)
    
    # 可视化参数
    parser.add_argument('--show-analysis', action='store_true')
    parser.add_argument('--save-all', action='store_true')
    
    args = parser.parse_args()
    
    # 确保权重和为1
    total_weight = args.weight_recall + args.weight_precision
    args.weight_recall = args.weight_recall / total_weight
    args.weight_precision = args.weight_precision / total_weight
    
    BACKGROUND_COLOR = (255, 255, 255)  # 白色背景
    
    # =============================================================================
    # 模型初始化
    # =============================================================================
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    print(f"加载模型: depth_anything_v2_{args.encoder}")
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    # =============================================================================
    # 加载图片和标注
    # =============================================================================
    
    print(f"加载图片: {args.img_path}")
    raw_image = cv2.imread(args.img_path)
    if raw_image is None:
        print(f"错误: 无法读取图片 {args.img_path}")
        exit(1)
    
    print(f"加载标注: {args.json_path}")
    polygons = load_labelme_polygon(args.json_path)
    if not polygons:
        print("错误: JSON文件中没有找到多边形标注")
        exit(1)
    
    print(f"找到 {len(polygons)} 个多边形标注")
    
    # =============================================================================
    # 深度估计
    # =============================================================================
    
    print("进行深度估计...")
    depth_raw = depth_anything.infer_image(raw_image, args.input_size)
    depth_normalized = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min()) * 255
    
    # =============================================================================
    # 处理每个多边形
    # =============================================================================
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # 存储所有多边形的评估结果
    all_metrics = []
    
    for idx, polygon_data in enumerate(polygons):
        polygon_points = polygon_data['points']
        label = polygon_data['label']
        
        print(f"\n{'='*70}")
        print(f"处理多边形 {idx+1}: {label}")
        print(f"{'='*70}")
        
        # 创建多边形掩码
        polygon_mask = create_mask_from_polygon(raw_image.shape, polygon_points)
        
        # 分析深度分布
        print("\n📊 分析深度分布...")
        inside_stats, outside_stats, inside_depth, outside_depth = \
            analyze_depth_distribution(depth_normalized, polygon_mask)
        
        if inside_stats is None:
            print("警告: 无法分析深度值")
            continue
        
        # 打印统计信息
        print(f"\n多边形内深度统计:")
        print(f"  均值: {inside_stats['mean']:.2f} ± {inside_stats['std']:.2f}")
        print(f"  范围: [{inside_stats['min']:.2f}, {inside_stats['max']:.2f}]")
        
        if outside_stats:
            print(f"\n多边形外深度统计:")
            print(f"  均值: {outside_stats['mean']:.2f} ± {outside_stats['std']:.2f}")
            mean_diff = abs(inside_stats['mean'] - outside_stats['mean'])
            print(f"\n内外均值差异: {mean_diff:.2f}")
        
        # 智能优化深度范围
        print(f"\n🔍 优化深度范围 (方法: {args.optimize_method})...")
        depth_min, depth_max, opt_metrics = optimize_depth_range_smart(
            depth_normalized, polygon_mask, inside_stats, outside_stats,
            method=args.optimize_method,
            weight_recall=args.weight_recall,
            weight_precision=args.weight_precision
        )
        
        print(f"\n🎯 优化后的深度范围:")
        print(f"   DEPTH_MIN = {depth_min:.2f}")
        print(f"   DEPTH_MAX = {depth_max:.2f}")
        
        # =============================================================================
        # 应用深度过滤和后处理
        # =============================================================================
        
        print("\n⚙️ 应用深度过滤...")
        
        # 创建深度掩码
        depth_mask = (depth_normalized >= depth_min) & (depth_normalized <= depth_max)
        depth_mask = depth_mask.astype(np.uint8) * 255
        
        # 形态学处理
        kernel = np.ones((args.morph_size, args.morph_size), np.uint8)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel)
        
        # 保留最大连通组件
        if args.keep_largest:
            print("🎯 提取最大主体...")
            depth_mask, max_area = keep_largest_component(depth_mask)
            print(f"   最大主体面积: {max_area:,} 像素")
        else:
            # 去除小区域（但保留多个组件）
            if args.min_area > 0:
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    depth_mask, connectivity=8
                )
                new_mask = np.zeros_like(depth_mask)
                kept_components = 0
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= args.min_area:
                        new_mask[labels == i] = 255
                        kept_components += 1
                depth_mask = new_mask
                print(f"   保留了 {kept_components} 个连通组件")
        
        # 边缘平滑
        if args.smooth_edges:
            depth_mask = cv2.GaussianBlur(depth_mask, (7, 7), 2)
            _, depth_mask = cv2.threshold(depth_mask, 127, 255, cv2.THRESH_BINARY)
        
        # =============================================================================
        # 计算mIoU和PA
        # =============================================================================
        
        print("\n📐 计算评估指标...")
        iou, pa, metrics = calculate_iou_and_pa(depth_mask, polygon_mask)
        
        print(f"\n🎖️ 核心评估指标:")
        print(f"   IoU (交并比): {iou:.3f}")
        print(f"   mIoU (平均IoU): {metrics['miou']:.3f}")
        print(f"     - 前景IoU: {metrics['fg_iou']:.3f}")
        print(f"     - 背景IoU: {metrics['bg_iou']:.3f}")
        print(f"   PA (像素准确率): {pa:.2%}")
        print(f"\n📊 其他指标:")
        print(f"   Precision: {metrics['precision']:.2%}")
        print(f"   Recall: {metrics['recall']:.2%}")
        print(f"   F1 Score: {metrics['f1']:.3f}")
        
        # 质量评估
        print(f"\n🏆 综合评价:")
        if iou >= 0.7 and pa >= 0.85:
            print("   ✅ 优秀: IoU和PA都很高，分割质量极佳")
        elif iou >= 0.5 and pa >= 0.75:
            print("   👍 良好: 分割质量不错，可以接受")
        elif iou >= 0.3 and pa >= 0.65:
            print("   ⚠️ 一般: 有改进空间，考虑调整参数")
        else:
            print("   ❌ 较差: 需要重新调整策略或参数")
        
        # 存储评估结果
        all_metrics.append({
            'label': label,
            'iou': iou,
            'miou': metrics['miou'],
            'pa': pa,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })
        
        # 可视化增强分析
        if args.show_analysis:
            viz_path = os.path.join(args.outdir, f"enhanced_analysis_{label}_{idx}.png")
            visualize_optimization_result_enhanced(
                depth_normalized, polygon_mask, depth_min, depth_max,
                inside_depth, outside_depth, depth_mask, metrics, viz_path
            )
        
        # =============================================================================
        # 保存结果
        # =============================================================================
        
        # 准备可视化
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        depth_vis = depth_normalized.astype(np.uint8)
        depth_vis = (cmap(depth_vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # 应用掩码
        background_raw = np.ones_like(raw_image) * np.array(BACKGROUND_COLOR, dtype=np.uint8)
        background_depth = np.ones_like(depth_vis) * np.array(BACKGROUND_COLOR, dtype=np.uint8)
        
        mask_3channel = np.stack([depth_mask] * 3, axis=-1) / 255.0
        filtered_raw = (raw_image * mask_3channel + background_raw * (1 - mask_3channel)).astype(np.uint8)
        filtered_depth = (depth_vis * mask_3channel + background_depth * (1 - mask_3channel)).astype(np.uint8)
        
        # 创建对比图
        comparison = np.zeros_like(raw_image)
        comparison[(polygon_mask > 127) & (depth_mask > 127)] = [0, 255, 0]  # 绿色: TP
        comparison[(polygon_mask <= 127) & (depth_mask > 127)] = [0, 0, 255]  # 红色: FP
        comparison[(polygon_mask > 127) & (depth_mask <= 127)] = [0, 255, 255]  # 黄色: FN
        
        # 保存结果
        base_name = os.path.splitext(os.path.basename(args.img_path))[0]
        
        result_path = os.path.join(args.outdir, f"{base_name}_result_{label}_{idx}.png")
        cv2.imwrite(result_path, filtered_raw)
        print(f"\n✅ 主要结果保存: {result_path}")
        
        comparison_path = os.path.join(args.outdir, f"{base_name}_comparison_{label}_{idx}.png")
        cv2.imwrite(comparison_path, comparison)
        print(f"✅ 对比图保存: {comparison_path}")
        
        if args.save_all:
            combined = cv2.hconcat([filtered_raw, comparison])
            combined_path = os.path.join(args.outdir, f"{base_name}_combined_{label}_{idx}.png")
            cv2.imwrite(combined_path, combined)
            print(f"✅ 组合图保存: {combined_path}")
        
        # 保存详细配置和评估结果
        config = {
            'image_path': args.img_path,
            'json_path': args.json_path,
            'polygon_label': label,
            'optimization_method': args.optimize_method,
            'keep_largest_component': args.keep_largest,
            'depth_range': {
                'DEPTH_MIN': float(depth_min),
                'DEPTH_MAX': float(depth_max)
            },
            'evaluation_metrics': {
                'IoU': float(iou),
                'mIoU': float(metrics['miou']),
                'PA': float(pa),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'F1': float(metrics['f1']),
                'TP': int(metrics['tp']),
                'FP': int(metrics['fp']),
                'FN': int(metrics['fn']),
                'TN': int(metrics['tn'])
            },
            'statistics': {
                'inside_mean': float(inside_stats['mean']),
                'inside_std': float(inside_stats['std']),
                'inside_median': float(inside_stats['median'])
            }
        }
        
        config_path = os.path.join(args.outdir, f"{base_name}_metrics_{label}_{idx}.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✅ 评估结果保存: {config_path}")
    
    # =============================================================================
    # 汇总所有多边形的结果
    # =============================================================================
    
    if len(all_metrics) > 0:
        print(f"\n{'='*70}")
        print("📊 所有多边形评估汇总")
        print(f"{'='*70}")
        
        # 计算平均值
        avg_iou = np.mean([m['iou'] for m in all_metrics])
        avg_miou = np.mean([m['miou'] for m in all_metrics])
        avg_pa = np.mean([m['pa'] for m in all_metrics])
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])
        avg_f1 = np.mean([m['f1'] for m in all_metrics])
        
        print(f"\n平均指标:")
        print(f"  平均 IoU: {avg_iou:.3f}")
        print(f"  平均 mIoU: {avg_miou:.3f}")
        print(f"  平均 PA: {avg_pa:.2%}")
        print(f"  平均 Precision: {avg_precision:.2%}")
        print(f"  平均 Recall: {avg_recall:.2%}")
        print(f"  平均 F1: {avg_f1:.3f}")
        
        print(f"\n各多边形详情:")
        for m in all_metrics:
            print(f"  {m['label']:15s} - IoU: {m['iou']:.3f}, mIoU: {m['miou']:.3f}, PA: {m['pa']:.2%}")
        
        # 保存汇总结果
        summary = {
            'total_polygons': len(all_metrics),
            'average_metrics': {
                'avg_IoU': float(avg_iou),
                'avg_mIoU': float(avg_miou),
                'avg_PA': float(avg_pa),
                'avg_precision': float(avg_precision),
                'avg_recall': float(avg_recall),
                'avg_F1': float(avg_f1)
            },
            'individual_results': all_metrics
        }
        
        summary_path = os.path.join(args.outdir, f"{base_name}_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 汇总报告保存: {summary_path}")
    
    print(f"\n📁 所有结果保存在: {args.outdir}")
    print("\n💡 优化建议:")
    print("1. 使用 --keep-largest 只保留最大主体（默认开启）")
    print("2. 查看详细分析: --show-analysis")
    print("3. 如需更高召回率: --weight-recall 0.9")
    print("4. 如需更高精确率: --weight-precision 0.7")
    print("5. IoU和PA是关键评估指标，IoU>0.5且PA>0.75表示良好分割")