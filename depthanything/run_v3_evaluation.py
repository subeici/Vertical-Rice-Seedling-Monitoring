"""
Official Implementation for Paper:
"End-Edge-Cloud Collaborative Group Phenotype Recognition and Diagnosis for Rice Seedlings in Vertical Rice Seedlings Cultivation"

Module: Perception Front-end (V3 - One-shot Depth Prior Batch Evaluation)

Description:
Implements the One-shot Calibration and Zero-shot Generalization paradigm using Depth Anything V3.
Features V3-specific model loading and depth inversion patches to align with geometric priors.
"""

import argparse
import cv2
import glob
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import torch
from scipy import optimize
from scipy.stats import gaussian_kde

# ======= V3 Specific Imports =======
from depth_anything_3.api import DepthAnything3
from safetensors.torch import load_file


def load_labelme_polygon(json_path):
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
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], 255)
    return mask


def create_global_mask(image_shape, polygons):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for p in polygons:
        cv2.fillPoly(mask, [p['points']], 255)
    return mask


def keep_largest_component(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if num_labels <= 1:
        return mask, 0
    max_area = 0
    max_label = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = i
    new_mask = np.zeros_like(mask)
    new_mask[labels == max_label] = 255
    return new_mask, max_area


def calculate_iou_and_pa(pred_mask, gt_mask):
    pred_binary = pred_mask > 127
    gt_binary = gt_mask > 127

    tp = np.sum(pred_binary & gt_binary)
    fp = np.sum(pred_binary & ~gt_binary)
    fn = np.sum(~pred_binary & gt_binary)
    tn = np.sum(~pred_binary & ~gt_binary)

    intersection = tp
    union = tp + fp + fn
    iou = intersection / union if union > 0 else 0

    total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
    correct_pixels = tp + tn
    pa = correct_pixels / total_pixels if total_pixels > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    bg_intersection = tn
    bg_union = tn + fp + fn
    bg_iou = bg_intersection / bg_union if bg_union > 0 else 0
    fg_iou = iou
    miou = (bg_iou + fg_iou) / 2

    metrics = {
        'iou': iou, 'miou': miou, 'bg_iou': bg_iou, 'fg_iou': fg_iou,
        'pa': pa, 'precision': precision, 'recall': recall, 'f1': f1,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
    }
    return iou, pa, metrics


def analyze_depth_distribution(depth_map, polygon_mask):
    inside_depth = depth_map[polygon_mask > 127]
    outside_mask = polygon_mask <= 127
    outside_depth = depth_map[outside_mask]

    if len(inside_depth) == 0:
        return None, None, None, None

    inside_stats = {
        'min': np.min(inside_depth), 'max': np.max(inside_depth),
        'mean': np.mean(inside_depth), 'std': np.std(inside_depth),
        'median': np.median(inside_depth),
        'percentiles': {
            1: np.percentile(inside_depth, 1), 5: np.percentile(inside_depth, 5),
            10: np.percentile(inside_depth, 10), 25: np.percentile(inside_depth, 25),
            75: np.percentile(inside_depth, 75), 90: np.percentile(inside_depth, 90),
            95: np.percentile(inside_depth, 95), 99: np.percentile(inside_depth, 99)
        }
    }

    outside_stats = None
    if len(outside_depth) > 0:
        outside_stats = {
            'min': np.min(outside_depth), 'max': np.max(outside_depth),
            'mean': np.mean(outside_depth), 'std': np.std(outside_depth),
            'median': np.median(outside_depth),
            'percentiles': {
                1: np.percentile(outside_depth, 1), 5: np.percentile(outside_depth, 5),
                10: np.percentile(outside_depth, 10), 25: np.percentile(outside_depth, 25),
                75: np.percentile(outside_depth, 75), 90: np.percentile(outside_depth, 90),
                95: np.percentile(outside_depth, 95), 99: np.percentile(outside_depth, 99)
            }
        }
    return inside_stats, outside_stats, inside_depth, outside_depth


def calculate_score(depth_min, depth_max, depth_map, polygon_mask, weight_recall=0.7, weight_precision=0.3):
    depth_mask = (depth_map >= depth_min) & (depth_map <= depth_max)
    inside_mask = polygon_mask > 127
    inside_total = np.sum(inside_mask)
    inside_retained = np.sum(depth_mask & inside_mask)
    recall = inside_retained / inside_total if inside_total > 0 else 0

    outside_mask = polygon_mask <= 127
    outside_total = np.sum(outside_mask)
    outside_detected = np.sum(depth_mask & outside_mask)
    false_positive_rate = outside_detected / outside_total if outside_total > 0 else 0

    total_detected = np.sum(depth_mask)
    precision = inside_retained / total_detected if total_detected > 0 else 0

    score = weight_recall * recall + weight_precision * precision
    return score, recall, precision, false_positive_rate


def optimize_depth_range_smart(depth_map, polygon_mask, inside_stats, outside_stats,
                               method='grid_search', weight_recall=0.7, weight_precision=0.3):
    best_depth_min = inside_stats['mean'] - inside_stats['std']
    best_depth_max = inside_stats['mean'] + inside_stats['std']
    best_score = 0
    best_metrics = {}

    if method == 'grid_search':
        min_candidates = [
            inside_stats['percentiles'][1], inside_stats['percentiles'][5],
            inside_stats['percentiles'][10], inside_stats['mean'] - 2 * inside_stats['std'],
                                             inside_stats['mean'] - 1.5 * inside_stats['std'],
                                             inside_stats['mean'] - inside_stats['std']
        ]
        max_candidates = [
            inside_stats['mean'] + inside_stats['std'], inside_stats['mean'] + 1.5 * inside_stats['std'],
            inside_stats['mean'] + 2 * inside_stats['std'], inside_stats['percentiles'][90],
            inside_stats['percentiles'][95], inside_stats['percentiles'][99]
        ]

        for d_min in min_candidates:
            for d_max in max_candidates:
                if d_min < d_max:
                    score, recall, precision, fpr = calculate_score(
                        d_min, d_max, depth_map, polygon_mask, weight_recall, weight_precision)
                    if score > best_score:
                        best_score = score
                        best_depth_min = d_min
                        best_depth_max = d_max
                        best_metrics = {'recall': recall, 'precision': precision, 'fpr': fpr, 'score': score}

    elif method == 'differential':
        if outside_stats:
            if inside_stats['mean'] > outside_stats['mean']:
                best_depth_min = max(inside_stats['percentiles'][10], outside_stats['percentiles'][75])
                best_depth_max = inside_stats['percentiles'][95]
            else:
                best_depth_min = inside_stats['percentiles'][5]
                best_depth_max = min(inside_stats['percentiles'][90], outside_stats['percentiles'][25])
        else:
            best_depth_min = inside_stats['percentiles'][10]
            best_depth_max = inside_stats['percentiles'][90]

    elif method == 'adaptive':
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
            best_depth_min, best_depth_max, depth_map, polygon_mask, weight_recall, weight_precision)
        best_metrics = {'recall': recall, 'precision': precision, 'fpr': fpr, 'score': score}

    best_depth_min = max(0, best_depth_min)
    best_depth_max = min(255, best_depth_max)
    return best_depth_min, best_depth_max, best_metrics


def visualize_optimization_result_enhanced(depth_map, polygon_mask, depth_min, depth_max,
                                           inside_depth, outside_depth, final_mask, metrics, save_path=None):
    fig = plt.figure(figsize=(18, 12))

    ax1 = plt.subplot(3, 3, 1)
    bins = np.linspace(0, 255, 50)
    ax1.hist(inside_depth, bins=bins, alpha=0.5, label='Inside Polygon', color='green', density=True)
    if len(outside_depth) > 0:
        ax1.hist(outside_depth, bins=bins, alpha=0.5, label='Outside Polygon', color='red', density=True)
    ax1.axvline(depth_min, color='blue', linestyle='--', linewidth=2, label=f'Min: {depth_min:.1f}')
    ax1.axvline(depth_max, color='blue', linestyle='--', linewidth=2, label=f'Max: {depth_max:.1f}')
    ax1.set_xlabel('Depth Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Depth Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 3, 2)
    im2 = ax2.imshow(depth_map, cmap='Spectral_r')
    ax2.set_title('Original Depth Map (V3 Reverted)')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = plt.subplot(3, 3, 3)
    ax3.imshow(polygon_mask, cmap='gray')
    ax3.set_title('Ground Truth (Polygon)')

    ax4 = plt.subplot(3, 3, 4)
    depth_filtered = ((depth_map >= depth_min) & (depth_map <= depth_max)).astype(float)
    ax4.imshow(depth_filtered, cmap='RdYlGn')
    ax4.set_title(f'Initial Depth Filter\n[{depth_min:.1f}, {depth_max:.1f}]')

    ax5 = plt.subplot(3, 3, 5)
    ax5.imshow(final_mask, cmap='gray')
    ax5.set_title('Final Mask\n(Largest Component)')

    ax6 = plt.subplot(3, 3, 6)
    confusion = np.zeros(depth_map.shape + (3,))
    confusion[(polygon_mask > 127) & (final_mask > 127)] = [0, 1, 0]
    confusion[(polygon_mask <= 127) & (final_mask > 127)] = [1, 0, 0]
    confusion[(polygon_mask > 127) & (final_mask <= 127)] = [1, 1, 0]
    ax6.imshow(confusion)
    ax6.set_title('Confusion Matrix\n(Green:TP, Red:FP, Yellow:FN)')

    ax7 = plt.subplot(3, 3, 7)
    intersection = (polygon_mask > 127) & (final_mask > 127)
    union = (polygon_mask > 127) | (final_mask > 127)
    iou_vis = np.zeros(depth_map.shape + (3,))
    iou_vis[union] = [0.3, 0.3, 0.3]
    iou_vis[intersection] = [0, 1, 0]
    ax7.imshow(iou_vis)
    ax7.set_title(f'IoU Visualization\nIoU = {metrics["iou"]:.4f}')

    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    metrics_text = f"""📊 Performance Metrics:

IoU: {metrics['iou']:.4f}
mIoU: {metrics['miou']:.4f}
PA (Pixel Acc): {metrics['pa']:.4f}
Precision: {metrics['precision']:.4f}
Recall: {metrics['recall']:.4f}
F1 Score: {metrics['f1']:.4f}

Confusion Matrix:
  TP: {metrics['tp']:,}
  FP: {metrics['fp']:,}
  FN: {metrics['fn']:,}
  TN: {metrics['tn']:,}
"""
    ax8.text(0.05, 0.95, metrics_text, transform=ax8.transAxes, fontsize=10, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    if metrics['iou'] >= 0.7 and metrics['pa'] >= 0.85:
        grade, grade_color = "🏆 Excellent", 'green'
    elif metrics['iou'] >= 0.5 and metrics['pa'] >= 0.75:
        grade, grade_color = "✅ Good", 'yellowgreen'
    elif metrics['iou'] >= 0.3 and metrics['pa'] >= 0.65:
        grade, grade_color = "⚠️ Fair", 'orange'
    else:
        grade, grade_color = "❌ Poor", 'red'

    eval_text = f"🎯 Overall Evaluation:\n\n{grade}"
    ax9.text(0.5, 0.5, eval_text, transform=ax9.transAxes, fontsize=11, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor=grade_color, alpha=0.2))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='One-shot Depth Prior Batch Evaluation for Depth Anything V3')

    # Core Logic Arguments
    parser.add_argument('--ref-img', type=str, required=True, help='Path to Reference image')
    parser.add_argument('--ref-json', type=str, required=True, help='Path to Reference LabelMe JSON')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing testing images and JSONs')

    # V3 Specific Arguments
    parser.add_argument('--config-dir', type=str, default='checkpoints/da3-large',
                        help='Directory containing V3 config.json')
    parser.add_argument('--weights-file', type=str, default='checkpoints/depth_anything_v3_vitl.safetensors',
                        help='Path to safetensors weights')
    parser.add_argument('--outdir', type=str, default='./v3_oneshot_eval_results')

    parser.add_argument('--optimize-method', type=str, default='grid_search',
                        choices=['grid_search', 'differential', 'adaptive'])
    parser.add_argument('--weight-recall', type=float, default=0.7)
    parser.add_argument('--weight-precision', type=float, default=0.3)
    parser.add_argument('--morph-size', type=int, default=5)
    parser.add_argument('--smooth-edges', action='store_true')
    parser.add_argument('--keep-largest', action='store_true', default=True)
    parser.add_argument('--min-area', type=int, default=500)
    parser.add_argument('--show-analysis', action='store_true')
    parser.add_argument('--save-all', action='store_true')

    args = parser.parse_args()

    total_weight = args.weight_recall + args.weight_precision
    args.weight_recall /= total_weight
    args.weight_precision /= total_weight
    BACKGROUND_COLOR = (255, 255, 255)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # =============================================================================
    # V3 Local Model Setup Patch
    # =============================================================================
    safetensor_dst = os.path.join(args.config_dir, "model.safetensors")
    if not os.path.exists(safetensor_dst):
        if os.path.exists(args.weights_file):
            print(f"🔧 Applying V3 setup patch: Linking weights to config directory...")
            try:
                os.symlink(os.path.abspath(args.weights_file), os.path.abspath(safetensor_dst))
            except Exception:
                shutil.copy(args.weights_file, safetensor_dst)
        else:
            print(f"❌ Weights file not found: {args.weights_file}")
            exit(1)

    print(f"📦 Loading Depth Anything V3 Model...")
    depth_anything = DepthAnything3.from_pretrained(args.config_dir).to(DEVICE).eval()

    os.makedirs(args.outdir, exist_ok=True)

    # ---------------------------------------------------------
    # STAGE 1: ONE-SHOT CALIBRATION
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 1: ONE-SHOT DEPTH PRIOR CALIBRATION")
    print("=" * 70)

    ref_raw = cv2.imread(args.ref_img)
    ref_polygons = load_labelme_polygon(args.ref_json)

    ref_mask = create_global_mask(ref_raw.shape, ref_polygons)

    print("Running V3 depth estimation on reference image...")
    with torch.no_grad():
        prediction = depth_anything.inference([args.ref_img])

    ref_depth_raw = prediction.depth[0]

    # Align dimensions if necessary
    if ref_depth_raw.shape != ref_raw.shape[:2]:
        ref_depth_raw = cv2.resize(ref_depth_raw, (ref_raw.shape[1], ref_raw.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Normalization and Crucial V3 Inversion Patch
    ref_depth_norm = (ref_depth_raw - ref_depth_raw.min()) / (ref_depth_raw.max() - ref_depth_raw.min()) * 255.0
    ref_depth_norm = 255.0 - ref_depth_norm

    inside_stats, outside_stats, _, _ = analyze_depth_distribution(ref_depth_norm, ref_mask)
    GLOBAL_DEPTH_MIN, GLOBAL_DEPTH_MAX, _ = optimize_depth_range_smart(
        ref_depth_norm, ref_mask, inside_stats, outside_stats,
        method=args.optimize_method, weight_recall=args.weight_recall, weight_precision=args.weight_precision
    )

    print(
        f"✅ Calibration Complete! Global Depth Prior Locked: MIN = {GLOBAL_DEPTH_MIN:.4f}, MAX = {GLOBAL_DEPTH_MAX:.4f}")

    # ---------------------------------------------------------
    # STAGE 2: ZERO-SHOT BATCH EVALUATION
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 2: ZERO-SHOT BATCH EVALUATION")
    print("=" * 70)

    image_extensions = ['*.jpg', '*.png', '*.jpeg']
    img_list = []
    for ext in image_extensions:
        img_list.extend(glob.glob(os.path.join(args.input_dir, ext)))

    print(f"Found {len(img_list)} test images. Commencing evaluation...")
    all_metrics = []

    for img_path in img_list:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(args.input_dir, f"{base_name}.json")

        if not os.path.exists(json_path):
            continue

        raw_image = cv2.imread(img_path)
        polygons = load_labelme_polygon(json_path)

        with torch.no_grad():
            prediction = depth_anything.inference([img_path])

        test_depth_raw = prediction.depth[0]
        if test_depth_raw.shape != raw_image.shape[:2]:
            test_depth_raw = cv2.resize(test_depth_raw, (raw_image.shape[1], raw_image.shape[0]),
                                        interpolation=cv2.INTER_LINEAR)

        test_depth_norm = (test_depth_raw - test_depth_raw.min()) / (
                    test_depth_raw.max() - test_depth_raw.min()) * 255.0
        test_depth_norm = 255.0 - test_depth_norm

        depth_mask = (test_depth_norm >= GLOBAL_DEPTH_MIN) & (test_depth_norm <= GLOBAL_DEPTH_MAX)
        depth_mask = depth_mask.astype(np.uint8) * 255

        kernel = np.ones((args.morph_size, args.morph_size), np.uint8)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel)

        if args.keep_largest:
            depth_mask, max_area = keep_largest_component(depth_mask)

        if args.smooth_edges:
            depth_mask = cv2.GaussianBlur(depth_mask, (7, 7), 2)
            _, depth_mask = cv2.threshold(depth_mask, 127, 255, cv2.THRESH_BINARY)

        for idx, polygon_data in enumerate(polygons):
            label = polygon_data['label']
            polygon_mask = create_mask_from_polygon(raw_image.shape, polygon_data['points'])

            iou, pa, metrics = calculate_iou_and_pa(depth_mask, polygon_mask)
            print(f"Image: {base_name} | Label: {label} | IoU: {iou:.4f} | PA: {pa:.4f}")

            all_metrics.append({
                'image': base_name, 'label': label, 'iou': iou, 'miou': metrics['miou'], 'pa': pa,
                'precision': metrics['precision'], 'recall': metrics['recall'], 'f1': metrics['f1']
            })

            comparison = np.zeros_like(raw_image)
            comparison[(polygon_mask > 127) & (depth_mask > 127)] = [0, 255, 0]
            comparison[(polygon_mask <= 127) & (depth_mask > 127)] = [0, 0, 255]
            comparison[(polygon_mask > 127) & (depth_mask <= 127)] = [0, 255, 255]

            result_path = os.path.join(args.outdir, f"{base_name}_v3_result_{label}_{idx}.png")
            mask_3channel = np.stack([depth_mask] * 3, axis=-1) / 255.0
            background_raw = np.ones_like(raw_image) * np.array(BACKGROUND_COLOR, dtype=np.uint8)
            filtered_raw = (raw_image * mask_3channel + background_raw * (1 - mask_3channel)).astype(np.uint8)
            cv2.imwrite(result_path, filtered_raw)

            if args.save_all:
                combined = cv2.hconcat([filtered_raw, comparison])
                cv2.imwrite(os.path.join(args.outdir, f"{base_name}_v3_combined_{label}_{idx}.png"), combined)

    if len(all_metrics) > 0:
        print(f"\n{'=' * 70}\n📊 OVERALL DATASET SUMMARY (V3)\n{'=' * 70}")
        avg_iou = np.mean([m['iou'] for m in all_metrics])
        avg_miou = np.mean([m['miou'] for m in all_metrics])
        avg_pa = np.mean([m['pa'] for m in all_metrics])
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])
        avg_f1 = np.mean([m['f1'] for m in all_metrics])

        print(f"  Global Average IoU:       {avg_iou:.4f}")
        print(f"  Global Average mIoU:      {avg_miou:.4f}")
        print(f"  Global Average PA:        {avg_pa:.4f}")
        print(f"  Global Average Precision: {avg_precision:.4f}")
        print(f"  Global Average Recall:    {avg_recall:.4f}")
        print(f"  Global Average F1 Score:  {avg_f1:.4f}")

        summary = {
            'calibration_depth_range': {'min': float(GLOBAL_DEPTH_MIN), 'max': float(GLOBAL_DEPTH_MAX)},
            'total_polygons_evaluated': len(all_metrics),
            'average_metrics': {
                'avg_IoU': float(avg_iou), 'avg_mIoU': float(avg_miou), 'avg_PA': float(avg_pa),
                'avg_precision': float(avg_precision), 'avg_recall': float(avg_recall), 'avg_F1': float(avg_f1)
            },
            'individual_results': all_metrics
        }
        summary_path = os.path.join(args.outdir, "FINAL_BATCH_SUMMARY_V3.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Summary Report Saved: {summary_path}")