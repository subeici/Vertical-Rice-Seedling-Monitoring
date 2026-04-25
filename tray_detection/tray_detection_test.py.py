"""
Official Implementation for Paper:
"End-Edge-Cloud Collaborative Group Phenotype Recognition and Diagnosis for Rice Seedlings in Vertical Rice Seedlings Cultivation"

Module: Perception Front-end (Inference & Visualization)

Original Script Description:
YOLOv11 single image and batch testing script for seedling tray detection.
"""

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np


def test_single_image(image_path='output_00001164.png',
                      model_path='runs/detect/tray_detection/weights/best.pt',
                      conf_threshold=0.25,
                      save_result=True):
    """
    Test a single image for tray detection.

    Args:
        image_path: Path to the image
        model_path: Path to the trained model weights
        conf_threshold: Confidence threshold for detection
        save_result: Whether to save the visualization results
    """
    print("=" * 60)
    print("YOLO Seedling Tray Detection Test")
    print("=" * 60)

    # Check if files exist
    if not Path(image_path).exists():
        print(f"❌ Image does not exist: {image_path}")
        return

    if not Path(model_path).exists():
        print(f"❌ Model does not exist: {model_path}")
        return

    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Confidence Threshold: {conf_threshold}")
    print("-" * 60)

    # Load model
    print("Loading model...")
    model = YOLO(model_path)

    # Run inference
    print("Running detection...")
    results = model(
        image_path,
        conf=conf_threshold,
        iou=0.45,
        imgsz=640,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Parse results
    print("\nDetection Results:")
    print("-" * 60)

    detections = []
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            print(f"✅ Detected {len(r.boxes)} seedling trays\n")

            for i, box in enumerate(r.boxes):
                # Get coordinates and confidence
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()

                x1, y1, x2, y2 = xyxy
                width = x2 - x1
                height = y2 - y1
                area = width * height

                print(f"Tray {i + 1}:")
                print(f"  Confidence: {conf:.3f}")
                print(f"  Bounding Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
                print(f"  Size: {width:.0f} × {height:.0f} pixels")
                print(f"  Area: {area:.0f} pixels²")
                print()

                detections.append({
                    'bbox': xyxy,
                    'conf': conf,
                    'class': int(cls)
                })
        else:
            print("⚠️ No seedling trays detected")

    # Visualize results
    if save_result and detections:
        print("-" * 60)
        print("Generating visualization results...")

        # Read original image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Original image
        ax1.imshow(img_rgb)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # Detection results
        ax2.imshow(img_rgb)
        ax2.set_title(f'Detection Results ({len(detections)} trays)')
        ax2.axis('off')

        # Draw bounding boxes
        colors = plt.cm.rainbow(np.linspace(0, 1, len(detections)))

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']

            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=colors[idx],
                facecolor='none'
            )
            ax2.add_patch(rect)

            # Add label
            label = f'#{idx + 1} {conf:.2f}'
            ax2.text(
                x1, y1 - 5, label,
                color='white',
                fontsize=10,
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=colors[idx],
                    alpha=0.7
                )
            )

        plt.tight_layout()

        # Save results
        output_path = 'detection_result.jpg'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {output_path}")

        # Display (if in GUI environment)
        try:
            plt.show()
        except:
            pass

    # Save YOLO format results
    if save_result:
        # Save predicted image
        for r in results:
            r.save(filename='yolo_result.jpg')
            print(f"✅ YOLO result saved to: yolo_result.jpg")

    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)

    return detections


def batch_test(image_folder='YOLOv11',
               model_path='YOLOv11/runs/detect/tray_detection/weights/best.pt',
               pattern='output_*.png'):
    """
    Batch test images in a specified folder.
    """
    from glob import glob

    print("=" * 60)
    print("Batch Testing")
    print("=" * 60)

    # Find all matching images
    image_paths = glob(f"{image_folder}/{pattern}")
    print(f"Found {len(image_paths)} images")

    if not image_paths:
        print("❌ No matching images found")
        return

    # Load model
    model = YOLO(model_path)

    all_results = {}
    total_detections = 0

    for img_path in image_paths[:5]:  # Only test the first 5 images
        print(f"\nTesting: {Path(img_path).name}")

        results = model(img_path, conf=0.25, imgsz=640)

        for r in results:
            if r.boxes is not None:
                num_boxes = len(r.boxes)
                print(f"  Detected: {num_boxes} trays")
                all_results[img_path] = num_boxes
                total_detections += num_boxes
            else:
                print(f"  Detected: 0 trays")
                all_results[img_path] = 0

    print("\n" + "=" * 60)
    print("Batch Testing Summary:")
    print(f"  Images tested: {len(all_results)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average per image: {total_detections / len(all_results):.1f} trays")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description='Test YOLO Detection Model')
    parser.add_argument('--image', type=str, default='output_00001164.png',
                        help='Path to test image')
    parser.add_argument('--model', type=str, default='runs/detect/tray_detection/weights/best.pt',
                        help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--batch', action='store_true',
                        help='Enable batch testing mode')

    args = parser.parse_args()

    # Display device info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    if args.batch:
        # Batch testing
        batch_test(model_path=args.model)
    else:
        # Single image testing
        test_single_image(
            image_path=args.image,
            model_path=args.model,
            conf_threshold=args.conf,
            save_result=True
        )