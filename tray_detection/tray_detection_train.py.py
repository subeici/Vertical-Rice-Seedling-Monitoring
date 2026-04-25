"""
Official Implementation for Paper:
"End-Edge-Cloud Collaborative Group Phenotype Recognition and Diagnosis for Rice Seedlings in Vertical Rice Seedlings Cultivation"

Module: Perception Front-end (Tray Detection Training & Evaluation)

Original Script Description:
YOLOv11 seedling tray detection training script.
Handles multi-task label format compatibility.
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import shutil
import os
from tqdm import tqdm


def preprocess_labels(source_dir='datasets/tray_dataset', target_dir='datasets/tray_yolo_clean'):
    """
    Pre-process label files, keeping only the first 5 values for object detection.
    """
    print("=" * 60)
    print("Pre-processing label files...")
    print("=" * 60)

    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Create target directory structure
    for split in ['train', 'val']:
        (target_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (target_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Process each dataset split
    for split in ['train', 'val']:
        source_images = source_path / 'images' / split
        source_labels = source_path / 'labels' / split
        target_images = target_path / 'images' / split
        target_labels = target_path / 'labels' / split

        if not source_images.exists():
            print(f"⚠️ {split} set does not exist, skipping.")
            continue

        # Get all images
        image_files = list(source_images.glob('*.jpg')) + \
                      list(source_images.glob('*.png')) + \
                      list(source_images.glob('*.jpeg'))

        print(f"\nProcessing {split} set: {len(image_files)} images")

        valid_count = 0
        for img_path in tqdm(image_files, desc=f"Processing {split}"):
            # Corresponding label file
            label_path = source_labels / f"{img_path.stem}.txt"

            if not label_path.exists():
                continue

            # Read and process labels
            new_labels = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Keep only the first 5 values: class_id, x_center, y_center, width, height
                        new_line = ' '.join(parts[:5])
                        new_labels.append(new_line)

            # If valid labels exist, save the file
            if new_labels:
                # Copy image
                shutil.copy2(img_path, target_images / img_path.name)

                # Save processed labels
                target_label = target_labels / f"{img_path.stem}.txt"
                with open(target_label, 'w') as f:
                    f.write('\n'.join(new_labels))

                valid_count += 1

        print(f"  ✓ {split} set: Successfully processed {valid_count} images")

    print(f"\n✅ Pre-processing complete! Data saved to: {target_path}")
    return str(target_path)


def create_dataset_yaml(data_root, save_path='tray_dataset.yaml'):
    """
    Create YOLO dataset configuration file.
    """
    yaml_content = f"""# Seedling tray detection dataset config
path: {os.path.abspath(data_root)}
train: images/train
val: images/val

# Classes
nc: 1
names: 
  0: tray
"""

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"✅ Configuration file created: {save_path}")
    return save_path


def train_yolo(data_root):
    """
    Train YOLOv11 detection model.
    """
    print("\n" + "=" * 60)
    print("YOLOv11 Training")
    print("=" * 60)

    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # Create configuration file
    yaml_path = create_dataset_yaml(data_root)

    # Training configuration
    print("\nTraining configuration:")
    config = {
        'model': 'yolo11n.pt',  # nano version
        'data': yaml_path,
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'workers': 4,  # reduce workers to avoid issues
        'device': device,
        'project': 'runs/detect',
        'name': 'tray_detection',
        'exist_ok': True,
        'patience': 30,
        'save': True,
        'save_period': 20,
        'amp': True,
        'verbose': True,

        # Optimizer
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,

        # Loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,

        # Data augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,

        'seed': 42,
        'close_mosaic': 10,
    }

    # Print key parameters
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch']}")
    print(f"  Image size: {config['imgsz']}")
    print(f"  Learning rate: {config['lr0']} -> {config['lr0'] * config['lrf']}")

    # Load model
    print(f"\nLoading model: {config['model']}")
    model = YOLO(config['model'])

    # Start training
    print("\nStarting training...")
    print("-" * 60)

    results = model.train(**config)

    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)

    # Best model path
    best_model_path = Path(config['project']) / config['name'] / 'weights' / 'best.pt'
    print(f"Best model: {best_model_path}")

    return model, results, best_model_path


def validate_model(model_path, yaml_path):
    """
    Validate model.
    """
    print("\n" + "=" * 60)
    print("Validating model")
    print("=" * 60)

    model = YOLO(model_path)

    results = model.val(
        data=yaml_path,
        imgsz=640,
        batch=16,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("\nValidation results:")
    if hasattr(results, 'box'):
        print(f"  mAP@50: {results.box.map50:.4f}")
        print(f"  mAP@50-95: {results.box.map:.4f}")
        if hasattr(results.box, 'mp'):
            print(f"  Precision: {results.box.mp:.4f}")
        if hasattr(results.box, 'mr'):
            print(f"  Recall: {results.box.mr:.4f}")

    return results


def test_model(model_path, test_image=None):
    """
    Test model.
    """
    print("\n" + "=" * 60)
    print("Testing model")
    print("=" * 60)

    if not Path(model_path).exists():
        print(f"❌ Model does not exist: {model_path}")
        return

    model = YOLO(model_path)

    # If no test image is specified, select one from the validation set
    if test_image is None:
        val_images = list(Path('datasets/tray_yolo_clean/images/val').glob('*'))
        if val_images:
            test_image = str(val_images[0])
            print(f"Using validation set image: {test_image}")
        else:
            print("❌ No test image found")
            return

    print(f"Test image: {test_image}")

    # Inference
    results = model(
        test_image,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        save=True,
        save_txt=True
    )

    # Display results
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            print(f"\n✅ Detected {len(r.boxes)} seedling trays:")
            for i, box in enumerate(r.boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                print(
                    f"  Tray {i + 1}: Confidence={conf:.3f}, BBox=[{xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f}]")
        else:
            print("⚠️ No seedling trays detected")

    print(f"\nResults saved to: runs/detect/predict/")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv11 Seedling Tray Detection Training')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'validate'],
                        help='Execution mode')
    parser.add_argument('--source', type=str, default='datasets/tray_dataset',
                        help='Original dataset path')
    parser.add_argument('--model', type=str, default=None,
                        help='Model path (used during testing)')
    parser.add_argument('--image', type=str, default=None,
                        help='Test image path')
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='Skip pre-processing (if already processed)')

    args = parser.parse_args()

    # Processed dataset path
    processed_data = 'datasets/tray_yolo_clean'

    if args.mode == 'train':
        # Step 1: Pre-process data
        if not args.skip_preprocess or not Path(processed_data).exists():
            data_root = preprocess_labels(args.source, processed_data)
        else:
            data_root = processed_data
            print(f"✓ Using processed data: {data_root}")

        # Step 2: Train
        model, results, best_path = train_yolo(data_root)

        # Step 3: Validate
        yaml_path = 'tray_dataset.yaml'
        validate_model(best_path, yaml_path)

        # Step 4: Test
        test_model(best_path)

    elif args.mode == 'test':
        model_path = args.model or 'runs/detect/tray_detection/weights/best.pt'
        test_model(model_path, args.image)

    elif args.mode == 'validate':
        model_path = args.model or 'runs/detect/tray_detection/weights/best.pt'
        yaml_path = 'tray_dataset.yaml'
        validate_model(model_path, yaml_path)

    print("\n" + "=" * 60)
    print("✨ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()