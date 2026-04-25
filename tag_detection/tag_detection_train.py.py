"""
Official Implementation for Paper:
"End-Edge-Cloud Collaborative Group Phenotype Recognition and Diagnosis for Rice Seedlings in Vertical Rice Seedlings Cultivation"

Module: Perception Front-end (Identification Tag Detection)
Original Script Description: 
YOLO training script explicitly optimized for small object (identification tag) detection.
"""

import sys
import argparse
import os
from ultralytics import YOLO

def main(opt):
    # Load model configuration file
    yaml = opt.cfg  
    model = YOLO(yaml)  # Load yaml file directly for training

    # Set model scale to large for better small object feature extraction
    model.model.yaml['scale'] = 'large'  

    # Output model information
    model.info()  

    # Train the model
    results = model.train( 
        data=r'tag_dataset.yaml',  # Path to the dataset configuration file
        epochs=170,                # Set training epochs to 170
        imgsz=832,                 # Increase image size to 832 (improves small object detection precision)
        batch=64,                  # Increase batch size to 64 (improves training efficiency)
        workers=12,                # Set number of worker threads to 12
        device='',                 # Leave empty to auto-select device (CPU/GPU)
        optimizer='Adam',          # Use Adam optimizer
        lr0=0.0001,                # Set initial learning rate to 0.0001 (prevents overfitting)
        lrf=0.1,                   # Set learning rate decay factor to 0.1
        weight_decay=0.0005,       # Set weight decay to 0.0005 (increases regularization)
        project='runs/train',      # Set project save path
        name='exp',                # Set experiment name
        amp=True,                  # Use Automatic Mixed Precision (AMP) training
        label_smoothing=0.1,       # Set label smoothing value to 0.1
    )

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov8.yaml', help='model.yaml path')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)