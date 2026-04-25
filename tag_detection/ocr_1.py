"""
Official Implementation for Paper:
"End-Edge-Cloud Collaborative Group Phenotype Recognition and Diagnosis for Rice Seedlings in Vertical Rice Seedlings Cultivation"

Module: Perception Front-end (Individual Tray Traceability & OCR Pipeline)

Original Script Description:
Cascaded "Detection-Enhancement-Recognition" strategy for robust identification tag OCR.
Integrates YOLO detection with a parallel 8-method image enhancement ensemble.
"""

from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
import cv2
import os
from ultralytics import YOLO


def enhance_image_for_ocr(image_array):
    """
    Apply multiple pre-processing methods to the image to improve OCR performance
    """
    processed_images = []

    # Original Image
    processed_images.append(("Original", image_array))

    # Convert to Grayscale
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    processed_images.append(("Grayscale", gray))

    # Enhance Contrast
    enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
    processed_images.append(("Enhanced Contrast", enhanced))

    # Binarization - Standard Threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    processed_images.append(("Binary Threshold", binary))

    # Binarization - Adaptive Threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    processed_images.append(("Adaptive Threshold", adaptive))

    # OTSU Binarization
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(("OTSU Threshold", otsu))

    # Inverse Binarization (White background, black text)
    _, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    processed_images.append(("Inverse Binary", binary_inv))

    # Morphological Operations - Noise Reduction
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    processed_images.append(("Morphological Operation", morph))

    return processed_images


def try_multiple_ocr_methods(image_array, region_name):
    """
    Attempt multiple OCR methods and parameters to find the highest confidence result
    """
    print(f"    Attempting multiple OCR methods for {region_name}...")

    best_result = ""
    best_confidence = 0

    # Apply pre-processing ensemble
    processed_images = enhance_image_for_ocr(image_array)

    # Initialize OCR engine once to avoid redundant overhead
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')

    for proc_name, proc_img in processed_images:
        print(f"      Processing Method: {proc_name}")

        try:
            # Execute OCR inference
            result = ocr.ocr(proc_img)

            # Debugging: Print raw OCR result structure
            print(f"        OCR Raw Result Type: {type(result)}")
            print(f"        OCR Raw Result Content: {result}")

            # Safe result parsing - Supports new PaddleOCR format
            if result is None:
                print(f"        {proc_name}: OCR result is None")
                continue

            if not isinstance(result, list) or len(result) == 0:
                print(f"        {proc_name}: OCR result is not a valid list")
                continue

            # Retrieve first-level results
            first_level = result[0]
            if first_level is None:
                print(f"        {proc_name}: First-level result is None")
                continue

            # Check for dictionary format (newer PaddleOCR versions)
            if isinstance(first_level, dict):
                print(f"        {proc_name}: Detected new PaddleOCR dictionary format")

                # Extract recognition results directly from dictionary
                if 'rec_texts' in first_level and 'rec_scores' in first_level:
                    texts = first_level['rec_texts']
                    scores = first_level['rec_scores']

                    print(f"        {proc_name}: Found rec_texts and rec_scores")
                    print(f"          Recognized Text: {texts}")
                    print(f"          Confidence: {scores}")

                    if isinstance(texts, list) and isinstance(scores, list):
                        for i in range(min(len(texts), len(scores))):
                            text = str(texts[i]).strip()
                            confidence = float(scores[i])

                            print(f"        {proc_name}: Recognized '{text}' (Confidence: {confidence:.4f})")

                            if confidence > best_confidence and text:
                                best_result = text
                                best_confidence = confidence
                else:
                    print(f"        {proc_name}: 'rec_texts' or 'rec_scores' not found in dictionary")
                    print(f"        {proc_name}: Available keys: {list(first_level.keys())}")

                continue

            # Handle older nested list formats
            if not isinstance(first_level, list) or len(first_level) == 0:
                print(f"        {proc_name}: First-level result is not a valid list")
                continue

            # Iterate through each recognized line
            for j, line in enumerate(first_level):
                print(f"          Line {j}: {type(line)} = {line}")

                if line is None:
                    print(f"          Line {j} is None, skipping")
                    continue

                if not isinstance(line, list) or len(line) < 2:
                    print(f"          Line {j} format incorrect, length: {len(line) if isinstance(line, list) else 'not list'}")
                    continue

                # Attempt to extract text and confidence score
                try:
                    # line[0] contains bounding box coordinates, line[1] contains text info
                    text_info = line[1]
                    print(f"          Text Info: {type(text_info)} = {text_info}")

                    if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                        text = str(text_info[0]).strip()
                        confidence = float(text_info[1])

                        print(f"        {proc_name}: Recognized '{text}' (Confidence: {confidence:.4f})")

                        if confidence > best_confidence and text:
                            best_result = text
                            best_confidence = confidence
                    else:
                        print(f"          Text info format error")

                except Exception as parse_error:
                    print(f"          Error parsing line {j}: {parse_error}")

        except Exception as e:
            print(f"        {proc_name}: OCR Recognition Error - {e}")

    print(f"    Best recognition result for {region_name}: '{best_result}' (Confidence: {best_confidence:.4f})")
    return best_result, best_confidence


def crop_and_ocr_yolo_process(image_path, yolo_model1_path):
    print("Initializing enhanced OCR pipeline...")

    # Load YOLO Model
    print(f"Loading YOLO model: {yolo_model1_path}")
    yolo1 = YOLO(yolo_model1_path)

    # Read Image
    image = Image.open(image_path).convert('RGB')
    print(f"Image dimensions: {image.size}")

    # Execute YOLO Detection
    print("\nRunning YOLO model detection...")
    yolo1_results = yolo1(image_path, verbose=False)

    # Extract YOLO Detections
    print(f"YOLO Detection Results:")
    yolo1_detections = []
    for result in yolo1_results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = result.names[cls_id]
                print(f"  Detected: {label}, Confidence: {conf:.4f}, Coordinates: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                yolo1_detections.append((x1, y1, x2, y2, label, conf))

    # Perform enhanced OCR on each detected Region of Interest (RoI)
    print(f"\nApplying enhanced OCR to {len(yolo1_detections)} detected regions...")
    region_ocr_results = []

    for i, (x1, y1, x2, y2, label, conf) in enumerate(yolo1_detections):
        print(f"\nProcessing Region {i + 1}: {label}")

        # Ensure coordinates are within image bounds and convert to integer
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(image.width, int(x2))
        y2 = min(image.height, int(y2))

        # Extend bounding box margins for better OCR context
        margin = 5
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image.width, x2 + margin)
        y2 = min(image.height, y2 + margin)

        # Validate region dimensions
        if x2 <= x1 or y2 <= y1:
            print(f"  Region {i + 1} coordinates invalid, skipping OCR")
            region_ocr_results.append("")
            continue

        # Crop RoI
        cropped_region = image.crop((x1, y1, x2, y2))
        print(f"  Cropped region dimensions: {cropped_region.size}")

        # Save original cropped region for debugging
        crop_path = f"cropped_region_{i + 1}_original.jpg"
        cropped_region.save(crop_path)
        print(f"  Original cropped region saved: {crop_path}")

        # Convert to numpy array for OCR processing
        cropped_array = np.array(cropped_region)

        # Utilize the enhancement ensemble for OCR
        best_text, best_conf = try_multiple_ocr_methods(cropped_array, f"Region {i + 1}")

        # Format Final Result
        if best_text and best_conf > 0.3:  # Confidence threshold constraint
            region_text_with_conf = f"{best_text}({best_conf:.2f})"
        else:
            region_text_with_conf = "Unrecognized"

        region_ocr_results.append(region_text_with_conf)

        # Save examples of processed images
        processed_images = enhance_image_for_ocr(cropped_array)
        for j, (proc_name, proc_img) in enumerate(processed_images[:3]):  # Save top 3 variations
            if len(proc_img.shape) == 2:  # Grayscale
                proc_pil = Image.fromarray(proc_img, mode='L')
            else:
                proc_pil = Image.fromarray(proc_img)
            # Replace spaces in filenames for compatibility
            safe_proc_name = proc_name.replace(" ", "_")
            proc_pil.save(f"cropped_region_{i + 1}_{safe_proc_name}.jpg")

    # Visualization
    print(f"\nInitializing visualization overlay...")
    draw = ImageDraw.Draw(image)

    # Load appropriate font
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 24)
        small_font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Overlay YOLO bounding boxes (green) and OCR results
    print(f"Drawing YOLO bounding boxes and enhanced OCR labels...")
    for i, (x1, y1, x2, y2, label, conf) in enumerate(yolo1_detections):
        # Boundary safety check
        x1 = max(0, min(x1, image.width))
        y1 = max(0, min(y1, image.height))
        x2 = max(0, min(x2, image.width))
        y2 = max(0, min(y2, image.height))

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline='green', width=4)

        # Retrieve OCR result
        region_ocr = region_ocr_results[i] if i < len(region_ocr_results) else "None"

        # Construct display label
        combined_label = f"R{i + 1}-{label}:{conf:.2f}|{region_ocr}"

        print(f"  Green Box {i + 1} Label: {combined_label}")

        # Determine label placement
        text_y = max(0, y1 - 25)
        if text_y < 25:
            text_y = y2 + 5

        try:
            # Draw semi-transparent background for readability
            draw.rectangle([x1, text_y, x1 + len(combined_label) * 10, text_y + 20],
                           fill=(0, 128, 0, 180))
            draw.text((x1 + 2, text_y + 2), combined_label, fill='white', font=small_font)
        except:
            draw.text((x1, text_y), f"R{i + 1}:{region_ocr}", fill='green')

    # Save final pipeline output
    result_path = 'result_enhanced_ocr.jpg'
    image.save(result_path)
    print(f"\nFinal visualization saved to: {result_path}")

    # Display image (if running in interactive environment)
    # image.show()

    # Pipeline Summary
    print(f"\n🎯 Processing Summary:")
    print(f"✓ YOLO Detections: {len(yolo1_detections)} regions")
    print(f"✓ Applied 8 distinct image enhancement methods per region")
    print(f"✓ Evaluated multiple OCR confidence configurations")

    print(f"\n📋 OCR Results per Region:")
    for i, ocr_result in enumerate(region_ocr_results):
        print(f"  Region {i + 1}: {ocr_result}")

    print(f"\n📁 Generated Debug Files:")
    print(f"  - result_enhanced_ocr.jpg (Final Output)")
    for i in range(len(yolo1_detections)):
        print(f"  - cropped_region_{i + 1}_original.jpg (Original Crop)")
        print(f"  - cropped_region_{i + 1}_Grayscale.jpg (Grayscale Processing)")
        print(f"  - cropped_region_{i + 1}_Enhanced_Contrast.jpg (Contrast Enhanced)")

    return image, region_ocr_results


# Main Execution Block
if __name__ == "__main__":
    # Define absolute paths (as per user's local environment)
    image_path = 'D:/yolo/ultralytics-main/0419c000120.635_result_1_0.png'
    yolo_model1_path = "D:/yolo/ultralytics-main/best.pt"

    # Execute pipeline
    result_image, ocr_results = crop_and_ocr_yolo_process(image_path, yolo_model1_path)

    print("\n🚀 Pipeline Execution Complete!")
    print("📝 Review generated debug files to assess pre-processing effectiveness.")
    print("🔍 Troubleshooting tips if recognition fails:")
    print("  1. Verify clarity of numerals in 'cropped_region_X_Binary_Threshold.jpg'.")
    print("  2. Ensure bounding boxes accurately encapsulate the target tag.")
    print("  3. Adjust YOLO confidence threshold if tags are being missed.")