import cv2
import numpy as np
import pandas as pd
import os
import base64
from ultralytics import YOLO

#Configurations n stuff
PAD_ORDER = [
    "Leukocytes", "Nitrite", "Urobilinogen", "Protein", "pH",
    "Blood", "Specific Gravity", "Ketone", "Bilirubin", "Calcium"
]

DELTA_E_THRESHOLD = 15.0  #Delta_e confuses me

CSV_PATH = "urine-strip-colorchart2.csv" 

DEBUG_CROP_FOLDER = "parameter_crop" 

MODEL = YOLO("models/best.pt")

#1. IMAGE CAPTURE AND STANDARDIZATION
def preprocess_image(image_path):
    """Load and return the image without preprocessing."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not loaded")
    return img

#2. DETECTING/ISOLATING EACH TEST PAD

def create_pad_mask(roi):
    """Create HSV-based mask to isolate colored pad region."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    
    # Exclude very light pixels (background) and very dark pixels
    mask = (v > 50) & (v < 200) & (s > 30)
    return (mask * 255).astype(np.uint8)


def segment_pads(strip_img):
    """Detect and isolate test pads using YOLO model."""
    results = MODEL.predict(
        source=strip_img,
        conf=0.35,
        iou=0.45,
        imgsz=640,
        verbose=False
    )
    
    detections = results[0]
    
    if len(detections.boxes) == 0:
        raise ValueError("No test pads detected!")
    
    # Extract detections with coordinates and sort by x-position (left to right)
    pad_detections = []
    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        # Extract YOLO confidence score
        yolo_conf = float(box.conf[0]) if hasattr(box.conf, '__len__') and len(box.conf) > 0 else float(box.conf)
        if 0 <= class_id < len(PAD_ORDER):
            parameter = PAD_ORDER[class_id]
        else:
            parameter = f"unknown_class_{class_id}"
        pad_detections.append((x1, y1, x2, y2, parameter, yolo_conf))
    
    #Sort n crop
    pad_detections.sort(key=lambda d: d[0])

    pads = []
    for x1, y1, x2, y2, parameter, yolo_conf in pad_detections:
        # Crop pad ROI
        roi = strip_img[y1:y2, x1:x2]
        
        # Skip invalid crops
        if roi.size == 0:
            continue
        
        # Create HSV mask
        mask = create_pad_mask(roi)
        
        pads.append({
            "parameter": parameter,
            "roi": roi,
            "mask": mask,
            "yolo_confidence": yolo_conf,
            "box_coords": (x1, y1, x2, y2)
        })
    
    if not pads:
        raise ValueError("No valid pad crops created!")
    
    return pads


#3. COLOR EXTRACTION

def extract_lab_color(pad_data, min_pixels=500):
    roi = pad_data["roi"]
    mask = pad_data["mask"]

    valid_pixels = roi[mask == 255]
    if len(valid_pixels) < min_pixels:
        return None, "low_pixels"

    valid_lab = cv2.cvtColor(valid_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)

    median_lab = np.median(valid_lab, axis=0)

    std_lab = np.std(valid_lab, axis=0)
    if np.any(std_lab > 30):  # tune as needed
        return None, "high_variance"

    return median_lab.astype(float), "good"

# 4. COLOR → BIOMARKER ESTIMATION

def load_reference_chart(csv_path):
    df = pd.read_csv(csv_path, skiprows=1, names=['parameter', 'filename', 'R', 'G', 'B', 'range', 'value', 'normal_range', 'status'])
    refs = {}
    for param in PAD_ORDER:
        subset = df[df["parameter"] == param]
        lab_list = []
        for idx, row in subset.iterrows():
            rgb = np.uint8([[ [row["R"], row["G"], row["B"]] ]])
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0][0]
            normal_range_val = row.get("normal_range", "N/A")
            status_val = row.get("status", "Unknown")
            lab_list.append((lab.astype(float), row["value"], normal_range_val, status_val))
        refs[param] = lab_list
    return refs

REFERENCE_CHART = load_reference_chart(CSV_PATH)

def estimate_biomarker(median_lab, parameter):
    if parameter not in REFERENCE_CHART or not REFERENCE_CHART[parameter]:
        return "unknown_param", 0.0, "N/A", "Unknown"

    refs = REFERENCE_CHART[parameter]
    distances = []
    for ref_lab, value, normal_range, status in refs:
        delta_e = np.sqrt(np.sum((median_lab - ref_lab)**2))
        distances.append((delta_e, value, normal_range, status))

    min_dist, best_value, best_normal_range, best_status = min(distances, key=lambda x: x[0])

    if min_dist > DELTA_E_THRESHOLD:
        return f"{best_value} (unclear)", min_dist, best_normal_range, best_status

    return best_value, min_dist, best_normal_range, best_status

#DEBUG: SAVE PARAMETER CROPS

def save_debug_crops(pads, debug_folder=DEBUG_CROP_FOLDER):

    for i, pad in enumerate(pads):
        parameter_name = pad["parameter"]
        roi = pad["roi"]
        mask = pad["mask"]  
        
        safe_name = parameter_name.replace(" ", "_").replace("/", "_")
        prefix = f"{i+1:02d}_"

        #Raw ROI
        raw_filename = f"{prefix}raw_{safe_name}.jpg"
        raw_filepath = os.path.join(debug_folder, raw_filename)
        cv2.imwrite(raw_filepath, roi)
        
        #Mask
        mask_filename = f"{prefix}mask_{safe_name}.jpg"
        mask_filepath = os.path.join(debug_folder, mask_filename)
        cv2.imwrite(mask_filepath, mask)
        
        #Masked version
        masked_roi = roi.copy()
        #Set pixels where mask == 0 to black
        masked_roi[mask == 0] = [0, 0, 0]
        
        masked_filename = f"{prefix}masked_{safe_name}.jpg"
        masked_filepath = os.path.join(debug_folder, masked_filename)
        cv2.imwrite(masked_filepath, masked_roi)
    
    return debug_folder


def draw_boxes_on_image(strip_img, pads):
    """Draw bounding boxes on the original image (no labels). Returns base64-encoded JPEG."""
    img_copy = strip_img.copy()
    for pad in pads:
        coords = pad.get("box_coords")
        if coords:
            x1, y1, x2, y2 = coords
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    _, buffer = cv2.imencode(".jpg", img_copy)
    return base64.b64encode(buffer).decode("utf-8")


#FULL PIPELINE

def analyze_urine_strip(image_path, save_debug=True):
    strip = preprocess_image(image_path)
    pads = segment_pads(strip)
    #Debug crops
    if save_debug:
        try:
            debug_folder = save_debug_crops(pads)
        except Exception as e:
            print(f"Warning: Failed to save debug crops: {e}")

    results = []
    for pad in pads:
        lab_color, status = extract_lab_color(pad)
        # YOLO confidence from pad or None if not available
        yolo_conf = pad.get("yolo_confidence", None)
        
        if status != "good":
            results.append({
                "parameter": pad["parameter"],
                "value": "Rejected",
                "reason": status,
                "delta_e": None,
                "yolo_confidence": yolo_conf,
                "color_confidence": "low",
                "normal_range": "N/A",
                "status": "Unknown"
            })
            continue

        value, dist, normal_range, status_val = estimate_biomarker(lab_color, pad["parameter"])
        color_confidence = "high" if dist < DELTA_E_THRESHOLD / 2 else "medium" if dist < DELTA_E_THRESHOLD else "low"

        results.append({
            "parameter": pad["parameter"],
            "value": value,
            "delta_e": round(dist, 2),
            "yolo_confidence": yolo_conf,
            "color_confidence": color_confidence,
            "normal_range": normal_range,
            "status": status_val
        })

    annotated_image_b64 = draw_boxes_on_image(strip, pads)
    return {"results": results, "annotated_image": annotated_image_b64}
