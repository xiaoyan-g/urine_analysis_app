import cv2
import numpy as np
import pandas as pd
import os

#Configurations n stuff
PAD_ORDER = [
    "Leukocytes", "Nitrite", "Urobilinogen", "Protein", "pH",
    "Blood", "Specific Gravity", "Ketone", "Bilirubin", "Calcium"
]

PAD_RELATIVE_STARTS = [
    0.065,  # Leukocytes
    0.145,  # Nitrite
    0.225,  # Urobilinogen
    0.305,  # Protein
    0.385,  # pH
    0.465,  # Blood
    0.545,  # Specific Gravity
    0.625,  # Ketone
    0.705,  # Bilirubin
    0.785   # Calcium
]
PAD_HEIGHT_FRACTION = 0.06  #~8% of total height

DELTA_E_THRESHOLD = 15.0  #Delta_e confuses me

CSV_PATH = "urine-strip-colorchart2.csv" 

DEBUG_CROP_FOLDER = "parameter_crop" 

#1. IMAGE CAPTURE AND STANDARDIZATION
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not loaded")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    corner_pixels = np.concatenate([
        gray[0:50, 0:50].ravel(),
        gray[0:50, w-50:w].ravel(),
        gray[h-50:h, 0:50].ravel(),
        gray[h-50:h, w-50:w].ravel()
    ])
    bg_brightness = np.mean(corner_pixels)

    #white background (> bg_brightness - tolerance)
    _, thresh = cv2.threshold(gray, int(bg_brightness - 30), 255, cv2.THRESH_BINARY_INV)

    #tallest vertical
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No strip detected")

    candidates = []
    for cnt in contours:
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
        aspect = h_cnt / float(w_cnt) if w_cnt > 0 else 0
        if aspect > 8 and h_cnt > 0.7 * h:  
            candidates.append((h_cnt, x, y, w_cnt, h_cnt))

    if not candidates:
        raise ValueError("No valid strip found")

    _, x, y, w_cnt, h_cnt = max(candidates, key=lambda t: t[0])

    #Crop
    margin = int(0.05 * w_cnt)
    cropped = img[max(0, y - margin): y + h_cnt + margin,
                  max(0, x - margin): x + w_cnt + margin]

    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    bg_samples = np.mean([
        cropped_gray[10:60, 10:60],
        cropped_gray[10:60, -60:-10],
        cropped_gray[-60:-10, 10:60],
        cropped_gray[-60:-10, -60:-10]
    ])
    scale_factor = 255.0 / (bg_samples + 1e-6)
    cropped_balanced = np.clip(cropped.astype(np.float32) * scale_factor, 0, 255).astype(np.uint8)

    return cropped_balanced


#2. DETECTING/ISOLATING EACH TEST PAD

def segment_pads(strip_img):
    h, w, _ = strip_img.shape
    pads = []

    for i, start_frac in enumerate(PAD_RELATIVE_STARTS):
        y_start = int(start_frac * h)
        y_end = int((start_frac + PAD_HEIGHT_FRACTION) * h)
        
        x_start = int(0.15 * w)
        x_end = int(0.85 * w)

        roi = strip_img[y_start:y_end, x_start:x_end].copy()

        # Convert to LAB
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

        #exclude near-white and very dark
        lower = np.array([20, 100, 100], dtype=np.uint8)
        upper = np.array([255, 200, 200], dtype=np.uint8)
        mask = cv2.inRange(lab_roi, lower, upper)

        #Morphology on mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        #Largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            final_mask = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(final_mask, [largest], -1, 255, -1)
        else:
            final_mask = mask

        pads.append({
            "parameter": PAD_ORDER[i],
            "roi": roi,
            "mask": final_mask,
            "y_range": (y_start, y_end)
        })

    return pads


#3. COLOR EXTRACTION

def extract_lab_color(pad_data, min_pixels=500):
    roi = pad_data["roi"]
    mask = pad_data["mask"]

    valid_pixels = roi[mask == 255]
    if len(valid_pixels) < min_pixels:
        return None, "low_pixels"

    #Convert valid pixels to LAB
    valid_lab = cv2.cvtColor(valid_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)

    #Median
    median_lab = np.median(valid_lab, axis=0)

    # variance check
    std_lab = np.std(valid_lab, axis=0)
    if np.any(std_lab > 30):  # tune as needed
        return None, "high_variance"

    return median_lab.astype(float), "good"

# 4. COLOR → BIOMARKER ESTIMATION

def load_reference_chart(csv_path):
    df = pd.read_csv(csv_path, header=None, names=['parameter', 'filename', 'R', 'G', 'B', 'range', 'value'])
    refs = {}
    for param in PAD_ORDER:
        subset = df[df["parameter"] == param]
        lab_list = []
        for _, row in subset.iterrows():
            rgb = np.uint8([[ [row["R"], row["G"], row["B"]] ]])
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0][0]
            lab_list.append((lab.astype(float), row["value"]))
        refs[param] = lab_list
    return refs

REFERENCE_CHART = load_reference_chart(CSV_PATH)

def estimate_biomarker(median_lab, parameter):
    if parameter not in REFERENCE_CHART or not REFERENCE_CHART[parameter]:
        return "unknown_param", 0.0

    refs = REFERENCE_CHART[parameter]
    distances = []
    for ref_lab, value in refs:
        delta_e = np.sqrt(np.sum((median_lab - ref_lab)**2))
        distances.append((delta_e, value))

    min_dist, best_value = min(distances, key=lambda x: x[0])

    if min_dist > DELTA_E_THRESHOLD:
        return f"{best_value} (unclear)", min_dist

    return best_value, min_dist

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
        
        #Masked version
        masked_filename = f"{prefix}masked_{safe_name}.jpg"
        masked_filepath = os.path.join(debug_folder, masked_filename)
        cv2.imwrite(masked_filepath, masked_roi)
    
    return debug_folder

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
        if status != "good":
            results.append({
                "parameter": pad["parameter"],
                "value": "Rejected",
                "reason": status,
                "confidence": "low"
            })
            continue

        value, dist = estimate_biomarker(lab_color, pad["parameter"])
        confidence = "high" if dist < DELTA_E_THRESHOLD / 2 else "medium" if dist < DELTA_E_THRESHOLD else "low"

        results.append({
            "parameter": pad["parameter"],
            "value": value,
            "delta_e": round(dist, 2),
            "confidence": confidence
        })

    return results
