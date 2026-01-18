# Urine Test Strip Analyzer

## What it does
This app analyzes urine test strips from photos and tells you the levels of 10 different health markers.

## Features
- Upload photos of urine test strips
- Get instant analysis results using a web-based interface
- Shows confidence levels for each result

## Parameters Tested
The app analyzes these 10 parameters:
1. Leukocytes (white blood cells)
2. Nitrite (infection indicator)
3. Urobilinogen (liver function)
4. Protein (kidney health)
5. pH (acidity level)
6. Blood (hematuria)
7. Specific Gravity (concentration)
8. Ketone (metabolism)
9. Bilirubin (liver function)
10. Calcium (mineral balance)

## How to run
1. Install Python dependencies: `pip install opencv-python numpy pandas fastapi uvicorn python-multipart`
2. Run the app: `python app.py`
3. Open browser to `http://localhost:8000`

## Important Note
This is for educational purposes only. Not for medical diagnosis.
