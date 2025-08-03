import csv
import os
import argparse
import io

try:
    import requests
    from tqdm import tqdm
    import numpy as np
    from PIL import Image
    from paddleocr import PaddleOCR
except ImportError as e:
    print(f"Error: Required package '{e.name}' is not installed.")
    print(f"Please ensure requests, tqdm, numpy, Pillow, and paddleocr are installed.")
    exit(1)

# --- Configuration ---
URL_COLUMN = "product_url"
NEW_COLUMN_NAME = "paddle_ocr_detection"
PROCESSED_DATA_DIR = "data/processed"
# ---------------------

def perform_ocr_on_cleaned_data(cleaned_filepath: str):
    """
    Processes a cleaned CSV, performs OCR on each image, and saves to a new file
    in the processed data directory, skipping rows with no detected text.
    """
    if not os.path.exists(cleaned_filepath):
        print(f"Error: Input file not found at {cleaned_filepath}")
        return

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    input_filename = os.path.basename(cleaned_filepath)
    ocr_output_path = os.path.join(PROCESSED_DATA_DIR, input_filename)

    print(f"\nPerforming OCR on {cleaned_filepath}...")
    print(f"OCR results will be saved to: {ocr_output_path}")

    ocr_engine = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False)

    try:
        with open(cleaned_filepath, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            rows = list(reader)
            original_fieldnames = reader.fieldnames
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    new_fieldnames = original_fieldnames + [NEW_COLUMN_NAME]

    with open(ocr_output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=new_fieldnames, extrasaction='ignore')
        writer.writeheader()

        for row in tqdm(rows, desc="Performing OCR"):
            image_url = row.get(URL_COLUMN)
            detected_text = ''

            if image_url:
                try:
                    response = requests.get(image_url, stream=True, timeout=10)
                    response.raise_for_status()
                    image_bytes = io.BytesIO(response.content)
                    image = Image.open(image_bytes).convert("RGB")
                    
                    result = ocr_engine.ocr(np.array(image), cls=False)
                    
                    if result and result[0]:
                        recognized_text_parts = [res[1][0] for res in result[0]]
                        detected_text = ' '.join(recognized_text_parts).strip()

                except (requests.RequestException, IOError, Image.UnidentifiedImageError) as e:
                    print(f"\nCould not process image {image_url}: {e}")
                    detected_text = 'IMAGE_PROCESSING_ERROR'
            
            # Only write rows where OCR successfully detected text
            if detected_text and detected_text != 'IMAGE_PROCESSING_ERROR':
                row[NEW_COLUMN_NAME] = detected_text
                writer.writerow(row)

    print(f"\nFinished OCR processing. Results saved to {ocr_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform OCR on a cleaned CSV file and save it to the processed directory."
    )
    parser.add_argument(
        "cleaned_file", 
        help="The path to the cleaned CSV file (e.g., data/cleaned/results20k.csv)."
    )
    args = parser.parse_args()

    perform_ocr_on_cleaned_data(args.cleaned_file)