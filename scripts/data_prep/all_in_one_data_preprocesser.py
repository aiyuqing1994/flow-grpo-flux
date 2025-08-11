import csv
import os
import argparse
import io
import random
import numpy as np

try:
    import requests
    from tqdm import tqdm
    from PIL import Image
    from paddleocr import PaddleOCR
except ImportError as e:
    print(f"Error: Required package '{e.name}' is not installed.")
    print(f"Please ensure requests, tqdm, Pillow, and paddleocr are installed.")
    exit(1)

# --- Configuration ---
URL_COLUMN = "product_url"
IMAGE_DIR = "data/images"
CLEANED_DATA_DIR = "data/cleaned"
DELETED_DATA_DIR = "data/deleted"
DESCRIPTION_COLUMN = "product_showcase_description"
NEW_COLUMN_NAME = "paddle_ocr_detection"
# ---------------------

def get_image_path(image_url: str) -> str:
    """Constructs the local path for a given image URL."""
    filename = image_url.split("/")[-1]
    return os.path.join(IMAGE_DIR, filename)

def is_invalid_prompt(description: str) -> bool:
    """
    Checks if a description contains multi-scene or multi-segment markers (case-insensitive).
    """
    if not description:
        return True
    lower_desc = description.lower()
    has_scene_markers = "scene 1" in lower_desc and "scene 2" in lower_desc
    has_segment_markers = "segment 1" in lower_desc and "segment 2" in lower_desc
    return has_scene_markers or has_segment_markers

def download_verify_and_ocr(row: dict, ocr_engine, ocr_cache: dict) -> dict:
    """
    Downloads an image, verifies it, performs OCR with caching, and returns the row with an updated
    status ('processed' or 'deleted').
    """
    image_url = row.get(URL_COLUMN)
    if not image_url:
        row['status'] = 'deleted'
        row['reason'] = 'Missing image URL'
        return row

    # Check cache first
    if image_url in ocr_cache:
        cached_result = ocr_cache[image_url]
        if cached_result['status'] == 'processed':
            row.update(cached_result)
        else:
            row['status'] = 'deleted'
            row['reason'] = cached_result.get('reason', 'Cached as deleted')
        return row

    image_path = get_image_path(image_url)
    
    if os.path.exists(image_path):
        try:
            image = Image.open(image_path).convert("RGB")
        except (IOError, Image.UnidentifiedImageError) as e:
            row['status'] = 'deleted'
            row['reason'] = f"Could not open existing image: {e}"
            ocr_cache[image_url] = row
            return row
    else:
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise ValueError(f"URL is not an image: {content_type}")

            image_bytes = io.BytesIO(response.content)
            
            with Image.open(image_bytes) as img:
                img.verify()
            
            image_bytes.seek(0)
            image = Image.open(image_bytes).convert("RGB")

            with open(image_path, "wb") as f:
                f.write(image_bytes.read())
                
        except (requests.RequestException, IOError, Image.UnidentifiedImageError, ValueError) as e:
            row['status'] = 'deleted'
            row['reason'] = str(e)
            ocr_cache[image_url] = row
            return row

    # --- Perform OCR ---
    try:
        result = ocr_engine.ocr(np.array(image), cls=False)
        detected_text = ''
        if result and result[0]:
            recognized_text_parts = [res[1][0] for res in result[0]]
            detected_text = ' '.join(recognized_text_parts).strip()
        
        if not detected_text:
            row['status'] = 'deleted'
            row['reason'] = 'Empty OCR result'
        else:
            row[NEW_COLUMN_NAME] = detected_text
            row['status'] = 'processed'

    except Exception as e:
        row['status'] = 'deleted'
        row['reason'] = f"OCR failed: {e}"

    # Cache the result
    ocr_cache[image_url] = {
        'status': row['status'],
        'reason': row.get('reason'),
        NEW_COLUMN_NAME: row.get(NEW_COLUMN_NAME)
    }
        
    return row

def load_ocr_cache(cache_csv: str) -> dict:
    """Loads a CSV file into an OCR cache dictionary."""
    if not cache_csv or not os.path.exists(cache_csv):
        return {}
    
    print(f"Loading OCR cache from {cache_csv}...")
    cache = {}
    try:
        with open(cache_csv, 'r', encoding='latin-1') as f:
            file_content = f.read().replace('\x00', '')
        infile = io.StringIO(file_content)
        reader = csv.DictReader(infile)
        for row in reader:
            image_url = row.get(URL_COLUMN)
            ocr_text = row.get(NEW_COLUMN_NAME)
            if image_url and ocr_text:
                cache[image_url] = {
                    'status': 'processed',
                    NEW_COLUMN_NAME: ocr_text
                }
    except Exception as e:
        print(f"Warning: Could not load cache file: {e}")
    
    print(f"Loaded {len(cache)} entries into OCR cache.")
    return cache

def process_images_from_csv(input_csv: str, cache_csv: str = None):
    """
    Processes a CSV file to download, validate, and OCR images sequentially.
    Saves cleaned and deleted data into separate CSV files.
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
    os.makedirs(DELETED_DATA_DIR, exist_ok=True)

    base_filename = os.path.basename(input_csv)
    cleaned_output_path = os.path.join(CLEANED_DATA_DIR, base_filename)
    deleted_output_path = os.path.join(DELETED_DATA_DIR, base_filename)

    print(f"Processing {input_csv}...")
    print(f"Cleaned data will be saved to: {cleaned_output_path}")
    print(f"Deleted data will be saved to: {deleted_output_path}")

    try:
        with open(input_csv, 'r', encoding='latin-1') as f:
            file_content = f.read().replace('\x00', '')
        infile = io.StringIO(file_content)
        reader = csv.DictReader(infile)
        rows = list(reader)
        original_fieldnames = reader.fieldnames
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return

    new_fieldnames = original_fieldnames + [NEW_COLUMN_NAME, 'status', 'reason']
    
    # --- Initial Filtering and Deduplication ---
    processed_combinations = set()
    unique_rows = []
    deleted_rows = []
    for row in rows:
        description = row.get(DESCRIPTION_COLUMN, "")
        if is_invalid_prompt(description):
            row['status'] = 'deleted'
            row['reason'] = 'Invalid prompt'
            deleted_rows.append(row)
            continue

        image_url = row.get(URL_COLUMN)
        combination = (image_url, description)
        if combination in processed_combinations:
            row['status'] = 'deleted'
            row['reason'] = 'Duplicate entry'
            deleted_rows.append(row)
        else:
            processed_combinations.add(combination)
            unique_rows.append(row)

    random.shuffle(unique_rows)

    cleaned_rows = []
    ocr_cache = load_ocr_cache(cache_csv)
    
    print("Initializing PaddleOCR engine...")
    ocr_engine = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
    print("OCR engine loaded.")

    for row in tqdm(unique_rows, desc="Processing images"):
        processed_row = download_verify_and_ocr(row, ocr_engine, ocr_cache)
        status = processed_row.get('status')

        if status == 'processed':
            cleaned_rows.append(processed_row)
        else:
            deleted_rows.append(processed_row)

    # --- Writing Cleaned Data ---
    if cleaned_rows:
        with open(cleaned_output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=original_fieldnames + [NEW_COLUMN_NAME], extrasaction='ignore')
            writer.writeheader()
            writer.writerows(cleaned_rows)
        print(f"\nSaved {len(cleaned_rows)} cleaned entries to {cleaned_output_path}")

    # --- Writing Deleted Data ---
    if deleted_rows:
        with open(deleted_output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(deleted_rows)
        print(f"Saved {len(deleted_rows)} deleted entries to {deleted_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download, validate, and OCR images from a CSV file."
    )
    parser.add_argument(
        "input_csv", 
        help="The path to the input CSV file (e.g., data/raw/results20k.csv)."
    )
    parser.add_argument(
        "--cache_csv",
        help="Path to an existing CSV file to use as an OCR cache."
    )
    args = parser.parse_args()

    process_images_from_csv(args.input_csv, args.cache_csv)
