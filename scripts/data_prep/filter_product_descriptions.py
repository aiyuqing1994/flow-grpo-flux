

import csv
import os
import argparse
import io

try:
    import requests
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Error: Required package '{e.name}' is not installed.")
    print("Please install it using: pip install requests tqdm matplotlib numpy")
    exit(1)

# --- Configuration ---
URL_COLUMN = "product_url"
DESCRIPTION_COLUMN = "product_showcase_description"
# ---------------------

def is_url_valid(url: str) -> bool:
    """
    Checks if a URL is valid and accessible.
    """
    if not url or not url.startswith(('http://', 'https://')):
        return False
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def is_invalid_prompt(description: str) -> bool:
    """
    Checks if a description contains multi-scene or multi-segment markers (case-insensitive).
    """
    if not description:
        return False
    lower_desc = description.lower()
    has_scene_markers = "scene 1" in lower_desc and "scene 2" in lower_desc
    has_segment_markers = "segment 1" in lower_desc and "segment 2" in lower_desc
    return has_scene_markers or has_segment_markers

def filter_csv(input_filename: str):
    """
    Filters the specified CSV file and returns the path to the cleaned file.
    """
    os.makedirs("data/cleaned", exist_ok=True)
    os.makedirs("data/deleted", exist_ok=True)

    input_filepath = os.path.join("data/raw", input_filename)
    cleaned_filepath = os.path.join("data/cleaned", input_filename)
    deleted_filepath = os.path.join("data/deleted", input_filename)

    try:
        with open(input_filepath, 'r', encoding='latin-1') as f:
            file_content = f.read().replace('\x00', '')
        infile = io.StringIO(file_content)
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        rows = list(reader)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return None

    if not fieldnames or URL_COLUMN not in fieldnames or DESCRIPTION_COLUMN not in fieldnames:
        print(f"Error: CSV must contain '{URL_COLUMN}' and '{DESCRIPTION_COLUMN}' columns.")
        return None

    with open(cleaned_filepath, 'w', newline='', encoding='utf-8') as cleaned_file, \
         open(deleted_filepath, 'w', newline='', encoding='utf-8') as deleted_file:
        cleaned_writer = csv.DictWriter(cleaned_file, fieldnames=fieldnames, extrasaction='ignore')
        deleted_writer = csv.DictWriter(deleted_file, fieldnames=fieldnames, extrasaction='ignore')
        cleaned_writer.writeheader()
        deleted_writer.writeheader()

        for row in tqdm(rows, desc=f"Filtering {input_filename}"):
            image_url = row.get(URL_COLUMN, "")
            description = row.get(DESCRIPTION_COLUMN, "")
            if is_url_valid(image_url) and not is_invalid_prompt(description):
                cleaned_writer.writerow(row)
            else:
                deleted_writer.writerow(row)
    
    print(f"\nFinished filtering {input_filename}.")
    print(f"Cleaned data saved to: {cleaned_filepath}")
    print(f"Deleted data saved to: {deleted_filepath}")
    return cleaned_filepath

def visualize_prompt_lengths(cleaned_filepath):
    """
    Generates a histogram of prompt lengths with dynamic tick spacing.
    """
    if not cleaned_filepath:
        print("Skipping visualization: Cleaned file not generated.")
        return

    print(f"\nGenerating prompt length visualization...")
    prompt_lengths = []
    with open(cleaned_filepath, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            description = row.get(DESCRIPTION_COLUMN, "")
            prompt_lengths.append(len(description.split()))

    if not prompt_lengths:
        print("No valid prompts found to visualize.")
        return

    max_len = max(prompt_lengths)
    
    # Determine a dynamic step for bins and ticks to avoid overcrowding
    if max_len > 500:
        step = 50
    elif max_len > 200:
        step = 20
    else:
        step = 10

    bins = np.arange(0, max_len + step, step)

    plt.figure(figsize=(12, 7))
    plt.hist(prompt_lengths, bins=bins, edgecolor='black')
    plt.title('Distribution of Prompt Lengths (Word Count)')
    plt.xlabel('Prompt Length (words)')
    plt.ylabel('Number of Prompts')
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(bins)
    plt.tight_layout() # Adjust layout to prevent labels overlapping

    output_path = os.path.splitext(cleaned_filepath)[0] + '_length_distribution.png'
    plt.savefig(output_path)
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter a CSV file by validating image URLs and checking for multi-scene text prompts."
    )
    parser.add_argument("input_file", help="The name of the input CSV file in the data/raw directory.")
    args = parser.parse_args()

    cleaned_file = filter_csv(args.input_file)
    if cleaned_file:
        visualize_prompt_lengths(cleaned_file)
