#!/usr/bin/env python3
"""
Download images from batch job results

This script:
- Reads the batch job results
- Downloads images from the results
- Saves images to the output directory
"""

import argparse
import base64
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Download images from batch job results')
    parser.add_argument('--jsonl_path', type=str, required=True, help='Path to the batch job results')
    parser.add_argument('--output_dir', type=str, default='/home/dongwook/vllm_inference/images', help='Path to the output directory')
    return parser.parse_args()


def read_jsonl(file_path):
    """Read JSONL file and return as list"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def download_image_from_response(response_data, output_dir=None, filename_prefix=None):
    """
    Extract and save image from response data
    
    Args:
        response_data: response dictionary
        output_dir: Directory to save images
        filename_prefix: Filename prefix
    
    Returns:
        Saved file path or None
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path.cwd()
    
    try:
        first_candidate = response_data['response']['candidates'][0]
        response = first_candidate['content']['parts'][0]['inlineData']
        
        file_extension = response['mimeType'].split('/')[-1]  # jpeg
        
        if filename_prefix:
            filename = output_dir / f"{filename_prefix}.{file_extension}"
        else:
            filename = output_dir / f"downloaded_image.{file_extension}"
        
        # Base64 decode
        image_data = base64.b64decode(response['data'])
        
        # Save file in binary write mode
        with open(filename, 'wb') as f:
            f.write(image_data)
        
        print(f"Image saved to '{filename}'.")
        return str(filename)
        
    except Exception as e:
        print(f"Error occurred while saving: {e}")
        return None


def main():
    args = parse_args()
    
    jsonl_path = args.jsonl_path
    output_dir = args.output_dir
    
    print(f"Reading JSONL file: {jsonl_path}")
    data = read_jsonl(jsonl_path)
    print(f"Total {len(data)} records found.")
    
    if not data:
        print("No data found.")
        return
    
    print(f"Extracting images from all records... (output directory: {output_dir})")
    success_count = 0
    fail_count = 0
    
    for idx, record in enumerate(data):
        # Use custom_id if available, otherwise use index
        filename_prefix = record.get('custom_id', f'image_{idx}')
        
        result = download_image_from_response(record, output_dir=output_dir, filename_prefix=filename_prefix)
        if result:
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\nComplete: {success_count} succeeded, {fail_count} failed")


if __name__ == '__main__':
    main()