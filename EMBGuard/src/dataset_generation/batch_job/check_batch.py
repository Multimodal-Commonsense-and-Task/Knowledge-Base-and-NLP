#!/usr/bin/env python3
"""
Google GenAI Batch job status check script
Usage: python check_batch.py [batch_name] [--output_dir OUTPUT_DIR]
"""

import sys
import os
import argparse
import dotenv
from pathlib import Path
from google import genai

dotenv.load_dotenv()

# Add project root to Python path (needed before importing utils)
current_dir = os.path.dirname(os.path.abspath(__file__))
# check_batch.py is at: src/dataset_generation/batch_job/check_batch.py
# Need to go up 3 levels to reach project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import config utilities
try:
    from utils.config import get_config
except ImportError:
    def get_config():
        return {}


def check_batch_status(batch_name, output_dir=None):
    """Check batch job status"""
    try:
        # Set API key - check config.yaml first, then environment variable
        config = get_config()
        api_key = config.get("gemini", {}).get("key") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("❌ Gemini API key is not set.")
            print("   Please set config.yaml (gemini.key) or GOOGLE_API_KEY environment variable.")
            print(f"   Debug: config loaded: {bool(config)}, gemini section: {bool(config.get('gemini'))}")
            return
            
        client = genai.Client(api_key=api_key)
        
        # Get batch job information
        print(f"Checking batch job: {batch_name}")
        job = client.batches.get(name=batch_name)
        
        # Output status
        print(f"\nBatch job status")
        print(f"  Status: {job.state}")
        print(f"  Created time: {job.create_time}")
        
        if hasattr(job, 'start_time') and job.start_time:
            print(f"  Start time: {job.start_time}")
        if hasattr(job, 'end_time') and job.end_time:
            print(f"  End time: {job.end_time}")
            
        # Save results
        if hasattr(job, 'dest') and job.dest:
            result_file_name = job.dest.file_name
            print(f"Downloading result file: {result_file_name}")
            file_content_bytes = client.files.download(file=result_file_name)
            file_content = file_content_bytes.decode('utf-8')
            
            # Set output directory
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                save_path = output_path / f"batch_result_{job.name.split('/')[-1]}.jsonl"
            else:
                save_path = f"batch_result_{job.name.split('/')[-1]}.jsonl"
            
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(file_content)
            print(f"Result file saved: {save_path}")
        else:
            print(f"\n📁 Result file: Not yet created")
            
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("   Batch job may not exist or there may be an issue with API key.")

def parse_args():
    parser = argparse.ArgumentParser(description='Check Google GenAI Batch job status')
    parser.add_argument('batch_name', nargs='?', help='Batch job name (e.g., batches/xxx)')
    parser.add_argument('--output_dir', '-o', type=str, help='Directory to save result file')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if not args.batch_name:
        # Default: find status files in current directory
        status_files = []
        for root, dirs, files in os.walk("."):
            for file in files:
                if file == "current_batch_status.json":
                    status_files.append(os.path.join(root, file))
        
        if status_files:
            import json
            print(f"🔍 Found batch status files: {len(status_files)}")
            for status_file in status_files:
                try:
                    with open(status_file, 'r') as f:
                        status = json.load(f)
                        batch_name = status.get('job_name')
                        if batch_name:
                            print(f"\n📂 {status_file}")
                            check_batch_status(batch_name, args.output_dir)
                except Exception as e:
                    print(f"❌ Failed to read {status_file}: {e}")
        else:
            print("Usage: python check_batch.py [batch_name] [--output_dir OUTPUT_DIR]")
            print("Example: python check_batch.py batches/pjsxqx05e6ccavnqhlljn65b76bfh1wgrokl --output_dir /path/to/results")
    else:
        check_batch_status(args.batch_name, args.output_dir)
