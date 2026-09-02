#!/usr/bin/env python3
"""
Google GenAI Batch job cancellation script
Usage: 
  python cancel_batch.py --iterate_name train
  python cancel_batch.py --batch_job batches/xxx
  python cancel_batch.py --timestamp 1234567890
"""

import sys
import os
import argparse
import json
import glob
from pathlib import Path
from google import genai

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from utils.config import get_config
except ImportError:
    def get_config():
        return {}


def get_api_key():
    """Get API key from config or environment"""
    config = get_config()
    api_key = config.get("gemini", {}).get("key") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key is not set.\n"
            "Please set config.yaml (gemini.key) or GOOGLE_API_KEY environment variable."
        )
    return api_key


def find_batch_jobs_from_iterate(iterate_name, results_dir="dataset_generation_output"):
    """Find all batch jobs from iterate directory"""
    iterate_dir = os.path.join(results_dir, iterate_name)
    batch_base_dir = os.path.join(iterate_dir, "raw", "batch")
    
    if not os.path.exists(batch_base_dir):
        return []
    
    batch_jobs = []
    
    # Find all timestamp directories
    timestamp_dirs = [d for d in os.listdir(batch_base_dir) if os.path.isdir(os.path.join(batch_base_dir, d))]
    
    for timestamp in timestamp_dirs:
        timestamp_dir = os.path.join(batch_base_dir, timestamp)
        
        # Find all job_*.txt files
        job_files = glob.glob(os.path.join(timestamp_dir, "job_*.txt"))
        
        for job_file in job_files:
            try:
                with open(job_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract job ID from content
                    for line in content.split('\n'):
                        if line.startswith('Job ID:'):
                            job_id = line.split('Job ID:')[1].strip()
                            batch_jobs.append({
                                'job_id': job_id,
                                'job_file': job_file,
                                'timestamp': timestamp
                            })
                            break
            except Exception as e:
                print(f"Warning: Could not read {job_file}: {e}")
    
    return batch_jobs


def find_batch_jobs_from_timestamp(timestamp, results_dir="dataset_generation_output"):
    """Find all batch jobs from a specific timestamp"""
    # Search in all iterate directories
    results_path = Path(results_dir)
    if not results_path.exists():
        return []
    
    batch_jobs = []
    
    for iterate_dir in results_path.iterdir():
        if not iterate_dir.is_dir():
            continue
        
        batch_base_dir = iterate_dir / "raw" / "batch" / str(timestamp)
        if not batch_base_dir.exists():
            continue
        
        # Find all job_*.txt files
        job_files = list(batch_base_dir.glob("job_*.txt"))
        
        for job_file in job_files:
            try:
                with open(job_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for line in content.split('\n'):
                        if line.startswith('Job ID:'):
                            job_id = line.split('Job ID:')[1].strip()
                            batch_jobs.append({
                                'job_id': job_id,
                                'job_file': str(job_file),
                                'timestamp': timestamp,
                                'iterate_name': iterate_dir.name
                            })
                            break
            except Exception as e:
                print(f"Warning: Could not read {job_file}: {e}")
    
    return batch_jobs


def cancel_batch_job(job_id, client):
    """Cancel a single batch job"""
    try:
        print(f"  Cancelling: {job_id}")
        job = client.batches.cancel(name=job_id)
        print(f"  ✓ Cancellation complete: {job.state}")
        return True
    except Exception as e:
        print(f"  ✗ Cancellation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Cancel Google GenAI Batch job')
    parser.add_argument('--iterate_name', type=str, help='Iterate name (e.g., train, debug)')
    parser.add_argument('--batch_job', type=str, help='Specific batch job ID (e.g., batches/xxx)')
    parser.add_argument('--timestamp', type=str, help='Timestamp to find batch jobs')
    parser.add_argument('--results_dir', type=str, default='dataset_generation_output',
                       help='Results directory (default: dataset_generation_output)')
    parser.add_argument('--dry_run', action='store_true', help='Show what would be cancelled without actually cancelling')
    
    args = parser.parse_args()
    
    # Get API key
    try:
        api_key = get_api_key()
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    client = genai.Client(api_key=api_key)
    
    batch_jobs = []
    
    # Find batch jobs based on arguments
    if args.batch_job:
        # Single batch job
        batch_jobs = [{'job_id': args.batch_job}]
    elif args.iterate_name:
        # Find from iterate name
        print(f"🔍 Finding batch jobs for iterate: {args.iterate_name}")
        batch_jobs = find_batch_jobs_from_iterate(args.iterate_name, args.results_dir)
    elif args.timestamp:
        # Find from timestamp
        print(f"🔍 Finding batch jobs for timestamp: {args.timestamp}")
        batch_jobs = find_batch_jobs_from_timestamp(args.timestamp, args.results_dir)
    else:
        print("❌ Error: One of --iterate_name, --batch_job, or --timestamp must be provided")
        parser.print_help()
        return
    
    if not batch_jobs:
        print("❌ No batch jobs found")
        return
    
    print(f"\n📋 Found {len(batch_jobs)} batch job(s):")
    for i, job_info in enumerate(batch_jobs, 1):
        job_id = job_info['job_id']
        print(f"  {i}. {job_id}")
        if 'iterate_name' in job_info:
            print(f"     Iterate: {job_info['iterate_name']}")
        if 'timestamp' in job_info:
            print(f"     Timestamp: {job_info['timestamp']}")
    
    if args.dry_run:
        print("\n🔍 Dry run mode - no jobs will be cancelled")
        return
    
    # Confirm cancellation
    print(f"\n⚠️  Are you sure you want to cancel {len(batch_jobs)} batch job(s)?")
    response = input("Type 'yes' to confirm: ")
    
    if response.lower() != 'yes':
        print("Cancelled by user")
        return
    
    # Cancel batch jobs
    print(f"\n🚫 Cancelling batch jobs...")
    success_count = 0
    fail_count = 0
    
    for job_info in batch_jobs:
        job_id = job_info['job_id']
        if cancel_batch_job(job_id, client):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n✓ Complete: {success_count} cancelled successfully, {fail_count} failed")


if __name__ == "__main__":
    main()
