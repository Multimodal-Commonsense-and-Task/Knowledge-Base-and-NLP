"""
Batch generation using Google GenAI batch API.
Extracted from inference.py for better modularity.
"""
import os
import json
import time
from tqdm import tqdm
from google import genai
from google.genai import types

# Add project root to Python path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from utils.config import get_config
except ImportError:
    def get_config():
        return {}


def generate_batch(all_model_input, stop, args):
    """
    Generate batch job using Google GenAI batch API.
    
    Args:
        all_model_input: List of [prompt, metadata] tuples
        stop: Stop sequence (not used for batch)
        args: Arguments object containing:
            - model_name: Model name to use
            - save_dir: Directory to save batch files
            - iterate_name: Optional iterate name for organizing results
            - api_key: Optional API key (if not provided, will try config.yaml or env var)
    
    Returns:
        List of batch job info dictionaries
    """
    # Get API key from config.yaml or environment variable
    api_key = getattr(args, "api_key", None)
    if not api_key:
        config = get_config()
        api_key = config.get("gemini", {}).get("key") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key is required. Set it in config.yaml (gemini.key) or GOOGLE_API_KEY environment variable")
    client = genai.Client(api_key=api_key)
    
    save_dir = getattr(args, "save_dir", ".")
    batch_dir = os.path.join(save_dir, getattr(args, "iterate_name", "batch"))
    os.makedirs(batch_dir, exist_ok=True)
    
    timestamp = int(time.time())
    input_filename = os.path.join(batch_dir, f"batch_input_{timestamp}.jsonl")
    
    print(f"\n📝 Creating batch request file...")
    print(f"   Total {len(all_model_input)} image generation requests")
    print(f"   File: {input_filename}")
    
    all_batch_job_info = []
    
    
    with open(input_filename, "w", encoding="utf-8") as f:
        for i, item in enumerate(tqdm(all_model_input, desc="Creating JSONL")):
            prompt, metadata = item[0], item[1]
            req_id = f"{metadata['id']}-{metadata['merge_source']}"
            # req_id = f"req-{i}" # Not using req_id, but matching id is important for easy mapping
            request_body = {
                "custom_id": req_id,
                "request": {
                    "model": f"models/{args.model_name}",
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "response_modalities": ["IMAGE"], 
                        "image_config": {
                            "aspect_ratio": "4:3",
                            "image_size": "2K"
                        }
                    }
                }
            }
            f.write(json.dumps(request_body, ensure_ascii=False) + "\n")
            
            batch_job_info = {
                "req_id": req_id,
                "model_input": prompt,
                **metadata,
            }
            all_batch_job_info.append(batch_job_info)
            
    
    print(f"   ✅ Saved {len(all_model_input)} requests")
    
    # 2. Save scenario batch information
    batch_scenarios_file = os.path.join(batch_dir, f"batch_scenarios_{timestamp}.json")
    with open(batch_scenarios_file, "w", encoding="utf-8") as f:
        json.dump(all_batch_job_info, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Saved batch scenario information: {batch_scenarios_file}")
    
    # 3. Upload and create batch job
    print(f"\n🚀 Uploading to Google server and creating batch job...")
    batch_file = client.files.upload(
        file=input_filename, 
        config={'mime_type': 'text/plain'}
    )
    
    job = client.batches.create(
        model=args.model_name,
        src=batch_file.name,
        config=types.CreateBatchJobConfig(display_name=f"job_{timestamp}")
    )
    
    # Add job information to batch info
    for item in all_batch_job_info:
        item["batch_job"] = job.name
        item["batch_job_id"] = job.name.split('/')[-1] if '/' in job.name else job.name
        item["timestamp"] = timestamp
    
    job_info_file = os.path.join(batch_dir, f"job_{timestamp}.txt")
    # Extract merge_source from first scenario
    merge_source = "unknown"
    if all_model_input and len(all_model_input) > 0:
        first_scenario = all_model_input[0][1]  # [model_input, metadata]
        merge_source = first_scenario.get('merge_source', 'unknown')
    
    job_info = f"""

Job ID: {job.name}
Job Short ID: {job.name.split('/')[-1] if '/' in job.name else job.name}
Model: {args.model_name}
Requests: {len(all_model_input)}
Timestamp: {timestamp}
Created: {time.strftime('%Y-%m-%d %H:%M:%S')}
Merge Source: {merge_source}

Input File: {input_filename}

Use this to check:
   python src/batch_job/check_batch.py {job.name}

""" 
    with open(job_info_file, "w", encoding="utf-8") as f:
        f.write(job_info)
    
    return all_batch_job_info

