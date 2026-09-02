"""
Merge LoRA adapter into base model for Qwen3-VL.
This script merges a LoRA adapter trained with LlamaFactory into the base model.

Important: For VLM models like Qwen3-VL, this script also copies:
- preprocessor_config.json (image processor config)
- chat_template.jinja (if exists)
- Any other processor-related files
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoProcessor,
)
from peft import PeftModel, PeftConfig
from huggingface_hub import snapshot_download, hf_hub_download

# Add LlamaFactory to path for multimodal model loading
PROJECT_ROOT = Path(__file__).parents[2]
LLAMAFACTORY_DIR = PROJECT_ROOT / "LlamaFactory"
sys.path.insert(0, str(LLAMAFACTORY_DIR / "src"))


def get_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base Qwen3-VL model"
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        required=True,
        help="Path or Hugging Face ID of the base model (e.g., Qwen/Qwen3-VL-4B-Instruct)",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the LoRA adapter directory (output_dir from training)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the merged model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint to merge (e.g., 'checkpoint-1000'). If not specified, uses the latest checkpoint or adapter_model.bin",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for the merged model (default: auto)",
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="5GB",
        help="Maximum shard size for model saving (default: 5GB)",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        default=True,
        help="Use safe serialization (safetensors) for saving (default: True)",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_false",
        dest="safe_serialization",
        help="Disable safe serialization (use .bin files)",
    )
    return parser.parse_args()


def resolve_checkpoint_path(adapter_path: Path, checkpoint: str = None) -> Path:
    """Resolve the adapter checkpoint path."""
    adapter_path = Path(adapter_path)
    
    if checkpoint:
        # Use specific checkpoint
        checkpoint_path = adapter_path / checkpoint
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        adapter_model_path = checkpoint_path / "adapter_model.bin"
        adapter_config_path = checkpoint_path / "adapter_config.json"
        if not adapter_model_path.exists() and not (checkpoint_path / "adapter_model.safetensors").exists():
            # Try alternative paths
            adapter_model_path = checkpoint_path / "pytorch_model.bin"
            if not adapter_model_path.exists():
                raise FileNotFoundError(
                    f"Adapter model file not found in checkpoint: {checkpoint_path}"
                )
        return checkpoint_path
    else:
        # Use latest checkpoint or adapter in root
        if (adapter_path / "adapter_model.bin").exists() or (adapter_path / "adapter_model.safetensors").exists():
            return adapter_path
        
        # Find latest checkpoint
        checkpoints = sorted(
            [d for d in adapter_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0,
            reverse=True
        )
        
        if checkpoints:
            print(f"Using latest checkpoint: {checkpoints[0].name}")
            return checkpoints[0]
        else:
            raise FileNotFoundError(
                f"No adapter found in {adapter_path}. Please specify --checkpoint or ensure adapter files exist."
            )


def get_model_class(config):
    """Determine the appropriate AutoModel class based on config type."""
    # Check for image-text models first
    if type(config) in AutoModelForImageTextToText._model_mapping.keys():
        return AutoModelForImageTextToText
    elif type(config) in AutoModelForVision2Seq._model_mapping.keys():
        return AutoModelForVision2Seq
    else:
        # Default to causal LM
        return AutoModelForCausalLM


def load_model_with_adapter(
    base_model_name_or_path: str,
    adapter_path: Path,
    torch_dtype: str = "auto",
):
    """Load base model and merge LoRA adapter."""
    print(f"Loading base model from: {base_model_name_or_path}")
    
    # Load config first to determine model type
    print("Loading model configuration...")
    config = AutoConfig.from_pretrained(base_model_name_or_path, trust_remote_code=True)
    
    # Determine the appropriate model class
    model_class = get_model_class(config)
    print(f"Using model class: {model_class.__name__}")
    
    # Determine dtype
    if torch_dtype == "auto":
        # Try to infer from model config or use bfloat16 for efficiency
        if hasattr(config, "torch_dtype") and config.torch_dtype:
            dtype = getattr(torch, str(config.torch_dtype).replace("torch.", ""))
        else:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        dtype = getattr(torch, torch_dtype)
    
    print(f"Using dtype: {dtype}")
    
    # Load base model with appropriate class
    print("Loading base model...")
    base_model = model_class.from_pretrained(
        base_model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Load tokenizer and processor (for multimodal models)
    print("Loading tokenizer and processor...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=True,
        )
    except:
        tokenizer = None
        print("Warning: Could not load tokenizer")
    
    try:
        processor = AutoProcessor.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=True,
        )
    except:
        processor = None
        print("Warning: Could not load processor")
    
    # Load and merge adapter
    print(f"Loading LoRA adapter from: {adapter_path}")
    
    # Check if adapter_path contains adapter files
    if (adapter_path / "adapter_model.bin").exists() or (adapter_path / "adapter_model.safetensors").exists():
        # Direct adapter path
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
    else:
        # Try to find adapter in subdirectories
        adapter_config = adapter_path / "adapter_config.json"
        if adapter_config.exists():
            model = PeftModel.from_pretrained(base_model, str(adapter_path))
        else:
            raise FileNotFoundError(
                f"Adapter files not found in {adapter_path}. "
                f"Expected adapter_model.bin or adapter_model.safetensors"
            )
    
    print("Merging adapter into base model...")
    merged_model = model.merge_and_unload()
    
    return merged_model, tokenizer, processor


def copy_vlm_config_files(base_model_name_or_path: str, output_dir: Path):
    """
    Copy VLM-related configuration files from the base model to the output directory.
    
    For VLM models like Qwen3-VL, these files are essential for inference:
    - preprocessor_config.json: Image processor configuration
    - chat_template.jinja: Chat template (sometimes separate from tokenizer_config)
    
    This function handles both local model paths and Hugging Face model IDs.
    """
    output_dir = Path(output_dir)
    
    # List of VLM-related files that should be copied if they don't exist
    # For Qwen3-VL / Qwen2.5-VL models, these are essential for inference
    vlm_files = [
        "preprocessor_config.json",       # Image processor configuration
        "video_preprocessor_config.json", # Video processor configuration (Qwen VL specific)
        "chat_template.jinja",            # Chat template (sometimes separate from tokenizer_config)
    ]
    
    base_path = Path(base_model_name_or_path)
    is_local = base_path.exists() and base_path.is_dir()
    
    for filename in vlm_files:
        output_file = output_dir / filename
        
        # Skip if file already exists in output directory
        if output_file.exists():
            print(f"✓ {filename} already exists")
            continue
        
        try:
            if is_local:
                # Local model path
                source_file = base_path / filename
                if source_file.exists():
                    shutil.copy2(source_file, output_file)
                    print(f"✓ Copied {filename} from local base model")
            else:
                # Hugging Face model ID - download the specific file
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=base_model_name_or_path,
                        filename=filename,
                        local_dir=None,  # Use cache
                    )
                    shutil.copy2(downloaded_path, output_file)
                    print(f"✓ Downloaded and copied {filename} from Hugging Face")
                except Exception as e:
                    # File might not exist in the repo (e.g., chat_template.jinja)
                    print(f"ℹ {filename} not found in base model (this may be normal): {e}")
        except Exception as e:
            print(f"⚠ Warning: Could not copy {filename}: {e}")
    
    # Ensure preprocessor_config.json has correct image_processor_type for Qwen models
    preprocessor_config_path = output_dir / "preprocessor_config.json"
    if preprocessor_config_path.exists():
        try:
            with open(preprocessor_config_path, "r") as f:
                config = json.load(f)
            
            # Check if it's a Qwen VL model and fix image_processor_type if needed
            image_processor_type = config.get("image_processor_type", "")
            if "Qwen" in image_processor_type:
                # Ensure compatibility with transformers
                # Some versions use Qwen2_5_VLImageProcessor which may not be recognized
                print(f"ℹ Image processor type: {image_processor_type}")
            
            # Make sure the config is saved properly
            with open(preprocessor_config_path, "w") as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            print(f"⚠ Warning: Could not verify preprocessor_config.json: {e}")


def main():
    args = get_args()
    
    # Resolve adapter checkpoint path
    adapter_path = resolve_checkpoint_path(args.adapter_path, args.checkpoint)
    print(f"Using adapter from: {adapter_path}")
    
    # Load and merge model
    merged_model, tokenizer, processor = load_model_with_adapter(
        args.base_model_name_or_path,
        adapter_path,
        args.torch_dtype,
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save merged model
    print(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(
        output_dir,
        max_shard_size=args.max_shard_size,
        safe_serialization=args.safe_serialization,
    )
    
    # Save tokenizer and processor
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
        print("✓ Tokenizer saved")
    
    if processor is not None:
        processor.save_pretrained(output_dir)
        print("✓ Processor saved")
    elif tokenizer is not None:
        # For some models, processor is the same as tokenizer
        try:
            processor = AutoProcessor.from_pretrained(
                args.base_model_name_or_path,
                trust_remote_code=True,
            )
            processor.save_pretrained(output_dir)
            print("✓ Processor saved")
        except:
            pass
    
    # Copy additional VLM-related files from base model
    # These files are often missing after merge but required for inference
    copy_vlm_config_files(args.base_model_name_or_path, output_dir)
    
    print("")
    print("=" * 60)
    print("✓ Model merge completed successfully!")
    print("=" * 60)
    print(f"Base model: {args.base_model_name_or_path}")
    print(f"Adapter: {adapter_path}")
    print(f"Merged model saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
