#!/usr/bin/env python3
"""
Upload merged model directories to Hugging Face Hub.

Usage:
    python -m src.hf_utils.upload_merged_models \
        --org <organization_name> \
        --models <model_path1> <model_path2> ... \
        [--repo-prefix <prefix>] \
        [--private] \
        [--token <token>]

Example:
    # Upload multiple merged models
    python -m src.hf_utils.upload_merged_models \
        --org EMBGuard \
        --models \
            /path/to/merged_checkpoint-1000 \
            /path/to/merged_checkpoint-2000 \
        --repo-prefix qwen3-vl-4b-embhazard-merged

    # Upload with custom repo names
    python -m src.hf_utils.upload_merged_models \
        --org EMBGuard \
        --models \
            /path/to/merged_checkpoint-1000 \
            /path/to/merged_checkpoint-2000 \
        --repo-names \
            qwen3-vl-4b-embhazard-merged-cp1000 \
            qwen3-vl-4b-embhazard-merged-cp2000
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from huggingface_hub import HfApi, login, create_repo
from tqdm import tqdm


def get_project_root():
    """Get the project root directory."""
    script_dir = Path(__file__).parent
    return script_dir.parent.parent


def validate_model_directory(model_path: Path) -> bool:
    """Validate that the model directory contains necessary files."""
    required_files = ["config.json"]
    # Check for either safetensors or pytorch_model files
    has_model_files = (
        any(model_path.glob("*.safetensors")) or
        any(model_path.glob("pytorch_model*.bin")) or
        any(model_path.glob("model*.safetensors"))
    )
    
    if not has_model_files:
        print(f"  ⚠️  Warning: No model files found in {model_path}")
        print(f"     Expected: *.safetensors or pytorch_model*.bin")
    
    has_config = (model_path / "config.json").exists()
    if not has_config:
        print(f"  ⚠️  Warning: config.json not found in {model_path}")
    
    return has_model_files and has_config


def generate_repo_name(model_path: Path, repo_prefix: Optional[str] = None) -> str:
    """Generate a repository name from model path."""
    model_name = model_path.name
    
    # Remove common suffixes
    if model_name.endswith("_merged"):
        model_name = model_name[:-7]
    
    # If repo_prefix is provided, use it
    if repo_prefix:
        # Extract checkpoint number if present
        if "checkpoint-" in model_name:
            checkpoint_num = model_name.split("checkpoint-")[-1]
            return f"{repo_prefix}-cp{checkpoint_num}"
        else:
            return f"{repo_prefix}"
    
    # Auto-generate from path
    # Try to extract meaningful name from path
    parts = model_path.parts
    for i, part in enumerate(parts):
        if "merged" in part.lower() or "checkpoint" in part.lower():
            # Use the part and following parts
            relevant_parts = parts[i:]
            name = "-".join(relevant_parts)
            # Clean up the name
            name = name.replace("_merged", "").replace("_", "-").lower()
            return name
    
    # Fallback: use directory name
    return model_name.replace("_", "-").lower()


def upload_model_to_hub(
    model_path: Path,
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> bool:
    """
    Upload a model directory to Hugging Face Hub.
    
    Args:
        model_path: Local path to the model directory
        repo_id: Hugging Face repository ID (org/repo-name)
        private: Whether the repository should be private
        token: Hugging Face token
        commit_message: Commit message for the upload
        
    Returns:
        True if successful, False otherwise
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"❌ Error: Model directory not found: {model_path}")
        return False
    
    if not model_path.is_dir():
        print(f"❌ Error: Path is not a directory: {model_path}")
        return False
    
    # Validate model directory
    if not validate_model_directory(model_path):
        print(f"❌ Error: Invalid model directory: {model_path}")
        return False
    
    print(f"\n📤 Uploading model: {model_path.name}")
    print(f"   Repository: {repo_id}")
    print(f"   Private: {private}")
    
    try:
        api = HfApi(token=token)
        
        # Create repository if it doesn't exist
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                token=token,
                exist_ok=True,
            )
            print(f"   ✓ Repository created/verified: {repo_id}")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not create repository: {e}")
            print(f"   Continuing anyway (repository may already exist)...")
        
        # Upload the model directory
        if commit_message is None:
            commit_message = f"Upload merged model: {model_path.name}"
        
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=commit_message,
        )
        
        print(f"   ✓ Successfully uploaded to {repo_id}")
        print(f"   View at: https://huggingface.co/{repo_id}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error uploading {model_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload merged model directories to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload multiple models with auto-generated repo names
  python -m src.hf_utils.upload_merged_models \\
      --org EMBGuard \\
      --models \\
          /path/to/merged_checkpoint-1000 \\
          /path/to/merged_checkpoint-2000 \\
      --repo-prefix qwen3-vl-4b-embhazard-merged

  # Upload with custom repo names
  python -m src.hf_utils.upload_merged_models \\
      --org EMBGuard \\
      --models \\
          /path/to/merged_checkpoint-1000 \\
          /path/to/merged_checkpoint-2000 \\
      --repo-names \\
          qwen3-vl-4b-embhazard-merged-cp1000 \\
          qwen3-vl-4b-embhazard-merged-cp2000
        """
    )
    
    parser.add_argument(
        "--org",
        type=str,
        required=True,
        help="Hugging Face organization name"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Paths to merged model directories (can specify multiple)"
    )
    
    parser.add_argument(
        "--repo-prefix",
        type=str,
        default=None,
        help="Prefix for repository names (auto-generates repo names with checkpoint numbers)"
    )
    
    parser.add_argument(
        "--repo-names",
        type=str,
        nargs="+",
        default=None,
        help="Custom repository names (must match number of --models). "
             "If not provided, names are auto-generated from paths."
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repositories private"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN environment variable)"
    )
    
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Custom commit message (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Get token from environment if not provided
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    if not token:
        print("⚠️  Warning: No Hugging Face token provided.")
        print("   Set HF_TOKEN environment variable or use --token")
        print("   Or login with: huggingface-cli login")
        print("   Attempting to proceed with cached credentials...")
    else:
        # Login with token
        try:
            login(token=token)
            print("✓ Logged in to Hugging Face")
        except Exception as e:
            print(f"⚠️  Warning: Login failed: {e}")
            print("   Continuing with cached credentials...")
    
    # Validate model paths
    model_paths = [Path(p) for p in args.models]
    valid_paths = []
    for model_path in model_paths:
        if not model_path.exists():
            print(f"❌ Error: Model path not found: {model_path}")
            continue
        if not model_path.is_dir():
            print(f"❌ Error: Path is not a directory: {model_path}")
            continue
        valid_paths.append(model_path)
    
    if not valid_paths:
        print("❌ Error: No valid model paths provided")
        sys.exit(1)
    
    # Generate repository names
    repo_names = []
    if args.repo_names:
        if len(args.repo_names) != len(valid_paths):
            print(f"❌ Error: Number of --repo-names ({len(args.repo_names)}) "
                  f"does not match number of --models ({len(valid_paths)})")
            sys.exit(1)
        repo_names = args.repo_names
    else:
        # Auto-generate repo names
        for model_path in valid_paths:
            repo_name = generate_repo_name(model_path, args.repo_prefix)
            repo_names.append(repo_name)
    
    # Build repo IDs
    repo_ids = [f"{args.org}/{name}" for name in repo_names]
    
    # Display upload plan
    print("\n" + "=" * 60)
    print("Upload Plan")
    print("=" * 60)
    print(f"Organization: {args.org}")
    print(f"Private: {args.private}")
    print(f"Models to upload: {len(valid_paths)}")
    print()
    for model_path, repo_id in zip(valid_paths, repo_ids):
        print(f"  {model_path.name}")
        print(f"    → {repo_id}")
    print("=" * 60)
    
    # Confirm before proceeding
    try:
        response = input("\nProceed with upload? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Upload cancelled.")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\nUpload cancelled.")
        sys.exit(0)
    
    # Upload each model
    success_count = 0
    failed_count = 0
    
    for model_path, repo_id in zip(valid_paths, repo_ids):
        success = upload_model_to_hub(
            model_path=model_path,
            repo_id=repo_id,
            private=args.private,
            token=token,
            commit_message=args.commit_message,
        )
        if success:
            success_count += 1
        else:
            failed_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Upload Summary")
    print("=" * 60)
    print(f"✓ Successful: {success_count}")
    if failed_count > 0:
        print(f"❌ Failed: {failed_count}")
    print("=" * 60)
    
    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
