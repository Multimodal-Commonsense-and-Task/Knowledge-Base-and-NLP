"""
Main entry point for dataset generation pipeline.
Uses task-based architecture with multiprocessing workers for parallel inference.
"""
import argparse
import json
import os
import sys
from typing import List, Dict, Any, Tuple

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # src/run_dataset_generation.py -> src -> project_root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from utils.config import get_config
except ImportError:
    def get_config():
        return {}

from src.dataset_generation.task import get_task, TASK_REGISTRY
from src.dataset_generation.utils.json_utils import parse_json_prediction
from src.dataset_generation.utils.batch_generation import generate_batch
from src.models import create_model
from src.inference_utils import run_parallel_inference
from tqdm import tqdm


def get_default_prompt_path():
    """Get default prompt path from config.yaml"""
    config = get_config()
    dataset_gen_config = config.get("dataset_generation", {})
    default_path = dataset_gen_config.get("prompt_path", "./src/dataset_generation/resources/prompt/prompt.yaml")
    
    # If relative path, make it absolute relative to project root
    if not os.path.isabs(default_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # dataset_generation.py -> src -> project_root
        default_path = os.path.join(project_root, default_path)
    
    return default_path


def get_default_shots_path():
    """Get default shots path from config.yaml"""
    config = get_config()
    dataset_gen_config = config.get("dataset_generation", {})
    
    # Get shots_file from config, fallback to default
    shots_file = dataset_gen_config.get("shots_file", "shots_w_mechanism.json")
    
    # If shots_file is already an absolute path, use it directly
    if os.path.isabs(shots_file):
        return shots_file
    
    # Otherwise, combine with source_dir
    source_dir = dataset_gen_config.get("source_dir", "src/dataset_generation/resources/source")
    
    # If relative path, make it absolute relative to project root
    if not os.path.isabs(source_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # dataset_generation.py -> src -> project_root
        shots_path = os.path.join(project_root, source_dir, shots_file)
    else:
        shots_path = os.path.join(source_dir, shots_file)
    
    return shots_path


def get_default_taxonomy_path():
    """Get default taxonomy path from config.yaml"""
    config = get_config()
    dataset_gen_config = config.get("dataset_generation", {})
    
    # Get taxonomy_file from config, fallback to default
    taxonomy_file = dataset_gen_config.get("taxonomy_file", "taxonomy.json")
    
    # If taxonomy_file is already an absolute path, use it directly
    if os.path.isabs(taxonomy_file):
        return taxonomy_file
    
    # Otherwise, combine with source_dir
    source_dir = dataset_gen_config.get("source_dir", "src/dataset_generation/resources/source")
    
    # If relative path, make it absolute relative to project root
    if not os.path.isabs(source_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # dataset_generation.py -> src -> project_root
        taxonomy_path = os.path.join(project_root, source_dir, taxonomy_file)
    else:
        taxonomy_path = os.path.join(source_dir, taxonomy_file)
    
    return taxonomy_path


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset generation pipeline with task-based architecture")
    parser.add_argument("--task", type=str, required=True, 
                       choices=list(TASK_REGISTRY.keys()),
                       help="Task name to execute")
    parser.add_argument("--model_name", type=str, default=None, required=True)
    parser.add_argument("--provider", type=str, default="openai",
                       choices=["openai", "openrouter", "vllm", "claude", "gemini"],
                       help="LLM provider")
    parser.add_argument("--model_port", type=int, default=None,
                       help="Port for local vLLM server")
    parser.add_argument("--model_type", type=str, default="gpt", 
                       choices=["local", "gpt", "anthropic", "openrouter_image"],
                       help="Legacy model type (for backward compatibility)")
    parser.add_argument("--data_path", type=str, default=None, 
                       help="Data file path (absolute path or relative name without .json)")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--file_index", type=str, default="1", 
                       help="starting index for output file numbering (can be int or str)")
    parser.add_argument("--save_dir", type=str, default="./dataset_generation_output")
    parser.add_argument("--risk_type", type=str, default=None, 
                       help="risk type for taxonomy_to_scenario (e.g., fire_risk, electrical_risk)")
    parser.add_argument("--mechanism", type=str, default=None, 
                       help="specific mechanism for taxonomy_to_scenario")
    parser.add_argument("--num", type=int, default=20, 
                       help="number of scenarios to generate (default: 20)")
    parser.add_argument("--iterate_name", type=str, default=None, 
                       help="iterate name for organizing results (e.g., pilot_full_changed)")
    parser.add_argument("--batch", type=bool, default=False, 
                       help="use Google GenAI batch API for large-scale inference")
    parser.add_argument("--reasoning_effort", type=str, default=None,
                       choices=["low", "medium", "high"],
                       help="Reasoning effort for GPT-5 family models")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of worker processes for parallel inference")
    parser.add_argument("--temperature", type=float, default=None,
                       help="Temperature for generation")
    parser.add_argument("--max_tokens", type=int, default=None,
                       help="Maximum tokens for generation")
    parser.add_argument("--include_original", type=lambda x: x.lower() == 'true', default=True,
                       help="For action_augmentation: include original scenarios in output (default: True)")
    parser.add_argument("--scenarios_file", type=str, default=None,
                       help="Input scenarios file path (for scenario_to_graph and gather_results)")
    parser.add_argument("--input_scenarios_file", type=str, default=None,
                       help="Input scenarios file for hazard_augmentation (to create pairs if needed)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for hazard pair creation (default: 42)")
    parser.add_argument("--pairs_per_mechanism", type=int, default=1,
                       help="Number of pairs to select per mechanism combination for hazard_augmentation (default: 1)")
    parser.add_argument("--use-few-shot", action="store_true", default=True,
                       help="For embguard_train_data_construction: include few-shot examples (default: True)")
    parser.add_argument("--no-use-few-shot", dest="use_few_shot", action="store_false",
                       help="For embguard_train_data_construction: exclude few-shot examples")
    return parser.parse_args()


def create_model_config(args) -> Dict[str, Any]:
    """Create model configuration dictionary for models.py"""
    config = {
        "model_name": args.model_name,
    }
    
    # Get API keys from config.yaml
    config_yaml = get_config()
    
    # Set provider-specific config
    if args.provider == "openai":
        config["api_key"] = config_yaml.get("openai", {}).get("key") or os.getenv("OPENAI_API_KEY")
        config["temperature"] = args.temperature if args.temperature is not None else 1.0
        config["max_tokens"] = args.max_tokens if args.max_tokens is not None else 16384
    elif args.provider == "openrouter":
        config["api_key"] = config_yaml.get("openrouter", {}).get("key") or os.getenv("OPENROUTER_API_KEY")
        config["base_url"] = "https://openrouter.ai/api/v1"
        config["temperature"] = args.temperature if args.temperature is not None else 0.7
        config["max_tokens"] = args.max_tokens if args.max_tokens is not None else 16384
    elif args.provider == "vllm":
        if args.model_port is None:
            raise ValueError("--model_port is required for vllm provider")
        config["base_url"] = f"http://localhost:{args.model_port}/v1"
        config["api_key"] = "EMPTY"
        config["temperature"] = args.temperature if args.temperature is not None else 0.5
        config["max_tokens"] = args.max_tokens if args.max_tokens is not None else 16384
    elif args.provider == "claude":
        config["api_key"] = config_yaml.get("anthropic", {}).get("key") or os.getenv("ANTHROPIC_API_KEY")
        config["temperature"] = args.temperature if args.temperature is not None else 0.3
        config["max_tokens"] = args.max_tokens if args.max_tokens is not None else 16384
    elif args.provider == "gemini":
        config["api_key"] = config_yaml.get("gemini", {}).get("key") or os.getenv("GOOGLE_API_KEY")
        config["temperature"] = args.temperature if args.temperature is not None else 0.7
        config["max_tokens"] = args.max_tokens if args.max_tokens is not None else 16384
    
    return config


def prompt_to_messages(prompt: str, task_name: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert prompt string to messages format for models.py"""
    messages = [{"role": "user", "content": prompt}]
    
    # For VQA and guardrail, add image to messages
    if task_name in ["vqa", "guardrail"]:
        image_path = metadata.get('image_path') or metadata.get('image')
        if image_path:
            # Add image to the message
            messages[0]["images"] = [image_path]
    
    return messages


def process_task_item(config: Dict[str, Any], item: Dict[str, Any]) -> Tuple[Dict[str, Any], float, Dict[str, int]]:
    """
    Process a single task item in worker process.
    
    Args:
        config: Configuration dictionary containing:
            - task_name: Task name
            - model_config: Model configuration dict
            - provider: Model provider name
            - args_dict: Arguments dict (for recreating task)
        item: Dictionary containing:
            - idx: Item index
            - model_input: Prepared model input (prompt string)
            - metadata: Item metadata
    
    Returns:
        Tuple of (result_dict, cost, usage_dict)
    """
    # Recreate task and model in worker process (avoid pickle issues)
    from src.dataset_generation.task import get_task
    import argparse
    
    task_name = config["task_name"]
    model_config = config["model_config"]
    provider = config["provider"]
    args_dict = config["args_dict"]
    
    # Recreate args object from dict
    args = argparse.Namespace(**args_dict)
    
    # Create model in worker
    model = create_model(provider, model_config)
    
    # Recreate task instance
    task = get_task(task_name, args)
    
    idx = item["idx"]
    model_input_str = item["model_input"]  # Already prepared prompt string
    metadata = item["metadata"]
    
    try:
        # Convert prompt to messages format
        messages = prompt_to_messages(model_input_str, task_name, metadata)
        
        # Generate using models.py interface
        response = model.generate(messages)
        
        # Format response to match expected format
        formatted_response = {
            "content": response["content"],
            "usage": response["usage"],
            "cost": response["cost"]
        }
        
        # Process result using task
        result = task.process_result(formatted_response, [model_input_str, metadata])
        
        return result, response["cost"], response["usage"]
    except Exception as e:
        print(f"Error processing item {idx}: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0, {}


# worker_main is imported from src.inference_utils


# run_parallel_inference is imported from src.inference_utils
# This function converts all_model_inputs to work_items format and calls the shared function
def run_parallel_inference_for_dataset_generation(
    all_model_inputs: List,
    config: Dict[str, Any],
    num_workers: int = 4,
    description: str = "Running inference"
) -> Tuple[List[Dict[str, Any]], float, Dict[str, int]]:
    """
    Run parallel inference on model inputs using multiprocessing.
    Wrapper around run_parallel_inference that converts all_model_inputs to work_items format.
    
    Args:
        all_model_inputs: List of [model_input_str, metadata] tuples
        config: Configuration dictionary
        num_workers: Number of worker processes
        description: Description for progress bar
        
    Returns:
        Tuple of (list of results, total_cost, total_usage)
    """
    # Prepare work items with indices
    work_items = []
    for idx, (model_input_str, metadata) in enumerate(all_model_inputs):
        work_items.append({
            "idx": idx,
            "model_input": model_input_str,
            "metadata": metadata
        })
    
    # Use shared run_parallel_inference function
    return run_parallel_inference(
        dataset=work_items,
        config=config,
        num_workers=num_workers,
        description=description,
        process_func=process_task_item
    )


def determine_save_path(args, task):
    """Determine save path and filename based on iterate_name and task."""
    if args.iterate_name:
        save_path = os.path.join(args.save_dir, args.iterate_name)
        output_filename = f"{task.task_name}.json"
    else:
        # Existing structure (backward compatible)
        if task.task_name == 'taxonomy_to_scenario':
            taxonomy_risk_type = args.risk_type.replace(" ", "_") if args.risk_type else "all_risks"
            if args.mechanism:
                mechanism_folder = args.mechanism.replace(" ", "_").replace("/", "_")
                save_path = os.path.join(args.save_dir, taxonomy_risk_type, mechanism_folder)
            else:
                save_path = os.path.join(args.save_dir, taxonomy_risk_type)
        else:
            save_path = args.save_dir
        output_filename = f"{task.task_name}_{args.file_index}.json"
    
    return save_path, output_filename


def _normalize_provider_from_model_type(model_type: str, provider: str) -> str:
    """Normalize provider from legacy model_type if provider is default"""
    # If provider is still default (openai) and model_type is set, use model_type to infer provider
    if provider == "openai" and model_type:
        model_type_to_provider = {
            "gpt": "openai",
            "anthropic": "claude",
            "openrouter_image": "gemini",  # openrouter_image typically uses gemini
            "local": "vllm"
        }
        return model_type_to_provider.get(model_type, provider)
    return provider


def main(args):
    """Main execution function."""
    # Set prompt_path and shots_path from config (always use config, no argparse needed)
    args.prompt_path = get_default_prompt_path()
    args.shots_path = get_default_shots_path()
    
    # Set default data_path for taxonomy_to_scenario if not provided
    if args.task == 'taxonomy_to_scenario' and not args.data_path:
        args.data_path = get_default_taxonomy_path()
    
    # Normalize provider from model_type if needed
    args.provider = _normalize_provider_from_model_type(args.model_type, args.provider)
    
    # For embguard_train_data_construction, model_name is not required
    if args.task == "embguard_train_data_construction" and not args.model_name:
        args.model_name = "dummy"
    
    # Get task instance
    task = get_task(args.task, args)
    
    print(f"Executing task: {task.task_name}")
    
    # Special handling for embguard_train_data_construction - no inference needed
    if task.task_name == "embguard_train_data_construction":
        # Load data
        print("Loading data...")
        dataset = task.load_data()
        print(f"Loaded {len(dataset)} items")
        
        # Determine output path
        iterate_base = args.iterate_name.split('/')[0] if args.iterate_name else "train"
        iterate_output_dir = os.path.join(args.save_dir, iterate_base)
        os.makedirs(iterate_output_dir, exist_ok=True)
        
        output_file = os.path.join(iterate_output_dir, "embguard_train_data.json")
        if hasattr(args, 'output_file') and args.output_file:
            output_file = args.output_file
        
        # Set use_few_shot in args for task to access
        if not hasattr(args, 'use_few_shot'):
            args.use_few_shot = True
        
        # Directly construct training data (no inference needed)
        print("Constructing training data in OpenAI format...")
        result = task.gather_results(
            results_file="",  # Not used
            scenarios_file=args.data_path,  # CSV file path
            output_file=output_file
        )
        
        print(f"\n✓ Training data construction completed")
        print(f"  Output: {output_file}")
        print(f"  Total examples: {result['metadata']['total_examples']}")
        return
    
    # Load data
    print("Loading data...")
    dataset = task.load_data()
    print(f"Loaded {len(dataset)} items")
    
    # Create prompter
    print("Creating prompter...")
    prompter = task.create_prompter()
    
    # Prepare model inputs (returns list of [prompt_string, metadata] tuples)
    print("Preparing model inputs...")
    all_model_inputs = task.prepare_model_inputs(dataset, prompter)
    
    # Determine save path
    save_path, output_filename = determine_save_path(args, task)
    os.makedirs(save_path, exist_ok=True)
    
    # Generate results
    if args.batch:
        print("Creating batch job...")
        all_results = generate_batch(all_model_inputs, None, args)
    else:
        # Create model using models.py
        print("Creating model...")
        model_config = create_model_config(args)
        
        # Convert args to dict for multiprocessing (Namespace objects aren't pickleable)
        args_dict = vars(args)
        
        # Prepare worker config
        # Note: We'll create model in each worker to avoid pickle issues
        worker_config = {
            "task_name": task.task_name,
            "model_config": model_config,
            "provider": args.provider,
            "args_dict": args_dict,  # For recreating task in worker
        }
        
        print(f"Generating responses with {args.num_workers} workers...")
        all_results, total_cost, total_usage = run_parallel_inference_for_dataset_generation(
            all_model_inputs,
            worker_config,
            num_workers=args.num_workers,
            description=f"Processing {task.task_name}"
        )
        
        print(f"\nTotal cost: ${total_cost:.4f}")
        print(f"Total usage: {total_usage}")
    
    # Save results
    results_file_path = None
    if task.task_name == 'taxonomy_to_scenario' and not args.iterate_name:
        # Group by risk_type and mechanism
        def _folder_name(value, default_name):
            if not value:
                return default_name
            return str(value).replace(" ", "_").replace("/", "_")
        
        grouped = {}
        for result in all_results:
            risk_value = result.get("type") or result.get("risk_type")
            risk_folder = _folder_name(risk_value, "all_risks")
            
            mech_value = result.get("mechanism")
            if isinstance(mech_value, list):
                if len(mech_value) == 1 and isinstance(mech_value[0], dict):
                    mech_value = mech_value[0].get("name")
                else:
                    mech_value = None
            elif isinstance(mech_value, dict):
                mech_value = mech_value.get("name")
            mech_folder = _folder_name(mech_value, "all_mechanisms")
            
            group_key = (risk_folder, mech_folder)
            grouped.setdefault(group_key, []).append(result)
        
        for (risk_folder, mech_folder), results in grouped.items():
            group_path = os.path.join(args.save_dir, risk_folder, mech_folder)
            os.makedirs(group_path, exist_ok=True)
            results_file_path = os.path.join(group_path, output_filename)
            with open(results_file_path, "w", encoding="UTF-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
    else:
        results_file_path = os.path.join(save_path, output_filename)
        with open(results_file_path, "w", encoding="UTF-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print(f"Results saved to: {results_file_path}")
    
    # Automatically run gathering if iterate_name is provided (new structure)
    if args.iterate_name and results_file_path:
        print(f"\nRunning gathering for task: {task.task_name}...")
        
        try:
            # Extract base iterate name (e.g., "debug" from "debug/raw")
            iterate_base = args.iterate_name.split('/')[0]
            iterate_output_dir = os.path.join(args.save_dir, iterate_base)
            os.makedirs(iterate_output_dir, exist_ok=True)
            
            # Prepare gather_results parameters based on task
            if task.task_name == 'taxonomy_to_scenario':
                # taxonomy_to_scenario: gather from shots and generated results
                scenarios_file = args.shots_path
                output_file = os.path.join(iterate_output_dir, "scenarios.json")
                
                gathered = task.gather_results(
                    results_file=results_file_path,
                    scenarios_file=scenarios_file,
                    output_file=output_file,
                    shots_file=args.shots_path,
                    results_dir=args.save_dir,
                    iterate_name=args.iterate_name,
                    source='all'
                )
                print(f"✓ Gathering completed: {output_file}")
            else:
                # Other tasks: gather from previous scenarios file
                # Determine scenarios_file based on task
                scenarios_file = None
                output_file = None
                
                if task.task_name == 'scenario_to_graph':
                    # scenario_to_graph: Priority: args.scenarios_file > auto-detect
                    if hasattr(args, 'scenarios_file') and args.scenarios_file:
                        scenarios_file = args.scenarios_file
                        if not os.path.isabs(scenarios_file):
                            # If relative path, resolve relative to iterate directory
                            scenarios_file = os.path.join(iterate_output_dir, scenarios_file)
                    else:
                        # Auto-detect: use scenarios_generated.json or scenarios.json
                        scenarios_file = os.path.join(iterate_output_dir, "scenarios_generated.json")
                        if not os.path.exists(scenarios_file):
                            scenarios_file = os.path.join(iterate_output_dir, "scenarios.json")
                    output_file = os.path.join(iterate_output_dir, "graphs.json")
                elif task.task_name == 'scene_normalization':
                    # scene_normalization: use graphs.json from previous step
                    scenarios_file = os.path.join(iterate_output_dir, "graphs.json")
                    output_file = os.path.join(iterate_output_dir, "graphs_normalized.json")
                elif task.task_name == 'scene_augmentation':
                    # scene_augmentation: use graphs_normalized.json from previous step
                    scenarios_file = os.path.join(iterate_output_dir, "graphs_normalized.json")
                    output_file = os.path.join(iterate_output_dir, "graphs_scene_augmented.json")
                elif task.task_name == 'hazard_removal':
                    # hazard_removal: use input file (data_path) or auto-detect
                    if args.data_path:
                        # If data_path is provided, use it as scenarios_file
                        if os.path.isabs(args.data_path):
                            scenarios_file = args.data_path
                        else:
                            # Relative path: resolve relative to iterate directory
                            scenarios_file = os.path.join(iterate_output_dir, args.data_path)
                    else:
                        # Auto-detect: use graphs_scene_augmented.json from previous step
                        scenarios_file = os.path.join(iterate_output_dir, "graphs_scene_augmented.json")
                    
                    # Generate output filename based on input filename
                    if args.data_path:
                        input_basename = os.path.basename(args.data_path)
                        if input_basename.endswith('.json'):
                            output_basename = input_basename.replace('.json', '_hazard_removed.json')
                        else:
                            output_basename = f"{input_basename}_hazard_removed.json"
                        output_file = os.path.join(iterate_output_dir, output_basename)
                    else:
                        # Default output filename
                        output_file = os.path.join(iterate_output_dir, "graphs_hazard_removed.json")
                elif task.task_name == 'hazard_augmentation':
                    # hazard_augmentation: determine input scenarios file and pairs file
                    input_scenarios_file = None
                    if hasattr(args, 'input_scenarios_file') and args.input_scenarios_file:
                        input_scenarios_file = args.input_scenarios_file
                        if not os.path.isabs(input_scenarios_file):
                            input_scenarios_file = os.path.join(iterate_output_dir, input_scenarios_file)
                    else:
                        # Auto-detect: try graphs_normalized.json first, then graphs.json
                        potential_file = os.path.join(iterate_output_dir, "graphs_normalized.json")
                        if os.path.exists(potential_file):
                            input_scenarios_file = potential_file
                        else:
                            potential_file = os.path.join(iterate_output_dir, "graphs.json")
                            if os.path.exists(potential_file):
                                input_scenarios_file = potential_file
                    
                    # Pairs file will be auto-created in load_data if needed
                    pairs_file = os.path.join(iterate_output_dir, "raw", "hazard_pairs_by_room.json")
                    
                    # Generate output filename based on input filename
                    if input_scenarios_file:
                        input_basename = os.path.basename(input_scenarios_file)
                        if input_basename.endswith('.json'):
                            output_basename = input_basename.replace('.json', '_hazard_augmented.json')
                        else:
                            output_basename = f"{input_basename}_hazard_augmented.json"
                        output_file = os.path.join(iterate_output_dir, output_basename)
                    else:
                        # Default output filename
                        output_file = os.path.join(iterate_output_dir, "graphs_hazard_augmented.json")
                    
                    # Gather results (pairs creation and splitting are handled in task)
                    gathered = task.gather_results(
                        results_file=results_file_path,
                        scenarios_file=input_scenarios_file,  # Used for reference
                        output_file=output_file,
                        pairs_file=pairs_file,
                        input_scenarios_file=input_scenarios_file,
                        **{}
                    )
                    print(f"✓ Gathering completed: {output_file}")
                    print(f"✓ Split scenarios: {output_file.replace('.json', '_split.json')}")
                elif task.task_name == 'graph_to_text':
                    # graph_to_text: use the same input file (data_path) that was used for inference
                    if args.data_path:
                        # If data_path is provided, use it as scenarios_file
                        if os.path.isabs(args.data_path):
                            scenarios_file = args.data_path
                        else:
                            # Relative path: resolve relative to iterate directory
                            scenarios_file = os.path.join(iterate_output_dir, args.data_path)
                    else:
                        # Fallback: use graphs_final.json or graphs.json
                        scenarios_file = os.path.join(iterate_output_dir, "graphs_final.json")
                        if not os.path.exists(scenarios_file):
                            scenarios_file = os.path.join(iterate_output_dir, "graphs.json")
                    output_file = os.path.join(iterate_output_dir, "texts_generated.json")
                elif task.task_name == 'action_augmentation':
                    # action_augmentation: use the same input file that was used for inference
                    scenarios_file = args.data_path if args.data_path else os.path.join(iterate_output_dir, "texts_with_images_scene_augmented.json")
                    # Determine output file name based on input
                    if args.data_path:
                        base_name = os.path.basename(args.data_path)
                        if base_name.endswith('.json'):
                            output_file = os.path.join(iterate_output_dir, base_name.replace('.json', '_safe.json'))
                        else:
                            output_file = os.path.join(iterate_output_dir, "texts_with_images_action_augmented.json")
                    else:
                        output_file = os.path.join(iterate_output_dir, "texts_with_images_action_augmented.json")
                else:
                    # Default: look for scenarios.json
                    scenarios_file = os.path.join(iterate_output_dir, "scenarios.json")
                    output_file = os.path.join(iterate_output_dir, f"{task.task_name}_scenarios.json")
                
                if scenarios_file and os.path.exists(scenarios_file):
                    # For action_augmentation, pass include_original flag
                    gather_kwargs = {}
                    if task.task_name == 'action_augmentation':
                        gather_kwargs['include_original'] = getattr(args, 'include_original', True)
                    
                    gathered = task.gather_results(
                        results_file=results_file_path,
                        scenarios_file=scenarios_file,
                        output_file=output_file,
                        **gather_kwargs
                    )
                    print(f"✓ Gathering completed: {output_file}")
                else:
                    print(f"Note: Previous scenarios file not found: {scenarios_file}. Skipping gathering.")
        except Exception as e:
            print(f"Warning: Gathering failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("Done!")

