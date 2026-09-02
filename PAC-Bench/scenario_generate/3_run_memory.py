"""
Stage 3: Memory Generation

Generate Agent Memories (raw data format memory) from Previous Stage Results (Requirements)

Usage:
    python 3_run_memory.py --input result/2_requirements/20241222_120000
    python 3_run_memory.py --input result/2_requirements/20241222_120000/Domain/requirements_0.json
    python 3_run_memory.py --input result/2_requirements/20241222_120000 --max_workers 8
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import Settings
from src.generators.stage3_memory_generator import MemoryGenerator
from src.llm.factory import create_llm_client
from src.prompts.prompt_loader import PromptLoader
from src.utils.logging import setup_logging, get_logger
from src.utils.parallel import run_parallel
from src.utils.file_io import load_stage_files, save_stage_result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Requirements → Memories (agent raw data memory) Generation"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to previous stage results (file or directory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: settings.paths.memory_output_dir)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic"],
        default=None,
        help="LLM provider (LLM_PROVIDER in .env or default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (LLM_MODEL in .env or use default)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level (LOG_LEVEL in .env or default: INFO)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Number of parallel workers (PIPELINE_MAX_WORKERS in .env or default: 4)",
    )
    
    args = parser.parse_args()
    
    # Load settings (read default values from .env)
    settings = Settings.load()
    
    # Override with argparse arguments if present, otherwise use values from .env
    if args.provider:
        settings.llm.provider = args.provider
    if args.model:
        settings.llm.model = args.model
    if args.log_level:
        settings.log_level = args.log_level
    
    # Create output directory (including timestamp) - use default path from settings
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output) if args.output else settings.paths.memory_output_dir
    output_dir = base_output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save log file in the same folder
    log_file = output_dir / "run.log"
    setup_logging(level=settings.log_level, log_file=log_file)
    logger = get_logger(__name__)
    logger.info(f"Log file: {log_file}")
    
    # Load previous stage results (requirements files)
    try:
        requirements_results = load_stage_files(args.input, "requirements")
        logger.info(f"Loaded {len(requirements_results)} requirements result(s)")
    except Exception as e:
        logger.error(f"Failed to load requirements results: {e}")
        sys.exit(1)
    
    # Determine number of parallel workers
    max_workers = args.max_workers or settings.pipeline.max_workers
    
    # Initialize LLM client and generator (use path from settings)
    prompt_loader = PromptLoader(settings.paths.prompts_dir)
    
    def process_requirements_result(item: tuple[str, int, Path, dict]) -> tuple[str, bool, str]:
        """Process a single requirements_#.json file (for parallel execution)"""
        domain_name, index, source_path, data = item
        try:
            # Create a separate LLM client for each thread
            llm_client = create_llm_client(settings)
            generator = MemoryGenerator(llm_client, prompt_loader)
            
            logger.info(f"[START] Generating memories for domain: {domain_name}, index: {index}")
            
            # Call LLM to generate memories
            memories = generator.generate_from_dict(data, domain_name)
            
            # JSON format: Maintain existing data + inject memory into agent_a/agent_b
            output_data = data.copy()
            
            # Add memory to agent_a
            if "agent_a" not in output_data["scenario"]:
                output_data["scenario"]["agent_a"] = {}
            output_data["scenario"]["agent_a"]["memory"] = memories["agent_a"]["memory"]
            
            # Add memory to agent_b
            if "agent_b" not in output_data["scenario"]:
                output_data["scenario"]["agent_b"] = {}
            output_data["scenario"]["agent_b"]["memory"] = memories["agent_b"]["memory"]
            
            # Save using common utility
            output_path = save_stage_result(
                output_dir=output_dir,
                domain_name=domain_name,
                stage_prefix="memory",
                index=index,
                data=output_data,
            )
            
            agent_a_count = len(memories["agent_a"]["memory"])
            agent_b_count = len(memories["agent_b"]["memory"])
            
            logger.info(f"[END] Saved: {output_path}")
            logger.info(f"  - Domain: {domain_name}")
            logger.info(f"  - Memories: agent_a={agent_a_count}, agent_b={agent_b_count}")
            
            return (f"{domain_name}/index_{index}", True, str(output_path))
        
        except Exception as e:
            import traceback
            logger.error(f"[ERROR] Failed to generate memories for {domain_name}/index_{index}: {e}")
            traceback.print_exc()
            return (f"{domain_name}/index_{index}", False, str(e))
    
    # Run parallel processing
    success_count, fail_count = run_parallel(
        items=requirements_results,
        process_fn=process_requirements_result,
        max_workers=max_workers,
        logger=logger,
        task_name="requirements",
    )
    
    print(f"\n✅ Stage 3 complete! Results saved at: {output_dir}")
    print(f"   Success: {success_count}, Failure: {fail_count}")
    print(f"Execute next stage: python 4_run_constraint.py --input {output_dir}")


if __name__ == "__main__":
    main()

