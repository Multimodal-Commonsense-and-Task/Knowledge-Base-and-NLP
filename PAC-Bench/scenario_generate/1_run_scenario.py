"""
Stage 1: Scenario Generation

Generate Scenarios (description, agents, goal) from Domain Names

Usage:
    # Basic: Use data/domains/domain.json file
    python 1_run_scenario.py
    
    # Use a different file
    python 1_run_scenario.py --domain_file domain_original.json
    
    # Specify a single domain name directly
    python 1_run_scenario.py --domain "Energy Equipment and Services"
    
    # Use both file and domain name together
    python 1_run_scenario.py --domain_file domain.json --domain "Custom Domain"
    
    # Specify the number of scenarios
    python 1_run_scenario.py --num_scenarios 10
    
    # Specify the number of parallel workers (default: PIPELINE_MAX_WORKERS in .env or 4)
    python 1_run_scenario.py --max_workers 8
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import Settings
from src.generators.stage1_scenario_generator import ScenarioGenerator
from src.llm.factory import create_llm_client
from src.prompts.prompt_loader import PromptLoader
from src.utils.logging import setup_logging, get_logger
from src.utils.parallel import run_parallel
from src.utils.file_io import save_stage_result, remove_index_from_scenario


def load_domain_names(domains_dir: Path, domain_arg: str = None, domain_file: str = None) -> list[str]:
    """
    Load list of domain names
    
    Args:
        domains_dir: Default path for domain files (settings.paths.domains_dir)
        domain_arg: Single domain name (optional)
        domain_file: Filename within the domains_dir folder (optional, default: domain.json)
    
    Returns:
        list[str]: List of domain names
    """
    domains = []
    
    # Use default value if domain_file is not specified
    if domain_file is None:
        domain_file = "domain.json"
    
    # Construct file path (relative to domains_dir folder)
    if domain_file:
        # If not an absolute path, use domains_dir folder as base
        if not Path(domain_file).is_absolute():
            path = domains_dir / domain_file
        else:
            path = Path(domain_file)
        
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # List of domain names
                domains = [name for name in data if isinstance(name, str)]
            elif isinstance(data, dict) and "name" in data:
                # Single domain object
                domains = [data["name"]]
        else:
            raise FileNotFoundError(f"Domain file not found: {path}")
    
    # Add domain_arg if present (higher priority than file)
    if domain_arg:
        domains.append(domain_arg)
    
    if not domains:
        raise ValueError(f"No domain specified. Use --domain or ensure domain.json exists in {domains_dir}/")
    
    return domains


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Domain → Scenario (description, agents, goal) Generation"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Single domain name (optional, can be used with --domain_file)",
    )
    parser.add_argument(
        "--domain_file",
        type=str,
        default=None,
        help="JSON filename within the domains folder (e.g., domain.json, domain_original.json). Default: domain.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: settings.paths.scenario_output_dir)",
    )
    parser.add_argument(
        "--num_scenarios",
        type=int,
        default=4,
        help="Number of scenarios to generate per domain (default: 4)",
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
    base_output_dir = Path(args.output) if args.output else settings.paths.scenario_output_dir
    output_dir = base_output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save log file in the same folder
    log_file = output_dir / "run.log"
    setup_logging(level=settings.log_level, log_file=log_file)
    logger = get_logger(__name__)
    logger.info(f"Log file: {log_file}")
    
    # Load domain names (use path from settings)
    try:
        domain_names = load_domain_names(settings.paths.domains_dir, args.domain, args.domain_file)
        # logger.info(f"Loaded {len(domain_names)} domain(s) from {'--domain' if args.domain else f'{settings.paths.domains_dir}/{args.domain_file or 'domain.json'}'}")
    except Exception as e:
        logger.error(f"Failed to load domains: {e}")
        sys.exit(1)
    
    # Determine number of parallel workers
    max_workers = args.max_workers or settings.pipeline.max_workers
    
    # Initialize LLM client and generator (use path from settings)
    prompt_loader = PromptLoader(settings.paths.prompts_dir)
    
    def process_domain(domain_name: str) -> tuple[str, bool, str]:
        """Process a single domain (for parallel execution)"""
        try:
            # Create a separate LLM client for each thread
            llm_client = create_llm_client(settings)
            generator = ScenarioGenerator(
                llm_client, 
                prompt_loader, 
                num_scenarios=args.num_scenarios
            )
            
            logger.info(f"[START] Generating scenarios for domain: {domain_name}")
            scenarios = generator.generate(domain_name)
            
            # Save each scenario as an individual file by index
            saved_files = []
            for scenario_dict in scenarios:
                # Extract index then remove (used only for filename)
                scenario_index, scenario_dict = remove_index_from_scenario(scenario_dict)
                
                # Save using common utility
                output_path = save_stage_result(
                    output_dir=output_dir,
                    domain_name=domain_name,
                    stage_prefix="scenario",
                    index=scenario_index,
                    data=scenario_dict,
                )
                
                saved_files.append(str(output_path))
                description = scenario_dict.get("scenario", {}).get("description", "")[:50]
                logger.info(f"  - Saved scenario {scenario_index}: {output_path}")
                logger.info(f"    {description}...")
            
            logger.info(f"[END] Saved {len(saved_files)} scenario file(s) for domain: {domain_name}")
            
            return (domain_name, True, f"{len(saved_files)} files")
        
        except Exception as e:
            import traceback
            logger.error(f"[ERROR] Failed to generate scenarios for {domain_name}: {e}")
            traceback.print_exc()
            return (domain_name, False, str(e))
    
    # Run parallel processing
    success_count, fail_count = run_parallel(
        items=domain_names,
        process_fn=process_domain,
        max_workers=max_workers,
        logger=logger,
        task_name="domain",
    )
    
    print(f"\n✅ Stage 1 complete! Results saved at: {output_dir}")
    print(f"   Success: {success_count}, Failure: {fail_count}")
    print(f"Execute next stage: python 2_run_criteria.py --input {output_dir}")


if __name__ == "__main__":
    main()
