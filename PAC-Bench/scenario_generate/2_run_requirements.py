"""
Stage 2: Requirements Generation

Generate requirements from previous stage results (Scenarios with goals)

Usage:
    python 2_run_requirements.py --input result/1_scenario/20241222_120000
    python 2_run_requirements.py --input result/1_scenario/20241222_120000/Domain/scenario_0.json
    python 2_run_requirements.py --input result/1_scenario/20241222_120000 --max_workers 8
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import Settings
from src.generators.stage2_requirements_generator import RequirementsGenerator
from src.llm.factory import create_llm_client
from src.prompts.prompt_loader import PromptLoader
from src.utils.logging import setup_logging, get_logger
from src.utils.parallel import run_parallel
from src.utils.file_io import load_stage_files, save_stage_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: Goals → Requirements Generation"
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
        help="Output directory (default: settings.paths.requirements_output_dir)",
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

    settings = Settings.load()

    if args.provider:
        settings.llm.provider = args.provider
    if args.model:
        settings.llm.model = args.model
    if args.log_level:
        settings.log_level = args.log_level

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output) if args.output else settings.paths.requirements_output_dir
    output_dir = base_output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "run.log"
    setup_logging(level=settings.log_level, log_file=log_file)
    logger = get_logger(__name__)
    logger.info(f"Log file: {log_file}")

    try:
        scenario_results = load_stage_files(args.input, "scenario")
        logger.info(f"Loaded {len(scenario_results)} scenario result(s)")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(f"Failed to load scenario results: {exc}")
        sys.exit(1)

    max_workers = args.max_workers or settings.pipeline.max_workers
    prompt_loader = PromptLoader(settings.paths.prompts_dir)

    def process_scenario_result(item: tuple[str, int, Path, dict]) -> tuple[str, bool, str]:
        """Process a single scenario_#.json file (for parallel execution)"""
        domain_name, index, source_path, data = item
        try:
            llm_client = create_llm_client(settings)
            generator = RequirementsGenerator(
                llm_client,
                prompt_loader,
                max_requirements=settings.requirements.max_count,
            )

            logger.info(f"[START] Generating requirements for domain: {domain_name}, index: {index}")
            requirements = generator.generate_from_dict(data, domain_name)

            output_data = data.copy()
            output_data.setdefault("scenario", {}).setdefault("goal", {})
            output_data["scenario"]["goal"]["requirements"] = requirements

            output_path = save_stage_result(
                output_dir=output_dir,
                domain_name=domain_name,
                stage_prefix="requirements",
                index=index,
                data=output_data,
            )

            logger.info(f"[END] Saved: {output_path}")
            logger.info(f"  - Domain: {domain_name}")
            logger.info(f"  - Requirements count: {len(requirements)}")

            return (f"{domain_name}/index_{index}", True, str(output_path))

        except Exception as exc:  # pragma: no cover - defensive logging
            import traceback

            logger.error(f"[ERROR] Failed to generate requirements for {domain_name}/index_{index}: {exc}")
            traceback.print_exc()
            return (f"{domain_name}/index_{index}", False, str(exc))

    success_count, fail_count = run_parallel(
        items=scenario_results,
        process_fn=process_scenario_result,
        max_workers=max_workers,
        logger=logger,
        task_name="scenario",
    )

    print(f"\n✅ Stage 2 complete! Results saved at: {output_dir}")
    print(f"   Success: {success_count}, Failure: {fail_count}")
    print(f"Execute next stage: python 3_run_memory.py --input {output_dir}")


if __name__ == "__main__":
    main()
