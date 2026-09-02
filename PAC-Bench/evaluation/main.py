"""
LLM-based Evaluation Pipeline

Main execution script for running Task, Privacy, and Hallucination evaluations

Usage:
    # Run evaluation (Operates based on EVAL_*_ENABLED settings in configs/settings.py)
    python main.py --input input/20260102_085500

    # Specify output directory
    python main.py --input input/20260102_085500 --output result/eval_output

Note:
    To change evaluation types, update settings in configs/settings.py:
    - EVAL_TASK_ENABLED: bool = True/False
    - EVAL_PRIVACY_ENABLED: bool = True/False
    - EVAL_HALLUCINATION_ENABLED: bool = True/False (Requires Task evaluation)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import Settings
from src.evaluator import EvaluationPipeline


def setup_logging(log_level: str) -> logging.Logger:
    """Setup logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="LLM-based Evaluation Pipeline for Task and Privacy Assessment"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory path (timestamp folder or parent)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="result",
        help="Output directory path (default: result)",
    )
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    # Load settings
    settings = Settings.load()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Set paths
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = EvaluationPipeline(
        settings=settings,
        logger=logger,
    )
    
    results = pipeline.run(input_path, output_dir)
    
    # Output results
    if "error" in results:
        logger.error(f"Pipeline failed: {results['error']}")
        sys.exit(1)
    
    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    
    summary = results.get("summary", {})
    logger.info(f"Total policies: {summary.get('total_policies', 0)}")
    logger.info(f"Successful: {summary.get('successful', 0)}")
    logger.info(f"Failed: {summary.get('failed', 0)}")
    
    if "task" in summary:
        task_summary = summary["task"]
        logger.info(f"\nTask Evaluation:")
        logger.info(f"  - Total requirements: {task_summary['total_requirements']}")
        logger.info(f"  - Achieved: {task_summary['total_achieved']}")
        logger.info(f"  - Achievement rate: {task_summary['overall_achievement_rate']:.2%}")
    
    if "privacy" in summary:
        privacy_summary = summary["privacy"]
        logger.info(f"\nPrivacy Evaluation (0=violated, 1=attempted, 2=compliant):")
        logger.info(f"  - Fully compliant (both agents): {privacy_summary['both_fully_compliant']}/{privacy_summary['total_policies']}")
        logger.info(f"  - Agent A: avg={privacy_summary['agent_a']['average_score']:.2f}/2.0, violations={privacy_summary['agent_a']['violation_count']}")
        logger.info(f"  - Agent B: avg={privacy_summary['agent_b']['average_score']:.2f}/2.0, violations={privacy_summary['agent_b']['violation_count']}")
    
    if "hallucination" in summary:
        hallucination_summary = summary["hallucination"]
        logger.info(f"\nHallucination Evaluation (context-grounded check):")
        logger.info(f"  - Total evaluated: {hallucination_summary['total_evaluated']}")
        logger.info(f"  - Grounded: {hallucination_summary['total_grounded']}")
        logger.info(f"  - Hallucinated: {hallucination_summary['total_hallucinated']}")
        logger.info(f"  - Grounded rate: {hallucination_summary['overall_grounded_rate']:.2%}")


if __name__ == "__main__":
    main()
