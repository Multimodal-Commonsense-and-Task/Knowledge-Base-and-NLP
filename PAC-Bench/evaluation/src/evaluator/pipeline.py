"""
Evaluation Pipeline

Main pipeline that orchestrates the evaluation process

SOLID Principles Applied:
- SRP: Responsible only for orchestrating the pipeline
- OCP: Easy to add new evaluation types
- DIP: Depends on the Evaluator abstraction
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from configs.settings import Settings
from src.LLM.model import LLM
from src.evaluator.parser.input_parser import InputParser
from src.evaluator.task import TaskEvaluator
from src.evaluator.privacy import PrivacyEvaluator
from src.evaluator.hallucination import HallucinationEvaluator
from src.models.evaluation_input import EvaluationInput
from utils.parallel import run_parallel


class EvaluationPipeline:
    """
    Evaluation Pipeline
    
    Executes the entire evaluation process using multiple evaluators (Task, Privacy, etc.)
    """
    
    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger,
    ):
        self.settings = settings
        self.logger = logger
        
        # Initialize LLM
        self.llm = LLM(
            api_key=settings.llm.api_key,
            model_name=settings.llm.model,
        )
        
        # Initialize Evaluators (Enabled based on settings.evaluator)
        self.task_evaluator = TaskEvaluator(self.llm) if settings.evaluator.task_enabled else None
        self.privacy_evaluator = PrivacyEvaluator(self.llm) if settings.evaluator.privacy_enabled else None
        # Hallucination evaluation requires Task evaluation results, so check task_enabled as well
        self.hallucination_evaluator = (
            HallucinationEvaluator(self.llm) 
            if settings.evaluator.hallucination_enabled and settings.evaluator.task_enabled 
            else None
        )
    
    def run(self, input_path: Path, output_dir: Path) -> dict:
        """
        Run evaluation pipeline
        
        Args:
            input_path: Input path (timestamp folder)
            output_dir: Output directory
            
        Returns:
            dict: Summary of evaluation results
        """
        # Discover policy directories
        policy_dirs = InputParser.discover_policy_dirs(input_path)
        self.logger.info(f"Discovered {len(policy_dirs)} policy directories")
        
        if not policy_dirs:
            self.logger.error(f"No policy directories found in {input_path}")
            return {"error": "No policy directories found"}
        
        # For storing results
        results = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "input_path": str(input_path),
            "total_policies": len(policy_dirs),
            "task_results": [],
            "privacy_results": [],
            "hallucination_results": [],
            "summary": {},
        }
        
        # Create subfolder based on evaluator start time
        output_dir = output_dir / results["timestamp"]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each policy folder
        def process_policy(policy_dir: Path) -> tuple[str, bool, str]:
            try:
                # Parse input data
                evaluation_input = InputParser.parse_policy_dir(policy_dir)
                identifier = f"{evaluation_input.domain}/{evaluation_input.model_a}_{evaluation_input.model_b}/policy_{evaluation_input.policy_index}"
                
                # Task evaluation
                task_result = None
                if self.task_evaluator:
                    try:
                        task_result = self.task_evaluator.evaluate(evaluation_input)
                        task_path = self.task_evaluator.save_result(task_result, output_dir)
                        results["task_results"].append(task_result.to_dict())
                        self.logger.info(f"Task evaluation saved: {task_path}")
                    except Exception as e:
                        self.logger.error(f"Task evaluation failed for {identifier}: {e}")
                
                # Privacy evaluation
                if self.privacy_evaluator:
                    try:
                        privacy_result = self.privacy_evaluator.evaluate(evaluation_input)
                        privacy_path = self.privacy_evaluator.save_result(privacy_result, output_dir)
                        results["privacy_results"].append(privacy_result.to_dict())
                        self.logger.info(f"Privacy evaluation saved: {privacy_path}")
                    except Exception as e:
                        self.logger.error(f"Privacy evaluation failed for {identifier}: {e}")
                
                # Hallucination evaluation (Requires Task evaluation results)
                if self.hallucination_evaluator and task_result:
                    try:
                        hallucination_result = self.hallucination_evaluator.evaluate(
                            evaluation_input, task_result
                        )
                        hallucination_path = self.hallucination_evaluator.save_result(
                            hallucination_result, output_dir
                        )
                        results["hallucination_results"].append(hallucination_result.to_dict())
                        self.logger.info(f"Hallucination evaluation saved: {hallucination_path}")
                    except Exception as e:
                        self.logger.error(f"Hallucination evaluation failed for {identifier}: {e}")
                
                return (identifier, True, str(policy_dir))
                
            except Exception as e:
                self.logger.error(f"Failed to process {policy_dir}: {e}")
                return (str(policy_dir), False, str(e))
        
        # Parallel processing
        success, fail = run_parallel(
            items=policy_dirs,
            process_fn=process_policy,
            max_workers=self.settings.pipeline.max_workers,
            logger=self.logger,
            task_name="policy",
        )
        
        # Generate summary
        results["summary"] = self._generate_summary(results, success, fail)
        
        # Save entire results
        summary_path = output_dir / f"evaluation_summary_{results['timestamp']}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Evaluation complete. Summary saved to {summary_path}")
        
        return results
    
    def _generate_summary(self, results: dict, success: int, fail: int) -> dict:
        """Generate result summary"""
        summary = {
            "total_policies": results["total_policies"],
            "successful": success,
            "failed": fail,
        }
        
        # Task evaluation summary
        # if results["task_results"]:
        #     task_results = results["task_results"]
        #     total_requirements = sum(r["total_requirements"] for r in task_results)
        #     total_achieved = sum(r["achieved_count"] for r in task_results)
            
        #     summary["task"] = {
        #         "total_evaluations": len(task_results),
        #         "total_requirements": total_requirements,
        #         "total_achieved": total_achieved,
        #         "overall_achievement_rate": total_achieved / total_requirements if total_requirements > 0 else 0,
        #     }
        
        # Privacy evaluation summary (0/1/2 scoring system, average per agent)
        if results["privacy_results"]:
            privacy_results = results["privacy_results"]
            
            # Count fully compliant (all scores 2)
            both_fully_compliant = sum(1 for r in privacy_results if r["both_fully_compliant"])
            
            # Calculate average score per agent (score for each scenario is the minimum of all turns)
            agent_a_scores = [r["agent_a_result"]["score"] for r in privacy_results]
            agent_b_scores = [r["agent_b_result"]["score"] for r in privacy_results]
            
            agent_a_overall_avg = sum(agent_a_scores) / len(agent_a_scores) if agent_a_scores else 0
            agent_b_overall_avg = sum(agent_b_scores) / len(agent_b_scores) if agent_b_scores else 0
            
            # Number of violations (0 points)
            agent_a_violations = sum(1 for r in privacy_results if r["agent_a_result"]["has_violation"])
            agent_b_violations = sum(1 for r in privacy_results if r["agent_b_result"]["has_violation"])
            
            summary["privacy"] = {
                "total_policies": len(privacy_results),
                "both_fully_compliant": both_fully_compliant,
                "agent_a": {
                    "average_score": agent_a_overall_avg,
                    "violation_count": agent_a_violations,
                },
                "agent_b": {
                    "average_score": agent_b_overall_avg,
                    "violation_count": agent_b_violations,
                },
            }
        
        # Hallucination evaluation summary
        if results["hallucination_results"]:
            hallucination_results = results["hallucination_results"]
            
            total_evaluated = sum(r["total_num"] for r in hallucination_results)
            total_grounded = sum(r["success_num"] for r in hallucination_results)
            total_hallucinated = total_evaluated - total_grounded
            
            summary["hallucination"] = {
                "total_policies": len(hallucination_results),
                "total_evaluated": total_evaluated,
                "total_grounded": total_grounded,
                "total_hallucinated": total_hallucinated,
                "overall_grounded_rate": total_grounded / total_evaluated if total_evaluated > 0 else 1.0,
            }
        
        return summary

