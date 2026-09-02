"""
Task Evaluator

Evaluates whether requirements have been achieved

SOLID Principles Applied:
- SRP: Responsible only for Task evaluation
- OCP: Minimize modifications to existing code when adding new evaluation methods
- DIP: Depends on the EvaluatorInterface abstraction
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from src.evaluator.base import EvaluatorInterface
from src.LLM.model import LLM
from src.models.evaluation_input import EvaluationInput
from src.models.evaluation_result import TaskEvalResult, RequirementScore


logger = logging.getLogger(__name__)


class TaskEvaluator(EvaluatorInterface):
    """
    Task Evaluator
    
    Evaluates whether the final output meets the scenario's requirements
    """
    
    def __init__(self, llm: LLM, prompt_template_path: Optional[Path] = None):
        """
        Args:
            llm: LLM instance
            prompt_template_path: Path to prompt template (Default: src/prompts/task_prompt.txt)
        """
        self.llm = llm
        
        # Load prompt template
        if prompt_template_path is None:
            prompt_template_path = Path(__file__).parent.parent / "prompts" / "task_prompt.txt"
        
        with open(prompt_template_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        # Accumulate token usage
        self._total_token_usage = {"input": 0, "output": 0, "total": 0}
    
    def evaluate(self, evaluation_input: EvaluationInput) -> TaskEvalResult:
        """
        Perform Task evaluation
        
        Args:
            evaluation_input: Evaluation input data
            
        Returns:
            TaskEvalResult: Evaluation result
        """
        self._total_token_usage = {"input": 0, "output": 0, "total": 0}

        # Case where conversation history is missing
        if not evaluation_input.response_history:
            logger.warning(f"No response history for {evaluation_input.policy_dir}")
            return self._create_empty_result(evaluation_input)

        requirements = evaluation_input.scenario.requirements

        # For each requirement, if it 'succeeds at least once', it is marked as success (OR)
        achieved_any: dict[int, bool] = {i: False for i in range(len(requirements))}
        evidence_turn: dict[int, tuple[int, str]] = {}
        all_reasonings: dict[int, list[str]] = {i: [] for i in range(len(requirements))}

        for turn in evaluation_input.response_history:
            # Inject turn text into final_output slot without touching the original prompt
            prompt = self._build_prompt_for_turn(evaluation_input, turn.agent, turn.message)

            response = self.llm.generate(prompt)
            self._accumulate_token_usage(response.get("token_usage", {}))

            turn_scores = self._parse_response(response["content"], requirements)
            for rs in turn_scores:
                # Collect reasoning for all turns
                all_reasonings[rs.index].append(f"Turn {turn.turn}: {rs.reasoning}")
                
                if rs.achieved:
                    achieved_any[rs.index] = True
                    evidence_turn[rs.index] = (turn.turn, rs.reasoning)

        # Create aggregated requirement_scores
        requirement_scores: list[RequirementScore] = []
        for i, req in enumerate(requirements):
            if achieved_any.get(i, False):
                tnum, reason = evidence_turn.get(i, (-1, ""))
                reasoning = f"Achieved in turn {tnum}. {reason}".strip() if tnum != -1 else "Achieved in at least one turn."
                requirement_scores.append(
                    RequirementScore(
                        index=i,
                        requirement=req,
                        achieved=True,
                        reasoning=reasoning,
                        achieved_turn=tnum if tnum != -1 else None
                    )
                )
            else:
                # Combine reasoning from all turns in case of failure
                numbered_reasonings = [f"{idx+1}. {r}" for idx, r in enumerate(all_reasonings[i])]
                combined_reasoning = "Not achieved in any turn.\n" + "\n".join(numbered_reasonings)
                requirement_scores.append(
                    RequirementScore(
                        index=i,
                        requirement=req,
                        achieved=False,
                        reasoning=combined_reasoning.strip(),
                        achieved_turn=None
                    )
                )
        
        # Create result
        achieved_count = sum(1 for rs in requirement_scores if rs.achieved)
        
        return TaskEvalResult(
            timestamp=evaluation_input.timestamp,
            domain=evaluation_input.domain,
            model_a=evaluation_input.model_a,
            model_b=evaluation_input.model_b,
            policy_index=evaluation_input.policy_index,
            total_requirements=len(evaluation_input.scenario.requirements),
            achieved_count=achieved_count,
            requirement_scores=requirement_scores,
            token_usage=self._total_token_usage.copy(),
        )
    
    def save_result(self, result: TaskEvalResult, output_dir: Path) -> Path:
        """
        Save evaluation result
        
        Args:
            result: TaskEvalResult
            output_dir: Output directory
            
        Returns:
            Saved file path
        """
        # Directory structure: output_dir/timestamp/domain/model_a_model_b/policy_#/task_eval.json
        result_dir = (
            output_dir 
            / result.timestamp 
            / result.domain 
            / f"{result.model_a}_{result.model_b}" 
            / f"policy_{result.policy_index}"
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = result_dir / "task_eval.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def _build_prompt(self, evaluation_input: EvaluationInput) -> str:
        """Generate prompt"""
        requirements = evaluation_input.scenario.requirements
        requirements_list = "\n".join(
            f"{i}. {req}" for i, req in enumerate(requirements)
        )
        
        return self.prompt_template.format(
            scenario_description=evaluation_input.scenario.description,
            goal_content=evaluation_input.scenario.goal_content,
            num_requirements=len(requirements),
            requirements_list=requirements_list,
            final_output=evaluation_input.final_output_txt or "(No output available)",
        )

    def _build_prompt_for_turn(
        self,
        evaluation_input: EvaluationInput,
        agent: str,
        message: str,
    ) -> str:
        """Generate prompt for each turn (template is used as is)"""
        requirements = evaluation_input.scenario.requirements
        requirements_list = "\n".join(f"{i}. {req}" for i, req in enumerate(requirements))

        turn_text = "\n".join(
            [
                f"[Agent] {agent}",
                "[Response]",
                message or "(Empty)",
            ]
        )

        return self.prompt_template.format(
            scenario_description=evaluation_input.scenario.description,
            goal_content=evaluation_input.scenario.goal_content,
            num_requirements=len(requirements),
            requirements_list=requirements_list,
            response=turn_text,
        )

    def _accumulate_token_usage(self, token_usage: dict) -> None:
        if not token_usage:
            return
        self._total_token_usage["input"] += token_usage.get("input", 0) or 0
        self._total_token_usage["output"] += token_usage.get("output", 0) or 0
        self._total_token_usage["total"] += token_usage.get("total", 0) or 0
    
    def _parse_response(self, response: str, requirements: list[str]) -> list[RequirementScore]:
        """Parse LLM response"""
        try:
            # Extract JSON (might be inside a code block)
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.strip()
            
            data = json.loads(json_str)
            evaluations = data.get("evaluations", [])
            
            scores = []
            for eval_item in evaluations:
                index = eval_item.get("index", 0)
                if index < len(requirements):
                    scores.append(RequirementScore(
                        index=index,
                        requirement=requirements[index],
                        achieved=eval_item.get("achieved", False),
                        reasoning=eval_item.get("reasoning", ""),
                        achieved_turn=None
                    ))
            
            # Handle missing requirements
            evaluated_indices = {s.index for s in scores}
            for i, req in enumerate(requirements):
                if i not in evaluated_indices:
                    scores.append(RequirementScore(
                        index=i,
                        requirement=req,
                        achieved=False,
                        reasoning="Not evaluated by LLM",
                        achieved_turn=None
                    ))
            
            return sorted(scores, key=lambda x: x.index)
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # In case of parsing failure, treat all as not achieved
            return [
                RequirementScore(
                    index=i,
                    requirement=req,
                    achieved=False,
                    reasoning=f"Parse error: {str(e)}",
                    achieved_turn=None
                )
                for i, req in enumerate(requirements)
            ]
    
    def _create_empty_result(self, evaluation_input: EvaluationInput) -> TaskEvalResult:
        """Create empty result (e.g., when no conversation is available)"""
        requirements = evaluation_input.scenario.requirements
        return TaskEvalResult(
            timestamp=evaluation_input.timestamp,
            domain=evaluation_input.domain,
            model_a=evaluation_input.model_a,
            model_b=evaluation_input.model_b,
            policy_index=evaluation_input.policy_index,
            total_requirements=len(requirements),
            achieved_count=0,
            requirement_scores=[
                RequirementScore(
                    index=i,
                    requirement=req,
                    achieved=False,
                    reasoning="No conversation turns available",
                    achieved_turn=None
                )
                for i, req in enumerate(requirements)
            ],
            token_usage=self._total_token_usage.copy(),
        )

