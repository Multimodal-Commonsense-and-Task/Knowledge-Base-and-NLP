"""
Hallucination Evaluator

Evaluation of context-groundedness: Checks if successful requirements actually used the provided context.

Evaluation Logic:
1. Targets only the requirements that were successful in the Task evaluation and are associated with context.
2. Checks if the context was actually used in the conversation turn where the requirement was achieved.
3. If the context was not used, it is judged as a hallucination.

SOLID Principles Applied:
- SRP: Responsible only for Hallucination evaluation
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
from src.models.evaluation_result import (
    TaskEvalResult,
    HallucinationEvalResult,
    HallucinationScore,
)


logger = logging.getLogger(__name__)

MAX_PARSE_RETRIES = 10


class HallucinationEvaluator(EvaluatorInterface):
    """
    Hallucination Evaluator
    
    Evaluates whether context was actually used for successful requirements
    """
    
    def __init__(
        self,
        llm: LLM,
        prompt_template_path: Optional[Path] = None,
    ):
        """
        Args:
            llm: LLM instance
            prompt_template_path: Path to prompt template (Default: src/prompts/context_alignment_prompt.txt)
        """
        self.llm = llm
        
        # Load prompt template
        if prompt_template_path is None:
            prompt_template_path = Path(__file__).parent.parent / "prompts" / "context_alignment_prompt.txt"
        
        with open(prompt_template_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        
        # Accumulate token usage
        self._total_token_usage = {"input": 0, "output": 0, "total": 0}
    
    def evaluate(
        self,
        evaluation_input: EvaluationInput,
        task_result: TaskEvalResult,
    ) -> HallucinationEvalResult:
        """
        Perform Hallucination evaluation
        
        Args:
            evaluation_input: Evaluation input data
            task_result: Task evaluation result (contains successful requirements and achievement turn info)
            
        Returns:
            HallucinationEvalResult: Evaluation result
        """
        self._total_token_usage = {"input": 0, "output": 0, "total": 0}
        
        evaluations: list[HallucinationScore] = []
        
        # Target only successful requirements
        for req_score in task_result.requirement_scores:
            if not req_score.achieved:
                continue
            
            # Get contexts and privacy_policy corresponding to the requirement
            ctx_policy_data = evaluation_input.get_contexts_and_policies_for_requirement(
                req_score.index
            )
            
            contexts = ctx_policy_data["contexts"]
            privacy_policies = ctx_policy_data["privacy_policies"]
            
            # Not a target for hallucination check if there is no context
            if not contexts:
                continue
            
            # Get achieved turn
            achieved_turn = req_score.achieved_turn
            if achieved_turn is None:
                logger.warning(
                    f"Requirement {req_score.index} achieved but no turn info. Skipping."
                )
                continue
            
            # Find the message for that turn
            turn_message = None
            for turn in evaluation_input.response_history:
                if turn.turn == achieved_turn:
                    turn_message = turn.message
                    break
            
            if turn_message is None:
                logger.warning(
                    f"Turn {achieved_turn} not found in response_history. Skipping."
                )
                continue
            
            # Build prompt and call LLM
            prompt = self._build_prompt(
                requirement=evaluation_input.scenario.requirements[req_score.index],
                contexts=contexts,
                privacy_policies=privacy_policies,
                agent_response=turn_message,
            )
            
            parsed = None
            parse_error: Optional[Exception] = None
            for attempt in range(1, MAX_PARSE_RETRIES + 1):
                response = self.llm.generate(prompt)
                self._accumulate_token_usage(response.get("token_usage", {}))
                try:
                    parsed = self._parse_response(response.get("content", ""))
                    break
                except ValueError as err:
                    parse_error = err
                    logger.warning(
                        "Hallucination response parse failed (attempt %d/%d) for requirement %s: %s",
                        attempt,
                        MAX_PARSE_RETRIES,
                        req_score.index,
                        err,
                    )
                    print("LLM Response:", response.get("content", ""))
            if parsed is None:
                logger.error(
                    "Hallucination response parse permanently failed for requirement %s after %d attempts",
                    req_score.index,
                    MAX_PARSE_RETRIES,
                )
                parsed = {
                    "reasoning": f"Parse error after {MAX_PARSE_RETRIES} attempts: {parse_error}",
                    "evidences": [],
                    "evaluation": False,
                }
            
            evaluation_str = "grounded" if parsed.get("evaluation", False) else "hallucinated"
            
            evaluations.append(HallucinationScore(
                requirement_index=req_score.index,
                requirement=evaluation_input.scenario.requirements[req_score.index],
                turn=achieved_turn,
                evaluation=evaluation_str,
                evidences=parsed.get("evidences", []),
                reasoning=parsed.get("reasoning", ""),
            ))
        
        # Aggregate results
        total_num = len(evaluations)
        success_num = sum(1 for e in evaluations if e.evaluation == "grounded")
        
        return HallucinationEvalResult(
            timestamp=evaluation_input.timestamp,
            domain=evaluation_input.domain,
            model_a=evaluation_input.model_a,
            model_b=evaluation_input.model_b,
            policy_index=evaluation_input.policy_index,
            total_num=total_num,
            success_num=success_num,
            evaluations=evaluations,
            token_usage=self._total_token_usage.copy(),
        )
    
    def save_result(self, result: HallucinationEvalResult, output_dir: Path) -> Path:
        """
        Save evaluation result
        
        Args:
            result: HallucinationEvalResult
            output_dir: Output directory
            
        Returns:
            Saved file path
        """
        # Directory structure: output_dir/timestamp/domain/model_a_model_b/policy_#/hallucination_eval.json
        result_dir = (
            output_dir 
            / result.timestamp 
            / result.domain 
            / f"{result.model_a}_{result.model_b}" 
            / f"policy_{result.policy_index}"
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = result_dir / "hallucination_eval.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def _build_prompt(
        self,
        requirement: str,
        contexts: list,
        privacy_policies: list,
        agent_response: str,
    ) -> str:
        """Generate prompt"""
        # Combine Contexts
        context_str = "\n\n".join(
            f"[Context {i+1}]\n{ctx.content}" for i, ctx in enumerate(contexts)
        )
        
        # Combine Privacy Policies
        if privacy_policies:
            policy_str = "\n".join(
                f"- {pp.content}" for pp in privacy_policies
            )
        else:
            policy_str = "(No privacy policy defined)"
        
        # Replace Placeholders
        prompt = self.prompt_template.replace("$TASK_REQUIREMENT$", requirement)
        prompt = prompt.replace("$CONTEXT$", context_str)
        prompt = prompt.replace("$PRIVACY_POLICY$", policy_str)
        prompt = prompt.replace("$AGENT_RESPONSE$", agent_response)
        
        return prompt
    
    def _parse_response(self, response: str) -> dict:
        """Parse LLM response"""
        try:
            json_str = response.strip()
            data = json.loads(json_str)
            
            return {
                "reasoning": data.get("reasoning", ""),
                "evidences": data.get("evidences", []),
                "evaluation": data.get("evaluation", False),
            }
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"Failed to parse LLM response: {e}") from e
    
    def _accumulate_token_usage(self, token_usage: dict) -> None:
        """Accumulate token usage"""
        if not token_usage:
            return
        self._total_token_usage["input"] += token_usage.get("input", 0) or 0
        self._total_token_usage["output"] += token_usage.get("output", 0) or 0
        self._total_token_usage["total"] += token_usage.get("total", 0) or 0

