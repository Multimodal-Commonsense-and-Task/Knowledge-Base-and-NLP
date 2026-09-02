"""
Judge prompts for evaluating hazard predictions
Used in results evaluation to compare predicted hazards with ground truth
"""
from typing import Dict


# SYSTEM PROMPT for judge model
JUDGE_SYSTEM_PROMPT = "You are an expert evaluator for safety hazard descriptions. Respond only with valid JSON."


# USER PROMPT template for hazard evaluation
def get_hazard_judge_prompt(ground_truth_hazard: str, predicted_hazard: str) -> str:
    """
    Generate judge prompt for evaluating hazard prediction
    
    Args:
        ground_truth_hazard: Ground truth hazard from CSV
        predicted_hazard: Predicted hazard from model
        
    Returns:
        Formatted judge prompt string
    """
    return f"""You are evaluating whether a predicted hazard description matches the ground truth hazard description.

Ground Truth Hazard: {ground_truth_hazard}

Predicted Hazard: {predicted_hazard}

Task: Determine if the predicted hazard accurately describes or matches the ground truth hazard. They don't need to be identical, but should convey the same or very similar safety concern.

Respond with JSON only:
{{
    "match": true or false,
    "reasoning": "brief explanation"
}}"""


def get_hazard_judge_messages(ground_truth_hazard: str, predicted_hazard: str) -> list[Dict[str, str]]:
    """
    Generate judge messages for hazard evaluation
    
    Args:
        ground_truth_hazard: Ground truth hazard from CSV
        predicted_hazard: Predicted hazard from model
        
    Returns:
        List of message dictionaries (system and user)
    """
    return [
        {
            "role": "system",
            "content": JUDGE_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": get_hazard_judge_prompt(ground_truth_hazard, predicted_hazard)
        }
    ]

