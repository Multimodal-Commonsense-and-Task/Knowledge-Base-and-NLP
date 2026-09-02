"""
JSON parsing utilities for dataset generation.
Handles parsing JSON from LLM predictions with various edge cases.
"""
import json
import re
from typing import Optional, Dict, Any


def parse_json_prediction(prediction_text: str) -> Optional[Dict[Any, Any]]:
    """
    Parse JSON from prediction text, handling markdown code blocks and raw JSON.
    
    Handles various edge cases:
    - Markdown code blocks (```json ... ```)
    - Unquoted predicates in graph arrays
    - Truncated JSON (when max_tokens is exceeded)
    - List of predictions
    
    Args:
        prediction_text: Raw prediction text from LLM (str or list)
    
    Returns:
        Parsed JSON object/array, or None if parsing fails
    """
    if not prediction_text:
        return None
    
    # Handle list of predictions (when n > 1)
    if isinstance(prediction_text, list):
        # Try to parse each prediction
        parsed_list = []
        for pred in prediction_text:
            parsed = parse_json_prediction(pred)
            if parsed:
                parsed_list.append(parsed)
            else:
                parsed_list.append(pred)  # Keep original if parsing fails
        return parsed_list if parsed_list else None
    
    # Handle string prediction
    text = str(prediction_text).strip()
    
    # Remove markdown code blocks if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Fix unquoted predicates in graph arrays
    # Pattern: ["subject", unquoted_word(s), "object"] -> ["subject", "unquoted_word(s)", "object"]
    # This handles cases like: ["stovetop", on, "counter"] or ["x", part of, "y"]
    def fix_unquoted_predicates(match):
        """Add quotes around unquoted middle element in triplets"""
        subj = match.group(1)    # Quoted subject
        pred = match.group(2).strip()    # Unquoted predicate (may have spaces)
        obj = match.group(3)     # Quoted object
        return f'[{subj}, "{pred}", {obj}]'
    
    # Match pattern: ["...", word(s)_without_quotes, "..."]
    # The pattern looks for array elements where the middle element lacks quotes
    # Handles both single words (on, in, above) and phrases (part of, directly beneath)
    pattern = r'\[("[^"]*"),\s*([a-zA-Z][a-zA-Z0-9\s]*?),\s*("[^"]*")\]'
    text = re.sub(pattern, fix_unquoted_predicates, text)
    
    # Try to extract JSON object/array
    json_match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', text)
    if json_match:
        json_str = json_match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix truncated JSON (common when max_tokens is exceeded)
            # Look for incomplete scenarios array and try to close it
            if '"scenarios"' in json_str and '"scenarios": [' in json_str:
                # Try to find the last complete scenario object
                # Match complete scenario objects: {"hazard": "...", "action": "..."}
                scenario_pattern = r'\{\s*"hazard"\s*:\s*"[^"]*"\s*,\s*"action"\s*:\s*"[^"]*"\s*\}'
                scenarios = re.findall(scenario_pattern, json_str)
                if scenarios:
                    # Reconstruct valid JSON
                    fixed_json = '{"scenarios": [' + ','.join(scenarios) + ']}'
                    try:
                        return json.loads(fixed_json)
                    except json.JSONDecodeError:
                        pass
    
    # Try parsing the entire text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # If all parsing fails, return None
    return None


def count_jsonl_records(jsonl_path: str) -> int:
    """
    Count the number of records in a JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
    
    Returns:
        Number of non-empty records in the file
    """
    count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count

