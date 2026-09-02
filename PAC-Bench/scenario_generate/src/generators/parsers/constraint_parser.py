"""
Constraint Parser

Parser for parsing Privacy Constraint responses
New format: agent_a.privacy_constraint and agent_b.privacy_constraint, each containing an array of [{content, violation_keywords, memory_idx(list[int]), reason}, ...]
"""

from src.generators.parsers.base_parser import BaseJSONParser


class ConstraintParser(BaseJSONParser[dict]):
    """
    Privacy Constraint response parser
    
    Extracts Privacy Constraint lists for each agent from the LLM response.
    
    Expected format:
    {
        "agent_a": {
            "privacy_constraint": [{"content": "...", "violation_keywords": ["..."], "memory_idx": [0], "reason": "..."}, ...]
        },
        "agent_b": {
            "privacy_constraint": [{"content": "...", "violation_keywords": ["..."], "memory_idx": [1], "reason": "..."}, ...]
        }
    }
    """
    
    def parse(self, response_text: str) -> dict:
        """
        Converts the response text into a Privacy Constraint dictionary for each agent.
        
        Args:
            response_text: LLM response text
            
        Returns:
            dict: {"agent_a": {"privacy_constraint": [...]}, "agent_b": {"privacy_constraint": [...]}}
        """
        data = self.parse_json(response_text)
        
        if not isinstance(data, dict):
            raise ValueError("Response must be a JSON object")
        
        result = {
            "agent_a": {"privacy_constraint": []},
            "agent_b": {"privacy_constraint": []},
        }
        
        # Parse agent_a
        if "agent_a" in data:
            agent_a_data = data["agent_a"]
            if isinstance(agent_a_data, dict) and "privacy_constraint" in agent_a_data:
                result["agent_a"]["privacy_constraint"] = self._validate_constraint_list(agent_a_data["privacy_constraint"], "agent_a")
        
        # Parse agent_b
        if "agent_b" in data:
            agent_b_data = data["agent_b"]
            if isinstance(agent_b_data, dict) and "privacy_constraint" in agent_b_data:
                result["agent_b"]["privacy_constraint"] = self._validate_constraint_list(agent_b_data["privacy_constraint"], "agent_b")
        
        return result
    
    def _validate_constraint_list(self, constraint_list: list, agent_name: str) -> list[dict]:
        """Validate Privacy Constraint list"""
        if not isinstance(constraint_list, list):
            raise ValueError(f"{agent_name}.privacy_constraint must be a list")
        
        validated = []
        for constraint in constraint_list:
            if not isinstance(constraint, dict):
                raise ValueError(f"Each constraint in {agent_name}.privacy_constraint must be an object")
            
            # Validate content
            content = constraint.get("content")
            if not content:
                raise ValueError(f"Each constraint must have 'content'")
            
            # Validate memory_idx
            memory_idx = constraint.get("memory_idx")
            if memory_idx is None:
                raise ValueError("Each constraint must have 'memory_idx'")
            if isinstance(memory_idx, int):
                memory_idx = [memory_idx]
            if not isinstance(memory_idx, list) or len(memory_idx) == 0:
                raise ValueError("Each constraint 'memory_idx' must be a non-empty list of integers")
            if not all(isinstance(i, int) for i in memory_idx):
                raise ValueError("Each constraint 'memory_idx' must be a list of integers")
            
            # Validate reason (optional, include if present)
            reason = constraint.get("reason", "")

            # Validate violation_keywords (required)
            violation_keywords = constraint.get("violation_keywords")
            if not isinstance(violation_keywords, list) or len(violation_keywords) == 0:
                raise ValueError(f"Each constraint must have non-empty 'violation_keywords' list")
            if not all(isinstance(k, str) and k.strip() for k in violation_keywords):
                raise ValueError(f"Each constraint 'violation_keywords' must be a list of non-empty strings")
            
            validated.append({
                "content": content,
                "violation_keywords": violation_keywords,
                "memory_idx": memory_idx,
                "reason": reason,
            })
        
        return validated

