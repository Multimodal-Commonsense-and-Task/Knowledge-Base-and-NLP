"""
Memory Parser

Parser for parsing Memory responses
New format: arrays of [{requirements_index, content}, ...] in both agent_a.memory and agent_b.memory
"""

from src.generators.parsers.base_parser import BaseJSONParser


class MemoryParser(BaseJSONParser[dict]):
    """
    Memory response parser
    
    Extracts a list of Memories for each agent from the LLM response
    
    Expected format:
    {
        "agent_a": {
            "memory": [{"requirements_index": [0], "content": "..."}, ...]
        },
        "agent_b": {
            "memory": [{"requirements_index": [1], "content": "..."}, ...]
        }
    }
    """
    
    def parse(self, response_text: str) -> dict:
        """
        Converts response text into a Memory dict for each agent.
        
        Args:
            response_text: LLM response text
            
        Returns:
            dict: {"agent_a": {"memory": [...]}, "agent_b": {"memory": [...]}}
        """
        data = self.parse_json(response_text)
        
        if not isinstance(data, dict):
            raise ValueError("Response must be a JSON object")
        
        result = {
            "agent_a": {"memory": []},
            "agent_b": {"memory": []},
        }
        
        # Parse agent_a
        if "agent_a" in data:
            agent_a_data = data["agent_a"]
            if isinstance(agent_a_data, dict) and "memory" in agent_a_data:
                result["agent_a"]["memory"] = self._validate_memory_list(agent_a_data["memory"], "agent_a")
        
        # Parse agent_b
        if "agent_b" in data:
            agent_b_data = data["agent_b"]
            if isinstance(agent_b_data, dict) and "memory" in agent_b_data:
                result["agent_b"]["memory"] = self._validate_memory_list(agent_b_data["memory"], "agent_b")
        
        return result
    
    def _validate_memory_list(self, memory_list: list, agent_name: str) -> list[dict]:
        """Validate Memory list"""
        if not isinstance(memory_list, list):
            raise ValueError(f"{agent_name}.memory must be a list")
        
        validated = []
        for mem in memory_list:
            if not isinstance(mem, dict):
                raise ValueError(f"Each memory in {agent_name}.memory must be an object")
            
            # Validate requirements_index: allows int/str or list[int|str]
            requirements_index = mem.get("requirements_index")
            if requirements_index is None:
                raise ValueError("Each memory must have 'requirements_index'")

            # normalize to list[int]
            if isinstance(requirements_index, (int, str)):
                requirement_indices = [requirements_index]
            elif isinstance(requirements_index, list):
                requirement_indices = requirements_index
            else:
                raise ValueError("'requirements_index' must be int, str, or list")

            if not requirement_indices:
                raise ValueError("'requirements_index' list must be non-empty")

            normalized_indices: list[int] = []
            for mi in requirement_indices:
                if isinstance(mi, int):
                    normalized_indices.append(mi)
                elif isinstance(mi, str) and mi.strip().isdigit():
                    normalized_indices.append(int(mi.strip()))
                else:
                    raise ValueError("'requirements_index' items must be ints or digit strings")
            
            # Validate content
            content = mem.get("content")
            if not content:
                raise ValueError(f"Each memory must have 'content'")
            
            validated.append({
                "requirements_index": normalized_indices,
                "content": content,
            })
        
        return validated

