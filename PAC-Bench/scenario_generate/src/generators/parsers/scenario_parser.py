"""
Scenario Parser

Parser for parsing Scenario generation responses
Dictionary-based processing
"""

from src.generators.parsers.base_parser import BaseJSONParser


class ScenarioParser(BaseJSONParser[list[dict]]):
    """
    Scenario response parser
    
    Extracts a list of Scenario dictionaries from the LLM response.
    """
    
    def parse(self, response_text: str) -> list[dict]:
        """
        Parses the response text to extract a list of Scenario dictionaries.
        
        Args:
            response_text: LLM response text
            
        Returns:
            list[dict]: List of parsed Scenario dictionaries
        """
        data = self.parse_json(response_text)
        
        # Convert to list if it's not already a list
        if isinstance(data, dict):
            if "scenarios" in data:
                scenarios_data = data["scenarios"]
            elif "scenario" in data:
                scenarios_data = [data]
            else:
                scenarios_data = [data]
        elif isinstance(data, list):
            scenarios_data = data
        else:
            raise ValueError("Response must be a JSON array or object")
        
        scenarios = []
        for idx, item in enumerate(scenarios_data):
            scenario = self._validate_scenario(item, idx)
            scenarios.append(scenario)
        
        return scenarios
    
    def _validate_scenario(self, data: dict, default_index: int) -> dict:
        """Validate and normalize a single Scenario"""
        # If the "scenario" key exists, use the data within it
        if "scenario" in data:
            scenario_data = data["scenario"]
        else:
            scenario_data = data
        
        # Validate description
        if not scenario_data.get("description"):
            raise ValueError("Scenario must have 'description' field")
        
        # Validate agent_a
        agent_a = scenario_data.get("agent_a")
        if not agent_a:
            raise ValueError("Scenario must have 'agent_a' field")
        self._validate_agent(agent_a, "agent_a")
        
        # Validate agent_b
        agent_b = scenario_data.get("agent_b")
        if not agent_b:
            raise ValueError("Scenario must have 'agent_b' field")
        self._validate_agent(agent_b, "agent_b")
        
        # Validate goal
        goal = scenario_data.get("goal")
        if not goal:
            raise ValueError("Scenario must have 'goal' field")
        if not goal.get("content"):
            raise ValueError("Goal must have 'content' field")
        
        # Set index (use default if not present)
        if "index" not in scenario_data:
            scenario_data["index"] = default_index
        
        # Return in the final format
        return {"scenario": scenario_data}
    
    def _validate_agent(self, data: dict, agent_name: str) -> None:
        """Validate Agent"""
        profile = data.get("profile")
        if not profile:
            raise ValueError(f"{agent_name} must have 'profile' field")
        
        if not profile.get("organization"):
            raise ValueError(f"{agent_name} profile must have 'organization' field")
        
        if not profile.get("expertise"):
            raise ValueError(f"{agent_name} profile must have 'expertise' field")
