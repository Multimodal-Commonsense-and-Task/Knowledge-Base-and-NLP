"""
Scenario Generator

Stage 1: Domain → Scenarios (description, agents, goal) generation

Dictionary-based processing
"""

from typing import Optional

from src.core.interfaces.llm_client import LLMClient, LLMConfig
from src.generators.parsers.scenario_parser import ScenarioParser
from src.prompts.prompt_loader import PromptLoader


class ScenarioGenerator:
    """
    Scenario Generator
    
    Generates multiple Scenarios given a domain name.
    Each Scenario includes description, agent_a, agent_b, and goal.
    """
    
    DEFAULT_NUM_SCENARIOS = 5
    
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_loader: PromptLoader,
        config: Optional[LLMConfig] = None,
        num_scenarios: int = DEFAULT_NUM_SCENARIOS,
    ):
        """
        Args:
            llm_client: LLM client
            prompt_loader: Prompt loader
            config: LLM settings (Optional)
            num_scenarios: Number of scenarios to generate (default: 5)
        """
        self.llm_client = llm_client
        self.prompt_loader = prompt_loader
        self.parser = ScenarioParser()
        self.config = config
        self.num_scenarios = num_scenarios
    
    def generate(self, domain_name: str) -> list[dict]:
        """
        Generates a list of Scenario dictionaries given a domain name.
        
        Args:
            domain_name: Domain name (e.g., "Energy Equipment and Services")
            
        Returns:
            list[dict]: List of scenario dictionaries (each in the form {"scenario": {...}})
        """
        if not domain_name or not domain_name.strip():
            raise ValueError("Domain name cannot be empty")
        
        # Compose prompt
        prompt_template = self.prompt_loader.load("scenario_generation")
        prompt = self._build_prompt(prompt_template, domain_name)
        
        # Call LLM
        response = self.llm_client.generate(prompt, self.config)
        
        # Parse response
        scenarios = self.parser.parse(response.content)
        
        # Validate results
        if len(scenarios) == 0:
            raise ValueError("At least one scenario is required")
        
        return scenarios
    
    def _build_prompt(self, template: str, domain_name: str) -> str:
        """Incorporate domain information into the prompt template"""
        prompt = template.replace("$DOMAIN_NAME$", domain_name)
        prompt = prompt.replace("$NUM_SCENARIO$", str(self.num_scenarios))
        return prompt
