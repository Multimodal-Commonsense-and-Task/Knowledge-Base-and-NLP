"""
Constraint Generator

Stage 4: Memories → Privacy Constraints generation

Generates privacy constraints that each agent might violate based on their memory
"""

import json
from typing import Optional

from src.core.interfaces.llm_client import LLMClient, LLMConfig
from src.generators.parsers.constraint_parser import ConstraintParser
from src.prompts.prompt_loader import PromptLoader


class ConstraintGenerator:
    """
    Privacy Constraint Generator
    
    Generates privacy constraints for each agent based on the results from the previous stage (memory JSON).
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_loader: PromptLoader,
        config: Optional[LLMConfig] = None,
    ):
        """
        Args:
            llm_client: LLM client
            prompt_loader: Prompt loader
            config: LLM settings (Optional)
        """
        self.llm_client = llm_client
        self.prompt_loader = prompt_loader
        self.parser = ConstraintParser()
        self.config = config
    
    def generate_from_dict(self, data: dict, domain_name: str) -> dict:
        """
        Generates Privacy Constraint directly from a JSON dictionary
        
        Args:
            data: Results JSON from the memory stage (scenario format)
            domain_name: Domain name
            
        Returns:
            dict: {"agent_a": {"privacy_constraint": [...]}, "agent_b": {"privacy_constraint": [...]}}
        """
        prompt_template = self.prompt_loader.load("4_constraint_generation_prompt")
        
        # Compose prompt - Inject the entire JSON from the previous stage
        prompt = prompt_template.replace("$DOMAIN_NAME$", domain_name)
        prompt = prompt.replace("$MEMORY_JSON$", json.dumps(data, ensure_ascii=False, indent=2))
        
        # Call LLM
        response = self.llm_client.generate(prompt, self.config)
        parsed_constraints = self.parser.parse(response.content)
        
        return parsed_constraints

