"""
Memory Generator

Stage 3: Requirements → Memories (memory in raw data format) generation

Determines internal data required for each agent based on shared requirements
"""

import json
from typing import Optional

from src.core.interfaces.llm_client import LLMClient, LLMConfig
from src.generators.parsers.memory_parser import MemoryParser
from src.prompts.prompt_loader import PromptLoader


class MemoryGenerator:
    """Memory Generator (Requirements-based)"""
    
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
        self.parser = MemoryParser()
        self.config = config
    
    def generate_from_dict(self, data: dict, domain_name: str) -> dict:
        """Generate memories from scenario + requirements JSON."""
        prompt_template = self.prompt_loader.load("memory_generation")
        
        # Compose prompt - Inject the entire JSON from the previous stage
        prompt = prompt_template.replace("$DOMAIN_NAME$", domain_name)
        prompt = prompt.replace("$REQUIREMENTS_JSON$", json.dumps(data, ensure_ascii=False, indent=2))

        response = self.llm_client.generate(prompt, self.config)
        parsed_memories = self.parser.parse(response.content)

        return parsed_memories

