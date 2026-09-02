"""
Requirements Generator

Stage 2: Scenarios (with goals) → Requirements generation

Implemented according to a pipeline structure that cumulatively stores a single JSON
"""

import json
from typing import Optional

from src.core.interfaces.llm_client import LLMClient, LLMConfig
from src.generators.parsers.requirements_parser import RequirementsParser
from src.prompts.prompt_loader import PromptLoader


class RequirementsGenerator:
    """Requirements Generator"""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_loader: PromptLoader,
        config: Optional[LLMConfig] = None,
        max_requirements: int = 4,
    ) -> None:
        """Initialize generator."""
        self.llm_client = llm_client
        self.prompt_loader = prompt_loader
        self.parser = RequirementsParser(max_requirements=max_requirements)
        self.config = config
        self.max_requirements = max_requirements

    def generate_from_dict(self, data: dict, domain_name: str) -> list[str]:
        """Generate requirements from prior stage JSON."""
        prompt_template = self.prompt_loader.load("requirements_generation")

        prompt = prompt_template.replace("$DOMAIN_NAME$", domain_name)
        prompt = prompt.replace("$SCENARIO_JSON$", json.dumps(data, ensure_ascii=False, indent=2))
        prompt = prompt.replace("$MAX_REQUIREMENTS$", str(self.max_requirements))

        response = self.llm_client.generate(prompt, self.config)
        return self.parser.parse(response.content)
