"""
Requirements Parser

Parser for parsing Requirements responses
"""

from typing import Optional

from src.generators.parsers.base_parser import BaseJSONParser


class RequirementsParser(BaseJSONParser[list[str]]):
    """Requirements response parser"""

    def __init__(self, max_requirements: Optional[int] = None) -> None:
        super().__init__()
        self.max_requirements = max_requirements

    def parse(self, response_text: str) -> list[str]:
        """Converts LLM response into a list of requirements"""
        data = self.parse_json(response_text)

        if not isinstance(data, dict):
            raise ValueError("Response must be a JSON object")

        if "requirements" not in data:
            raise ValueError("Response must have 'requirements' key")

        requirements_data = data["requirements"]
        if not isinstance(requirements_data, list):
            raise ValueError('"requirements" must be an array')

        normalized: list[str] = []
        for idx, item in enumerate(requirements_data):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"Requirement at index {idx} must be a non-empty string")
            normalized.append(item.strip())

        if self.max_requirements is not None and len(normalized) > self.max_requirements:
            # raise ValueError(
            #     f"Response contains {len(normalized)} requirements which exceeds limit {self.max_requirements}"
            # )
            pass

        return normalized
