"""
Base JSON Parser

Base utility class for parsing JSON responses

SOLID principles applied:
- SRP: Responsible only for JSON extraction and basic parsing
- OCP: Extensible through inheritance
"""

import json
import re
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.core.interfaces.parser import ResponseParser

OutputT = TypeVar("OutputT")


class BaseJSONParser(ResponseParser[OutputT], ABC, Generic[OutputT]):
    """
    Base class for JSON response parsers
    
    Provides basic functionality to extract and parse JSON from LLM responses.
    Conversion to specific domain models is implemented in subclasses.
    """
    
    def extract_json(self, text: str) -> str:
        """
        Extract the JSON part from the text.
        
        Handles both code blocks (```json ... ```) and raw JSON.
        
        Args:
            text: Text containing JSON
            
        Returns:
            str: Extracted JSON string
        """
        cleaned = text.strip()
        
        # Extract ```json ... ``` block
        if "```json" in cleaned:
            start = cleaned.find("```json") + 7
            end = cleaned.find("```", start)
            if end != -1:
                return cleaned[start:end].strip()
        
        # Extract ``` ... ``` block
        if "```" in cleaned:
            start = cleaned.find("```") + 3
            end = cleaned.find("```", start)
            if end != -1:
                return cleaned[start:end].strip()
        
        # Extract JSON object/array pattern
        json_pattern = r'(\{[\s\S]*\}|\[[\s\S]*\])'
        match = re.search(json_pattern, cleaned)
        if match:
            return match.group(1).strip()
        
        return cleaned
    
    def parse_json(self, text: str) -> dict | list:
        """
        Extract and parse JSON from text.
        
        Args:
            text: Text containing JSON
            
        Returns:
            dict | list: Parsed JSON object
            
        Raises:
            ValueError: If JSON parsing fails
        """
        json_str = self.extract_json(text)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")
    
    @abstractmethod
    def parse(self, response_text: str) -> OutputT:
        """
        Convert response text to domain model.
        
        Must be implemented in subclasses.
        """
        pass

