"""
Response Parser Interface

Applying SOLID principles:
- SRP: The parser is responsible only for response parsing
- OCP: No modification of existing code is required when adding new parser types
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic

OutputT = TypeVar("OutputT")


class ResponseParser(ABC, Generic[OutputT]):
    """
    Abstract base class for response parsers
    
    Responsible for parsing LLM responses into specific data structures
    Implemented as concrete parsers for each domain model
    """
    
    @abstractmethod
    def parse(self, response_text: str) -> OutputT:
        """
        Parses the response text into a data object
        
        Args:
            response_text: LLM response text
            
        Returns:
            OutputT: Parsed data object
            
        Raises:
            ParseError: When parsing fails
        """
        pass
    
    @abstractmethod
    def extract_json(self, text: str) -> str:
        """
        Extracts JSON portion from text
        
        Args:
            text: Text containing JSON
            
        Returns:
            str: Extracted JSON string
        """
        pass

