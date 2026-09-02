"""
Interfaces module - Definition of abstract interfaces

Maintains only LLM client and parser interfaces
"""

from src.core.interfaces.llm_client import LLMClient, LLMConfig, LLMResponse
from src.core.interfaces.parser import ResponseParser

__all__ = [
    "LLMClient",
    "LLMConfig",
    "LLMResponse",
    "ResponseParser",
]
