"""
LLM module - LLM client implementations

SOLID principles applied:
- OCP: No need to modify existing code when adding new LLM services
- DIP: All clients implement the LLMClient interface
- LSP: All clients can replace LLMClient
"""

from src.llm.openai_client import OpenAIClient
from src.llm.anthropic_client import AnthropicClient
from src.llm.factory import create_llm_client

__all__ = [
    "OpenAIClient",
    "AnthropicClient",
    "create_llm_client",
]

