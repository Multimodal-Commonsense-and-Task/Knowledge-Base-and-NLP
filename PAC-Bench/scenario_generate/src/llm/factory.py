"""
LLM Client Factory

Factory for creating LLM clients

SOLID principles applied:
- SRP: Responsible only for creating LLM clients
- OCP: Extensible when adding new providers
"""

from src.config.settings import Settings
from src.core.interfaces.llm_client import LLMClient, LLMConfig
from src.llm.openai_client import OpenAIClient
from src.llm.anthropic_client import AnthropicClient


def create_llm_client(settings: Settings) -> LLMClient:
    """
    Create an appropriate LLM client based on settings
    
    Args:
        settings: Application settings
        
    Returns:
        LLMClient: Created LLM client
        
    Raises:
        ValueError: If the provider is unknown
    """
    config = LLMConfig(
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
    )
    
    if settings.llm.provider == "openai":
        return OpenAIClient(api_key=settings.llm.api_key, default_config=config)
    elif settings.llm.provider == "anthropic":
        return AnthropicClient(api_key=settings.llm.api_key, default_config=config)
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm.provider}")

