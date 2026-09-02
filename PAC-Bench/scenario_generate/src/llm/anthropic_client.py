"""
Anthropic Client

LLM client implementation using Anthropic API

SOLID principles applied:
- SRP: Responsible only for Anthropic API calls
- LSP: Fully implements the LLMClient interface
"""

import os
from typing import Optional

from src.core.interfaces.llm_client import LLMClient, LLMConfig, LLMResponse


class AnthropicClient(LLMClient):
    """
    Anthropic API Client
    
    Text generation using Claude models
    """
    
    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    DEFAULT_MAX_TOKENS = 4096
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_config: Optional[LLMConfig] = None,
    ):
        """
        Args:
            api_key: Anthropic API key (Optional, loaded from environment variables)
            default_config: Default LLM settings (Optional)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        # Lazy import to avoid dependency issues
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")
        
        self.default_config = default_config or LLMConfig(
            model=self.DEFAULT_MODEL,
            max_tokens=self.DEFAULT_MAX_TOKENS,
        )
    
    def generate(self, prompt: str, config: Optional[LLMConfig] = None) -> LLMResponse:
        """
        Generate text from a prompt
        
        Args:
            prompt: Input prompt
            config: LLM settings (Optional)
            
        Returns:
            LLMResponse: Generated response
        """
        messages = [{"role": "user", "content": prompt}]
        return self.generate_with_messages(messages, config)
    
    def generate_with_messages(
        self,
        messages: list[dict],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Generate text from a list of messages
        
        Args:
            messages: Message list
            config: LLM settings (Optional)
            
        Returns:
            LLMResponse: Generated response
        """
        cfg = config or self.default_config
        
        # Convert to Anthropic API format
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_message = content
            else:
                anthropic_messages.append({
                    "role": role,
                    "content": content,
                })
        
        kwargs = {
            "model": cfg.model,
            "messages": anthropic_messages,
            "max_tokens": cfg.max_tokens or self.DEFAULT_MAX_TOKENS,
            "temperature": cfg.temperature,
        }
        
        if system_message:
            kwargs["system"] = system_message
        
        response = self.client.messages.create(**kwargs)
        
        content = ""
        if response.content:
            content = response.content[0].text
        
        usage = None
        if response.usage:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        
        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
        )

