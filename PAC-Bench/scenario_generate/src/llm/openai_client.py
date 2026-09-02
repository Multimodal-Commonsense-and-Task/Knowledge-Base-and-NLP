"""
OpenAI Client

LLM client implementation using OpenAI API

SOLID principles applied:
- SRP: Responsible only for OpenAI API calls
- LSP: Fully implements the LLMClient interface
"""

import os
from typing import Optional

from openai import OpenAI

from src.core.interfaces.llm_client import LLMClient, LLMConfig, LLMResponse


class OpenAIClient(LLMClient):
    """
    OpenAI API Client
    
    Text generation using GPT models
    """
    
    DEFAULT_MODEL = "gpt-5.1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_config: Optional[LLMConfig] = None,
    ):
        """
        Args:
            api_key: OpenAI API key (Optional, loaded from environment variables)
            default_config: Default LLM settings (Optional)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.default_config = default_config or LLMConfig(model=self.DEFAULT_MODEL)
    
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
        
        # GPT-5 or later models do not support temperature and top_p
        is_gpt5_or_above = cfg.model.startswith("gpt-5") or cfg.model.startswith("o1")
        
        kwargs = {
            "model": cfg.model,
            "messages": messages,
        }
        
        # Set temperature and top_p only if not GPT-5 or later
        if not is_gpt5_or_above:
            kwargs["temperature"] = cfg.temperature
            kwargs["top_p"] = cfg.top_p
        
        # Set max_tokens
        # GPT-5 or later models support max_tokens, but some models (like o1) determine it automatically
        # If None, use model default maximum (remove limit)
        if cfg.max_tokens is not None:
            kwargs["max_tokens"] = cfg.max_tokens
        
        response = self.client.chat.completions.create(**kwargs)
        
        content = response.choices[0].message.content or ""
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
        )

