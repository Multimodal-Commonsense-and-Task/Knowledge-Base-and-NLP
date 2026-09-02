"""
LLM Client Interface

Applying SOLID principles:
- ISP: Defines only the minimum interface required for LLM calls
- DIP: Does not depend on specific LLM services (OpenAI, Anthropic, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """LLM response data class"""
    content: str
    model: str
    usage: Optional[dict] = None
    raw_response: Optional[dict] = None


@dataclass
class LLMConfig:
    """LLM configuration data class"""
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0


class LLMClient(ABC):
    """
    LLM client abstract interface
    
    Concrete LLM service implementations inherit and implement this interface
    - OpenAI, Anthropic, Local LLM, etc.
    """
    
    @abstractmethod
    def generate(self, prompt: str, config: Optional[LLMConfig] = None) -> LLMResponse:
        """
        Generates LLM response from a prompt
        
        Args:
            prompt: Input prompt
            config: LLM configuration (Optional)
            
        Returns:
            LLMResponse: LLM response object
        """
        pass
    
    @abstractmethod
    def generate_with_messages(
        self, 
        messages: list[dict], 
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Generates LLM response from a list of messages (Chat format)
        
        Args:
            messages: Message list [{"role": "user/assistant/system", "content": "..."}]
            config: LLM configuration (Optional)
            
        Returns:
            LLMResponse: LLM response object
        """
        pass

