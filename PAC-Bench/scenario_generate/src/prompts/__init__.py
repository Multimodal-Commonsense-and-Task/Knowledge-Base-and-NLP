"""
Prompts module - Prompt template management

SOLID principles applied:
- SRP: Responsible only for loading and managing prompts
"""

from src.prompts.prompt_loader import PromptLoader

__all__ = [
    "PromptLoader",
]

