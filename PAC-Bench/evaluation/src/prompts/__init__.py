"""
Prompts Package

Prompt template management
"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent

TASK_PROMPT_PATH = PROMPTS_DIR / "task_prompt.txt"
PRIVACY_PROMPT_PATH = PROMPTS_DIR / "privacy_prompt.txt"

__all__ = ["PROMPTS_DIR", "TASK_PROMPT_PATH", "PRIVACY_PROMPT_PATH"]

