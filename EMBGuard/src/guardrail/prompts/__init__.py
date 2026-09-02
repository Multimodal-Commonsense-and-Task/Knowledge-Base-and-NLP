"""
Guardrail prompts module
Contains prompt templates and few-shot examples for safety guardrail evaluation
"""

from .guardrail_prompt import (
    GUARDRAIL_SYSTEM_PROMPT,
    GUARDRAIL_USER_PROMPT,
    GUARDRAIL_SAFE_EXAMPLE,
    GUARDRAIL_UNSAFE_EXAMPLE,
    format_safe_example,
    format_unsafe_example,
    format_few_shot_examples,
    get_few_shot_messages,
    GUARDRAIL_PROMPT_TEMPLATE,
)

__all__ = [
    "GUARDRAIL_SYSTEM_PROMPT",
    "GUARDRAIL_USER_PROMPT",
    "GUARDRAIL_SAFE_EXAMPLE",
    "GUARDRAIL_UNSAFE_EXAMPLE",
    "format_safe_example",
    "format_unsafe_example",
    "format_few_shot_examples",
    "get_few_shot_messages",
    "GUARDRAIL_PROMPT_TEMPLATE",
]

