"""
Evaluator Module

Module responsible for LLM-based evaluation

SOLID Principles Applied:
- SRP: Each Evaluator is responsible for only one type of evaluation
- OCP: Can be extended with new evaluation types without modifying existing code
- LSP: All Evaluators implement a common interface
- ISP: Implement only the necessary interfaces
- DIP: Depend on abstractions (EvaluatorInterface)
"""

from src.evaluator.base import EvaluatorInterface
from src.evaluator.task import TaskEvaluator
from src.evaluator.privacy import PrivacyEvaluator
from src.evaluator.pipeline import EvaluationPipeline

__all__ = [
    "EvaluatorInterface",
    "TaskEvaluator",
    "PrivacyEvaluator",
    "EvaluationPipeline",
]
