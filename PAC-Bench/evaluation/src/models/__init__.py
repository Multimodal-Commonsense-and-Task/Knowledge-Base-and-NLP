"""
Data Models

Definition of data models used in evaluation
"""

from src.models.evaluation_input import (
    EvaluationInput,
    ScenarioData,
    AgentData,
    PrivacyPolicy,
    ContextItem,
    ResponseTurn,
)
from src.models.evaluation_result import (
    TaskEvalResult,
    RequirementScore,
    PrivacyEvalResult,
    AgentPrivacyResult,
    TurnPrivacyScore,
    PrivacyScore,
)

__all__ = [
    # Input models
    "EvaluationInput",
    "ScenarioData",
    "AgentData",
    "PrivacyPolicy",
    "ContextItem",
    "ResponseTurn",
    # Result models
    "TaskEvalResult",
    "RequirementScore",
    "PrivacyEvalResult",
    "AgentPrivacyResult",
    "TurnPrivacyScore",
    "PrivacyScore",
]
