"""
Settings

Application configuration management

=== How to Change Settings ===
1. Directly modify the values in the "User Settings" section below.
2. Manage API keys in the .env file (OPENAI_API_KEY, ANTHROPIC_API_KEY).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

from dotenv import load_dotenv


# =============================================================================
# User Settings (Change values here)
# =============================================================================

# LLM Settings
LLM_PROVIDER: Literal["openai", "anthropic"] = "openai"
LLM_MODEL: str = "gpt-5.1"
LLM_TEMPERATURE: float = 0.7
LLM_MAX_TOKENS: Optional[int] = None  # Use model default if None

# Pipeline Settings
PIPELINE_MAX_WORKERS: int = 25  # Number of parallel processing workers

# Evaluator Settings (True: Enabled, False: Disabled)
EVAL_TASK_ENABLED: bool = True  # Whether to enable Task evaluation
EVAL_PRIVACY_ENABLED: bool = True  # Whether to enable Privacy evaluation
EVAL_HALLUCINATION_ENABLED: bool = True  # Whether to enable Hallucination evaluation (Requires Task evaluation)

# Requirements Settings
REQUIREMENTS_MAX_COUNT: int = 5  # Maximum number of requirements to generate

# Logging Settings
LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR

# =============================================================================
# You do not need to modify below this line
# =============================================================================

# Project root directory (calculated based on settings.py)
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class LLMSettings:
    """LLM related settings"""
    provider: Literal["openai", "anthropic"] = LLM_PROVIDER
    model: str = LLM_MODEL
    temperature: float = LLM_TEMPERATURE
    max_tokens: Optional[int] = LLM_MAX_TOKENS
    api_key: Optional[str] = None


@dataclass
class EvaluatorSettings:
    """Evaluator activation settings"""
    task_enabled: bool = EVAL_TASK_ENABLED
    privacy_enabled: bool = EVAL_PRIVACY_ENABLED
    hallucination_enabled: bool = EVAL_HALLUCINATION_ENABLED


@dataclass
class PipelineSettings:
    """Pipeline related settings"""
    max_workers: int = PIPELINE_MAX_WORKERS


@dataclass
class RequirementsSettings:
    """Requirements related settings"""
    max_count: int = REQUIREMENTS_MAX_COUNT  # Maximum number of requirements to generate


@dataclass
class PathSettings:
    """Path related settings - manage all paths centrally"""
    base_dir: Path = field(default_factory=lambda: PROJECT_ROOT)
    
    # Prompt paths
    prompts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "src" / "prompts" / "templates")
    
    # Data paths
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    domains_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "domains")
    
    # Result paths (per stage)
    result_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "result")
    scenario_output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "result" / "1_scenario")
    requirements_output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "result" / "2_requirements")
    context_output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "result" / "3_context")
    policy_output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "result" / "4_policy")


@dataclass
class Settings:
    """Application-wide settings"""
    # LLM settings
    llm: LLMSettings = field(default_factory=LLMSettings)
    
    # Pipeline settings
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    
    # Evaluator settings
    evaluator: EvaluatorSettings = field(default_factory=EvaluatorSettings)
    
    # Requirements settings
    requirements: RequirementsSettings = field(default_factory=RequirementsSettings)
    
    # Path settings
    paths: PathSettings = field(default_factory=PathSettings)
    
    # Logging settings
    log_level: str = LOG_LEVEL
    
    def __post_init__(self):
        """Create directories"""
        self.paths.result_dir.mkdir(parents=True, exist_ok=True)
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load(cls, env_file: Optional[Path] = None) -> "Settings":
        """
        Load settings (API keys are read from .env)
        
        Args:
            env_file: Path to .env file (Optional, Default: .env in project root)
            
        Returns:
            Settings: Loaded settings
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        # Select API key based on provider (read from .env)
        if LLM_PROVIDER == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif LLM_PROVIDER == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            api_key = None
        
        return cls(
            llm=LLMSettings(
                provider=LLM_PROVIDER,
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                api_key=api_key,
            ),
            pipeline=PipelineSettings(
                max_workers=PIPELINE_MAX_WORKERS,
            ),
            evaluator=EvaluatorSettings(
                task_enabled=EVAL_TASK_ENABLED,
                privacy_enabled=EVAL_PRIVACY_ENABLED,
                hallucination_enabled=EVAL_HALLUCINATION_ENABLED,
            ),
            requirements=RequirementsSettings(
                max_count=REQUIREMENTS_MAX_COUNT,
            ),
            log_level=LOG_LEVEL,
        )
