"""
Prompt Loader

Prompt template loading and management

SOLID principles applied:
- SRP: Responsible only for loading prompts
- OCP: No need to modify existing code when adding new prompt types
"""

from pathlib import Path
from typing import Optional


class PromptLoader:
    """
    Prompt template loader
    
    Loads and manages prompt template files
    """
    
    # Default prompt filename mapping
    DEFAULT_PROMPT_FILES = {
        "scenario_generation": "1_scenario_generation_prompt.txt",
        "requirements_generation": "2_requirements_generation_prompt.txt",
        "milestone_generation": "2_criteria_generation_prompt.txt",  # backward compatibility
        "memory_generation": "3_memory_generation_prompt.txt",
        "constraint_generation": "4_constraint_generation_prompt.txt",
    }
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Args:
            prompts_dir: Path to the prompt templates directory
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent / "templates"
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[str, str] = {}
    
    def load(self, prompt_name: str) -> str:
        """
        Load prompt template
        
        Args:
            prompt_name: Prompt name (e.g., "goal_generation")
            
        Returns:
            str: Prompt template string
            
        Raises:
            FileNotFoundError: If the prompt file does not exist
        """
        # Check cache
        if prompt_name in self._cache:
            return self._cache[prompt_name]
        
        # Determine filename
        filename = self.DEFAULT_PROMPT_FILES.get(prompt_name, f"{prompt_name}.txt")
        prompt_path = self.prompts_dir / filename
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        content = prompt_path.read_text(encoding="utf-8")
        self._cache[prompt_name] = content
        return content
    
    def load_custom(self, file_path: Path) -> str:
        """
        Load custom prompt file
        
        Args:
            file_path: Path to the prompt file
            
        Returns:
            str: Prompt template string
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")
    
    def clear_cache(self) -> None:
        """Clear cache"""
        self._cache.clear()
