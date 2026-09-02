"""
Evaluator Interface

Definition of the abstract evaluator interface

SOLID Principles Applied:
- DIP: High-level modules do not depend on low-level modules, but both depend on abstractions
- ISP: Interface Segregation - Define only the minimum necessary methods
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.models.evaluation_input import EvaluationInput


class EvaluatorInterface(ABC):
    """
    Abstract Evaluator Interface
    
    All Evaluators must implement this interface
    """
    
    @abstractmethod
    def evaluate(self, evaluation_input: EvaluationInput) -> Any:
        """
        Perform evaluation
        
        Args:
            evaluation_input: Evaluation input data
            
        Returns:
            Evaluation result (specific type defined in implementation)
        """
        pass
    
    @abstractmethod
    def save_result(self, result: Any, output_dir: Path) -> Path:
        """
        Save evaluation result
        
        Args:
            result: Evaluation result
            output_dir: Output directory
            
        Returns:
            Saved file path
        """
        pass

