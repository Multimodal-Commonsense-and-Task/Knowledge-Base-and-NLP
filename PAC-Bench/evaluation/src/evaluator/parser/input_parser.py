"""
Input Parser

Parsing evaluation input data

SOLID Principles Applied:
- SRP: Responsible only for parsing input data
- OCP: Minimize modifications to existing code when adding new parsing logic
"""

import json
import re
import tempfile
from pathlib import Path
from typing import Optional

from src.models.evaluation_input import (
    EvaluationInput,
    ScenarioData,
    AgentData,
    PrivacyPolicy,
    ContextItem,
    ResponseTurn,
)
from utils.file_converter import FileConverter


class InputParser:
    """
    Evaluation Input Data Parsing Class
    
    Reads scenario.json, result.json from the policy folder
    and converts them into EvaluationInput objects.
    """
    
    @staticmethod
    def parse_policy_dir(policy_dir: Path) -> EvaluationInput:
        """
        Create EvaluationInput object from the policy folder
        
        Args:
            policy_dir: Policy folder path (e.g., .../20260102_085500/Biotechnology/gpt-5.1_gpt-5.1/policy_0)
            
        Returns:
            EvaluationInput: Parsed evaluation input data
        """
        # Extract meta information from the path
        path_info = InputParser._extract_path_info(policy_dir)
        
        # Load scenario.json
        scenario_path = policy_dir / "scenario.json"
        with open(scenario_path, "r", encoding="utf-8") as f:
            scenario_raw = json.load(f)
        
        # Load result.json
        result_path = policy_dir / "result.json"
        with open(result_path, "r", encoding="utf-8") as f:
            result_raw = json.load(f)
        
        # Parse scenario data
        scenario = InputParser._parse_scenario(scenario_raw)
        
        # Parse agent data
        agent_a = InputParser._parse_agent("agent_a", scenario_raw["scenario"]["agent_a"])
        agent_b = InputParser._parse_agent("agent_b", scenario_raw["scenario"]["agent_b"])
        
        # Parse conversation history
        response_history = InputParser._parse_response_history(result_raw)
        
        # Generate final output txt (shared folder)
        final_output_txt = InputParser._get_final_output_txt(policy_dir)
        
        return EvaluationInput(
            policy_dir=policy_dir,
            timestamp=path_info["timestamp"],
            domain=path_info["domain"],
            model_a=path_info["model_a"],
            model_b=path_info["model_b"],
            policy_index=path_info["policy_index"],
            scenario=scenario,
            agent_a=agent_a,
            agent_b=agent_b,
            response_history=response_history,
            final_output_txt=final_output_txt,
            status=result_raw.get("status", ""),
        )
    
    @staticmethod
    def _extract_path_info(policy_dir: Path) -> dict:
        """
        Extract meta information from the path
        
        Example: .../20260102_085500/Biotechnology/gpt-5.1_gpt-5.1/policy_0
        """
        parts = policy_dir.parts
        
        # Extract index from policy_#
        policy_name = parts[-1]  # policy_0
        policy_index = int(policy_name.split("_")[1])
        
        # Parse model_a_model_b
        models_str = parts[-2]  # gpt-5.1_gpt-5.1
        # Cannot simply split by the first _. Model names may contain _.
        # Patterns like gpt-5.1_gpt-5.1, llama3-70b_llama3-70b
        # Need to handle both cases where the two model names are the same or different.
        model_a, model_b = InputParser._parse_model_names(models_str)
        
        # Domain
        domain = parts[-3]  # Biotechnology
        
        # Timestamp
        timestamp = parts[-4]  # 20260102_085500
        
        return {
            "timestamp": timestamp,
            "domain": domain,
            "model_a": model_a,
            "model_b": model_b,
            "policy_index": policy_index,
        }
    
    @staticmethod
    def _parse_model_names(models_str: str) -> tuple[str, str]:
        """
        Parse model name string
        
        Format: model_a_model_b (But- can be included in model names)
        Example: gpt-5.1_gpt-5.1, llama3-70b_ministral-14b
        
        Strategy: Split by _, then match known model name patterns.
        """
        # Known model name patterns
        known_patterns = [
            r"gpt-\d+\.\d+",           # gpt-5.1
            r"llama3-\d+b",             # llama3-70b
            r"ministral-\d+b",          # ministral-14b
            r"qwen3-\d+b",              # qwen3-32b
            r"claude-\d+-\d+-\w+",      # claude-3-5-sonnet etc.
        ]
        
        # Combine patterns
        combined_pattern = "|".join(f"({p})" for p in known_patterns)
        matches = re.findall(combined_pattern, models_str)
        
        if matches and len(matches) >= 2:
            # Each match is returned as a tuple (per group)
            model_names = []
            for match in matches:
                for group in match:
                    if group:
                        model_names.append(group)
                        break
            if len(model_names) >= 2:
                return model_names[0], model_names[1]
        
        # Try simple splitting if pattern matching fails
        # Find and split at the middle _ (since the two model names might be the same)
        parts = models_str.split("_")
        if len(parts) == 2:
            return parts[0], parts[1]
        
        # Search for middle _
        mid = len(models_str) // 2
        for i in range(len(models_str) // 2):
            if models_str[mid + i] == "_":
                return models_str[:mid + i], models_str[mid + i + 1:]
            if models_str[mid - i] == "_":
                return models_str[:mid - i], models_str[mid - i + 1:]
        
        # Last resort
        return models_str, models_str
    
    @staticmethod
    def _parse_scenario(scenario_raw: dict) -> ScenarioData:
        """Parse scenario data"""
        scenario = scenario_raw["scenario"]
        return ScenarioData(
            description=scenario.get("description", ""),
            goal_content=scenario["goal"]["content"],
            requirements=scenario["goal"]["requirements"],
        )
    
    @staticmethod
    def _parse_agent(name: str, agent_raw: dict) -> AgentData:
        """Parse agent data"""
        # Parse context
        context_items = []
        for ctx in agent_raw.get("context", []):
            context_items.append(ContextItem(
                requirements_index=ctx.get("requirements_index", []),
                content=ctx.get("content", ""),
            ))
        
        # Parse privacy policy
        privacy_policies = []
        for pp in agent_raw.get("privacy_policy", []):
            privacy_policies.append(PrivacyPolicy(
                content=pp.get("content", ""),
                violation_keywords=pp.get("violation_keywords", []),
                context_idx=pp.get("context_idx", []),
                reason=pp.get("reason", ""),
            ))
        
        return AgentData(
            name=name,
            organization=agent_raw["profile"]["organization"],
            expertise=agent_raw["profile"]["expertise"],
            context=context_items,
            privacy_policy=privacy_policies,
        )
    
    @staticmethod
    def _parse_response_history(result_raw: dict) -> list[ResponseTurn]:
        """Parse conversation history"""
        history = result_raw.get("response_history", [])
        turns = []
        for item in history:
            turns.append(ResponseTurn(
                turn=item.get("turn", 0),
                agent=item.get("agent", ""),
                message=item.get("message", ""),
            ))
        return turns
    
    @staticmethod
    def _get_final_output_txt(policy_dir: Path) -> Optional[str]:
        """
        Convert contents of the shared folder in the last workspace_at_# folder to txt
        
        Uses FileConverter for conversion.
        """
        # Find workspace_at_# folders
        workspace_dirs = sorted(
            [d for d in policy_dir.iterdir() 
             if d.is_dir() and d.name.startswith("workspace_at_")],
            key=lambda x: int(x.name.split("_")[-1])
        )
        
        if not workspace_dirs:
            return None
        
        # Last workspace
        last_workspace = workspace_dirs[-1]
        shared_dir = last_workspace / "shared"
        
        if not shared_dir.exists():
            return None
        
        # Check if shared folder is empty
        shared_files = list(shared_dir.iterdir())
        if not shared_files:
            return None
        
        # Convert with FileConverter
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_output = Path(temp_dir) / "converted"
                converter = FileConverter(
                    input_dir=str(shared_dir),
                    output_dir=str(temp_output),
                    to_txt=False,
                )
                converter.convert()
                
                # Read conversion_report.txt
                report_path = temp_output / "conversion_report.txt"
                if report_path.exists():
                    return report_path.read_text(encoding="utf-8")
        except Exception:
            pass
        
        return None
    
    @staticmethod
    def discover_policy_dirs(input_path: Path) -> list[Path]:
        """
        Discover all policy folders in the input path
        
        Args:
            input_path: Input path (timestamp folder or its parent)
            
        Returns:
            list[Path]: List of policy folder paths
        """
        policy_dirs = []
        
        # Find folders matching the policy_# pattern
        for policy_dir in input_path.rglob("policy_*"):
            if policy_dir.is_dir():
                # Check if scenario.json and result.json exist
                if (policy_dir / "scenario.json").exists() and (policy_dir / "result.json").exists():
                    policy_dirs.append(policy_dir)
        
        return sorted(policy_dirs)

