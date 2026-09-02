"""
Evaluation Input Models

Definition of evaluation input data models

SOLID Principles Applied:
- SRP: Responsible only for defining the input data structure
- OCP: Minimize modifications to existing code when adding new fields
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ContextItem:
    """Agent's context item"""
    requirements_index: list[int]
    content: str


@dataclass
class PrivacyPolicy:
    """Agent's privacy policy"""
    content: str
    violation_keywords: list[str]
    context_idx: list[int]
    reason: str


@dataclass
class AgentData:
    """Agent data"""
    name: str  # "agent_a" or "agent_b"
    organization: str
    expertise: str
    context: list[ContextItem]
    privacy_policy: list[PrivacyPolicy]


@dataclass
class ResponseTurn:
    """Conversation turn data"""
    turn: int
    agent: str  # "agent_a" or "agent_b"
    message: str


@dataclass
class ScenarioData:
    """Scenario data"""
    description: str
    goal_content: str
    requirements: list[str]


@dataclass
class EvaluationInput:
    """
    Integrated evaluation input data model
    
    Contains all evaluation input data for a single policy folder
    """
    # Path information
    policy_dir: Path
    timestamp: str
    domain: str
    model_a: str
    model_b: str
    policy_index: int
    
    # Scenario data
    scenario: ScenarioData
    
    # Agent data
    agent_a: AgentData
    agent_b: AgentData
    
    # Conversation history
    response_history: list[ResponseTurn]
    
    # Final output (shared folder contents converted to txt)
    final_output_txt: Optional[str] = None
    
    # Metadata
    status: str = ""
    
    def get_agent(self, agent_name: str) -> AgentData:
        """Returns agent data by agent name"""
        if agent_name == "agent_a":
            return self.agent_a
        elif agent_name == "agent_b":
            return self.agent_b
        else:
            raise ValueError(f"Unknown agent: {agent_name}")
    
    def get_context_by_indices(self, agent_name: str, indices: list[int]) -> list[ContextItem]:
        """Returns contexts corresponding to specific indices for a specific agent"""
        agent = self.get_agent(agent_name)
        return [ctx for i, ctx in enumerate(agent.context) if i in indices]
    
    def get_agent_turns(self, agent_name: str) -> list[ResponseTurn]:
        """Returns only the conversation turns for a specific agent"""
        return [turn for turn in self.response_history if turn.agent == agent_name]
    
    def get_contexts_and_policies_for_requirement(
        self, requirement_index: int
    ) -> dict[str, list]:
        """
        Returns contexts and privacy policies corresponding to a specific requirement
        
        Args:
            requirement_index: requirement index
            
        Returns:
            {
                "contexts": [ContextItem, ...],
                "privacy_policies": [PrivacyPolicy, ...]
            }
        """
        contexts: list[ContextItem] = []
        privacy_policies: list[PrivacyPolicy] = []
        context_indices_found: set[tuple[str, int]] = set()  # (agent_name, context_idx)
        
        # Search in both agent_a and agent_b
        for agent_name, agent_data in [("agent_a", self.agent_a), ("agent_b", self.agent_b)]:
            # 1. Find context containing requirement_index
            for ctx_idx, ctx in enumerate(agent_data.context):
                if requirement_index in ctx.requirements_index:
                    contexts.append(ctx)
                    context_indices_found.add((agent_name, ctx_idx))
            
            # 2. Find privacy_policy corresponding to the found contexts
            for policy in agent_data.privacy_policy:
                for ctx_idx in policy.context_idx:
                    if (agent_name, ctx_idx) in context_indices_found:
                        if policy not in privacy_policies:
                            privacy_policies.append(policy)
                        break
        
        return {
            "contexts": contexts,
            "privacy_policies": privacy_policies,
        }

