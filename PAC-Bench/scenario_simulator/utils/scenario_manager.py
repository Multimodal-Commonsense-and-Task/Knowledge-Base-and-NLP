import json
from typing import Any


class ScenarioManager:
    def __init__(
        self,
        scenario_path: str,
        max_turns: int = 20,
        max_tool_calls: int = 10,
    ):
        self.scenario = self._load_scenario(scenario_path)
        self.max_turns_half = max_turns // 2
        self.max_tool_calls = max_tool_calls
    
    def _load_scenario(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as file:
            scenario = json.load(file)
        return scenario
    
    def get_scenario(self) -> dict:
        return self.scenario
    
    @staticmethod
    def _as_str(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _format_numbered(items: list[str]) -> str:
        lines: list[str] = []
        for idx, item in enumerate(items, start=1):
            item_str = ScenarioManager._as_str(item).strip()
            if not item_str:
                continue
            lines.append(f"{idx}. {item_str}")
        return "\n".join(lines)

    @staticmethod
    def _format_bullets(items: list[str]) -> str:
        lines: list[str] = []
        for item in items:
            item_str = ScenarioManager._as_str(item).strip()
            if not item_str:
                continue
            lines.append(f"- {item_str}")
        return "\n".join(lines)

    @staticmethod
    def _format_context_sections(items: list[str]) -> str:
        sections: list[str] = []
        for item in items:
            item_str = ScenarioManager._as_str(item).strip()
            if not item_str:
                continue
            sections.append(item_str)
        return "\n---\n".join(sections)

    # TODO : choose one of the two implementations below(milestones vs. requirements)
    # Just below one is milestones-based

    # def get_agent_context(self, agent_name: str) -> dict[str, str]:
    #     if agent_name not in ("agent_a", "agent_b"):
    #         raise ValueError("agent_name must be either 'agent_a' or 'agent_b'")

    #     partner_name = "agent_b" if agent_name == "agent_a" else "agent_a"

    #     # Support both schemas:
    #     # 1) New: {"scenario": {...}} (preferred)
    #     # 2) Old: {...} (scenario fields at root)
    #     scenario_root = self.scenario.get("scenario", self.scenario)

    #     description = self._as_str(scenario_root.get("description", ""))

    #     goal_obj = scenario_root.get("goal", {})
    #     goal_content = ""
    #     milestones: list[str] = []
    #     if isinstance(goal_obj, dict):
    #         goal_content = self._as_str(goal_obj.get("content", ""))
    #         milestones = goal_obj.get("milestones", [])
    #     else:
    #         goal_content = self._as_str(goal_obj)

    #     agent_block = scenario_root.get(agent_name, {})
    #     profile = agent_block.get("profile", {}) if isinstance(agent_block, dict) else {}

    #     partner_block = scenario_root.get(partner_name, {})
    #     partner_profile = partner_block.get("profile", {}) if isinstance(partner_block, dict) else {}

    #     organization = ""
    #     expertise = ""
    #     if isinstance(profile, dict):
    #         organization = self._as_str(profile.get("organization", ""))
    #         expertise = self._as_str(profile.get("expertise", ""))

    #     partner_organization = ""
    #     partner_expertise = ""
    #     if isinstance(partner_profile, dict):
    #         partner_organization = self._as_str(partner_profile.get("organization", ""))
    #         partner_expertise = self._as_str(partner_profile.get("expertise", ""))

    #     context_items: list[str] = []
    #     if isinstance(agent_block, dict):
    #         raw_context = agent_block.get("context", [])
    #         if isinstance(raw_context, list):
    #             for item in raw_context:
    #                 if isinstance(item, dict):
    #                     context_items.append(self._as_str(item.get("content", "")))
    #                 else:
    #                     context_items.append(self._as_str(item))

    #     privacy_items: list[str] = []
    #     if isinstance(agent_block, dict):
    #         raw_privacy = agent_block.get("privacy_policy", [])
    #         if isinstance(raw_privacy, list):
    #             for item in raw_privacy:
    #                 if isinstance(item, dict):
    #                     privacy_items.append(self._as_str(item.get("content", "")))
    #                 else:
    #                     privacy_items.append(self._as_str(item))

    #     return {
    #         "description": description,
    #         "goal": goal_content,
    #         "milestones": self._format_numbered(milestones),
    #         "organization": organization,
    #         "expertise": expertise,
    #         "partner_organization": partner_organization,
    #         "partner_expertise": partner_expertise,
    #         "context": self._format_context_sections(context_items),
    #         "privacy_policy": self._format_bullets(privacy_items),
    #         "max_turns_half": str(self.max_turns_half),
    #         "max_tool_calls": str(self.max_tool_calls),
    #     }

    def get_agent_context(self, agent_name: str) -> dict[str, str]:
        if agent_name not in ("agent_a", "agent_b"):
            raise ValueError("agent_name must be either 'agent_a' or 'agent_b'")

        partner_name = "agent_b" if agent_name == "agent_a" else "agent_a"

        # Support both schemas:
        # 1) New: {"scenario": {...}} (preferred)
        # 2) Old: {...} (scenario fields at root)
        scenario_root = self.scenario.get("scenario", self.scenario)

        description = self._as_str(scenario_root.get("description", ""))

        goal_obj = scenario_root.get("goal", {})
        goal_content = ""
        requirements: list[str] = []
        if isinstance(goal_obj, dict):
            goal_content = self._as_str(goal_obj.get("content", ""))
            requirements = goal_obj.get("requirements", [])
        else:
            goal_content = self._as_str(goal_obj)

        agent_block = scenario_root.get(agent_name, {})
        profile = agent_block.get("profile", {}) if isinstance(agent_block, dict) else {}

        partner_block = scenario_root.get(partner_name, {})
        partner_profile = partner_block.get("profile", {}) if isinstance(partner_block, dict) else {}

        organization = ""
        expertise = ""
        if isinstance(profile, dict):
            organization = self._as_str(profile.get("organization", ""))
            expertise = self._as_str(profile.get("expertise", ""))

        partner_organization = ""
        partner_expertise = ""
        if isinstance(partner_profile, dict):
            partner_organization = self._as_str(partner_profile.get("organization", ""))
            partner_expertise = self._as_str(partner_profile.get("expertise", ""))

        context_items: list[str] = []
        if isinstance(agent_block, dict):
            raw_context = agent_block.get("context", [])
            if isinstance(raw_context, list):
                for item in raw_context:
                    if isinstance(item, dict):
                        context_items.append(self._as_str(item.get("content", "")))
                    else:
                        context_items.append(self._as_str(item))

        privacy_items: list[str] = []
        if isinstance(agent_block, dict):
            raw_privacy = agent_block.get("privacy_policy", [])
            if isinstance(raw_privacy, list):
                for item in raw_privacy:
                    if isinstance(item, dict):
                        privacy_items.append(self._as_str(item.get("content", "")))
                    else:
                        privacy_items.append(self._as_str(item))

        return {
            "description": description,
            "goal": goal_content,
            "requirements": self._format_numbered(requirements),
            "organization": organization,
            "expertise": expertise,
            "partner_organization": partner_organization,
            "partner_expertise": partner_expertise,
            "context": self._format_context_sections(context_items),
            "privacy_policy": self._format_bullets(privacy_items),
            "max_turns_half": str(self.max_turns_half),
            "max_tool_calls": str(self.max_tool_calls),
        }

if __name__ == "__main__":
    manager = ScenarioManager("scenarios/Test/policy_0.json")
    scenario = manager.get_scenario()
    print("Full scenario:", json.dumps(scenario, indent=2))

    for agent in ("agent_a", "agent_b"):
        context = manager.get_agent_context(agent)
        print(f"\nContext for {agent}:")
        for key, value in context.items():
            print(f"{key}:\n{value}\n")