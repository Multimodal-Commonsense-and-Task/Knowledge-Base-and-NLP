import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class SimulationResultWriter:
    """Persist simulation artifacts and their associated workspace snapshots."""

    def __init__(self, base_result_dir: str | Path):
        self.base_result_dir = Path(base_result_dir)
        self.base_result_dir.mkdir(parents=True, exist_ok=True)

    def _format_token_usage(self, token_usage_by_agent: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        formatted: Dict[str, Dict[str, int]] = {}
        for agent_name, usage in token_usage_by_agent.items():
            formatted[agent_name] = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            totals["input_tokens"] += formatted[agent_name]["input_tokens"]
            totals["output_tokens"] += formatted[agent_name]["output_tokens"]
            totals["total_tokens"] += formatted[agent_name]["total_tokens"]

        formatted["total"] = totals
        return formatted

    def _workspace_root(self, session_id: int) -> Path:
        template = os.getenv("WORKSPACE_SHARED")
        if not template:
            raise ValueError("WORKSPACE_SHARED environment variable is not set.")

        shared_path = Path(template.format(session_id=session_id))
        return shared_path.parent

    def _write_result_json(
        self,
        scenario_dir: Path,
        results: List[Dict[str, Any]],
        response_history: List[Dict[str, Any]],
        tool_usage_history: Dict[str, List[Dict[str, Any]]],
        milestone_completion: Dict[str, Any],
        status: str,
        error: Dict[str, Any] | None,
        workspace_snapshot_error: Dict[str, Any] | None,
    ) -> Path:
        payload = {
            "status": status,
            "results": results,
            "milestone_completion": milestone_completion,
            "response_history": response_history,
            "tool_usage_history": tool_usage_history,
        }

        if error is not None:
            payload["error"] = error

        if workspace_snapshot_error is not None:
            payload["workspace_snapshot_error"] = workspace_snapshot_error

        output_path = scenario_dir / "result.json"
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        return output_path

    def _write_metadata_json(
        self,
        *,
        scenario_dir: Path,
        scenario_name: str,
        session_id: int,
        token_usage_by_agent: Dict[str, Dict[str, int]],
        created_at: str,
    ) -> Path:
        payload = {
            "created_at": created_at,
            "scenario": scenario_name,
            "session_id": session_id,
            "token_usage": self._format_token_usage(token_usage_by_agent),
        }

        output_path = scenario_dir / "metadata.json"
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        return output_path

    def _write_agent_prompts(self, *, scenario_dir: Path, agent_prompts: Dict[str, str]) -> Path:
        prompts_dir = scenario_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        for agent_name, prompt in agent_prompts.items():
            output_path = prompts_dir / f"{agent_name}.txt"
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(prompt or "")

        return prompts_dir

    def _write_scenario_json(self, scenario_path: Path, scenario_dir: Path) -> Path:
        output_path = scenario_dir / "scenario.json"
        shutil.copyfile(scenario_path, output_path)
        return output_path

    def _copy_workspace_snapshot(self, session_id: int, destination: Path) -> Path:
        source_workspace = self._workspace_root(session_id)
        if not source_workspace.exists():
            raise ValueError(f"Workspace for session_id {session_id} does not exist at {source_workspace}.")

        if destination.exists():
            shutil.rmtree(destination)

        shutil.copytree(source_workspace, destination)
        return destination

    def save(
        self,
        *,
        scenario_path: str | Path,
        scenario_name: str,
        session_id: int,
        results: List[Dict[str, Any]],
        milestone_completion: Dict[str, Any],
        response_history: List[Dict[str, Any]],
        tool_usage_history: Dict[str, List[Dict[str, Any]]],
        token_usage_by_agent: Dict[str, Dict[str, int]],
        agent_prompts: Dict[str, str] | None = None,
        error: Dict[str, Any] | None = None,
    ) -> Path:
        scenario_path = Path(scenario_path)
        scenario_dir = self.base_result_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        created_at = datetime.now(timezone.utc).isoformat()
        self._write_scenario_json(scenario_path=scenario_path, scenario_dir=scenario_dir)

        if agent_prompts is not None:
            self._write_agent_prompts(scenario_dir=scenario_dir, agent_prompts=agent_prompts)

        workspace_error: Dict[str, Any] | None = None
        try:
            self._copy_workspace_snapshot(session_id, scenario_dir / "workspace")
        except Exception as exc:
            workspace_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "repr": repr(exc),
            }

        status = "error" if error is not None else "success"
        self._write_result_json(
            scenario_dir=scenario_dir,
            results=results,
            response_history=response_history,
            tool_usage_history=tool_usage_history,
            status=status,
            error=error,
            workspace_snapshot_error=workspace_error,
            milestone_completion=milestone_completion,
        )

        self._write_metadata_json(
            scenario_dir=scenario_dir,
            scenario_name=scenario_name,
            session_id=session_id,
            token_usage_by_agent=token_usage_by_agent,
            created_at=created_at,
        )

        return scenario_dir
