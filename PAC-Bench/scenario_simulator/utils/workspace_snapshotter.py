from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


class WorkspaceSnapshotter:
    """Copies the current workspace directory into a turn-scoped snapshot folder.

    By default, the workspace root is resolved from WORKSPACE_SHARED (same convention
    as utils.result_writer.SimulationResultWriter).

    Snapshot folder name: workspace_at_<turn_num>
    """

    def __init__(
        self,
        *,
        session_id: int,
        output_dir: str | Path,
        workspace_root: Optional[str | Path] = None,
    ):
        self.session_id = int(session_id)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.workspace_root = (
            Path(workspace_root) if workspace_root is not None else self._resolve_workspace_root(self.session_id)
        )

    @staticmethod
    def _resolve_workspace_root(session_id: int) -> Path:
        template = os.getenv("WORKSPACE_SHARED")
        if not template:
            raise ValueError("WORKSPACE_SHARED environment variable is not set.")

        shared_path = Path(template.format(session_id=session_id))
        return shared_path.parent

    def snapshot(self, turn_num: int) -> Path:
        source = self.workspace_root
        if not source.exists():
            raise ValueError(f"Workspace for session_id {self.session_id} does not exist at {source}.")

        destination = self.output_dir / f"workspace_at_{int(turn_num)}"
        if destination.exists():
            shutil.rmtree(destination)

        shutil.copytree(source, destination)
        return destination
