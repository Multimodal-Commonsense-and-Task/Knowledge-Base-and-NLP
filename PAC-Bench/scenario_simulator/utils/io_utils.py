from pathlib import Path
from typing import List

from natsort import natsorted


def list_json_files(directory: Path) -> List[Path]:
    """List all JSON files in the given directory."""
    json_files = [file for file in directory.iterdir() if file.is_file() and file.suffix == ".json"]
    return natsorted(json_files)