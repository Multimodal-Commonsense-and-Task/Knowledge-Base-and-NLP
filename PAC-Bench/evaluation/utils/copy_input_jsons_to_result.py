from __future__ import annotations

import shutil
from pathlib import Path


INPUT_DIR = "/home/acradin/kdh/MAS/evaluation/input/change_priv_H"
RESULT_DIR = "/home/acradin/kdh/MAS/evaluation/result/change_priv_H/change_priv_H"


FILES_TO_COPY = ("scenario.json", "result.json")


def _iter_policy_dirs(result_root: Path) -> list[Path]:
	policy_dirs: list[Path] = []
	# Expected structure: result_root/<domain>/<model_pair>/<policy_idx>/
	for domain_dir in sorted(result_root.iterdir()):
		if not domain_dir.is_dir():
			continue
		for model_pair_dir in sorted(domain_dir.iterdir()):
			if not model_pair_dir.is_dir():
				continue
			for policy_dir in sorted(model_pair_dir.iterdir()):
				if policy_dir.is_dir():
					policy_dirs.append(policy_dir)
	return policy_dirs


def copy_input_jsons_to_result(input_dir: str | Path, result_dir: str | Path) -> None:
	input_root = Path(input_dir).expanduser().resolve()
	result_root = Path(result_dir).expanduser().resolve()

	if not input_root.exists() or not input_root.is_dir():
		raise FileNotFoundError(f"INPUT_DIR not found or not a directory: {input_root}")
	if not result_root.exists() or not result_root.is_dir():
		raise FileNotFoundError(f"RESULT_DIR not found or not a directory: {result_root}")

	policy_dirs = _iter_policy_dirs(result_root)
	if not policy_dirs:
		print(f"No policy_idx directories found under: {result_root}")
		return

	copied = 0
	missing_policy_dir = 0
	missing_files = 0

	for result_policy_dir in policy_dirs:
		rel = result_policy_dir.relative_to(result_root)
		input_policy_dir = input_root / rel

		if not input_policy_dir.exists() or not input_policy_dir.is_dir():
			missing_policy_dir += 1
			print(f"[WARN] Missing input policy dir: {input_policy_dir}")
			continue

		for filename in FILES_TO_COPY:
			src = input_policy_dir / filename
			dst = result_policy_dir / filename

			if not src.exists() or not src.is_file():
				missing_files += 1
				print(f"[WARN] Missing file: {src}")
				continue

			shutil.copy2(src, dst)
			copied += 1

	print(
		"Done. "
		f"policy_dirs={len(policy_dirs)}, copied_files={copied}, "
		f"missing_policy_dirs={missing_policy_dir}, missing_files={missing_files}"
	)


if __name__ == "__main__":
	copy_input_jsons_to_result(INPUT_DIR, RESULT_DIR)