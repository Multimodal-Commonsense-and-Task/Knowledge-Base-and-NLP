#!/usr/bin/env bash
set -euo pipefail

# Run from the parent directory of this script (project root)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
cd "${project_root}"

# session id range: [0, 20)
for sid in $(seq 0 19); do
	project="mas_${sid}"
	echo "[down] project=${project} SESSION_ID=${sid}"
	sudo docker compose -p "${project}" down

done

echo "[down] done"
