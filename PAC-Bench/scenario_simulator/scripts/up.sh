#!/usr/bin/env bash
set -euo pipefail

# Run from the parent directory of this script (project root)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
cd "${project_root}"

# Ranges:
# - session id: [0, 20)
# - agent A port: [8100, 8120)
# - agent B port: [8200, 8220)

for sid in $(seq 0 19); do
	a_port=$((8100 + sid))
	b_port=$((8200 + sid))
	project_name="mas_${sid}"

	echo "[up] project=${project_name} SESSION_ID=${sid} AGENT_A_PORT=${a_port} AGENT_B_PORT=${b_port}"

	sudo env SESSION_ID="${sid}" AGENT_A_PORT="${a_port}" AGENT_B_PORT="${b_port}" \
		docker compose -p "${project_name}" up -d
done

echo "[up] done"
