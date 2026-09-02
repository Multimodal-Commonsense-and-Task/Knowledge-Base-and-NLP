#!/usr/bin/env bash
set -euo pipefail

# Run from repo root (parent of this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- Config (override via env if needed) ----
: "${SCENARIO_DIR:=scenarios/20260102_083142_10_req/Biotechnology}"
: "${RESULT_DIR:=results}"

# Ranges must be interval strings like "[0, 20)" or "[0, 20]"
: "${SESSION_RANGE:=[0,1)}"
: "${PORT_A_RANGE:=[8100,8101)}"
: "${PORT_B_RANGE:=[8200,8201)}"

: "${MODEL_AGENT_A:=ministral-14b}"
: "${MODEL_AGENT_B:=ministral-14b}"
: "${MAX_TURNS:=10}"
: "${MAX_STEPS:=10}"
: "${MAX_TOOL_CALLS:=5}"
	
python main.py \
	--scenario-dir "${SCENARIO_DIR}" \
	--result-dir "${RESULT_DIR}" \
	--session-range "${SESSION_RANGE}" \
	--port-a-range "${PORT_A_RANGE}" \
	--port-b-range "${PORT_B_RANGE}" \
	--model-agent-a "${MODEL_AGENT_A}" \
	--model-agent-b "${MODEL_AGENT_B}" \
	--max-turns "${MAX_TURNS}" \
	--max-steps "${MAX_STEPS}" \
	--max-tool-calls "${MAX_TOOL_CALLS}"

