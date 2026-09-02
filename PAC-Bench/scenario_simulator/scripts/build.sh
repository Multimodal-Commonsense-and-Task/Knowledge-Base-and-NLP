#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root (parent of this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

sudo env SESSION_ID="${SESSION_ID:-0}" AGENT_A_PORT="${AGENT_A_PORT:-8000}" AGENT_B_PORT="${AGENT_B_PORT:-8001}" docker compose build
