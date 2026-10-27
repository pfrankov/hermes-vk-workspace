#!/usr/bin/env bash
# A fresh test-only environment on the supported host's frozen dependency graph.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${HERMES_SOURCE_PATH:?Set HERMES_SOURCE_PATH to the pinned Hermes checkout}"
HERMES_SOURCE_PATH="$(cd "$HERMES_SOURCE_PATH" && pwd)"
uv python install 3.14.10
PYTHON="$(uv python find 3.14.10)"
cd "$ROOT"
uv lock --check
cd "$HERMES_SOURCE_PATH"
"$PYTHON" -m pm.build_env --source "$HERMES_SOURCE_PATH" --out "$ROOT/.venv" \
  --python "$PYTHON" --extra messaging --group dev
