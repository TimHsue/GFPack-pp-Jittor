#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"


PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" -m src.evaluate "$@"
