#!/usr/bin/env bash
# Execute a notebook in place with the given virtualenv, keeping every output.
# usage: scripts/run_notebook.sh <venv-name> <notebook.ipynb>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venvs/$1"; NB="$2"
start=$(date +%s)
"$VENV/bin/jupyter" nbconvert --to notebook --execute --inplace --allow-errors \
  --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=python3 "$NB"
echo "done $NB in $(( $(date +%s) - start ))s"
