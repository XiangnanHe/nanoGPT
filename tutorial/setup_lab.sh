#!/bin/bash
# One-time setup of an isolated venv for the notebook + dashboard (re-uses the torch of the base python via --system-site-packages).
# usage: BASE_PY=/miniforge/envs/iris/bin/python bash setup_lab.sh
set -e
BASE_PY=${BASE_PY:-python3}
VENV=${VENV:-/mnt/task_runtime/venv_nanogpt}
$BASE_PY -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install -q -r "$(dirname "$0")/requirements_lab.txt" "numpy<2"
echo "ready: $VENV"
