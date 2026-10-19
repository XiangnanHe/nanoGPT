#!/bin/bash
# Starts JupyterLab (notebook) and the Streamlit dashboard.
#   JUPYTER_PORT (default 8888)  TENSORBOARD_PORT is reused for the dashboard (default 8501)
# On Bolt, open them with:  bolt task viewport <task_id> JUPYTER_PORT   (and TENSORBOARD_PORT)
cd "$(dirname "$0")"
VENV=${VENV:-/mnt/task_runtime/venv_nanogpt}
JP=${JUPYTER_PORT:-8888}; DP=${DASHBOARD_PORT:-${TENSORBOARD_PORT:-8501}}
nohup $VENV/bin/streamlit run dashboard.py --server.port $DP --server.address 0.0.0.0 --server.headless true > dashboard.log 2>&1 &
echo "dashboard : http://localhost:$DP"
echo "notebook  : http://localhost:$JP/lab/tree/tutorial/nanogpt_tutorial.ipynb"
exec $VENV/bin/jupyter lab --ip 0.0.0.0 --port $JP --no-browser --allow-root --notebook-dir=.. 
