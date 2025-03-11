#!/bin/bash

set -e  # Exit the script if any statement returns a non-true return value

# Set up Jupyter runtime directory with correct permissions
setup_jupyter_runtime() {
    JUPYTER_RUNTIME_DIR="/app/.local/share/jupyter/runtime"  # Customize this path
    mkdir -p "$JUPYTER_RUNTIME_DIR"
    chmod 700 "$JUPYTER_RUNTIME_DIR"
    export JUPYTER_RUNTIME_DIR="$JUPYTER_RUNTIME_DIR"  # Export so JupyterLab knows where to find it

    # Static files directory
    JUPYTER_PATH="/opt/venv/share/jupyter"
    mkdir -p "$JUPYTER_PATH"
    export JUPYTER_PATH="$JUPYTER_PATH"
    
    # Data directory
    JUPYTER_DATA_DIR="/app/.local/share/jupyter"
    mkdir -p "$JUPYTER_DATA_DIR"
    export JUPYTER_DATA_DIR="$JUPYTER_DATA_DIR"

    # Notebooks directory
    JUPYTER_NOTEBOOK_DIR="/app/notebooks"
    mkdir -p "$JUPYTER_NOTEBOOK_DIR"
    export JUPYTER_NOTEBOOK_DIR="$JUPYTER_NOTEBOOK_DIR"

    # Set permissions
    chown -R root:root "$JUPYTER_RUNTIME_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_NOTEBOOK_DIR"
}

# Start JupyterLab
start_jupyter() {
    echo "🚀 Starting JupyterLab..."
    jupyter lab --ip=0.0.0.0 \
                --port=8888 \
                --no-browser \
                --allow-root \
                --ServerApp.allow_origin=* \
                --IdentityProvider.token="$JUPYTER_PASSWORD" \
                --ServerApp.allow_credentials=true \
                --ServerApp.root_dir='/app' \
                --ServerApp.notebook_dir='/app/notebooks' \
                --FileContentsManager.delete_to_trash=False \
                --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' \
                --ServerApp.disable_check_xsrf=False \
                --LabApp.dev_mode=False &
    echo "✅ JupyterLab started!"
}

# Start FastAPI and Celery
start_fastapi_celery() {
    echo "🚀 Starting FastAPI and Celery..."
    source /opt/venv/bin/activate
    uvicorn main:app --host 0.0.0.0 --port 8000 &
    celery -A tasks worker --loglevel=info --concurrency=1 --pool=solo &
    echo "✅ FastAPI and Celery started!"
}

# Trap SIGTERM and SIGINT to shutdown child processes
trap 'kill $(jobs -p)' SIGTERM SIGINT

setup_jupyter_runtime
start_fastapi_celery
start_jupyter

# Keep container running
wait -n
