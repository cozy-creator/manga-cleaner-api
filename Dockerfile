# Base image with CUDA 12.4.1 support
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install system dependencies
RUN apt update && apt install -y --no-install-recommends \
    git curl libgl1-mesa-glx ffmpeg build-essential python3.11 python3.11-dev python3-pip python3.11-venv \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv $VIRTUAL_ENV

# Ensure pip is installed inside the venv
RUN $VIRTUAL_ENV/bin/python -m ensurepip

# Upgrade pip inside the venv
RUN $VIRTUAL_ENV/bin/pip install --no-cache-dir --upgrade pip

# 🔥 Install PyTorch inside the virtual environment
RUN $VIRTUAL_ENV/bin/pip install --no-cache-dir torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu124

# Install Jupyter Lab
RUN $VIRTUAL_ENV/bin/pip install --no-cache-dir jupyterlab

# Copy project files
COPY . /app

# ✅ Install all dependencies
RUN $VIRTUAL_ENV/bin/pip install --no-cache-dir -r requirements.txt

# Expose ports for FastAPI, Celery, and Jupyter Notebook
EXPOSE 8000 8888

# Start FastAPI, Celery, and Jupyter Notebook
CMD ["bash", "-c", "source $VIRTUAL_ENV/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000 & celery -A tasks worker --loglevel=info --concurrency=1 --pool=solo & jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"]
