# Base image with CUDA 12.4.1 support
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install essential dependencies
RUN apt update && apt install -y --no-install-recommends \
    git curl libgl1-mesa-glx ffmpeg build-essential python3.11 python3.11-dev python3-pip python3.11-venv \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch for CUDA 12.4
RUN pip install --no-cache-dir --upgrade pip
# Install PyTorch for CUDA 12.4 if possible
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        pip install -U --no-cache-dir torch torchvision torchaudio \
        xformers --index-url https://download.pytorch.org/whl/cu124; \
    else \
        echo "xformers is unavailable on $TARGETARCH architecture"; \
        pip3 install torch torchvision torchaudio; \
    fi

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 8000

# Start FastAPI and Celery worker
CMD ["bash", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & celery -A tasks worker --loglevel=info --concurrency=1 --pool=solo"]
