FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt update && \
    apt install -y git curl libgl1-mesa-glx ffmpeg tzdata && \
    rm -rf /var/lib/apt/lists/*


COPY . /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["bash", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & celery -A tasks worker --loglevel=info --concurrency=1 --pool=solo"]
