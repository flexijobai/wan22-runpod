FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg git wget \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies for Wan2.2 via diffusers
# Note: base image has torch 2.4.0 pre-installed, do NOT upgrade it
RUN pip install --no-cache-dir --no-deps "diffusers>=0.35.0" && \
    pip install --no-cache-dir \
    transformers \
    accelerate \
    safetensors \
    sentencepiece \
    ftfy \
    huggingface_hub \
    Pillow \
    imageio[ffmpeg] \
    runpod

COPY handler.py /app/handler.py

WORKDIR /app

CMD ["python", "-u", "/app/handler.py"]
