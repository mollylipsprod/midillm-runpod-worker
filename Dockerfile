# MIDI-LLM worker for RunPod Serverless (queue endpoint).
# Weights are NOT baked in: RunPod "Cached model" (slseanwu/MIDI-LLM_Llama-3.2-1B)
# or a one-time download at start. torch 2.8.0 from PyPI ships CUDA 12.8 libs.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY midillm_core.py handler.py ./

CMD ["python", "-u", "handler.py"]
