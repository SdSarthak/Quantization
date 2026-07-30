# GPU image: CUDA runtime + cuDNN, needed by bitsandbytes for int8/int4.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models/hf \
    AIRAVATA_HOST=0.0.0.0 \
    AIRAVATA_PORT=8000 \
    AIRAVATA_CACHE_DIR=/models/cache \
    AIRAVATA_QUANTIZED_MODEL_PATH=/models/quantized \
    AIRAVATA_BENCHMARK_DIR=/models/benchmarks

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.10 python3-pip curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY airavata_quant ./airavata_quant
COPY airavata_quantization_service.py ./
RUN pip install --no-cache-dir --no-deps -e .

# Weights are large and mutable: keep them on a volume, not in the image.
VOLUME ["/models"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["python", "-m", "airavata_quant"]
CMD ["serve"]
