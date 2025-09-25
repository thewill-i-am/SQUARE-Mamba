# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-cudnn9-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9" \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        git \
        libopenblas-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Install CUDA-enabled PyTorch first so requirements can rely on it
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision torchaudio

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["python3", "SQUARE_Mamba/demo.py", "--mode", "SQUARE-Mamba", "--skip-test", "--no-plot"]
