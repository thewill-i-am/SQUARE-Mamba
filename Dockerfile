# syntax=docker/dockerfile:1
FROM python:3.10-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        build-essential \
        git \
        libopenblas-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# Install CPU-only PyTorch (built with NumPy 2 support) first so requirements can rely on it
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "SQUARE_Mamba/demo.py", "--mode", "SQUARE-Mamba", "--skip-test", "--no-plot"]
