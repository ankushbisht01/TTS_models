# ─────────────────────────────────────────────────────────────────────
# Dockerfile — Voice Cloning Server (GPU)
#
# Build:  docker build -t voice-cloning-server .
# Run:    docker run --gpus all -p 8000:8000 voice-cloning-server
# ─────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/models_cache

# System dependencies. Ubuntu 24.04 ships Python 3.12, which is inside the
# range this project supports (see requires-python in pyproject.toml).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    ffmpeg sox libsndfile1 libportaudio2 libsox-dev \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv rather than --break-system-packages, so pip never fights
# with apt-managed site-packages (PEP 668).
RUN python3 -m venv /opt/venv && pip install --upgrade pip setuptools wheel

# Create app directory
WORKDIR /app

# Copy project files. README.md is required: pyproject.toml declares it as the
# project readme, and the build fails without it.
COPY pyproject.toml README.md ./
COPY server/ server/
COPY scripts/ scripts/

# Install Python dependencies
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip install -e ".[all]"

# Create directories
RUN mkdir -p voices lora_adapters models_cache output

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["/opt/venv/bin/python", "-m", "uvicorn", "server.api_server:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
