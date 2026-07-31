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
    PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg sox libportaudio2 libsox-dev \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY server/ server/
COPY scripts/ scripts/

# Install Python dependencies
RUN python3 -m pip install --break-system-packages \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

RUN python3 -m pip install --break-system-packages -e ".[all]"

# Create directories
RUN mkdir -p voices lora_adapters models_cache output

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python3", "-m", "uvicorn", "server.api_server:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
