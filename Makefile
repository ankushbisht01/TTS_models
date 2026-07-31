.PHONY: help setup setup-all install-f5tts install-chatterbox install-qwen3tts \
       run run-dev test benchmark docker-build docker-up docker-down clean

SHELL := /bin/bash
PYTHON ?= python3
VENV := .venv
CUDA_VERSION ?= cu128

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Environment Setup ───────────────────────────────────────────────

setup: ## Set up Python venv + core dependencies (run this first)
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip setuptools wheel
	$(VENV)/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$(CUDA_VERSION)
	$(VENV)/bin/pip install -e ".[dev]"
	@echo ""
	@echo "✅ Core setup complete. Now install model backends:"
	@echo "   make install-f5tts       # Recommended primary model"
	@echo "   make install-chatterbox  # Production alternative"
	@echo "   make install-qwen3tts    # Real-time / low-latency"

setup-all: setup install-f5tts install-chatterbox ## Set up everything (core + F5-TTS + Chatterbox)

install-f5tts: ## Install F5-TTS backend
	$(VENV)/bin/pip install -e ".[f5tts]"
	@echo "✅ F5-TTS installed"

install-chatterbox: ## Install Chatterbox backend
	$(VENV)/bin/pip install -e ".[chatterbox]"
	@echo "✅ Chatterbox installed"

install-qwen3tts: ## Install Qwen3-TTS backend
	$(VENV)/bin/pip install -e ".[qwen3tts]"
	@echo "✅ Qwen3-TTS installed"

# ─── Running ─────────────────────────────────────────────────────────

run: ## Start the API server (production)
	$(VENV)/bin/uvicorn server.api_server:app \
		--host 0.0.0.0 --port 8000 --workers 1

run-dev: ## Start the API server (dev mode with auto-reload)
	$(VENV)/bin/uvicorn server.api_server:app \
		--host 0.0.0.0 --port 8000 --reload --log-level debug

# ─── Testing ─────────────────────────────────────────────────────────

test: ## Run all tests
	$(VENV)/bin/pytest tests/ -v

check-gpu: ## Verify GPU + CUDA + PyTorch setup
	$(VENV)/bin/python -c "\
		import torch; \
		print(f'PyTorch: {torch.__version__}'); \
		print(f'CUDA available: {torch.cuda.is_available()}'); \
		print(f'CUDA version: {torch.version.cuda}'); \
		print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); \
		print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else ''); \
		print(f'Arch list: {torch.cuda.get_arch_list()}'); \
	"

benchmark: ## Run model benchmarks
	$(VENV)/bin/python -m scripts.benchmark

# ─── Docker ──────────────────────────────────────────────────────────

docker-build: ## Build Docker image
	docker build -t voice-cloning-server:latest .

docker-up: ## Start services via Docker Compose
	docker compose up -d

docker-down: ## Stop Docker Compose services
	docker compose down

# ─── Maintenance ─────────────────────────────────────────────────────

clean: ## Remove build artifacts, caches, and generated files
	rm -rf $(VENV) build/ dist/ *.egg-info __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleaned"

lint: ## Run linter
	$(VENV)/bin/ruff check server/ scripts/ tests/

format: ## Auto-format code
	$(VENV)/bin/ruff format server/ scripts/ tests/
