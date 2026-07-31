# Voice Cloning Server 🎤

Self-hosted voice cloning & TTS server with an OpenAI-compatible API. Supports multiple frontier TTS backends (F5-TTS, Chatterbox, Qwen3-TTS) on NVIDIA GPUs.

## Features

- **Zero-shot voice cloning** — clone any voice from 5-15 seconds of reference audio
- **OpenAI-compatible API** — drop-in replacement for `/v1/audio/speech`
- **Multiple backends** — swap between F5-TTS, Chatterbox, and Qwen3-TTS
- **Voice profile management** — save, list, and reuse cloned voices
- **Emotion control** — adjust expressiveness (Chatterbox)
- **Voice design from text** — describe a voice to create it (Qwen3-TTS)
- **CLI tools** — command-line cloning, benchmarking, and dataset preparation
- **Docker support** — one-command deployment with GPU passthrough

## Supported Models

| Model | Params | License | Voice Cloning | Best For |
|:--|:--|:--|:--|:--|
| **F5-TTS** | ~330M | CC-BY-NC / Apache 2.0 | ✅ Zero-shot + LoRA | Highest quality, natural prosody |
| **Chatterbox** | ~300M | MIT | ✅ Zero-shot | Production use, emotion control |
| **Qwen3-TTS** | 0.6B–1.7B | Apache 2.0 | ✅ Zero-shot + voice design | Low-latency, multilingual |

## Hardware Requirements

| Component | Minimum | Recommended |
|:--|:--|:--|
| GPU | NVIDIA 8GB VRAM | NVIDIA 16-24GB VRAM |
| CUDA | 12.8+ | 12.8+ |
| PyTorch | 2.7+ | 2.7+ |
| RAM | 16 GB | 32+ GB |
| Python | 3.11+ | 3.12 |

> **Blackwell GPU users (RTX 5000/PRO 4000):** PyTorch ≥ 2.7 and CUDA ≥ 12.8 are **mandatory**. Older versions will fail with kernel errors.

## Quick Start

### 1. Clone & Setup

```bash
git clone <your-repo-url> voice-cloning-server
cd voice-cloning-server

# Run the full setup script (installs system deps, creates venv, installs PyTorch)
chmod +x setup/setup_environment.sh
./setup/setup_environment.sh
```

### 2. Install a TTS Backend

```bash
source .venv/bin/activate

# Option A: F5-TTS (recommended — highest quality cloning)
pip install -e ".[f5tts]"

# Option B: Chatterbox (MIT license, emotion control)
pip install -e ".[chatterbox]"

# Option C: Both
pip install -e ".[all]"
```

### 3. Start the Server

```bash
# Using Make
make run

# Or directly
uvicorn server.api_server:app --host 0.0.0.0 --port 8000
```

### 4. Clone a Voice

```bash
# Upload reference audio and create a voice profile
curl -X POST http://localhost:8000/v1/voices/create \
  -F "name=my-voice" \
  -F "audio=@reference_audio.wav" \
  -F "ref_text=This is what I said in the reference clip."

# Generate speech with the cloned voice
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "f5tts",
    "voice": "my_voice",
    "input": "Hello! This is my cloned voice speaking new text."
  }' \
  --output output.wav
```

### One-Shot Cloning (No Profile Needed)

```bash
curl -X POST http://localhost:8000/v1/clone \
  -F "text=Hello from a cloned voice!" \
  -F "ref_audio=@speaker.wav" \
  -F "ref_text=Transcript of the reference audio" \
  --output cloned_output.wav
```

## CLI Usage

```bash
# Quick voice clone from command line
python -m scripts.clone_voice \
  --ref speaker.wav \
  --text "Hello world" \
  --output output.wav \
  --backend f5tts

# Benchmark backends
python -m scripts.benchmark --backends f5tts,chatterbox

# Prepare dataset for fine-tuning
python -m scripts.prepare_dataset \
  --input-dir ./raw_audio/ \
  --output-dir ./training_data/ \
  --speaker-name "speaker1"
```

## Docker

```bash
# Build and run with GPU
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

## API Endpoints

| Method | Endpoint | Description |
|:--|:--|:--|
| `GET` | `/health` | Health check |
| `GET` | `/v1/status` | Server status + GPU info |
| `POST` | `/v1/audio/speech` | OpenAI-compatible TTS |
| `POST` | `/v1/generate` | Extended TTS with full control |
| `POST` | `/v1/clone` | One-shot cloning (upload audio inline) |
| `GET` | `/v1/voices` | List saved voice profiles |
| `POST` | `/v1/voices/create` | Create a voice profile |
| `DELETE` | `/v1/voices/{id}` | Delete a voice profile |
| `GET` | `/v1/voices/{id}/audio` | Download reference audio |
| `GET` | `/v1/backends` | List loaded backends |

## Configuration

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

Key settings:

| Variable | Default | Description |
|:--|:--|:--|
| `ACTIVE_BACKENDS` | `f5tts` | Backends to load (comma-separated) |
| `DEFAULT_BACKEND` | `f5tts` | Default for requests without explicit backend |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `PORT` | `8000` | Server port |

## Project Structure

```
├── server/
│   ├── api_server.py          # FastAPI server with all endpoints
│   ├── config.py              # Configuration (env vars / .env)
│   ├── voice_manager.py       # Voice profile CRUD
│   └── models/
│       ├── base.py            # Abstract backend interface
│       ├── f5tts_backend.py   # F5-TTS wrapper
│       ├── chatterbox_backend.py  # Chatterbox wrapper
│       └── qwen3tts_backend.py    # Qwen3-TTS wrapper
├── scripts/
│   ├── clone_voice.py         # CLI voice cloning tool
│   ├── benchmark.py           # Backend comparison
│   └── prepare_dataset.py     # Fine-tuning dataset prep
├── setup/
│   └── setup_environment.sh   # Full environment setup
├── tests/
│   └── test_api.py            # Test suite
├── voices/                    # Saved voice profiles
├── lora_adapters/             # Fine-tuned LoRA weights
├── pyproject.toml             # Python project config
├── Dockerfile                 # GPU container
├── docker-compose.yml         # Container orchestration
├── Makefile                   # Shortcuts
└── .env.example               # Configuration template
```

## License

This project scaffolding is MIT licensed. Individual TTS model weights have their own licenses — check each model's page before commercial use.
