#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# setup_environment.sh — Full environment setup for Voice Cloning Server
#
# This script sets up everything needed to run the server on a fresh
# Arch Linux machine with an NVIDIA Blackwell GPU.
#
# Usage:  chmod +x setup/setup_environment.sh && ./setup/setup_environment.sh
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERR]${NC}   $*"; }

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Voice Cloning Server — Environment Setup"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── 1. Check NVIDIA GPU & Driver ─────────────────────────────────────

info "Checking NVIDIA GPU..."
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    ok "GPU: ${GPU_NAME} | Driver: ${DRIVER} | VRAM: ${VRAM} MiB"
else
    err "nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi

# ── 2. Install System Dependencies (Arch Linux) ─────────────────────

info "Installing system dependencies..."
if command -v pacman &>/dev/null; then
    sudo pacman -S --needed --noconfirm \
        python python-pip \
        ffmpeg sox portaudio \
        git base-devel \
        2>/dev/null || warn "Some packages may already be installed"
    ok "System packages installed (Arch Linux)"
elif command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        python3 python3-pip python3-venv \
        ffmpeg sox libportaudio2 portaudio19-dev libsox-dev \
        git build-essential
    ok "System packages installed (Debian/Ubuntu)"
else
    warn "Unknown package manager. Install manually: python3, ffmpeg, sox, portaudio, git"
fi

# ── 3. Create Python Virtual Environment ─────────────────────────────

VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists at ${VENV_DIR}"
else
    info "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

# Activate
source "${VENV_DIR}/bin/activate"
info "Python: $(python --version) at $(which python)"

# Upgrade pip
pip install --upgrade pip setuptools wheel -q

# ── 4. Install PyTorch with CUDA 12.8 (Blackwell support) ───────────

info "Installing PyTorch >= 2.7 with CUDA 12.8 support..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 \
    -q

# Verify
info "Verifying PyTorch + CUDA..."
python -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.version.cuda}')
print(f'  GPU OK:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device:   {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'  VRAM:     {props.total_mem / 1e9:.1f} GB')
    arch_list = torch.cuda.get_arch_list()
    has_blackwell = any('sm_120' in a or 'sm_12' in a for a in arch_list)
    print(f'  Blackwell: {\"✅\" if has_blackwell else \"⚠️  sm_120 not in arch list\"}')
"

# ── 5. Install Voice Cloning Server ─────────────────────────────────

info "Installing voice-cloning-server and core dependencies..."
pip install -e ".[dev]" -q
ok "Core dependencies installed"

# ── 6. Done ──────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ Environment setup complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Install TTS backend(s):"
echo "     pip install -e '.[f5tts]'        # F5-TTS (recommended)"
echo "     pip install -e '.[chatterbox]'   # Chatterbox"
echo "     pip install -e '.[all]'          # All backends"
echo ""
echo "  3. Start the server:"
echo "     make run"
echo "     # or: uvicorn server.api_server:app --host 0.0.0.0 --port 8000"
echo ""
echo "  4. Verify GPU:"
echo "     make check-gpu"
echo ""
