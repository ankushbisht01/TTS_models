#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# setup_environment.sh — Full environment setup for Voice Cloning Server
#
# Sets up everything needed to run the server on a fresh Linux machine
# with an NVIDIA Blackwell GPU.
#
# Usage:  bash setup/setup_environment.sh
#
# Environment overrides:
#   PYTHON_BIN=/path/to/python3.12   Use a specific interpreter
#   RECREATE_VENV=1                  Rebuild .venv without prompting
#   CUDA_VERSION=cu128               PyTorch CUDA wheel index
#   SKIP_SYSTEM_DEPS=1               Don't touch the system package manager
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

# Always operate from the project root, no matter where we're invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="${PROJECT_ROOT}/.venv"
CUDA_VERSION="${CUDA_VERSION:-cu128}"

# Supported interpreter range (inclusive). Keep in sync with `requires-python`
# in pyproject.toml. 3.14 works: numba/llvmlite ship cp314 wheels, librosa and
# both TTS backends are pure-Python, and torch>=2.9 has cu128 cp314 builds.
PY_MIN_MINOR=11
PY_MAX_MINOR=14

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Voice Cloning Server — Environment Setup"
echo "═══════════════════════════════════════════════════════════"
echo ""
info "Project root: ${PROJECT_ROOT}"

# ── 1. Check NVIDIA GPU & Driver ─────────────────────────────────────

info "Checking NVIDIA GPU..."
HAS_GPU=0
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    HAS_GPU=1
    ok "GPU: ${GPU_NAME} | Driver: ${DRIVER} | VRAM: ${VRAM} MiB"
else
    warn "No usable NVIDIA GPU detected — installing CPU-only PyTorch."
    warn "Install the NVIDIA driver and re-run this script for GPU inference."
fi

# ── 2. Install System Dependencies ───────────────────────────────────

if [ "${SKIP_SYSTEM_DEPS:-0}" = "1" ]; then
    info "SKIP_SYSTEM_DEPS=1 — skipping system packages"
elif command -v pacman &>/dev/null; then
    info "Installing system dependencies (Arch Linux)..."
    sudo pacman -S --needed --noconfirm \
        python python-pip \
        ffmpeg sox libsndfile portaudio \
        git base-devel cmake \
        || warn "Some packages may already be installed"
    ok "System packages installed"
elif command -v apt-get &>/dev/null; then
    info "Installing system dependencies (Debian/Ubuntu)..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        python3 python3-pip python3-venv python3-dev \
        ffmpeg sox libsndfile1 libportaudio2 portaudio19-dev libsox-dev \
        git build-essential cmake
    ok "System packages installed"
else
    warn "Unknown package manager. Install manually:"
    warn "  python3, python3-venv, ffmpeg, sox, libsndfile, portaudio, git, a C toolchain"
fi

# ── 3. Select a compatible Python interpreter ────────────────────────

# Returns 0 if $1 is an interpreter inside the supported version range.
python_is_supported() {
    "$1" -c "
import sys
lo, hi = ${PY_MIN_MINOR}, ${PY_MAX_MINOR}
v = sys.version_info
raise SystemExit(0 if v[0] == 3 and lo <= v[1] <= hi else 1)
" 2>/dev/null
}

python_version_of() {
    "$1" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo "unknown"
}

info "Looking for a supported Python (3.${PY_MIN_MINOR}–3.${PY_MAX_MINOR})..."

if [ -n "${PYTHON_BIN:-}" ]; then
    if ! python_is_supported "$PYTHON_BIN"; then
        err "PYTHON_BIN=${PYTHON_BIN} is $(python_version_of "$PYTHON_BIN"), outside 3.${PY_MIN_MINOR}–3.${PY_MAX_MINOR}."
        exit 1
    fi
else
    # Prefer the system default; fall back to explicit versions. 3.14 is fine
    # and is the better choice on Blackwell (see the chatterbox note above).
    for candidate in python3 python3.14 python3.13 python3.12 python3.11 python; do
        if command -v "$candidate" &>/dev/null && python_is_supported "$candidate"; then
            PYTHON_BIN="$(command -v "$candidate")"
            break
        fi
    done
fi

# Nothing suitable on PATH — try uv, which can fetch a standalone CPython.
if [ -z "${PYTHON_BIN:-}" ] && command -v uv &>/dev/null; then
    info "No supported system Python found; provisioning 3.13 via uv..."
    uv python install 3.13
    PYTHON_BIN="$(uv python find 3.13)"
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    SYSTEM_PY="$(command -v python3 || true)"
    err "No supported Python interpreter found."
    if [ -n "$SYSTEM_PY" ]; then
        err "  System python3 is $(python_version_of "$SYSTEM_PY") — this project needs 3.${PY_MIN_MINOR}–3.${PY_MAX_MINOR}."
    fi
    echo ""
    err "Install one of these, then re-run:"
    err "  uv:     curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.13"
    err "  pyenv:  pyenv install 3.13 && PYTHON_BIN=\"\$(pyenv prefix 3.13)/bin/python\" bash setup/setup_environment.sh"
    exit 1
fi

ok "Using Python $(python_version_of "$PYTHON_BIN") at ${PYTHON_BIN}"

# ── 4. Create / validate the virtual environment ─────────────────────

if [ -d "$VENV_DIR" ]; then
    if [ ! -f "${VENV_DIR}/pyvenv.cfg" ]; then
        err "${VENV_DIR} exists but is not a virtualenv. Move it aside and re-run."
        exit 1
    fi
    if python_is_supported "${VENV_DIR}/bin/python"; then
        ok "Reusing existing venv (Python $(python_version_of "${VENV_DIR}/bin/python"))"
    else
        warn "Existing venv uses Python $(python_version_of "${VENV_DIR}/bin/python") — unsupported."
        if [ "${RECREATE_VENV:-0}" = "1" ]; then
            REPLY="y"
        elif [ -t 0 ]; then
            read -r -p "Delete ${VENV_DIR} and recreate it with ${PYTHON_BIN}? [Y/n] " REPLY
            REPLY="${REPLY:-y}"
        else
            err "Re-run with RECREATE_VENV=1 to rebuild it, or delete ${VENV_DIR} yourself."
            exit 1
        fi
        case "$REPLY" in
            [yY]*)
                info "Removing ${VENV_DIR}..."
                rm -rf "$VENV_DIR"
                ;;
            *)
                err "Aborted — cannot install into an unsupported interpreter."
                exit 1
                ;;
        esac
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment..."
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
        err "venv creation failed. On Debian/Ubuntu install the matching venv package,"
        err "e.g. sudo apt install python3.12-venv"
        exit 1
    fi
    ok "Virtual environment created"
fi

VENV_PY="${VENV_DIR}/bin/python"

info "Upgrading pip and wheel..."
$VENV_PY -m pip install --upgrade pip wheel

# Deliberately NOT `--upgrade setuptools`: torch pins an upper bound (torch
# 2.11 requires setuptools<82), and unconditionally jumping to the latest
# breaks an already-installed torch with a resolver conflict. Install only if
# the build-system floor in pyproject.toml isn't already satisfied.
$VENV_PY -m pip install 'setuptools>=77'

# ── 5. Install PyTorch ───────────────────────────────────────────────

if [ "$HAS_GPU" = "1" ]; then
    info "Installing PyTorch with CUDA (${CUDA_VERSION}) — this downloads several GB..."
    if ! $VENV_PY -m pip install torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"; then
        err "PyTorch install failed for ${CUDA_VERSION} on Python $(python_version_of "$VENV_PY")."
        err "Check which builds exist: https://download.pytorch.org/whl/${CUDA_VERSION}/torch/"
        err "Then retry with e.g. CUDA_VERSION=cu126 bash setup/setup_environment.sh"
        exit 1
    fi
else
    info "Installing CPU-only PyTorch..."
    $VENV_PY -m pip install torch torchvision torchaudio
fi

info "Verifying PyTorch..."
$VENV_PY - <<'PY'
import torch
print(f"  PyTorch:   {torch.__version__}")
print(f"  CUDA:      {torch.version.cuda}")
print(f"  GPU OK:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"  Device:    {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:      {props.total_memory / 1e9:.1f} GB")
    print(f"  Capability: sm_{props.major}{props.minor}")
    arch_list = torch.cuda.get_arch_list()
    print(f"  Arch list: {', '.join(arch_list)}")
    if f"sm_{props.major}{props.minor}" not in arch_list:
        print("  ⚠️  This build has no kernels for your GPU — reinstall torch "
              "from a newer CUDA index (e.g. CUDA_VERSION=cu128).")
    else:
        print("  ✅ GPU architecture supported by this build")
PY

# ── 6. Install Voice Cloning Server ──────────────────────────────────

info "Installing voice-cloning-server and core dependencies..."
$VENV_PY -m pip install -e ".[dev]"
ok "Core dependencies installed"

info "Verifying imports..."
$VENV_PY -c "
import fastapi, uvicorn, librosa, soundfile, transformers
from server.api_server import app
print('  server.api_server imports cleanly')
"

info "Checking for dependency conflicts..."
if ! $VENV_PY -m pip check; then
    warn "pip reports conflicts (see above). These are usually harmless, but if"
    warn "torch was downgraded, re-run: make check-gpu"
fi
ok "Verification passed"

# ── 7. Done ──────────────────────────────────────────────────────────

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
