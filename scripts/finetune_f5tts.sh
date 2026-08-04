#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# finetune_f5tts.sh — Fine-tune F5-TTS on a prepared dataset.
#
# Usage:
#   bash scripts/finetune_f5tts.sh <prepared-dataset-dir> <speaker-name>
#
# Example:
#   python -m scripts.prepare_dataset -i ./raw_audio -o ./training_data \
#       -s john --transcribe
#   bash scripts/finetune_f5tts.sh ./training_data john
#
# Environment overrides:
#   EPOCHS=100              Training epochs
#   BATCH_SIZE=3200         Frames per batch (batch_size_type=frame)
#   LEARNING_RATE=1e-5      LR — keep low for fine-tuning
#   SAVE_PER_UPDATES=2000   Checkpoint interval
#   BASE_MODEL=F5TTS_v1_Base
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()  { echo -e "${RED}[ERR]${NC}   $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 2 ]; then
    err "Usage: bash scripts/finetune_f5tts.sh <prepared-dataset-dir> <speaker-name>"
    exit 1
fi

if [ ! -d "$1" ]; then
    err "Dataset directory '$1' does not exist."
    err "Prepare it first:"
    err "  make prepare-dataset IN=./raw_audio OUT=$1 SPEAKER=$2"
    exit 1
fi

DATASET_DIR="$(cd "$1" && pwd)"
SPEAKER="$2"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-3200}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SAVE_PER_UPDATES="${SAVE_PER_UPDATES:-2000}"
BASE_MODEL="${BASE_MODEL:-F5TTS_v1_Base}"
TOKENIZER="pinyin"   # f5-tts default; the data dir name embeds this

# Prefer the project venv so this works without activating it first.
if [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
    PY="${PROJECT_ROOT}/.venv/bin/python"
else
    PY="$(command -v python3)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  F5-TTS Fine-Tuning"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── 1. Preflight ─────────────────────────────────────────────────────

if ! $PY -c "import f5_tts" 2>/dev/null; then
    err "f5-tts is not installed. Run: pip install -e '.[f5tts]'"
    exit 1
fi

METADATA="${DATASET_DIR}/metadata.csv"
if [ ! -f "$METADATA" ]; then
    err "No metadata.csv in ${DATASET_DIR}. Run scripts/prepare_dataset.py first."
    exit 1
fi

# Empty transcripts train the model to map audio to nothing — catch it here
# rather than after hours of GPU time.
EMPTY=$($PY - "$METADATA" <<'PY'
import csv, sys
with open(sys.argv[1], encoding="utf-8") as f:
    rows = list(csv.reader(f, delimiter="|"))
print(sum(1 for r in rows[1:] if len(r) < 2 or not r[1].strip()))
PY
)
TOTAL=$(( $(wc -l < "$METADATA") - 1 ))

if [ "$EMPTY" -gt 0 ]; then
    err "${EMPTY} of ${TOTAL} rows in metadata.csv have an empty transcript."
    err "Fill them in, or re-run prepare_dataset.py with --transcribe."
    exit 1
fi
ok "Dataset: ${TOTAL} segments, all transcribed"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    | head -1 | sed 's/^/[INFO]  GPU: /' || warn "No GPU detected — training will be unusably slow"

# ── 2. Resolve the paths f5-tts hard-codes ───────────────────────────
#
# f5-tts loads data from  <f5_tts pkg>/../../data/<dataset>_<tokenizer>
# and writes checkpoints to <f5_tts pkg>/../../ckpts/<dataset>
# (see f5_tts/model/dataset.py::load_dataset and train/finetune_cli.py).
# With a pip install those resolve inside the venv, so compute them rather
# than guessing.

DATA_ROOT=$($PY -c "
from importlib.resources import files
from pathlib import Path
print(Path(str(files('f5_tts').joinpath('../../data'))).resolve())
")
CKPT_ROOT=$($PY -c "
from importlib.resources import files
from pathlib import Path
print(Path(str(files('f5_tts').joinpath('../../ckpts'))).resolve())
")

PREPARED_DIR="${DATA_ROOT}/${SPEAKER}_${TOKENIZER}"
OUT_CKPT_DIR="${CKPT_ROOT}/${SPEAKER}"

info "Arrow dataset -> ${PREPARED_DIR}"
info "Checkpoints   -> ${OUT_CKPT_DIR}"

mkdir -p "$DATA_ROOT" "$CKPT_ROOT"

# ── 3. Build the Arrow dataset ───────────────────────────────────────

info "Converting CSV + wavs to Arrow format..."
$PY -m f5_tts.train.datasets.prepare_csv_wavs "$METADATA" "$PREPARED_DIR"
ok "Dataset converted"

if [ ! -f "${PREPARED_DIR}/duration.json" ]; then
    err "prepare_csv_wavs did not produce duration.json — conversion failed."
    exit 1
fi

# ── 4. Train ─────────────────────────────────────────────────────────

info "Starting fine-tuning (epochs=${EPOCHS}, lr=${LEARNING_RATE})..."
echo ""

$PY -m f5_tts.train.finetune_cli \
    --exp_name "$BASE_MODEL" \
    --dataset_name "$SPEAKER" \
    --finetune \
    --learning_rate "$LEARNING_RATE" \
    --batch_size_per_gpu "$BATCH_SIZE" \
    --batch_size_type frame \
    --epochs "$EPOCHS" \
    --save_per_updates "$SAVE_PER_UPDATES" \
    --last_per_updates 1000 \
    --keep_last_n_checkpoints 3 \
    --tokenizer "$TOKENIZER" \
    --log_samples

# ── 5. Done ──────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
ok "Fine-tuning complete"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Checkpoints in: ${OUT_CKPT_DIR}"
ls -1t "$OUT_CKPT_DIR" 2>/dev/null | head -5 | sed 's/^/  /'
echo ""
echo "To serve the fine-tuned model, add to .env and restart:"
echo ""
echo "  ACTIVE_BACKENDS=f5tts"
echo "  DEFAULT_BACKEND=f5tts"
echo "  F5TTS_CKPT_FILE=${OUT_CKPT_DIR}/model_last.safetensors"
echo ""
echo "Then generate as usual — the fine-tuned voice still needs a reference"
echo "clip, so keep using a /v1/voices profile for the same speaker."
echo ""
