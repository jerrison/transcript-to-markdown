#!/usr/bin/env bash
set -euo pipefail

echo "=== Transcript-to-Markdown Setup ==="
echo ""

# Check prerequisites
missing=()
if ! command -v ffmpeg &>/dev/null; then
    missing+=("ffmpeg (install via: brew install ffmpeg)")
fi
if ! command -v uv &>/dev/null; then
    missing+=("uv (install via: curl -LsSf https://astral.sh/uv/install.sh | sh)")
fi

if [ ${#missing[@]} -ne 0 ]; then
    echo "Missing prerequisites:"
    for m in "${missing[@]}"; do
        echo "  - $m"
    done
    exit 1
fi

echo "[1/4] Checking HuggingFace token..."
# Load token from .env if it exists
if [ -z "${HF_TOKEN:-}" ] && [ -f .env ]; then
    source .env
    export HF_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo ""
    echo "A HuggingFace token is required for pyannote speaker diarization."
    echo ""
    echo "Before continuing, you must:"
    echo "  1. Create a free account at https://huggingface.co"
    echo "  2. Accept model terms at BOTH of these pages:"
    echo "     - https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo "     - https://huggingface.co/pyannote/speaker-diarization-community-1"
    echo "  3. Create an access token at https://hf.co/settings/tokens"
    echo ""
    read -rp "Paste your HuggingFace token: " HF_TOKEN
    export HF_TOKEN
    echo "HF_TOKEN=$HF_TOKEN" > .env
    echo "  Token saved to .env"
    echo ""
elif [ ! -f .env ] || ! grep -q "HF_TOKEN" .env; then
    echo "HF_TOKEN=$HF_TOKEN" > .env
    echo "  Token saved to .env"
fi

echo "[2/4] Installing Python dependencies..."
uv sync

echo ""
echo "[3/4] Downloading Whisper large-v3-turbo model (~1.5 GB)..."
uv run python -c "
from faster_whisper import WhisperModel
print('  Downloading/verifying model...')
WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')
print('  Whisper model ready.')
"

echo ""
echo "[4/4] Downloading pyannote diarization model (~200 MB)..."
uv run python -c "
import os
from pyannote.audio import Pipeline
print('  Downloading/verifying model...')
Pipeline.from_pretrained(
    'pyannote/speaker-diarization-3.1',
    token=os.environ['HF_TOKEN'],
)
print('  Pyannote model ready.')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Usage:"
echo "  uv run python transcribe.py 00-unprocessed-audio-recordings/recording.wav"
