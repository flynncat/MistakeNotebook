#!/bin/bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
LOG="$ROOT/install.log"
exec > >(tee -a "$LOG") 2>&1

fail() {
    echo
    echo "============================================================"
    echo "Installation failed"
    echo "============================================================"
    echo "Review the error above or the log at: $LOG"
    echo "Fix the reported issue, then run Install.command again."
    read -r -p "Press Return to close..."
    exit 1
}
trap fail ERR

echo
echo "============================================================"
echo "Mistake Notebook - macOS installer"
echo "============================================================"
echo "Log: $LOG"
echo "The first installation downloads about 6-10GB."
echo "Keep this window open and maintain a stable network connection."
echo

echo "[1/7] Checking Xcode Command Line Tools..."
if ! xcode-select -p >/dev/null 2>&1; then
    echo "Opening the Apple installer. Complete the installation in that window."
    xcode-select --install >/dev/null 2>&1 || true
    for _ in $(seq 1 180); do
        if xcode-select -p >/dev/null 2>&1; then
            break
        fi
        sleep 10
    done
    if ! xcode-select -p >/dev/null 2>&1; then
        echo "Xcode Command Line Tools were not detected."
        echo "Complete the Apple installation, then run this file again."
        false
    fi
fi

echo "[2/7] Checking Homebrew..."
if ! command -v brew >/dev/null 2>&1; then
    echo "Installing Homebrew. macOS may request the administrator password."
    /bin/bash -c \
        "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
fi
command -v brew >/dev/null 2>&1

echo "[3/7] Installing Python 3.11, Git, and optional verification OCR..."
brew list python@3.11 >/dev/null 2>&1 || brew install python@3.11
brew list git >/dev/null 2>&1 || brew install git
brew list tesseract >/dev/null 2>&1 || brew install tesseract
brew list tesseract-lang >/dev/null 2>&1 || brew install tesseract-lang
PYTHON="$(brew --prefix python@3.11)/bin/python3.11"
"$PYTHON" --version
git --version

echo "[4/7] Creating the Python virtual environment..."
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    "$PYTHON" -m venv "$ROOT/.venv"
fi
VENV_PYTHON="$ROOT/.venv/bin/python"

echo "[5/7] Installing project dependencies..."
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -e '.[dev,v2]'

echo "[6/7] Installing OCR, formula, and document models..."
echo "This step can take a long time. Re-run this installer after an interruption."
export PYTORCH_ENABLE_MPS_FALLBACK=1
if [[ "$(uname -m)" == "arm64" ]]; then
    "$VENV_PYTHON" scripts/setup_runtime.py
else
    echo "Intel macOS detected; PaddleOCR is unavailable, using macOS Vision."
    "$VENV_PYTHON" scripts/setup_runtime.py --skip-paddle
fi

echo "[7/7] Running installation checks..."
"$VENV_PYTHON" -c \
    "from mistake_book.app import create_app; app=create_app(); print('Application import: OK')"
"$VENV_PYTHON" -c \
    "from mistake_book.font_selection import default_font_metrics; print('Chinese font:', default_font_metrics()['rendered_family'])"
"$VENV_PYTHON" -m pytest -q tests/test_portability.py

echo
echo "============================================================"
echo "Installation complete"
echo "============================================================"
echo "Start the application with:"
echo "  .venv/bin/mistake-book --root ."
echo
echo "Then open http://127.0.0.1:8765"
echo "The installer is safe to run again."
echo
read -r -p "Press Return to close..."
