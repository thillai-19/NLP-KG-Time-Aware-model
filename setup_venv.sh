#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/nlp_env}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found." >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
else
  echo "Using existing virtual environment at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading core packaging tools"
python -m pip install --upgrade "pip<25" "setuptools<82" wheel

echo "Installing project dependencies"
python -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo "Downloading spaCy English model"
python -m spacy download en_core_web_sm

echo
echo "Environment setup complete."
echo "Activate it with:"
echo "  source \"$VENV_DIR/bin/activate\""
echo
echo "Run the project with:"
echo "  python \"$SCRIPT_DIR/main.py\""
