#!/usr/bin/env bash
set -e
MODEL_DIR="src/models"
mkdir -p "$MODEL_DIR"

FILE="$MODEL_DIR/lid.176.bin"
URL="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

if [ ! -f "$FILE" ]; then
  echo "Downloading fastText language model..."
  curl -L -o "$FILE" "$URL"
fi
