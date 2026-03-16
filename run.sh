#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: bash run.sh <questions_txt_path> <predictions_out_path>" >&2
  exit 1
fi

QUESTIONS_PATH="$1"
PREDICTIONS_PATH="$2"

python3 -m src.main_predict \
  --questions "${QUESTIONS_PATH}" \
  --output "${PREDICTIONS_PATH}" \
  --config "config/runtime.yaml"
