#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?Usage: $0 /path/to/LibriSpeech [output_dir]}
OUT=${2:-data/lhotse}
for subset in train-clean-100 dev-clean dev-other test-clean test-other; do
  python scripts/prepare_lhotse.py librispeech --input-dir "$ROOT/$subset" --subset "$subset" --output-dir "$OUT"
done
