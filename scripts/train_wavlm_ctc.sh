#!/usr/bin/env bash
set -euo pipefail
python -m finetune.train_ctc \
  --train-manifest data/train.jsonl \
  --valid-manifest data/valid.jsonl \
  --model microsoft/wavlm-base-plus \
  --output-dir exp/wavlm_ctc
