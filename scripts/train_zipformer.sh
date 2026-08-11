#!/usr/bin/env bash
set -euo pipefail
: "${ICEFALL_ROOT:?Set ICEFALL_ROOT to your icefall checkout}"
python -m icefall_wavlm.train \
  --icefall-root "$ICEFALL_ROOT" \
  --icefall-recipe "$ICEFALL_ROOT/egs/librispeech/ASR/zipformer" \
  --bpe-model data/lang_bpe_500/bpe.model \
  --wavlm-model microsoft/wavlm-base-plus \
  --train-cuts data/fbank/librispeech_cuts_train-clean-100.jsonl.gz \
  --valid-cuts data/fbank/librispeech_cuts_dev-clean.jsonl.gz \
  --exp-dir exp/wavlm_zipformer
