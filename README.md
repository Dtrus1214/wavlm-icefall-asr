# WavLM + Icefall Zipformer Pruned RNN-T

Research project for replacing FBank/MFCC with self-supervised WavLM representations in end-to-end ASR.

## Architecture

`16-kHz waveform -> WavLM -> learnable layer weighting -> LayerNorm/Linear adapter -> Icefall Zipformer2 -> stateless predictor -> joiner -> pruned RNN-T`

This repository contains three stages:

1. **WavLM SSL pretraining / continued pretraining** using Microsoft's official WavLM/fairseq implementation.
2. **Supervised WavLM CTC fine-tuning** using Hugging Face Transformers.
3. **WavLM + Icefall Zipformer pruned RNN-T** training and waveform inference.

The Icefall integration deliberately bypasses the normal 80-bin FBank `Conv2dSubsampling` frontend. WavLM already produces a learned sequence near 50 Hz, so `SSLAdapter` replaces that frontend and presents the interface expected by Icefall's `AsrModel`.

## Requirements

Recommended: Linux, Python 3.10/3.11, CUDA GPU.

External repositories are intentionally not vendored:

```bash
git clone https://github.com/k2-fsa/icefall.git third_party/icefall
git clone https://github.com/microsoft/unilm.git third_party/unilm
```

Install Icefall/k2 according to the Icefall installation guide for your CUDA/PyTorch combination, then:

```bash
pip install -r requirements.txt
export ICEFALL_ROOT=$PWD/third_party/icefall
export WAVLM_OFFICIAL_ROOT=$PWD/third_party/unilm/wavlm
```

## Dataset manifest

For the standalone WavLM fine-tuning tools use JSONL:

```json
{"audio":"/abs/path/1.wav","text":"THE QUICK BROWN FOX"}
{"audio":"/abs/path/2.wav","text":"HELLO WORLD"}
```

The Icefall RNN-T stage uses the normal Lhotse manifests/BPE model from an Icefall recipe. For LibriSpeech, first run Icefall's standard `egs/librispeech/ASR/prepare.sh`.

## Stage 1: WavLM SSL pretraining / continued pretraining

Exact WavLM pretraining depends on HuBERT-style offline k-means targets and Microsoft's fairseq task. The launcher validates paths and invokes the official training code rather than reimplementing the objective inaccurately.

```bash
python -m wavlm_ssl.prepare_fairseq_manifest \
  --audio-dir /data/unlabeled_wavs \
  --out-dir data/wavlm_ssl

python -m wavlm_ssl.train_official \
  --official-root "$WAVLM_OFFICIAL_ROOT" \
  --data-dir data/wavlm_ssl \
  --label-dir /data/hubert_kmeans_labels \
  --config configs/wavlm_base_continue.yaml \
  --save-dir exp/wavlm_ssl
```

`configs/wavlm_base_continue.yaml` documents the expected knobs. The exact fairseq config name/override set can vary with the Microsoft checkout; `train_official.py --dry-run` prints the command.

## Stage 2: supervised WavLM CTC fine-tuning

```bash
python -m finetune.train_ctc \
  --train-manifest data/train.jsonl \
  --valid-manifest data/valid.jsonl \
  --model microsoft/wavlm-base-plus \
  --output-dir exp/wavlm_ctc

python -m finetune.infer_ctc \
  --model exp/wavlm_ctc/best \
  --wav test.wav
```

## Stage 3: WavLM + Zipformer + pruned RNN-T

Run from an Icefall LibriSpeech ASR working directory after `prepare.sh`, or pass absolute paths.

```bash
python -m icefall_wavlm.train \
  --icefall-root "$ICEFALL_ROOT" \
  --icefall-recipe "$ICEFALL_ROOT/egs/librispeech/ASR/zipformer" \
  --bpe-model data/lang_bpe_500/bpe.model \
  --wavlm-model microsoft/wavlm-base-plus \
  --train-cuts data/fbank/librispeech_cuts_train-clean-100.jsonl.gz \
  --valid-cuts data/fbank/librispeech_cuts_dev-clean.jsonl.gz \
  --exp-dir exp/wavlm_zipformer \
  --num-epochs 20
```

The manifests may be named differently in newer Icefall preparations; point `--train-cuts` and `--valid-cuts` at the generated Lhotse cut manifests.

Single-file inference:

```bash
python -m icefall_wavlm.recognize \
  --icefall-root "$ICEFALL_ROOT" \
  --icefall-recipe "$ICEFALL_ROOT/egs/librispeech/ASR/zipformer" \
  --bpe-model data/lang_bpe_500/bpe.model \
  --checkpoint exp/wavlm_zipformer/epoch-20.pt \
  --wav test.wav \
  --decoding-method greedy_search
```

## Freezing strategy

Defaults are conservative:

- WavLM convolutional feature extractor frozen.
- WavLM transformer frozen for `--freeze-wavlm-epochs 2` epochs.
- Then the transformer is unfrozen at a lower learning rate than Zipformer.
- A learnable softmax weighted sum of WavLM hidden layers is used unless `--wavlm-layer` selects one layer.

For small labeled corpora, increase frozen epochs. For very large corpora, unfreeze earlier.

## Important timing detail

FBank recipes commonly start near 100 Hz and Icefall's frontend subsamples. WavLM's convolutional frontend outputs near 50 Hz. This project therefore uses an adapter with optional `--adapter-stride` (default 1), then Zipformer2's own output downsampling. Do not blindly retain the FBank Conv2d frontend.

## Smoke tests

```bash
python -m compileall wavlm_ssl finetune icefall_wavlm utils tests
python -m unittest tests.test_manifest tests.test_adapter_math
```

## What is intentionally external

- WavLM official SSL pretraining implementation: Microsoft/unilm.
- Zipformer2, stateless decoder/joiner, Icefall optimizer/utilities: k2-fsa/icefall.
- k2 pruned RNN-T loss.

Keeping these upstream components external avoids silently diverging from the exact algorithms while this project supplies the WavLM-to-Icefall integration layer.
