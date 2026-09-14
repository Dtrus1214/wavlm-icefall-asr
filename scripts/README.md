# WavLM + Icefall Zipformer ASR

End-to-end ASR research project using raw waveform -> WavLM -> SSL adapter -> Icefall Zipformer -> pruned RNN-T.

Includes generic/LibriSpeech JSONL prep, Lhotse RecordingSet/SupervisionSet/CutSet prep, SentencePiece prep, WavLM frontend, Zipformer encoder integration, training validation, and inference utilities.

## Environment

Install compatible PyTorch and k2 first, then:

```bash
pip install -r requirements.txt
git clone https://github.com/k2-fsa/icefall.git third_party/icefall
export PYTHONPATH=$PWD/third_party/icefall:$PYTHONPATH
```

Icefall currently recommends the latest Lhotse from GitHub:

```bash
pip install git+https://github.com/lhotse-speech/lhotse
```

## WavLM

Default model: `microsoft/wavlm-base-plus`. Hugging Face will download it automatically.

```bash
huggingface-cli download microsoft/wavlm-base-plus --local-dir models/wavlm-base-plus
```

## LibriSpeech -> JSONL

```bash
python scripts/prepare_dataset.py librispeech --input-dir /data/LibriSpeech/train-clean-100 --output data/jsonl/train-clean-100.jsonl
```

## LibriSpeech -> Lhotse

```bash
python scripts/prepare_lhotse.py librispeech --input-dir /data/LibriSpeech/train-clean-100 --subset train-clean-100 --output-dir data/lhotse
```

Outputs:
- `librispeech_recordings_train-clean-100.jsonl.gz`
- `librispeech_supervisions_train-clean-100.jsonl.gz`
- `librispeech_cuts_train-clean-100.jsonl.gz`

## Generic WAV/TXT -> Lhotse

```bash
python scripts/prepare_lhotse.py generic --input-dir /data/my_dataset --subset train --output-dir data/lhotse --language en --speaker-from-parent
```

## SentencePiece

```bash
python scripts/train_sentencepiece.py --manifest data/lhotse/librispeech_supervisions_train-clean-100.jsonl.gz --output-dir data/lang_bpe_500 --vocab-size 500
```

## Inspect cuts

```bash
python scripts/inspect_lhotse.py --cuts data/lhotse/librispeech_cuts_train-clean-100.jsonl.gz
```

## Validate WavLM -> Zipformer

Add the Icefall LibriSpeech Zipformer recipe directory to `PYTHONPATH`, for example:

```bash
export PYTHONPATH=$PWD/third_party/icefall/egs/librispeech/ASR/zipformer:$PWD/third_party/icefall:$PYTHONPATH
```

Then:

```bash
python -m wavlm_zipformer.train --train-cuts data/lhotse/librispeech_cuts_train-clean-100.jsonl.gz --valid-cuts data/lhotse/librispeech_cuts_dev-clean.jsonl.gz --bpe-model data/lang_bpe_500/bpe.model
```

No FBank features are computed because WavLM consumes raw waveform.
