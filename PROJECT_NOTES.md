# Design notes

## Why there is no MFCC/FBank
The RNN-T stage requests raw samples from Lhotse via `AudioSamples()`. WavLM's convolutional feature encoder and transformer produce the acoustic representation.

## Why the normal Icefall Conv2dSubsampling is removed
Current Icefall Zipformer recipes treat `encoder_embed` as a Conv2d frontend that maps FBank frames to the first encoder dimension and subsamples them. This project replaces it with an identity embed because `WavLMFrontend` already projects learned SSL frames to the first Zipformer dimension.

## Exact WavLM pretraining
WavLM pretraining is not equivalent to a generic reconstruction loss. It uses a HuBERT-style masked prediction pipeline plus denoising/utterance mixing and gated relative position bias. Therefore Stage 1 launches Microsoft's official implementation. This is more reproducible than a short local approximation labeled as “WavLM training.”

## Current limitations
- The included custom `recognize.py` implements batch-size-1 greedy search. Icefall's richer modified/fast beam search can be adapted by wrapping the encoder output in the same way.
- Distributed training, Eden optimizer/scheduler, model averaging, and ONNX export are not duplicated in this compact integration recipe. They can be added by transplanting this frontend into the upstream Icefall training script.
- Exact upstream class signatures can evolve. The project targets the current Zipformer recipe structure in which `Zipformer2`, `Decoder`, `Joiner`, and `AsrModel` are importable from the recipe directory.
