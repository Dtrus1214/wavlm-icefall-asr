from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torchaudio


def load_audio(path: str | Path, sample_rate: int = 16000) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.ndim != 2:
        raise ValueError(f"Expected [channels, samples], got {tuple(wav.shape)}")
    wav = wav.mean(dim=0)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
        sr = sample_rate
    peak = wav.abs().max().clamp_min(1e-8)
    if peak > 1.0:
        wav = wav / peak
    return wav.contiguous(), sr
