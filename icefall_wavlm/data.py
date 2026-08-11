from __future__ import annotations
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler, K2SpeechRecognitionDataset, SimpleCutSampler
from lhotse.dataset.input_strategies import AudioSamples


def load_cuts(path: str | Path) -> CutSet:
    return CutSet.from_file(path)


def make_loader(cuts: CutSet, max_duration: float, shuffle: bool, num_workers: int = 4):
    dataset = K2SpeechRecognitionDataset(input_strategy=AudioSamples())
    sampler_cls = DynamicBucketingSampler if shuffle else SimpleCutSampler
    sampler = sampler_cls(cuts, max_duration=max_duration, shuffle=shuffle)
    return torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=num_workers)


def unpack_audio_batch(batch, device):
    # AudioSamples produces [B,S] and supervisions['num_samples'].
    x = batch["inputs"]
    if x.ndim == 3 and x.size(-1) == 1:
        x = x.squeeze(-1)
    if x.ndim != 2:
        raise ValueError(f"Expected raw samples [B,S], got {tuple(x.shape)}")
    return x.to(device), batch["supervisions"]["num_samples"].to(device)
