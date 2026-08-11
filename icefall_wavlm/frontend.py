from __future__ import annotations
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import WavLMModel


def wavlm_output_lengths(input_lengths: torch.Tensor, conv_kernel, conv_stride) -> torch.Tensor:
    lengths = input_lengths.clone()
    for kernel, stride in zip(conv_kernel, conv_stride):
        lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
    return lengths.clamp_min(0)


class WavLMFrontend(nn.Module):
    """Raw-waveform WavLM frontend that returns [B,T,C] SSL frames and lengths."""
    def __init__(
        self,
        model_name: str,
        output_dim: int,
        weighted_layers: bool = True,
        layer: int = -1,
        dropout: float = 0.1,
        adapter_stride: int = 1,
    ):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name)
        hidden = self.wavlm.config.hidden_size
        self.weighted_layers = weighted_layers
        self.layer = layer
        n_layers = self.wavlm.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.zeros(n_layers)) if weighted_layers else None
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.adapter_stride = int(adapter_stride)
        if self.adapter_stride < 1:
            raise ValueError("adapter_stride must be >= 1")
        self.stride_conv = (
            nn.Conv1d(output_dim, output_dim, kernel_size=3, stride=self.adapter_stride, padding=1)
            if self.adapter_stride > 1 else None
        )

    @property
    def feature_dim(self):
        return self.proj.out_features

    def freeze_feature_encoder(self):
        self.wavlm.feature_extractor._freeze_parameters()

    def set_wavlm_trainable(self, trainable: bool, keep_feature_encoder_frozen: bool = True):
        for p in self.wavlm.parameters():
            p.requires_grad = trainable
        if keep_feature_encoder_frozen:
            self.freeze_feature_encoder()

    def _select(self, hidden_states):
        if self.weighted_layers:
            w = torch.softmax(self.layer_weights, dim=0)
            return torch.stack([wi * hi for wi, hi in zip(w, hidden_states)], dim=0).sum(dim=0)
        return hidden_states[self.layer]

    def forward(self, waveforms: torch.Tensor, sample_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # waveforms: [B,S], padded with zero
        max_s = waveforms.size(1)
        arange = torch.arange(max_s, device=waveforms.device).unsqueeze(0)
        attention_mask = (arange < sample_lens.unsqueeze(1)).long()
        out = self.wavlm(
            input_values=waveforms,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        x = self._select(out.hidden_states)
        x = self.dropout(self.proj(self.norm(x)))
        lens = wavlm_output_lengths(
            sample_lens,
            self.wavlm.config.conv_kernel,
            self.wavlm.config.conv_stride,
        )
        if self.stride_conv is not None:
            x = self.stride_conv(x.transpose(1, 2)).transpose(1, 2)
            lens = torch.div(lens + self.adapter_stride - 1, self.adapter_stride, rounding_mode="floor")
        return x, lens


class IdentityEncoderEmbed(nn.Module):
    """Adapter matching Icefall AsrModel.encoder_embed(x, x_lens)."""
    def forward(self, x, x_lens):
        return x, x_lens
