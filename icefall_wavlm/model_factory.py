from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

from icefall_wavlm.frontend import IdentityEncoderEmbed, WavLMFrontend


@dataclass
class ModelConfig:
    wavlm_model: str = "microsoft/wavlm-base-plus"
    weighted_layers: bool = True
    wavlm_layer: int = -1
    adapter_stride: int = 1
    encoder_dims: tuple = (192, 256, 384, 512, 384, 256)
    downsampling_factors: tuple = (1, 2, 4, 8, 4, 2)
    num_encoder_layers: tuple = (2, 2, 3, 4, 3, 2)
    encoder_unmasked_dims: tuple = (192, 192, 256, 256, 256, 192)
    feedforward_dims: tuple = (512, 768, 1024, 1536, 1024, 768)
    num_heads: tuple = (4, 4, 4, 8, 4, 4)
    query_head_dim: tuple = (32,)
    value_head_dim: tuple = (12,)
    pos_head_dim: tuple = (4,)
    cnn_module_kernel: tuple = (31, 31, 15, 15, 15, 31)
    pos_dim: int = 48
    decoder_dim: int = 512
    joiner_dim: int = 512
    context_size: int = 2


class WavLMZipformerRNNT(nn.Module):
    def __init__(self, frontend: nn.Module, asr_model: nn.Module):
        super().__init__()
        self.frontend = frontend
        self.asr_model = asr_model

    @property
    def decoder(self): return self.asr_model.decoder
    @property
    def joiner(self): return self.asr_model.joiner

    def forward_encoder(self, waveforms, sample_lens):
        x, x_lens = self.frontend(waveforms, sample_lens)
        return self.asr_model.forward_encoder(x, x_lens)

    def forward_transducer(self, encoder_out, encoder_out_lens, y, y_lens, **kwargs):
        return self.asr_model.forward_transducer(encoder_out, encoder_out_lens, y, y_lens, **kwargs)


def build_model(cfg: ModelConfig, vocab_size: int, blank_id: int):
    # Imports resolve from --icefall-recipe after bootstrap.add_icefall_paths().
    from zipformer import Zipformer2
    from decoder import Decoder
    from joiner import Joiner
    from model import AsrModel

    first_dim = cfg.encoder_dims[0]
    frontend = WavLMFrontend(
        cfg.wavlm_model,
        output_dim=first_dim,
        weighted_layers=cfg.weighted_layers,
        layer=cfg.wavlm_layer,
        adapter_stride=cfg.adapter_stride,
    )
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=cfg.downsampling_factors,
        num_encoder_layers=cfg.num_encoder_layers,
        encoder_dim=cfg.encoder_dims,
        encoder_unmasked_dim=cfg.encoder_unmasked_dims,
        query_head_dim=cfg.query_head_dim,
        value_head_dim=cfg.value_head_dim,
        pos_head_dim=cfg.pos_head_dim,
        num_heads=cfg.num_heads,
        feedforward_dim=cfg.feedforward_dims,
        cnn_module_kernel=cfg.cnn_module_kernel,
        pos_dim=cfg.pos_dim,
        causal=False,
        chunk_size=(-1,),
        left_context_frames=(-1,),
    )
    decoder = Decoder(vocab_size=vocab_size, decoder_dim=cfg.decoder_dim, blank_id=blank_id, context_size=cfg.context_size)
    joiner = Joiner(
        encoder_dim=cfg.encoder_dims[-1],
        decoder_dim=cfg.decoder_dim,
        joiner_dim=cfg.joiner_dim,
        vocab_size=vocab_size,
    )
    asr = AsrModel(
        encoder_embed=IdentityEncoderEmbed(),
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=cfg.encoder_dims[-1],
        decoder_dim=cfg.decoder_dim,
        vocab_size=vocab_size,
        use_transducer=True,
        use_ctc=False,
        use_attention_decoder=False,
    )
    return WavLMZipformerRNNT(frontend, asr)
