from __future__ import annotations
import argparse
import sentencepiece as spm
import torch

from icefall_wavlm.bootstrap import add_icefall_paths
from icefall_wavlm.model_factory import ModelConfig, build_model
from utils.audio import load_audio


def greedy_search(model, encoder_out, encoder_out_lens, context_size, blank_id):
    # Batch size 1, stateless transducer greedy decoding.
    assert encoder_out.size(0) == 1
    hyp = [blank_id] * context_size
    T = int(encoder_out_lens[0])
    for t in range(T):
        decoder_input = torch.tensor([hyp[-context_size:]], device=encoder_out.device, dtype=torch.int64)
        dec = model.decoder(decoder_input, need_pad=False)
        enc = encoder_out[:, t:t+1, :]
        logits = model.joiner(enc.unsqueeze(2), dec.unsqueeze(1), project_input=True)
        token = int(logits[0,0,0].argmax())
        if token != blank_id:
            hyp.append(token)
    return hyp[context_size:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--icefall-root", required=True)
    p.add_argument("--icefall-recipe", required=True)
    p.add_argument("--bpe-model", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--wav", required=True)
    p.add_argument("--wavlm-model", default="microsoft/wavlm-base-plus")
    p.add_argument("--adapter-stride", type=int, default=1)
    p.add_argument("--wavlm-layer", type=int, default=-1)
    p.add_argument("--single-layer", action="store_true")
    p.add_argument("--decoding-method", choices=["greedy_search"], default="greedy_search")
    args = p.parse_args()

    add_icefall_paths(args.icefall_root, args.icefall_recipe)
    sp = spm.SentencePieceProcessor(model_file=args.bpe_model)
    blank_id = sp.piece_to_id("<blk>")
    if blank_id < 0: blank_id = 0
    cfg = ModelConfig(args.wavlm_model, not args.single_layer, args.wavlm_layer, args.adapter_stride)
    model = build_model(cfg, sp.get_piece_size(), blank_id)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    wav, _ = load_audio(args.wav)
    wav = wav.unsqueeze(0).to(device)
    lens = torch.tensor([wav.size(1)], device=device)
    with torch.inference_mode():
        enc, enc_lens = model.forward_encoder(wav, lens)
        ids = greedy_search(model, enc, enc_lens, cfg.context_size, blank_id)
    print(sp.decode(ids))


if __name__ == "__main__":
    main()
