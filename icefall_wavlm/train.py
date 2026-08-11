from __future__ import annotations
import argparse
import json
from pathlib import Path

import k2
import sentencepiece as spm
import torch
from torch.optim import AdamW

from icefall_wavlm.bootstrap import add_icefall_paths
from icefall_wavlm.data import load_cuts, make_loader, unpack_audio_batch
from icefall_wavlm.model_factory import ModelConfig, build_model


def save(path, model, optimizer, epoch, args):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "args": vars(args)}, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--icefall-root", required=True)
    p.add_argument("--icefall-recipe", required=True)
    p.add_argument("--bpe-model", required=True)
    p.add_argument("--wavlm-model", default="microsoft/wavlm-base-plus")
    p.add_argument("--train-cuts", required=True)
    p.add_argument("--valid-cuts", required=True)
    p.add_argument("--exp-dir", type=Path, default=Path("exp/wavlm_zipformer"))
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--max-duration", type=float, default=120.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--wavlm-lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--freeze-wavlm-epochs", type=int, default=2)
    p.add_argument("--adapter-stride", type=int, default=1)
    p.add_argument("--wavlm-layer", type=int, default=-1)
    p.add_argument("--single-layer", action="store_true")
    p.add_argument("--prune-range", type=int, default=5)
    p.add_argument("--simple-loss-scale", type=float, default=0.5)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume", type=Path)
    args = p.parse_args()

    add_icefall_paths(args.icefall_root, args.icefall_recipe)
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=args.bpe_model)
    blank_id = sp.piece_to_id("<blk>")
    if blank_id < 0:
        blank_id = 0
    cfg = ModelConfig(
        wavlm_model=args.wavlm_model,
        weighted_layers=not args.single_layer,
        wavlm_layer=args.wavlm_layer,
        adapter_stride=args.adapter_stride,
    )
    model = build_model(cfg, vocab_size=sp.get_piece_size(), blank_id=blank_id)
    model.frontend.freeze_feature_encoder()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    groups = [
        {"params": [p for n,p in model.named_parameters() if not n.startswith("frontend.wavlm.") and p.requires_grad], "lr": args.lr},
        {"params": [p for n,p in model.named_parameters() if n.startswith("frontend.wavlm.") and p.requires_grad], "lr": args.wavlm_lr},
    ]
    optimizer = AdamW(groups, weight_decay=args.weight_decay)
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    train_loader = make_loader(load_cuts(args.train_cuts), args.max_duration, True, args.num_workers)
    valid_loader = make_loader(load_cuts(args.valid_cuts), args.max_duration, False, args.num_workers)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    for epoch in range(start_epoch, args.num_epochs + 1):
        trainable = epoch > args.freeze_wavlm_epochs
        model.frontend.set_wavlm_trainable(trainable, keep_feature_encoder_frozen=True)
        model.train()
        running = 0.0
        count = 0
        for batch in train_loader:
            wav, wav_lens = unpack_audio_batch(batch, device)
            texts = batch["supervisions"]["text"]
            y_list = sp.encode(texts, out_type=int)
            y = k2.RaggedTensor(y_list).to(device)
            y_lens = torch.tensor([len(v) for v in y_list], device=device, dtype=torch.int64)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                enc, enc_lens = model.forward_encoder(wav, wav_lens)
                simple_loss, pruned_loss = model.forward_transducer(
                    enc, enc_lens, y, y_lens,
                    prune_range=args.prune_range,
                    am_scale=0.0,
                    lm_scale=0.25,
                )
                loss = args.simple_loss_scale * simple_loss + pruned_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().cpu())
            count += 1
            if count % 50 == 0:
                print(f"epoch={epoch} batch={count} loss={running/count:.4f}")

        model.eval()
        val, n = 0.0, 0
        with torch.inference_mode():
            for batch in valid_loader:
                wav, wav_lens = unpack_audio_batch(batch, device)
                texts = batch["supervisions"]["text"]
                y_list = sp.encode(texts, out_type=int)
                y = k2.RaggedTensor(y_list).to(device)
                y_lens = torch.tensor([len(v) for v in y_list], device=device, dtype=torch.int64)
                enc, enc_lens = model.forward_encoder(wav, wav_lens)
                simple, pruned = model.forward_transducer(enc, enc_lens, y, y_lens, prune_range=args.prune_range, am_scale=0.0, lm_scale=0.25)
                val += float((args.simple_loss_scale * simple + pruned).cpu()); n += 1
        print(f"epoch={epoch} train_loss={running/max(count,1):.4f} valid_loss={val/max(n,1):.4f}")
        save(args.exp_dir / f"epoch-{epoch}.pt", model, optimizer, epoch, args)


if __name__ == "__main__":
    main()
