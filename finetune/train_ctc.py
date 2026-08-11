from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    WavLMForCTC,
)
from finetune.data import ASRJsonlDataset


@dataclass
class CTCDataCollator:
    processor: object

    def __call__(self, features: List[Dict]):
        audio = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.pad(audio, padding=True, return_tensors="pt")
        text = [f["text"] for f in features]
        with self.processor.as_target_processor():
            labels = self.processor(text=text, padding=True, return_tensors="pt").input_ids
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
        batch["labels"] = labels
        return batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--valid-manifest", required=True)
    p.add_argument("--model", default="microsoft/wavlm-base-plus")
    p.add_argument("--output-dir", type=Path, default=Path("exp/wavlm_ctc"))
    p.add_argument("--epochs", type=float, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--freeze-feature-encoder", action="store_true", default=True)
    args = p.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model = WavLMForCTC.from_pretrained(
        args.model,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.08,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ASRJsonlDataset(args.train_manifest),
        eval_dataset=ASRJsonlDataset(args.valid_manifest),
        data_collator=CTCDataCollator(processor),
        processing_class=processor,
    )
    trainer.train()
    best = args.output_dir / "best"
    trainer.save_model(str(best))
    processor.save_pretrained(str(best))
    print(f"Saved: {best}")


if __name__ == "__main__":
    main()
