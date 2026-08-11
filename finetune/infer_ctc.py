from __future__ import annotations
import argparse
import torch
from transformers import AutoProcessor, WavLMForCTC
from utils.audio import load_audio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--wav", required=True)
    args = p.parse_args()
    processor = AutoProcessor.from_pretrained(args.model)
    model = WavLMForCTC.from_pretrained(args.model).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    wav, _ = load_audio(args.wav)
    inp = processor(wav.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.inference_mode():
        logits = model(input_values=inp.input_values.to(device), attention_mask=getattr(inp, "attention_mask", None).to(device) if getattr(inp, "attention_mask", None) is not None else None).logits
    ids = logits.argmax(-1)
    print(processor.batch_decode(ids)[0])


if __name__ == "__main__":
    main()
