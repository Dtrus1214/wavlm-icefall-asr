from __future__ import annotations

import json
from pathlib import Path
from torch.utils.data import Dataset
from utils.audio import load_audio


class ASRJsonlDataset(Dataset):
    def __init__(self, manifest: str | Path):
        self.rows = []
        with Path(manifest).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if "audio" not in row or "text" not in row:
                        raise ValueError("Each JSONL row needs 'audio' and 'text'")
                    self.rows.append(row)
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        wav, _ = load_audio(row["audio"])
        return {"input_values": wav, "text": row["text"]}
