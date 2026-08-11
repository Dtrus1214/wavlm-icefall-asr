from __future__ import annotations

import argparse
from pathlib import Path
import soundfile as sf

AUDIO_EXTS = {".wav", ".flac", ".ogg"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--valid-percent", type=float, default=1.0)
    args = p.parse_args()

    root = args.audio_dir.resolve()
    files = sorted(x for x in root.rglob("*") if x.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise SystemExit(f"No audio found below {root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_valid = max(1, round(len(files) * args.valid_percent / 100.0))
    splits = {"valid": files[:n_valid], "train": files[n_valid:] or files}
    for split, items in splits.items():
        with (args.out_dir / f"{split}.tsv").open("w", encoding="utf-8") as f:
            f.write(str(root) + "\n")
            for path in items:
                info = sf.info(str(path))
                rel = path.relative_to(root)
                f.write(f"{rel}\t{info.frames}\n")
        print(f"{split}: {len(items)} files -> {args.out_dir / (split + '.tsv')}")


if __name__ == "__main__":
    main()
