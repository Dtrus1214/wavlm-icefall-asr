from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def main() -> None:
    p = argparse.ArgumentParser(description="Launch Microsoft's official WavLM fairseq pretraining code")
    p.add_argument("--official-root", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--label-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--config-name", default="wavlm_base")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    train_py = args.official_root / "fairseq_cli" / "hydra_train.py"
    if not train_py.exists():
        # Common unilm/wavlm layout contains a fairseq subdirectory.
        alt = args.official_root / "fairseq" / "fairseq_cli" / "hydra_train.py"
        train_py = alt if alt.exists() else train_py
    if not train_py.exists():
        raise SystemExit(
            "Could not find fairseq_cli/hydra_train.py below --official-root. "
            "Use Microsoft's unilm/wavlm checkout and its documented fairseq environment."
        )

    overrides = [
        f"task.data={args.data_dir.resolve()}",
        f"task.label_dir={args.label_dir.resolve()}",
        f"checkpoint.save_dir={args.save_dir.resolve()}",
        f"optimization.max_update={int(cfg.get('max_update', 100000))}",
        f"optimization.update_freq=[{int(cfg.get('update_freq', 8))}]",
        f"dataset.max_tokens={int(cfg.get('max_tokens', 1400000))}",
        f"optimizer.lr=[{float(cfg.get('learning_rate', 1e-4))}]",
        f"lr_scheduler.warmup_updates={int(cfg.get('warmup_updates', 8000))}",
        f"checkpoint.save_interval_updates={int(cfg.get('save_interval_updates', 5000))}",
    ]
    cmd = [sys.executable, str(train_py), "--config-name", args.config_name, *overrides]
    print("COMMAND:\n" + " \\\n  ".join(map(str, cmd)))
    if not args.dry_run:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, cwd=args.official_root, check=True)


if __name__ == "__main__":
    main()
