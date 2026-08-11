from __future__ import annotations
import sys
from pathlib import Path


def add_icefall_paths(icefall_root: str | Path, recipe: str | Path) -> None:
    root = Path(icefall_root).resolve()
    recipe = Path(recipe).resolve()
    if not (root / "icefall").exists():
        raise FileNotFoundError(f"Not an Icefall checkout: {root}")
    if not (recipe / "zipformer.py").exists():
        raise FileNotFoundError(f"Not a Zipformer recipe directory: {recipe}")
    for p in (str(recipe), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
