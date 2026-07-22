"""Precompute MobileCLIP text features for the C++ app.

    python3 src/tools/precompute_text_features.py [config.yaml] [--out features.npy]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

SRC_ROOT = Path(__file__).resolve().parents[1]
SRC_PYTHON = SRC_ROOT / "python"
sys.path.insert(0, str(SRC_PYTHON))
import pyneat as neat
import config
from clip import TextEncoder

DEFAULT_CONFIG = SRC_ROOT / "common" / "config.yaml"

def _build_run_options(queue_depth):
    opt = neat.RunOptions()
    opt.queue_depth = queue_depth
    opt.overflow_policy = neat.OverflowPolicy.Block
    opt.preset = neat.RunPreset.Balanced
    opt.input_timeout_ms = 30000
    opt.startup_preflight = True
    return opt

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default=None,
                    help="output .npy path (default: clip.text_features_path from the config)")
    args = ap.parse_args()

    cfg = config.load_config(args.config)  # requires a non-empty prompt.text (text-prompt only app)

    raw = yaml.safe_load(Path(args.config).read_text())
    out = args.out or raw.get("clip", {}).get("text_features_path")
    if not out:
        out = str(Path(args.config).resolve().parent / "text_features.npy")
        print(f"[warn] clip.text_features_path not set in config; defaulting to {out}")
    if not str(out).endswith(".npy"):
        out = str(out) + ".npy"

    run_opt = _build_run_options(cfg.queue_depth)
    print(f"[precompute] encoding prompt {cfg.text!r} via {cfg.clip_text_path}", flush=True)
    encoder = TextEncoder(cfg.clip_text_path, cfg.clip_consts_path, run_opt)
    try:
        feats = np.ascontiguousarray(encoder.encode(cfg.text, cfg.timeout_ms), dtype=np.float32)
    finally:
        encoder.close()

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.save(out, feats)                                   # writes exactly `out` (already .npy)
    Path(out + ".prompt.txt").write_text(cfg.text)
    print(f"[precompute] wrote {feats.shape} float32 -> {out}")
    print(f"[precompute] set clip.text_features_path: {out} in the config for the C++ app.")


if __name__ == "__main__":
    main()
