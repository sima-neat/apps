#!/usr/bin/env python3
"""Split single-speaker Piper voices into streaming encoder/decoder pairs."""

import argparse
import sys
from pathlib import Path

import onnx


def _decoder_input(graph):
    initializers = {entry.name for entry in graph.initializer}
    for node in graph.node:
        if node.name.startswith("/dec/conv_pre"):
            inputs = [name for name in node.input if name not in initializers]
            if len(inputs) != 1:
                raise ValueError(f"ambiguous decoder input in {node.name}: {inputs}")
            return inputs[0]
    raise ValueError("no /dec/conv_pre node found; this is not a Piper voice model")


def split_voice(model_path):
    """Write ``.enc.onnx`` and ``.dec.onnx`` beside one Piper voice."""
    model_path = Path(model_path)
    encoder_path = model_path.with_suffix(".enc.onnx")
    decoder_path = model_path.with_suffix(".dec.onnx")
    if (
        encoder_path.exists()
        and decoder_path.exists()
        and min(encoder_path.stat().st_mtime, decoder_path.stat().st_mtime)
        >= model_path.stat().st_mtime
    ):
        print(f"✅ Already split: {model_path.name}")
        return
    encoder_path.unlink(missing_ok=True)
    decoder_path.unlink(missing_ok=True)

    model = onnx.load(str(model_path), load_external_data=False)
    if any(entry.name == "sid" for entry in model.graph.input):
        raise ValueError("multi-speaker voice; streaming supports single-speaker voices only")

    boundary = _decoder_input(model.graph)
    inputs = [entry.name for entry in model.graph.input]
    outputs = [entry.name for entry in model.graph.output]
    try:
        onnx.utils.extract_model(str(model_path), str(encoder_path), inputs, [boundary])
        onnx.utils.extract_model(str(model_path), str(decoder_path), [boundary], outputs)
    except Exception:
        encoder_path.unlink(missing_ok=True)
        decoder_path.unlink(missing_ok=True)
        raise
    print(f"🔪 Split {model_path.name} at {boundary}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Piper .onnx voice or directory of voices")
    args = parser.parse_args(argv)
    if args.path.is_dir():
        voices = [
            path for path in sorted(args.path.glob("*.onnx"))
            if not path.name.endswith((".enc.onnx", ".dec.onnx"))
        ]
    elif args.path.is_file() and args.path.suffix == ".onnx":
        voices = [args.path]
    else:
        print(f"not a Piper voice or directory: {args.path}", file=sys.stderr)
        return 2

    failed = []
    for model_path in voices:
        try:
            split_voice(model_path)
        except Exception as exc:  # noqa: BLE001 - report every failed voice together
            print(f"❌ Split failed for {model_path.name}: {exc}", file=sys.stderr)
            failed.append(model_path.name)
    if failed:
        print(
            f"❌ {len(failed)} of {len(voices)} voices could not be prepared: "
            f"{', '.join(failed)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
