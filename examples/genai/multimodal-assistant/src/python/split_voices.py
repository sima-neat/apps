#!/usr/bin/env python3
"""
Split Piper voice models into streaming encoder/decoder pairs.

The HiFi-GAN decoder is about three quarters of the runtime of a Piper voice.
Splitting the exported graph at the decoder input lets the app decode the latent
in slices and start playing audio while the rest of the sentence is still being
generated, instead of waiting for the whole sentence to finish.

Each `<voice>.onnx` produces `<voice>.enc.onnx` and `<voice>.dec.onnx` beside it.
PiperTTS streams from the pair where it exists and synthesizes a whole sentence
at a time where it does not, so a voice that cannot be split still speaks.

`voice_install.sh` runs this after downloading, so `setup.sh` covers it unless it
ran with `SPLIT_VOICES=0`. Run it by hand after dropping a voice into the assets
directory yourself, or to add streaming to an install without it:

    python split_voices.py ui/assets

Voices that already have both halves are left alone, so re-running is cheap.
"""

import argparse
import sys
from pathlib import Path

import onnx


def find_decoder_input(graph):
    """
    Return the name of the tensor that feeds the decoder.

    This is the single non-initializer input of the decoder's first convolution.
    Looking the node up by name keeps this working across voice qualities, which
    have different numbers of upsampling layers.
    """
    initializers = {entry.name for entry in graph.initializer}
    for node in graph.node:
        if node.name.startswith("/dec/conv_pre"):
            inputs = [name for name in node.input if name not in initializers]
            if len(inputs) != 1:
                raise ValueError(f"Ambiguous decoder input in {node.name}: {inputs}")
            return inputs[0]
    raise ValueError("No /dec/conv_pre node found; this is not a Piper voice model")


def split_voice(model_path):
    """Write the encoder and decoder halves of a voice model next to it."""
    encoder_path = model_path.with_suffix(".enc.onnx")
    decoder_path = model_path.with_suffix(".dec.onnx")

    if encoder_path.exists() and decoder_path.exists():
        print(f"✅ Already split: {model_path.name}")
        return

    model = onnx.load(str(model_path), load_external_data=False)

    # Multi-speaker voices condition the decoder on a speaker embedding, so the
    # decoder half needs `sid` as well as the latent and PiperTTS has no value to
    # feed it. Say that outright instead of failing on a dangling /emb_g node.
    if any(entry.name == "sid" for entry in model.graph.input):
        raise ValueError("multi-speaker voice (has a 'sid' input); "
                         "streaming synthesis supports single-speaker voices only")

    boundary = find_decoder_input(model.graph)

    # Take the graph inputs as they are: multi-speaker voices have an extra `sid`
    # input that a hardcoded list would silently drop.
    graph_inputs = [entry.name for entry in model.graph.input]
    graph_outputs = [entry.name for entry in model.graph.output]

    try:
        onnx.utils.extract_model(str(model_path), str(encoder_path), graph_inputs, [boundary])
        onnx.utils.extract_model(str(model_path), str(decoder_path), [boundary], graph_outputs)
    except Exception:
        # extract_model saves before it validates, so a failure can leave a file
        # that looks complete. Clear both halves or the next run skips this voice.
        encoder_path.unlink(missing_ok=True)
        decoder_path.unlink(missing_ok=True)
        raise

    print(f"🔪 Split {model_path.name} at {boundary}")
    print(f"  {encoder_path.name} ({encoder_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {decoder_path.name} ({decoder_path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("assets_dir", type=Path, help="Directory holding the voice .onnx files")
    args = parser.parse_args()

    if not args.assets_dir.is_dir():
        print(f"not a directory: {args.assets_dir}", file=sys.stderr)
        return 2

    voices = [path for path in sorted(args.assets_dir.glob("*.onnx"))
              if not path.name.endswith((".enc.onnx", ".dec.onnx"))]
    if not voices:
        print(f"no voice models found in {args.assets_dir}", file=sys.stderr)
        return 2

    failed = []
    for model_path in voices:
        try:
            split_voice(model_path)
        except Exception as e:
            print(f"❌ Split failed for {model_path.name}: {e}", file=sys.stderr)
            failed.append(model_path.name)

    if failed:
        sys.stdout.flush()
        print(f"\n⚠️  {len(failed)} of {len(voices)} voices could not be split and will be "
              f"spoken a whole sentence at a time: {', '.join(failed)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
