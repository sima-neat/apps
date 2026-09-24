#!/usr/bin/env python3
"""Compile an anomalib FastFlow ONNX export into the model package this example runs.

    activate-model-compiler
    python3 scripts/compile_model.py --model model.onnx --calib-images /data/my_product/train/good

Converts an anomalib 2.x export to bfloat16 and writes ``build/<name>/<name>_mpk.tar.gz``.
Set ``model.path`` to it and ``model.normalize`` to mean 0 and stddev 1.

The export's per-image LayerNorm and 5-D map average are patched first, because the Model
Compiler mishandles both. Use photos of real good parts for ``--calib-images``, even though
bfloat16 needs no quantisation scales: a synthetic sample saturates the map.
"""
import argparse
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.utils import Extractor
from onnxsim import simplify

from afe.apis.defines import CalibrationMethod, bfloat16_scheme, default_quantization, gen2_target
from afe.apis.loaded_net import load_model
from afe.ir.tensor_type import scalar_type_from_dtype
from afe.load.importers.general_importer import onnx_source


def prepare(src: Path, dst: Path) -> tuple[str, list[int]]:
    """Static batch of one, the anomaly_map output only, and the two patches."""
    model = onnx.load(str(src))
    (inp,) = model.graph.input
    shape = [d.dim_value or 1 for d in inp.type.tensor_type.shape.dim]
    model, ok = simplify(model, overwrite_input_shapes={inp.name: shape})
    assert ok, "onnxsim could not simplify the export"
    model = Extractor(model).extract_model([inp.name], ["anomaly_map"])
    graph = model.graph
    producer = {out: n for n in graph.node for out in n.output}
    assert producer["anomaly_map"].op_type == "Clip", "anomaly_map is not the 0..1 map: export with anomalib 2.x"
    assert patch_stacked_mean(graph, producer) == 1, "stacked mean not found: not an anomalib FastFlow export?"
    assert patch_layer_norm(graph) >= 2, "per-image LayerNorm not found: not an anomalib FastFlow export?"
    onnx.save(onnx.shape_inference.infer_shapes(model), str(dst))
    return inp.name, shape


def replace(graph, old: list, new: list) -> None:
    position = min(list(graph.node).index(n) for n in old)
    for node in old:
        graph.node.remove(node)
    for offset, node in enumerate(new):
        graph.node.insert(position + offset, node)


def patch_stacked_mean(graph, producer: dict) -> int:
    """mean(stack([a, b, c])) -> Concat on the channel axis + 1x1 Conv with weights 1/n.

    The map stays 4-D and the average runs on the MLA instead of splitting the package."""
    count = 0
    for mean in [n for n in graph.node if n.op_type == "ReduceMean"]:
        concat = producer.get(mean.input[0])
        if concat is None or concat.op_type != "Concat":
            continue
        unsqueezes = [producer[i] for i in concat.input]
        if any(u.op_type != "Unsqueeze" for u in unsqueezes):
            continue
        n, out = len(unsqueezes), mean.output[0]
        weights = numpy_helper.from_array(np.full((1, n, 1, 1), 1.0 / n, np.float32), out + "_weights")
        graph.initializer.append(weights)
        replace(graph, [mean, concat, *unsqueezes], [
            helper.make_node("Concat", [u.input[0] for u in unsqueezes], [out + "_stack"], axis=1),
            helper.make_node("Conv", [out + "_stack", weights.name], [out], kernel_shape=[1, 1]),
        ])
        count += 1
    return count


def patch_layer_norm(graph) -> int:
    """ReduceMean over (C, H, W) -> GlobalAveragePool + ReduceMean over C.

    Same numbers, but the compiler no longer recognises torch's decomposed LayerNorm
    (ReduceMean, Sub, Pow, ReduceMean, Add, Sqrt, Div), which it lowers to a per-channel
    normalisation whatever the axes say."""
    count = 0
    for mean in [n for n in graph.node if n.op_type == "ReduceMean"]:
        axes = next((helper.get_attribute_value(a) for a in mean.attribute if a.name == "axes"), None)
        if axes is None or sorted(a % 4 for a in axes) != [1, 2, 3]:
            continue
        pooled = mean.output[0] + "_pooled"
        replace(graph, [mean], [
            helper.make_node("GlobalAveragePool", [mean.input[0]], [pooled]),
            helper.make_node("ReduceMean", [pooled], [mean.output[0]], axes=[1], keepdims=1),
        ])
        count += 1
    return count


def samples(root: Path, height: int, width: int) -> list[np.ndarray]:
    """Good images the way the CVU feeds the model: bilinear resize, RGB, 0..1, NHWC."""
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    images = [cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (width, height)), cv2.COLOR_BGR2RGB) for p in paths[:50]]
    return [image[None].astype(np.float32) / 255.0 for image in images]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="FastFlow ONNX from anomalib export")
    parser.add_argument("--calib-images", type=Path, required=True, help="directory of known-good images")
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    args = parser.parse_args()
    if shutil.which("mla-iasm") is None:
        sys.exit("mla-iasm is not on PATH: run activate-model-compiler first")
    name = args.model.stem
    output_dir = args.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)

    input_name, shape = prepare(args.model, output_dir / f"{name}.prepared.onnx")
    calibration = [{input_name: s} for s in samples(args.calib_images, shape[2], shape[3])]
    if not calibration:
        sys.exit(f"no images under {args.calib_images}")

    net = load_model(onnx_source(model_path=str(output_dir / f"{name}.prepared.onnx"), shape_dict={input_name: shape},
                                 dtype_dict={input_name: scalar_type_from_dtype("float32")}), target=gen2_target)
    config = (default_quantization.with_activation_quantization(bfloat16_scheme())
              .with_weight_quantization(bfloat16_scheme()).with_calibration(CalibrationMethod.from_str("mse")))
    model = net.quantize(calibration_data=calibration, quantization_config=config, model_name=name,
                         log_level=logging.WARNING)
    model.compile(output_path=str(output_dir), batch_size=1, log_level=logging.INFO)
    package = output_dir / f"{name}_mpk.tar.gz"
    if not package.exists():
        sys.exit("the Model Compiler wrote no package; see its log above")
    print(f"\nPackage: {package}\nSet model.path to it and model.normalize to mean [0, 0, 0], stddev [1, 1, 1]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
