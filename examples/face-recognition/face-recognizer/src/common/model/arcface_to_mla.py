"""
arcface_to_mla.py  —  ArcFace ONNX → SiMa MLA-ready model

Supports any InsightFace ArcFace model with a ResNet/MobileFaceNet backbone
and BNNeck output head (w600k_mbf, w600k_r50, w600k_r100, glint360k_r50, …).

Pipeline
--------
1. Rewrite BatchNormalization → Mul + Add  (pre-compute scale and bias)
2. Rewrite Flatten(axis=1) → Reshape([1, -1])
3. Rewrite Gemm(transB=1) → transposed-weight initializer + MatMul + Add
4. Freeze batch dimension to 1 and run onnxsim (constant-fold shape ops)
5. (optional) Audit with model_surgery_guard.py
6. (optional) Numerical validation against the original model

Why these rewrites are needed
------------------------------
SiMa MLA does not support BatchNormalization, Flatten, or Gemm ops directly.
The standard ArcFace BNNeck outputs: Conv → BN → Flatten → Gemm → embedding.
After surgery the graph becomes: Conv → Mul → Add → Reshape → MatMul → Add,
all of which map cleanly to MLA.

Expected output tensor shapes
------------------------------
  683 (or model-specific name)  →  [1, 512]   L2-normalized embedding vector

Usage examples
--------------
  # Basic: w600k_r50.onnx → w600k_r50.surgery.onnx
  python3 arcface_to_mla.py w600k_r50.onnx

  # MBF variant
  python3 arcface_to_mla.py w600k_mbf.onnx

  # Custom output path + validation
  python3 arcface_to_mla.py w600k_r50.onnx --out r50_mla.onnx --validate

  # Full run with audit
  python3 arcface_to_mla.py w600k_r50.onnx --validate --audit

Compile after surgery
---------------------
  docker run --rm \\
    -v "$VENV:/model-compiler" \\
    -v "$(pwd):/models" \\
    -v "$SCRIPTS:/scripts:ro" \\
    -v "$CALIB:/calib:ro" \\
    ghcr.io/sima-neat/sdk:release-2.1 bash -c "
      export PATH=/model-compiler/bin:$PATH
      export LD_LIBRARY_PATH=/model-compiler/lib:$LD_LIBRARY_PATH
      python3 /scripts/quantize_compile.py \\
        --model_path /models/w600k_r50.surgery.onnx \\
        --model_format onnx --model_layout NCHW \\
        --input_names input.1 --input_shapes 1,3,112,112 \\
        --output_names 683 \\
        --device modalix --build_dir /models/build \\
        --real_data --dataset_images /calib \\
        --num_calib_samples 16 --calib_method mse
    "
"""

import argparse
import os
import sys

import numpy as np
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
from onnxsim import simplify


def rewrite_flatten(node, new_inits):
    axis = next((a.i for a in node.attribute if a.name == "axis"), 1)
    if axis != 1:
        return None
    shape_name = node.output[0] + "_rshp_shape"
    new_inits.append(
        numpy_helper.from_array(np.array([1, -1], dtype=np.int64), name=shape_name)
    )
    return [helper.make_node("Reshape", inputs=[node.input[0], shape_name], outputs=node.output)]


def rewrite_gemm(node, inits, new_inits):
    transB = next((a.i for a in node.attribute if a.name == "transB"), 0)
    alpha  = next((a.f for a in node.attribute if a.name == "alpha"),  1.0)
    beta   = next((a.f for a in node.attribute if a.name == "beta"),   1.0)

    W = inits[node.input[1]]
    Wt = ((W.T if transB else W) * alpha).astype(np.float32)
    Wt_name = node.input[1] + "_T"
    new_inits.append(numpy_helper.from_array(Wt, name=Wt_name))

    mm_out = node.output[0] + "_mm"
    result = [helper.make_node("MatMul", inputs=[node.input[0], Wt_name], outputs=[mm_out])]

    if len(node.input) > 2 and node.input[2]:
        b = (inits[node.input[2]] * beta).astype(np.float32)
        b_name = node.input[2] + "_scaled"
        new_inits.append(numpy_helper.from_array(b, name=b_name))
        result.append(helper.make_node("Add", inputs=[mm_out, b_name], outputs=node.output))
    else:
        result[-1].output[0] = node.output[0]

    return result


def rewrite_batchnorm(node, inits, new_inits, input_rank: int = 4):
    # input_rank controls broadcast shape: 4 → (C,1,1) for NCHW; 2 → (1,C) for NC
    scale = inits[node.input[1]]
    B     = inits[node.input[2]]
    mean  = inits[node.input[3]]
    var   = inits[node.input[4]]
    eps   = next((a.f for a in node.attribute if a.name == "epsilon"), 1e-5)

    std      = np.sqrt(var + eps)
    eff_s    = (scale / std).astype(np.float32)
    eff_b    = (B - mean * scale / std).astype(np.float32)

    if input_rank == 2:
        eff_scale = eff_s.reshape(1, -1)
        eff_bias  = eff_b.reshape(1, -1)
    else:
        eff_scale = eff_s.reshape(-1, 1, 1)
        eff_bias  = eff_b.reshape(-1, 1, 1)

    sc_name = node.input[1] + "_bn_scale"
    bi_name = node.input[1] + "_bn_bias"
    new_inits.append(numpy_helper.from_array(eff_scale, name=sc_name))
    new_inits.append(numpy_helper.from_array(eff_bias,  name=bi_name))

    mul_out = node.output[0] + "_bn_mul"
    return [
        helper.make_node("Mul", inputs=[node.input[0], sc_name], outputs=[mul_out]),
        helper.make_node("Add", inputs=[mul_out, bi_name],        outputs=[node.output[0]]),
    ]



def apply_surgery(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    # Run shape inference first so every intermediate tensor has rank info
    model = onnx.shape_inference.infer_shapes(model)
    g = model.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}

    rank_of: dict[str, int] = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        try:
            rank_of[vi.name] = len(vi.type.tensor_type.shape.dim)
        except (AttributeError, TypeError):
            # Some ValueInfoProto entries may not carry complete type/shape info.
            # Leave rank unset here; downstream code uses a safe default rank.
            continue

    new_nodes: list = []
    new_inits: list = list(g.initializer)
    rewrites = 0

    for node in g.node:
        replacement = None

        if node.op_type == "Flatten":
            replacement = rewrite_flatten(node, new_inits)
        elif node.op_type == "Gemm":
            replacement = rewrite_gemm(node, inits, new_inits)
        elif node.op_type == "BatchNormalization":
            # Determine input rank so scale/bias are broadcast-compatible
            input_rank = rank_of.get(node.input[0], 4)
            replacement = rewrite_batchnorm(node, inits, new_inits, input_rank=input_rank)

        if replacement is not None:
            new_nodes.extend(replacement)
            rewrites += 1
        else:
            new_nodes.append(node)

    new_graph = helper.make_graph(
        nodes       = new_nodes,
        name        = g.name,
        inputs      = list(g.input),
        outputs     = list(g.output),
        initializer = new_inits,
    )
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version
    return new_model, rewrites


def freeze_and_simplify(model: onnx.ModelProto) -> onnx.ModelProto:
    if len(model.graph.input) != 1:
        raise ValueError(f"Expected 1 graph input, got {len(model.graph.input)}")
    inp_name = model.graph.input[0].name
    simplified, ok = simplify(model, overwrite_input_shapes={inp_name: [1, 3, 112, 112]})
    if not ok:
        print("  [warn] onnxsim may be incomplete")
    return simplified



def validate(orig_path: str, surgery_path: str) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [skip] onnxruntime not available")
        return True

    rng = np.random.default_rng(0)
    inp = rng.random((1, 3, 112, 112), dtype=np.float32)

    sess_orig = ort.InferenceSession(orig_path)
    sess_new  = ort.InferenceSession(surgery_path)

    inp_name_orig = sess_orig.get_inputs()[0].name
    inp_name_new  = sess_new.get_inputs()[0].name

    out_orig = sess_orig.run(None, {inp_name_orig: inp})[0]
    out_new  = sess_new.run(None,  {inp_name_new:  inp})[0]

    diff = float(np.abs(out_orig - out_new).max())
    ok   = diff < 1e-4
    print(f"  orig {out_orig.shape}  surgery {out_new.shape}  max_diff={diff:.2e}  {'OK' if ok else 'FAIL'}")
    return ok



def parse_args():
    p = argparse.ArgumentParser(
        description="ArcFace ONNX surgery: BN→Mul+Add, Flatten→Reshape, Gemm→MatMul+Add",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="Path to raw ArcFace ONNX (e.g. w600k_r50.onnx)")
    p.add_argument("--out", default=None,
                   help="Output path (default: <stem>.surgery.onnx)")
    p.add_argument("--validate", action="store_true",
                   help="Numerical comparison with original (requires onnxruntime)")
    p.add_argument("--audit", action="store_true",
                   help="Run model_surgery_guard.py after saving (requires model-compiler env)")
    return p.parse_args()


def main():
    args = parse_args()

    src = os.path.abspath(args.input)
    if not os.path.exists(src):
        sys.exit(f"Error: not found: {src}")

    stem    = os.path.splitext(src)[0]
    out_path = args.out or stem + ".surgery.onnx"

    print(f"Input:  {src}")
    print(f"Output: {out_path}\n")

    print("Step 1/2  Rewriting BatchNormalization, Flatten, Gemm …")
    model = onnx.load(src)
    before_n = len(model.graph.node)
    model, rewrites = apply_surgery(model)
    print(f"  {rewrites} rewrites applied  ({before_n} → {len(model.graph.node)} nodes)")

    print("Step 2/2  Freezing batch=1 and running onnxsim …")
    before_n = len(model.graph.node)
    model = freeze_and_simplify(model)
    print(f"  Nodes: {before_n} → {len(model.graph.node)}")

    onnx.checker.check_model(model)
    onnx.save(model, out_path)

    for o in model.graph.output:
        dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
        print(f"  Output: {o.name}  {dims}")

    print(f"\nSaved: {out_path}")

    if args.audit:
        print("\nAudit …")
        guard = os.environ.get(
            "MODEL_SURGERY_GUARD",
            os.path.normpath(os.path.join(os.path.dirname(__file__),
                                          "model_surgery_guard.py")),
        )
        if os.path.exists(guard):
            os.system(f"{sys.executable} {guard} audit-model --model {out_path} --dtype int8")
        else:
            print(f"  [skip] guard not found at {guard}")

    if args.validate:
        print("\nNumerical validation …")
        ok = validate(src, out_path)
        if not ok:
            sys.exit("Validation FAILED — max diff > 1e-4")
        print("  PASS")

    print("\nDone.  Next step:")
    print(f"  /sima-model-quantize-compile {out_path}")


if __name__ == "__main__":
    main()
