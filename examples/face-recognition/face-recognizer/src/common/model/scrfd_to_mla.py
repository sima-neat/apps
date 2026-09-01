"""
scrfd_to_mla.py  —  SCRFD raw ONNX → SiMa single-segment MLA model

Pipeline
--------
1. Rename the 9 raw head output tensors to stride-named outputs
   (stride_{8,16,32}_{cls,bbox,kps})
2. Freeze input to a static spatial shape and run onnxsim to constant-fold
   all shape-dependent ops (Shape/Gather/Slice/Concat/Unsqueeze)
3. Cut Transpose+Reshape+Sigmoid postprocess tails from all 9 heads, leaving
   raw 4D NCHW tensors as outputs — these map entirely to MLA
4. Remove orphaned initializers left by the removed nodes
5. (optional) Audit with model_surgery_guard.py
6. (optional) Numerical validation against the original model

Why the cut is needed
---------------------
The SCRFD ONNX head outputs apply Transpose(perm=[2,3,0,1]) which moves the
batch axis from position 0 to position 2.  SiMa MLA rejects any Transpose or
Reshape that touches the batch axis, forcing those 9 nodes to compile as A65
ARM cortex stages.  Removing them gives a single-segment, zero-A65 MPK.

Output tensor shapes after surgery
-----------------------------------
  stride_8_cls   (1, 2,  80, 80)  raw class logits, apply sigmoid in runtime
  stride_16_cls  (1, 2,  40, 40)
  stride_32_cls  (1, 2,  20, 20)
  stride_8_bbox  (1, 8,  80, 80)  decoded bbox distances (l,t,r,b per anchor)
  stride_16_bbox (1, 8,  40, 40)
  stride_32_bbox (1, 8,  20, 20)
  stride_8_kps   (1, 20, 80, 80)  keypoint offsets (2*5 per anchor)
  stride_16_kps  (1, 20, 40, 40)
  stride_32_kps  (1, 20, 20, 20)

Runtime notes for the postprocessing app
-----------------------------------------
The MLA model outputs raw 4D NCHW tensors.  Apply the following to recover
the same flat 2D format that the original SCRFD model produced:

  # For each cls output (1,2,H,W): apply sigmoid then flatten
  cls = torch.sigmoid(cls_4d)                      # (1, 2, H, W)
  cls = cls.permute(2,3,0,1).reshape(-1, 1)        # → (H*W*2, 1) matching old format

  # For bbox/kps (1,C,H,W): just permute + reshape
  bbox = bbox_4d.permute(2,3,0,1).reshape(-1, 4)  # → (H*W*2, 4)
  kps  = kps_4d.permute(2,3,0,1).reshape(-1, 10) # → (H*W*2, 10)

Compiled artifact
-----------------
  facerecognition_models/build/scrfd_2.5g_bnkps.mla/scrfd_2.5g_bnkps.mla_mpk.tar.gz

Usage examples
--------------
  # Basic: raw SCRFD → MLA model in same directory
  python3 scrfd_to_mla.py scrfd_2.5g_bnkps.onnx

  # Custom input resolution
  python3 scrfd_to_mla.py scrfd_2.5g_bnkps.onnx --height 320 --width 320

  # Save intermediate static model and run numerical validation
  python3 scrfd_to_mla.py scrfd_2.5g_bnkps.onnx --save-static --validate

  # Custom output path
  python3 scrfd_to_mla.py scrfd_2.5g_bnkps.onnx --out /path/to/scrfd_mla.onnx
"""

import argparse
import os
import sys

import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
from onnxsim import simplify


HEAD_NAMES = [
    "stride_8_cls",  "stride_16_cls",  "stride_32_cls",
    "stride_8_bbox", "stride_16_bbox", "stride_32_bbox",
    "stride_8_kps",  "stride_16_kps",  "stride_32_kps",
]

# The ONNX model has 9 outputs ordered by stride then type:
#   0..2 = cls (8,16,32), 3..5 = bbox (8,16,32), 6..8 = kps (8,16,32)
OUTPUT_INDEX_TO_NAME = {i: name for i, name in enumerate(HEAD_NAMES)}



def rename_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    old_outputs = list(g.output)
    if len(old_outputs) != 9:
        raise ValueError(
            f"Expected 9 SCRFD outputs, got {len(old_outputs)}. "
            "Ensure you are passing the raw unmodified SCRFD ONNX."
        )

    rename = {old_outputs[i].name: HEAD_NAMES[i] for i in range(9)}

    for node in g.node:
        for j, inp in enumerate(node.input):
            if inp in rename:
                node.input[j] = rename[inp]
        for j, out in enumerate(node.output):
            if out in rename:
                node.output[j] = rename[out]

    del g.output[:]
    for i, old_vi in enumerate(old_outputs):
        new_vi = onnx.ValueInfoProto()
        new_vi.CopyFrom(old_vi)
        new_vi.name = HEAD_NAMES[i]
        g.output.append(new_vi)

    return model



def freeze_and_simplify(
    model: onnx.ModelProto, height: int, width: int
) -> onnx.ModelProto:
    g = model.graph
    if len(g.input) != 1:
        raise ValueError(f"Expected 1 graph input, got {len(g.input)}")

    input_name = g.input[0].name
    overwrite_shapes = {input_name: [1, 3, height, width]}

    simplified, ok = simplify(model, overwrite_input_shapes=overwrite_shapes)
    if not ok:
        print("  [warn] onnxsim reported simplification may be incomplete")
    return simplified



def cut_postprocess_heads(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    node_by_output = {out: node for node in g.node for out in node.output}
    all_vi = {vi.name: vi for vi in list(g.value_info) + list(g.input) + list(g.output)}

    raw_tensor_for: dict[str, str] = {}  # head_name -> raw 4D tensor name
    remove_ids: set[int] = set()

    for head_name in HEAD_NAMES:
        if head_name not in node_by_output:
            raise RuntimeError(
                f"Output '{head_name}' not found in graph. "
                "Run rename_outputs first."
            )
        node = node_by_output[head_name]
        # Walk back through Sigmoid / Reshape until we hit Transpose
        while node is not None:
            if node.op_type == "Transpose":
                raw_tensor_for[head_name] = node.input[0]
                remove_ids.add(id(node))
                break
            remove_ids.add(id(node))
            node = node_by_output.get(node.input[0]) if node.input else None

        if head_name not in raw_tensor_for:
            raise RuntimeError(
                f"Could not find Transpose node upstream of '{head_name}'. "
                "The model may already be in MLA format or has an unexpected graph structure."
            )

    tensor_rename = {v: k for k, v in raw_tensor_for.items()}

    def r(name: str) -> str:
        return tensor_rename.get(name, name)

    new_nodes = []
    for node in g.node:
        if id(node) in remove_ids:
            continue
        new_node = helper.make_node(
            node.op_type,
            inputs=[r(x) for x in node.input],
            outputs=[r(x) for x in node.output],
            name=node.name,
            domain=node.domain,
        )
        for attr in node.attribute:
            new_node.attribute.append(attr)
        new_nodes.append(new_node)

    new_outputs = []
    for head_name in HEAD_NAMES:
        raw = raw_tensor_for[head_name]
        vi = all_vi.get(raw)
        if vi is not None:
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            dtype = vi.type.tensor_type.elem_type
        else:
            dims = None
            dtype = TensorProto.FLOAT
        new_outputs.append(helper.make_tensor_value_info(head_name, dtype, dims))

    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=g.name,
        inputs=list(g.input),
        outputs=new_outputs,
        initializer=list(g.initializer),
    )
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version
    return new_model



def clean_initializers(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    used: set[str] = set()
    for n in g.node:
        used.update(n.input)
        used.update(n.output)
    used.update(i.name for i in g.input)
    used.update(o.name for o in g.output)

    before = len(g.initializer)
    kept = [init for init in g.initializer if init.name in used]
    del g.initializer[:]
    g.initializer.extend(kept)
    removed = before - len(g.initializer)
    if removed:
        print(f"  Removed {removed} unused initializer(s)")
    return model



def validate(orig_path: str, mla_path: str, height: int, width: int) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [skip] onnxruntime not available — skipping numerical validation")
        return True

    rng = np.random.default_rng(0)
    inp = rng.random((1, 3, height, width), dtype=np.float32)

    sess_orig = ort.InferenceSession(orig_path)
    sess_mla  = ort.InferenceSession(mla_path)

    # Index by position, not name: the surgery renames outputs to HEAD_NAMES
    # but the original model still has its native tensor names.
    out_orig = {
        HEAD_NAMES[i]: v
        for i, v in enumerate(sess_orig.run(None, {"input.1": inp}))
    }
    out_mla = {
        o.name: v
        for o, v in zip(sess_mla.get_outputs(), sess_mla.run(None, {"input.1": inp}))
    }

    # The MLA model outputs raw 4D NCHW; reconstruct the original 2D format:
    #   Transpose perm=[2,3,0,1]: (N,C,H,W) → (H,W,N,C)
    #   Reshape → original 2D shape
    #   Sigmoid (cls only)
    def reconstruct(mla_4d, target_shape, is_cls):
        t = mla_4d.transpose(2, 3, 0, 1)
        r = t.reshape(target_shape)
        if is_cls:
            r = 1.0 / (1.0 + np.exp(-r))
        return r

    all_ok = True
    print(f"  {'Head':<22} {'orig':<16} {'mla':<20} max_diff   ok?")
    print(f"  {'-'*72}")
    for name in HEAD_NAMES:
        orig = out_orig[name]
        mla4 = out_mla[name]
        rec  = reconstruct(mla4, orig.shape, "cls" in name)
        diff = float(np.abs(orig - rec).max())
        ok   = diff < 1e-5
        all_ok = all_ok and ok
        print(f"  {name:<22} {str(orig.shape):<16} {str(mla4.shape):<20} {diff:.2e}   {'OK' if ok else 'FAIL'}")

    return all_ok



def parse_args():
    p = argparse.ArgumentParser(
        description="Convert raw SCRFD ONNX to SiMa MLA-optimized model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="Path to scrfd_2.5g_bnkps.onnx (dynamic input)")
    p.add_argument(
        "--out",
        default=None,
        help="Output path for the MLA model (default: <stem>.mla.onnx in same dir)",
    )
    p.add_argument("--height", type=int, default=640, help="Input image height (default 640)")
    p.add_argument("--width",  type=int, default=640, help="Input image width  (default 640)")
    p.add_argument(
        "--save-static",
        action="store_true",
        help="Also save the intermediate static model as <stem>.static.onnx",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Run numerical comparison between original and MLA outputs (requires onnxruntime)",
    )
    p.add_argument(
        "--audit",
        action="store_true",
        help="Run model_surgery_guard.py audit after saving (requires model-compiler env)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    src = os.path.abspath(args.input)
    if not os.path.exists(src):
        sys.exit(f"Error: input not found: {src}")

    stem = os.path.splitext(src)[0]
    # Strip any existing .static suffix for a clean stem
    if stem.endswith(".static"):
        stem = stem[: -len(".static")]

    static_path = stem + ".static.onnx"
    mla_path    = args.out if args.out else stem + ".mla.onnx"

    print(f"Input:  {src}")
    print(f"Output: {mla_path}")
    print(f"Shape:  [1, 3, {args.height}, {args.width}]")
    print()

    print("Step 1/4  Renaming raw output tensors to stride names …")
    model = onnx.load(src)
    model = rename_outputs(model)
    print(f"  9 outputs renamed: {HEAD_NAMES[:3]} …")

    print(f"Step 2/4  Freezing input to [1,3,{args.height},{args.width}] and running onnxsim …")
    before_n = len(model.graph.node)
    model = freeze_and_simplify(model, args.height, args.width)
    after_n = len(model.graph.node)
    print(f"  Nodes: {before_n} → {after_n} (onnxsim folded {before_n - after_n})")

    if args.save_static:
        onnx.save(model, static_path)
        print(f"  Saved static model: {static_path}")

    print("Step 3/4  Cutting Transpose + Reshape + Sigmoid postprocess heads …")
    before_n = len(model.graph.node)
    model = cut_postprocess_heads(model)
    after_n = len(model.graph.node)
    print(f"  Nodes: {before_n} → {after_n} (removed {before_n - after_n})")
    for o in model.graph.output:
        dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
        print(f"  {o.name:<22} {dims}")

    print("Step 4/4  Cleaning unused initializers …")
    model = clean_initializers(model)

    onnx.checker.check_model(model)
    onnx.save(model, mla_path)
    print(f"\nSaved: {mla_path}")

    if args.audit:
        print("\nAudit …")
        guard = os.environ.get(
            "MODEL_SURGERY_GUARD",
            os.path.normpath(os.path.join(os.path.dirname(__file__),
                                          "model_surgery_guard.py")),
        )
        if os.path.exists(guard):
            os.system(f"{sys.executable} {guard} audit-model --model {mla_path} --dtype int8")
        else:
            print(f"  [skip] guard script not found at {guard}")

    if args.validate:
        print("\nNumerical validation …")
        ok = validate(src, mla_path, args.height, args.width)
        if not ok:
            sys.exit("Validation FAILED — outputs do not match within 1e-5")
        print("  PASS — all outputs match")

    print("\nDone.  Next step:")
    print(f"  /sima-model-quantize-compile {mla_path}")


if __name__ == "__main__":
    main()
