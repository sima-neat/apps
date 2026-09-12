#!/usr/bin/env python3
"""
prepare_models.py
=================
Download + graph surgery for both face-recognition models in one step.

Source formats
--------------
Both models are distributed as ONNX by their respective repositories.
No PyTorch-to-ONNX conversion is required.  If you have a raw PyTorch
checkpoint (.pt / .pth) instead, export it to ONNX first with
torch.onnx.export() before running this script.

Downloads
---------
  scrfd_2.5g_bnkps.onnx  — HuggingFace (hsuyabc/scrfd_2.5g_bnkps.onnx), ONNX
  w600k_r50.onnx          — InsightFace buffalo_l pack (GitHub releases v0.7), ONNX

Outputs (passed directly to compile_models.sh)
-----------------------------------------------
  scrfd_2.5g_bnkps.mla.onnx  — SCRFD: renamed stride outputs, static 640×640
                                 input, Transpose+Reshape+Sigmoid heads removed
                                 so the entire model fits in one MLA segment
  w600k_r50.surgery.onnx      — ArcFace R50: BN→Mul+Add, Flatten→Reshape,
                                 Gemm→MatMul+Add for MLA compatibility

What the script does (4 steps)
--------------------------------
  1. Download  — fetches scrfd_2.5g_bnkps.onnx from HuggingFace
                  and w600k_r50.onnx from the InsightFace buffalo_l zip;
                  skips files already present on disk.
  2. SCRFD     — calls scrfd_to_mla.py surgery:
                    rename 9 raw head outputs to stride_{8,16,32}_{cls,bbox,kps}
                    freeze input to [1,3,H,W] and onnxsim-fold (157→139 nodes)
                    cut Transpose+Reshape+Sigmoid postprocess tails from all 9 heads
                    remove orphaned initializers
  3. ArcFace   — calls arcface_to_mla.py surgery:
                    BatchNormalization → Mul + Add  (pre-computed scale/bias)
                    Flatten(axis=1)   → Reshape([1,-1])
                    Gemm(transB=1)    → MatMul + Add  (W.T stored as initializer)
  4. Validate  — runs both original and surgery models on random input and
                  asserts max numeric diff within tolerance.

Usage
-----
  # Run inside the Neat Development Environment with model-compiler venv active:
  source /sdk-extensions/model-compiler/bin/activate

  python3 examples/face-recognition/scripts/prepare_models.py \\
    --out-dir /workspace/face-recog-models

  # Different SCRFD input resolution (e.g. 320×320):
  python3 prepare_models.py --out-dir ./models --input-h 320 --input-w 320

Arguments
---------
  --out-dir DIR    Directory for all downloaded and processed files.
                   Default: same directory as this script.
  --input-h H      Height to freeze SCRFD input (default: 640).
  --input-w W      Width  to freeze SCRFD input (default: 640).
  --no-validate    Skip numerical validation (faster; not recommended).

Requirements (model-compiler venv)
-----------------------------------
  onnx, onnxsim, onnxruntime, requests
"""

import argparse
import importlib.util
import io
import os
import pathlib
import sys
import zipfile

import requests



_HERE = pathlib.Path(__file__).parent


def _load_module(name: str):
    path = _HERE / f"{name}.py"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found — it must be in the same directory as this script.")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod



def _download_bytes(url: str, desc: str) -> bytes:
    print(f"  Downloading {desc} …")
    r = requests.get(url, timeout=180, stream=True)
    r.raise_for_status()
    chunks = []
    total = int(r.headers.get("content-length", 0))
    done = 0
    for chunk in r.iter_content(chunk_size=1 << 20):
        chunks.append(chunk)
        done += len(chunk)
        if total:
            pct = done * 100 // total
            print(f"\r    {done >> 20} / {total >> 20} MB  ({pct}%)", end="", flush=True)
    print()
    return b"".join(chunks)


def download_scrfd(out_dir: str) -> str:
    dest = os.path.join(out_dir, "scrfd_2.5g_bnkps.onnx")
    if os.path.exists(dest):
        print(f"  [skip] {dest} already exists")
        return dest

    url = "https://huggingface.co/hsuyabc/scrfd_2.5g_bnkps.onnx/resolve/main/scrfd_2.5g_bnkps.onnx"
    data = _download_bytes(url, "scrfd_2.5g_bnkps.onnx (HuggingFace)")
    with open(dest, "wb") as f:
        f.write(data)
    print(f"  Saved → {dest}")
    return dest


def download_r50(out_dir: str) -> str:
    dest = os.path.join(out_dir, "w600k_r50.onnx")
    if os.path.exists(dest):
        print(f"  [skip] {dest} already exists")
        return dest

    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    data = _download_bytes(url, "buffalo_l.zip (InsightFace v0.7 — ~280 MB)")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        if "w600k_r50.onnx" not in names:
            sys.exit(
                f"ERROR: w600k_r50.onnx not found inside buffalo_l.zip. "
                f"Contents: {names}"
            )
        zf.extract("w600k_r50.onnx", path=out_dir)
    print(f"  Saved → {dest}")
    return dest



def process_scrfd(src: str, out_dir: str, input_h: int, input_w: int) -> str:
    import onnx
    sm = _load_module("scrfd_to_mla")

    dest = os.path.join(out_dir, "scrfd_2.5g_bnkps.mla.onnx")
    print(f"\n[SCRFD] Loading {src}")
    model = onnx.load(src)

    print("  Step 1/4  Renaming raw output tensors to stride names …")
    model = sm.rename_outputs(model)

    print(f"  Step 2/4  Freezing input to [1,3,{input_h},{input_w}] + onnxsim …")
    n_before = len(model.graph.node)
    model = sm.freeze_and_simplify(model, input_h, input_w)
    print(f"    Nodes: {n_before} → {len(model.graph.node)}")

    print("  Step 3/4  Cutting Transpose+Reshape+Sigmoid postprocess heads …")
    n_before = len(model.graph.node)
    model = sm.cut_postprocess_heads(model)
    print(f"    Nodes: {n_before} → {len(model.graph.node)}")

    print("  Step 4/4  Cleaning unused initializers …")
    model = sm.clean_initializers(model)

    onnx.checker.check_model(model)
    onnx.save(model, dest)
    print(f"  Saved → {dest}")
    return dest


def process_r50(src: str, out_dir: str) -> str:
    import onnx
    am = _load_module("arcface_to_mla")

    dest = os.path.join(out_dir, "w600k_r50.surgery.onnx")
    print(f"\n[ArcFace R50] Loading {src}")
    model = onnx.load(src)

    print("  Step 1/2  Rewriting BatchNormalization, Flatten, Gemm …")
    n_before = len(model.graph.node)
    model, rewrites = am.apply_surgery(model)
    print(f"    {rewrites} rewrites  ({n_before} → {len(model.graph.node)} nodes)")

    print("  Step 2/2  Freezing batch=1 and running onnxsim …")
    n_before = len(model.graph.node)
    model = am.freeze_and_simplify(model)
    print(f"    Nodes: {n_before} → {len(model.graph.node)}")

    onnx.checker.check_model(model)
    onnx.save(model, dest)
    print(f"  Saved → {dest}")
    return dest



def validate_all(
    scrfd_raw: str, scrfd_out: str, input_h: int, input_w: int,
    r50_raw: str, r50_out: str,
) -> None:
    sm = _load_module("scrfd_to_mla")
    am = _load_module("arcface_to_mla")

    print("\n[Validate] SCRFD …")
    ok_scrfd = sm.validate(scrfd_raw, scrfd_out, input_h, input_w)
    print(f"  SCRFD   {'PASS ✓' if ok_scrfd else 'FAIL ✗'}")

    print("[Validate] ArcFace R50 …")
    ok_r50 = am.validate(r50_raw, r50_out)
    print(f"  ArcFace {'PASS ✓' if ok_r50 else 'FAIL ✗'}")

    if not (ok_scrfd and ok_r50):
        sys.exit("Validation FAILED — see output above.")



def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir", default=str(_HERE),
        help="Directory to place all output files (default: script directory)",
    )
    parser.add_argument(
        "--input-h", type=int, default=640,
        help="SCRFD input height to freeze (default: 640)",
    )
    parser.add_argument(
        "--input-w", type=int, default=640,
        help="SCRFD input width to freeze (default: 640)",
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip numerical validation (not recommended)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print("Step 1/4 — Download source models (both already in ONNX format)")
    print("=" * 64)
    scrfd_raw = download_scrfd(out_dir)
    r50_raw   = download_r50(out_dir)

    print("\n" + "=" * 64)
    print("Step 2/4 — SCRFD graph surgery (scrfd_to_mla.py)")
    print("=" * 64)
    scrfd_out = process_scrfd(scrfd_raw, out_dir, args.input_h, args.input_w)

    print("\n" + "=" * 64)
    print("Step 3/4 — ArcFace R50 graph surgery (arcface_to_mla.py)")
    print("=" * 64)
    r50_out = process_r50(r50_raw, out_dir)

    if not args.no_validate:
        print("\n" + "=" * 64)
        print("Step 4/4 — Numerical validation")
        print("=" * 64)
        validate_all(scrfd_raw, scrfd_out, args.input_h, args.input_w, r50_raw, r50_out)

    print("\n" + "=" * 64)
    print("Done — files ready for compile_models.sh:")
    print(f"  {scrfd_out}")
    print(f"  {r50_out}")
    print("=" * 64)
    print("\nNext step:")
    print("  bash examples/face-recognition/scripts/compile_models.sh \\")
    print(f"    --models-dir {out_dir} \\")
    print("    --build-dir  <build-output-dir> \\")
    print("    --calib-dir  <calibration-images-dir>")


if __name__ == "__main__":
    main()
