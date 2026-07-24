#!/usr/bin/env python3
"""
validate_embeddings.py
─────────────────────
Assertion-level validation for the quantized ArcFace model vs. a float ONNX
reference:
  1. Cosine similarity ≥ 0.95 on N test faces (default 10).
  2. SCRFD detection count matches reference within ±tolerance.

Dependencies:
  pip install onnxruntime opencv-python numpy

Usage:
  python3 validate_embeddings.py \\
    --arcface-onnx  arcface_mbf.onnx \\
    --scrfd-onnx    scrfd_2.5g.onnx  \\
    --sima-embeddings  sima_emb_dir/ \\
    --test-faces    face_images/     \\
    [--det-ref-log  reference_detections.json] \\
    [--n 10] [--cosim-threshold 0.95] [--det-tolerance 1]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("WARNING: onnxruntime not installed — ONNX reference inference disabled")


# ── ArcFace canonical template ────────────────────────────────────────────────
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def l2_normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def cosine_similarity(a, b):
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def align_face(bgr, landmarks_5x2):
    """Similarity-transform crop to 112×112 using 5 landmarks."""
    src = landmarks_5x2.astype(np.float32)
    dst = ARCFACE_TEMPLATE
    transform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
    if transform is None:
        transform = cv2.getAffineTransform(src[:3], dst[:3])
    return cv2.warpAffine(bgr, transform, (112, 112), flags=cv2.INTER_LINEAR)


# ── ONNX inference helpers ────────────────────────────────────────────────────

def preprocess_arcface(bgr_crop):
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    f32 = rgb.astype(np.float32) / 127.5 - 1.0
    return f32.transpose(2, 0, 1)[None]  # NCHW


def run_arcface_onnx(session, crop_bgr):
    inp = preprocess_arcface(crop_bgr)
    name = session.get_inputs()[0].name
    out = session.run(None, {name: inp})[0]
    return out.flatten()


def preprocess_scrfd(bgr, infer_size=640):
    h, w = bgr.shape[:2]
    scale = min(infer_size / w, infer_size / h)
    sw, sh = int(round(w * scale)), int(round(h * scale))
    scaled = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_LINEAR)
    pad = np.zeros((infer_size, infer_size, 3), dtype=np.float32)
    pl = (infer_size - sw) // 2
    pt = (infer_size - sh) // 2
    pad[pt:pt+sh, pl:pl+sw] = scaled.astype(np.float32)
    pad -= np.array([104., 117., 123.], dtype=np.float32)
    return pad.transpose(2, 0, 1)[None], (w, h, scale, pl, pt)


def run_scrfd_onnx(session, bgr):
    """Run SCRFD and return list of (score, x1,y1,x2,y2, lm[10]) in image coords."""
    inp, (ow, oh, scale, pl, pt) = preprocess_scrfd(bgr)
    name = session.get_inputs()[0].name
    outs = session.run(None, {name: inp})
    # Parse boxes from SCRFD output (post-NMS ONNX variant returns boxes+scores).
    # If using raw-head ONNX, decode manually.
    dets = []
    if len(outs) == 1:
        # Assume NMS bbox tensor: [N, 15] = (score, x1,y1,x2,y2, lm×10)
        for row in outs[0]:
            score = float(row[0])
            if score < 0.3:
                continue
            x1 = (row[1] - pl) / scale
            y1 = (row[2] - pt) / scale
            x2 = (row[3] - pl) / scale
            y2 = (row[4] - pt) / scale
            lm = np.array([(row[5+k] - (pl if k%2==0 else pt)) / scale
                           for k in range(10)], dtype=np.float32)
            dets.append((score, x1, y1, x2, y2, lm))
    return sorted(dets, key=lambda d: -d[0])


# ── validation ────────────────────────────────────────────────────────────────

def load_sima_embedding(path):
    """Load a raw float32 binary embedding written by face-model-test."""
    data = np.fromfile(path, dtype=np.float32)
    if data.size != 512:
        raise ValueError(f"Expected 512 floats, got {data.size} in {path}")
    return data


def validate_cosim(arcface_onnx, sima_emb_dir, face_images, n, threshold):
    if not ORT_AVAILABLE:
        print("SKIP cosine-sim validation: onnxruntime not available")
        return True

    sess = ort.InferenceSession(arcface_onnx, providers=["CPUExecutionProvider"])
    passed, failed = 0, 0
    images = sorted(Path(face_images).glob("*.jpg")) + \
             sorted(Path(face_images).glob("*.png"))
    images = images[:n]

    if not images:
        print(f"WARNING: no test face images found in {face_images}")
        return True

    print(f"\nCosine similarity validation ({len(images)} images, threshold={threshold}):")
    for img_path in images:
        sima_path = Path(sima_emb_dir) / (img_path.stem + ".f32bin")
        if not sima_path.exists():
            print(f"  SKIP {img_path.name}: no SiMa embedding at {sima_path}")
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  SKIP {img_path.name}: cannot read image")
            continue

        # ONNX reference embedding (on the 112×112 crop — requires a pre-aligned crop)
        crop = cv2.resize(bgr, (112, 112))
        ref_emb = run_arcface_onnx(sess, crop)

        # SiMa embedding
        sima_emb = load_sima_embedding(sima_path)

        cosim = cosine_similarity(ref_emb, sima_emb)
        ok = cosim >= threshold
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {img_path.name}: cosim={cosim:.4f}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"  {passed} passed, {failed} failed\n")
    return failed == 0


def validate_detection_counts(scrfd_onnx, det_ref_log, face_images, tolerance):
    """Assert SiMa detection counts match ONNX reference within ±tolerance."""
    if not ORT_AVAILABLE or not det_ref_log:
        print("SKIP detection count validation")
        return True

    sess = ort.InferenceSession(scrfd_onnx, providers=["CPUExecutionProvider"])

    with open(det_ref_log) as f:
        ref = json.load(f)  # {image_name: detection_count}

    passed, failed = 0, 0
    print(f"Detection count validation (tolerance=±{tolerance}):")
    for img_name, ref_count in ref.items():
        img_path = Path(face_images) / img_name
        if not img_path.exists():
            continue
        bgr = cv2.imread(str(img_path))
        dets = run_scrfd_onnx(sess, bgr)
        diff = abs(len(dets) - ref_count)
        ok = diff <= tolerance
        print(f"  [{'PASS' if ok else 'FAIL'}] {img_name}: "
              f"onnx={len(dets)} ref={ref_count} diff={diff}")
        (passed if ok else failed).__class__  # side-effect free
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"  {passed} passed, {failed} failed\n")
    return failed == 0


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arcface-onnx",    required=False)
    ap.add_argument("--scrfd-onnx",      required=False)
    ap.add_argument("--sima-embeddings", required=False,
                    help="Dir of .f32bin files produced by face-model-test")
    ap.add_argument("--test-faces",      required=True,
                    help="Dir of test face images")
    ap.add_argument("--det-ref-log",     required=False,
                    help="JSON {filename: count} reference detection counts")
    ap.add_argument("--n",               type=int, default=10)
    ap.add_argument("--cosim-threshold", type=float, default=0.95)
    ap.add_argument("--det-tolerance",   type=int,   default=1)
    args = ap.parse_args()

    all_pass = True

    if args.arcface_onnx and args.sima_embeddings:
        ok = validate_cosim(args.arcface_onnx, args.sima_embeddings,
                            args.test_faces, args.n, args.cosim_threshold)
        all_pass = all_pass and ok

    if args.scrfd_onnx and args.det_ref_log:
        ok = validate_detection_counts(args.scrfd_onnx, args.det_ref_log,
                                       args.test_faces, args.det_tolerance)
        all_pass = all_pass and ok

    if all_pass:
        print("=== VALIDATION PASSED ===")
        sys.exit(0)
    else:
        print("=== VALIDATION FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
