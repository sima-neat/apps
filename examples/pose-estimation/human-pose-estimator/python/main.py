"""Config-driven OpenPose pose-estimation overlay pipeline using pyneat."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import math
from utils.config import AppConfig, load_app_config


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "common" / "config.yaml"

POSE_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), 
    (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), 
    (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]

PAF_CHANNELS = [
    (12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1),
    (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (28, 29), 
    (30, 31), (34, 35), (32, 33), (36, 37), (18, 19), (26, 27)
]
def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}

def tensor_to_numpy(t: pyneat.Tensor) -> np.ndarray:
    return np.asarray(t.to_numpy(copy=True))


def iter_tensors(sample: pyneat.Sample):
    if isinstance(sample, (list, tuple)):
        for item in sample:
            yield from iter_tensors(item)
        return
    if isinstance(sample, pyneat.Tensor):
        yield sample
        return
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        yield sample.tensor
    for tensor in getattr(sample, "tensors", []):
        yield tensor
    for field in getattr(sample, "fields", []):
        yield from iter_tensors(field)


def tensor_to_hwc_f32(t: pyneat.Tensor) -> np.ndarray:
    arr = tensor_to_numpy(t).astype(np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected tensor rank {arr.ndim}")
    return arr


def letterbox(img: np.ndarray, new_shape: tuple[int, int], color=(128, 128, 128)):
    shape = img.shape[:2]  
    
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    if shape[::-1] != new_unpad:  
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    pad_w = new_shape[1] - new_unpad[0]
    pad_h = new_shape[0] - new_unpad[1]
    
    top = int(math.floor(pad_h / 2.0))
    bottom = pad_h - top
    left = int(math.floor(pad_w / 2.0))
    right = pad_w - left
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, r, left, top


def get_all_keypoints_without_grouping(
    heatmap_tensor, infer_size: int, min_score: float, nms_radius: int
) -> list[dict]:
    h, w, c_hm = heatmap_tensor.shape
    stride_x = infer_size / float(w)
    stride_y = infer_size / float(h)

    keypoint_list = []
    kpt_id = 0
    
    for part_id in range(18):
        prob_map = heatmap_tensor[:, :, part_id].copy()
        
        prob_map[prob_map < min_score] = 0
        heatmap_with_borders = np.pad(prob_map, [(2, 2), (2, 2)], mode='constant')
        heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 1:heatmap_with_borders.shape[1]-1]
        heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 2:heatmap_with_borders.shape[1]]
        heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 0:heatmap_with_borders.shape[1]-2]
        heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1]-1]
        heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0]-2, 1:heatmap_with_borders.shape[1]-1]

        heatmap_peaks = (heatmap_center > heatmap_left) &\
                        (heatmap_center > heatmap_right) &\
                        (heatmap_center > heatmap_up) &\
                        (heatmap_center > heatmap_down)
        
        heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0]-1, 1:heatmap_center.shape[1]-1]
        
        y_coords, x_coords = np.nonzero(heatmap_peaks)
        scores = prob_map[y_coords, x_coords]
        
        candidates = list(zip(x_coords, y_coords, scores))
        candidates.sort(key=lambda x: x[0])
        
        suppressed = np.zeros(len(candidates), dtype=np.uint8)
        for i in range(len(candidates)):
            if suppressed[i]:
                continue
            for j in range(i+1, len(candidates)):
                if math.hypot(candidates[i][0] - candidates[j][0],
                              candidates[i][1] - candidates[j][1]) < nms_radius:
                    suppressed[j] = 1
        
        for i in range(len(candidates)):
            if suppressed[i]:
                continue
                
            x_c, y_c, s = candidates[i]

            cx = x_c * stride_x
            cy = y_c * stride_y
            
            keypoint_list.append({
                "id": kpt_id,
                "x": float(cx),
                "y": float(cy),
                "grid_x": int(x_c), 
                "grid_y": int(y_c),
                "score": float(s),
                "part_id": part_id
            })
            kpt_id += 1
            
    return keypoint_list

KP_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), 
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), 
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255), 
    (255, 0, 255), (255, 0, 170), (255, 0, 85)
]

def draw_keypoints(frame: np.ndarray, keypoints: list[dict], labels: list[str]) -> np.ndarray:
    for k in keypoints:
        cx, cy = int(k["x"]), int(k["y"])
        part_id = k["part_id"]
        score = k["score"]
        
        color = KP_COLORS[part_id % len(KP_COLORS)]
        label = labels[part_id] if part_id < len(labels) else str(part_id)
        text = f"{label} {score:.2f}"
        
        cv2.circle(frame, (cx, cy), radius=4, color=color, thickness=-1)
        cv2.circle(frame, (cx, cy), radius=4, color=(255, 255, 255), thickness=1)
        
        text_x = cx + 5
        text_y = cy - 5
        
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (text_x, text_y - th - 2), (text_x + tw, text_y + 2), color, -1)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
    return frame
def compute_paf_score(
    paf_tensor: np.ndarray,
    kpt_a: dict,
    kpt_b: dict,
    paf_x_idx: int,
    paf_y_idx: int,
    paf_min_score: float,
    paf_success_ratio: float,
    num_samples: int,
) -> float:
    """Calculates the line integral over the PAF vector field between two keypoints."""
    dx = kpt_b["grid_x"] - kpt_a["grid_x"]
    dy = kpt_b["grid_y"] - kpt_a["grid_y"]
    distance = np.hypot(dx, dy)
    
    if distance < 1e-5:
        return 0.0

    vec_x = dx / distance
    vec_y = dy / distance

    xs = np.linspace(kpt_a["grid_x"], kpt_b["grid_x"], num_samples)
    ys = np.linspace(kpt_a["grid_y"], kpt_b["grid_y"], num_samples)

    xs = np.clip(np.round(xs).astype(int), 0, paf_tensor.shape[1] - 1)
    ys = np.clip(np.round(ys).astype(int), 0, paf_tensor.shape[0] - 1)

    paf_vx = paf_tensor[ys, xs, paf_x_idx]
    paf_vy = paf_tensor[ys, xs, paf_y_idx]

    local_scores = paf_vx * vec_x + paf_vy * vec_y
    
    valid_mask = local_scores > paf_min_score
    valid_points = np.sum(valid_mask)
    
    if valid_points > 0:
        paf_score = np.sum(local_scores[valid_mask]) / valid_points
    else:
        paf_score = 0.0
    

    if valid_points > paf_success_ratio * num_samples and paf_score > 0.0:
        return float(paf_score)
        
    return 0.0

def group_keypoints_with_pafs(
    keypoints: list[dict],
    paf_tensor: np.ndarray,
    paf_min_score: float,
    paf_success_ratio: float,
    paf_num_samples: int,
    min_valid_joints: int,
    min_avg_person_score: float,
) -> list[dict]:
    """Uses greedy bipartite matching and graph merging to assemble people."""
    kpts_by_part = {i: [] for i in range(18)}
    for kpt in keypoints:
        kpts_by_part[kpt["part_id"]].append(kpt)

    people = [] 

    for pair_idx, (part_a, part_b) in enumerate(POSE_PAIRS):
        cands_a = kpts_by_part[part_a]
        cands_b = kpts_by_part[part_b]
        
        if not cands_a or not cands_b:
            continue
            
        paf_x_idx, paf_y_idx = PAF_CHANNELS[pair_idx]

        candidate_connections = []
        for a in cands_a:
            for b in cands_b:
                score = compute_paf_score(paf_tensor, a, b, paf_x_idx, paf_y_idx, 
                                         paf_min_score, paf_success_ratio, paf_num_samples)
                if score > 0.0:
                    candidate_connections.append({
                        "a": a, "b": b, "score": score, 
                        "a_id": a["id"], "b_id": b["id"]
                    })

        candidate_connections.sort(key=lambda x: x["score"], reverse=True)
        used_a = set()
        used_b = set()
        valid_connections = []
        
        for conn in candidate_connections:
            if conn["a_id"] not in used_a and conn["b_id"] not in used_b:
                valid_connections.append(conn)
                used_a.add(conn["a_id"])
                used_b.add(conn["b_id"])

        if pair_idx == 0:
            for conn in valid_connections:
                people.append({
                    "keypoints": {part_a: conn["a"], part_b: conn["b"]},
                    "total_score": conn["a"]["score"] + conn["b"]["score"] + conn["score"],
                    "valid_joints_count": 2
                })
        elif pair_idx in (17, 18):
            for conn in valid_connections:
                a_id = conn["a"]["id"]
                b_id = conn["b"]["id"]
                for p in people:
                    if part_a in p["keypoints"] and p["keypoints"][part_a]["id"] == a_id and part_b not in p["keypoints"]:
                        p["keypoints"][part_b] = conn["b"]
                    elif part_b in p["keypoints"] and p["keypoints"][part_b]["id"] == b_id and part_a not in p["keypoints"]:
                        p["keypoints"][part_a] = conn["a"]
        else:
            for conn in valid_connections:
                num = 0
                a_id = conn["a"]["id"]
                for p in people:
                    if part_a in p["keypoints"] and p["keypoints"][part_a]["id"] == a_id:
                        p["keypoints"][part_b] = conn["b"]
                        num += 1
                        p["valid_joints_count"] += 1
                        p["total_score"] += conn["b"]["score"] + conn["score"]
                
                if num == 0:
                    people.append({
                        "keypoints": {part_a: conn["a"], part_b: conn["b"]},
                        "total_score": conn["a"]["score"] + conn["b"]["score"] + conn["score"],
                        "valid_joints_count": 2
                    })

    final_people = []
    for p in people:
        kpts = p["keypoints"]
        valid_joints = p.get("valid_joints_count", len(kpts))
        if valid_joints < min_valid_joints:
            continue
            
        if (p["total_score"] / valid_joints) >= min_avg_person_score:
            final_people.append(p)

    return final_people

 
def draw_poses(frame: np.ndarray, people: list[dict]) -> np.ndarray:
    """Draws keypoints and skeleton bones for each person."""
    for person in people:
        kpts = person["keypoints"]
        
   
        for pair_idx, (part_a, part_b) in enumerate(POSE_PAIRS):
            if pair_idx in (17, 18):
                continue 
            if part_a in kpts and part_b in kpts:
                pt_a = (int(kpts[part_a]["x"]), int(kpts[part_a]["y"]))
                pt_b = (int(kpts[part_b]["x"]), int(kpts[part_b]["y"]))
                color = KP_COLORS[part_a % len(KP_COLORS)]
                cv2.line(frame, pt_a, pt_b, color, 2)
                
        for part_id, k in kpts.items():
            cx, cy = int(k["x"]), int(k["y"])
            color = KP_COLORS[part_id % len(KP_COLORS)]
            cv2.circle(frame, (cx, cy), radius=4, color=color, thickness=-1)
            cv2.circle(frame, (cx, cy), radius=4, color=(255, 255, 255), thickness=1)
            
    return frame
def scale_keypoints(keypoints: list[dict], r: float, left: int, top: int) -> list[dict]:
    scaled = []
    for k in keypoints:
        scaled.append(dict(
            x=(k["x"] - left) / r, 
            y=(k["y"] - top) / r,
            score=k["score"], 
            part_id=k["part_id"]
        ))
    return scaled

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenPose simple pose estimation pipeline")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to YAML configuration. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling: report end-to-end and model-inference timing",
    )
    return parser


def run_app(cfg: AppConfig, enable_profile: bool) -> int:
    global cv2, np, pyneat

    import cv2
    import numpy as np
    import pyneat

    input_dir = Path(cfg.io.input_dir)
    output_dir = Path(cfg.io.output_dir)
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in input_dir.iterdir() if p.is_file() and is_image(p))
    if not images:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return 3
    print(f"Found {len(images)} images")

    try:
        opt = pyneat.ModelOptions()
        opt.preprocess.kind = pyneat.InputKind.Image
        opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
        opt.preprocess.input_max_width = cfg.runtime.infer_size
        opt.preprocess.input_max_height = cfg.runtime.infer_size
        opt.preprocess.input_max_depth = 3

        model = pyneat.Model(cfg.model.path, opt)

        sess = pyneat.Session()
        sess.add(model.session())
        print(f"[BUILD] Pipeline:\n{sess.describe_backend()}")

        dummy = np.zeros((cfg.runtime.infer_size, cfg.runtime.infer_size, 3), dtype=np.uint8)
        t_dummy = pyneat.Tensor.from_numpy(
            dummy,
            copy=True,
            image_format=pyneat.PixelFormat.BGR,
            memory=pyneat.TensorMemory.EV74,
        )
        run = sess.build(t_dummy, pyneat.RunMode.Async)

        processed = 0
        total_e2e_time = 0.0
        total_infer_time = 0.0
        
        for img_path in images:
            e2e_start = time.perf_counter()
            
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"Skipping unreadable: {img_path.name}", file=sys.stderr)
                continue

            orig_h, orig_w = bgr.shape[:2]
            
            resized, r, pad_l, pad_t = letterbox(
                bgr, (cfg.runtime.infer_size, cfg.runtime.infer_size)
            )
            resized = np.ascontiguousarray(resized, dtype=np.uint8)

            t_in = pyneat.Tensor.from_numpy(
                resized,
                copy=True,
                image_format=pyneat.PixelFormat.BGR,
                memory=pyneat.TensorMemory.EV74,
            )

            infer_start = time.perf_counter()
            out_opt = run.run(t_in, timeout_ms=cfg.runtime.timeout_ms)
            infer_end = time.perf_counter()
            if out_opt is None:
                print(f"Inference failed for {img_path.name}", file=sys.stderr)
                continue
            
            tensors = list(iter_tensors(out_opt))
            
            # Validate that the model output contains the expected tensors
            if len(tensors) < 2:
                print(
                    f"Inference output for {img_path.name} does not contain the expected "
                    f"number of tensors (expected at least 2, got {len(tensors)}). Skipping.",
                    file=sys.stderr,
                )
                continue
            heatmap_tensor = tensor_to_hwc_f32(tensors[0])
            paf_tensor = tensor_to_hwc_f32(tensors[1])

            heatmap_tensor = cv2.resize(
                heatmap_tensor,
                (0, 0),
                fx=cfg.runtime.upsample_factor,
                fy=cfg.runtime.upsample_factor,
                interpolation=cv2.INTER_CUBIC,
            )
            paf_tensor = cv2.resize(
                paf_tensor,
                (0, 0),
                fx=cfg.runtime.upsample_factor,
                fy=cfg.runtime.upsample_factor,
                interpolation=cv2.INTER_CUBIC,
            )


            raw_kpts = get_all_keypoints_without_grouping(
                heatmap_tensor,
                cfg.runtime.infer_size,
                min_score=cfg.decode.keypoint_score,
                nms_radius=cfg.decode.nms_radius,
            )
            people = group_keypoints_with_pafs(
                raw_kpts,
                paf_tensor,
                paf_min_score=cfg.decode.paf_score,
                paf_success_ratio=cfg.decode.paf_success_ratio,
                paf_num_samples=cfg.decode.paf_samples,
                min_valid_joints=cfg.decode.min_valid_joints,
                min_avg_person_score=cfg.decode.min_avg_person_score,
            )

            for p in people:
                scaled_kpts = scale_keypoints(list(p["keypoints"].values()), r, pad_l, pad_t)
                new_kpts = {sk["part_id"]: sk for sk in scaled_kpts}
                p["keypoints"] = new_kpts

            draw_poses(bgr, people)

            out_path = output_dir / f"{img_path.stem}.png"
            cv2.imwrite(str(out_path), bgr)
            
            e2e_end = time.perf_counter()
            e2e_time = (e2e_end - e2e_start) * 1000  # ms
            infer_time = (infer_end - infer_start) * 1000  # ms
            total_e2e_time += e2e_time
            total_infer_time += infer_time
            
            processed += 1
            status = f"[{processed}/{len(images)}] {img_path.name} -> {out_path.name} ({len(people)} detections)"
            if enable_profile:
                status += f" | e2e={e2e_time:.2f}ms infer={infer_time:.2f}ms"
            print(status)

        run.close()
        print(f"Done: {processed} images processed")
        if enable_profile and processed > 0:
            print(f"[PROFILE] Average end-to-end: {total_e2e_time/processed:.2f}ms")
            print(f"[PROFILE] Average model inference: {total_infer_time/processed:.2f}ms")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr, flush=True)
        return 2

    try:
        cfg = load_app_config(config_path)
    except Exception as exc:
        print(f"Error: failed to load config {config_path}: {exc}", file=sys.stderr, flush=True)
        return 2

    return run_app(cfg, args.profile)


if __name__ == "__main__":
    raise SystemExit(main())
