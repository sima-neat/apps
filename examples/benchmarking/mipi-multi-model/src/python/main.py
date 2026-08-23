#!/usr/bin/env python3
"""Run a selected model directly from a strict zero-copy MIPI camera."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from model_profiles import (
    PROFILES,
    ModelPackage,
    ModelPackageError,
    Profile,
    inspect_package,
    profile_named,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
DEFAULT_MODELS_DIR = Path("models")
OUTPUT_NAME = "results"


@dataclass(frozen=True)
class AppConfig:
    profile: Profile
    model_path: Path
    frames: int
    timeout_ms: int


def _mapping(value, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_config(path: Path) -> AppConfig:
    try:
        with path.open("r", encoding="utf-8") as handle:
            root = _mapping(yaml.safe_load(handle), "config root")
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load {path}: {exc}") from exc

    model = _mapping(root.get("model"), "model")
    runtime = _mapping(root.get("runtime"), "runtime")
    profile_value = model.get("profile", "detect")
    path_value = model.get("path", "")
    if not isinstance(profile_value, str):
        raise TypeError("model.profile must be a string")
    if not isinstance(path_value, str):
        raise TypeError("model.path must be a string")

    profile = profile_named(profile_value)
    model_path = Path(path_value) if path_value.strip() else DEFAULT_MODELS_DIR / profile.archive
    return AppConfig(
        profile=profile,
        model_path=model_path,
        frames=_positive_integer(runtime.get("frames", 5), "runtime.frames"),
        timeout_ms=_positive_integer(runtime.get("timeout_ms", 30_000), "runtime.timeout_ms"),
    )


def model_options(pyneat, profile: Profile, package: ModelPackage):
    """Set only source information and model policy that cannot be inferred."""
    options = pyneat.ModelOptions()
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12

    if profile.preprocessing == "torchvision_ssdlite":
        preprocess = options.preprocess
        preprocess.kind = pyneat.InputKind.Image
        preprocess.enable = pyneat.AutoFlag.On
        preprocess.resize.enable = pyneat.AutoFlag.On
        preprocess.resize.mode = pyneat.ResizeMode.Stretch
        preprocess.normalize.enable = pyneat.AutoFlag.On
        preprocess.normalize.mean = [0.485, 0.456, 0.406]
        preprocess.normalize.stddev = [0.229, 0.224, 0.225]
        preprocess.normalize.has_explicit_stats = True
        preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
        options.num_classes = 91

    if profile.decode_type:
        try:
            options.decode_type = getattr(pyneat.BoxDecodeType, profile.decode_type)
        except AttributeError as exc:
            raise RuntimeError(
                f"installed Neat does not support {profile.decode_type} decoding"
            ) from exc

    # SDK 2.1.3 YOLO packages carry the generic `yolo` token. When selecting the
    # versioned decoder, preserve the decoder policy shipped inside the package.
    if profile.decoder_policy_required:
        policy = package.decoder_policy
        if policy is None:
            raise RuntimeError(f"{package.path.name} has no packaged decoder policy")
        options.score_threshold = policy.score_threshold
        options.nms_iou_threshold = policy.nms_iou_threshold
        options.top_k = policy.top_k
    return options


def make_graph(pyneat, profile: Profile, package: ModelPackage):
    model = pyneat.Model(str(package.path), model_options(pyneat, profile, package))
    route = pyneat.ModelRouteOptions()
    route.upstream_name = "camera"

    graph = pyneat.Graph(f"mipi_{profile.name}")
    graph.add(pyneat.nodes.camera_input())
    graph.add(model.graph(route))
    graph.add(pyneat.nodes.output(OUTPUT_NAME))
    return graph


def sample_tensors(sample) -> list:
    """Flatten Tensor, TensorSet, and Bundle samples."""
    tensors = []
    tensor = getattr(sample, "tensor", None)
    if tensor is not None:
        tensors.append(tensor)
    tensors.extend(list(getattr(sample, "tensors", []) or []))
    for field in list(getattr(sample, "fields", []) or []):
        tensors.extend(sample_tensors(field))
    return tensors


def _rows(tensor) -> int:
    shape = list(tensor.shape)
    return int(shape[0]) if shape else 0


def summarize_output(pyneat, profile: Profile, sample) -> str:
    tensors = sample_tensors(sample)
    if not tensors:
        raise RuntimeError(f"{profile.name} produced no output tensors")

    if profile.task == "detection":
        return f"detections={sum(_rows(item) for item in pyneat.decode_bbox(tensors))}"
    if profile.task == "pose":
        poses = sum(_rows(item.boxes) for item in pyneat.decode_pose(tensors))
        return f"poses={poses}"
    if profile.task == "segmentation":
        instances = sum(_rows(item.boxes) for item in pyneat.decode_segmentation(tensors))
        return f"instances={instances}"

    import numpy as np

    first = tensors[0]
    values = np.asarray(first.to_numpy(copy=True))
    if values.size == 0:
        raise RuntimeError(f"{profile.task} output tensor is empty")
    shape = [int(value) for value in first.shape]
    if profile.task == "classification":
        flattened = values.reshape(-1)
        class_id = int(np.argmax(flattened))
        return f"class_id={class_id} value={float(flattened[class_id]):.6g} shape={shape}"
    return (
        f"depth_shape={shape} min={float(np.nanmin(values)):.6g} "
        f"max={float(np.nanmax(values)):.6g}"
    )


def require_strict_zero_copy(backend: str) -> None:
    if "simaai-zero-copy-required=true" not in backend:
        raise RuntimeError("camera backend does not require zero-copy")
    if "neatcamerabridge" in backend.lower():
        raise RuntimeError("graph contains the forbidden CPU camera bridge")


def _apply_overrides(config: AppConfig, args) -> AppConfig:
    profile = profile_named(args.profile) if args.profile else config.profile
    model_path = args.model or (
        config.model_path
        if profile == config.profile
        else DEFAULT_MODELS_DIR / profile.archive
    )
    return replace(
        config,
        profile=profile,
        model_path=model_path,
        frames=args.frames if args.frames is not None else config.frames,
        timeout_ms=args.timeout_ms if args.timeout_ms is not None else config.timeout_ms,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--model", type=Path, help="override the selected profile's MPK")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--timeout-ms", type=int)
    parser.add_argument("--describe", action="store_true", help="print the negotiated backend")
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--validate-model-only", action="store_true")
    args = parser.parse_args(argv)

    if args.list_profiles:
        for profile in PROFILES.values():
            print(f"{profile.name:10} {profile.task:14} {profile.title} [{profile.source}]")
        return 0

    try:
        config = _apply_overrides(load_config(args.config), args)
        _positive_integer(config.frames, "frames")
        _positive_integer(config.timeout_ms, "timeout-ms")
        if args.validate_config_only:
            print(f"config valid: {args.config}")
            return 0

        package = inspect_package(config.model_path, config.profile)
        print(
            f"profile={config.profile.name} model={package.name} "
            f"sdk={package.sdk_version} package={package.path}"
        )
        if args.validate_model_only:
            return 0

        try:
            import pyneat
        except ImportError as exc:
            raise RuntimeError(
                "pyneat is not importable; run: source ~/pyneat/bin/activate"
            ) from exc

        graph = make_graph(pyneat, config.profile, package)
        backend = graph.describe_backend(False)
        require_strict_zero_copy(backend)
        if args.describe:
            print(backend)

        run = graph.build()
        try:
            for frame in range(config.frames):
                sample = run.pull(OUTPUT_NAME, config.timeout_ms)
                if sample is None:
                    raise TimeoutError(
                        run.last_error() or f"timed out waiting for {config.profile.name}"
                    )
                summary = summarize_output(pyneat, config.profile, sample)
                print(f"frame={frame} profile={config.profile.name} {summary}")
        finally:
            run.close()

        print(f"PASS strict-zero-copy MIPI -> {config.profile.title} -> {OUTPUT_NAME}")
        return 0
    except (ModelPackageError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
