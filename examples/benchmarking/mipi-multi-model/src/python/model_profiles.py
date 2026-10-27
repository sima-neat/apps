"""Model profiles and package handling for the MIPI multi-model example."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODEL_ZOO_VERSION = "3.0.0"


class ModelPackageError(RuntimeError):
    """A model profile or compiled package is invalid."""


@dataclass(frozen=True)
class Profile:
    name: str
    title: str
    task: str
    archive: str
    decode_type: str | None = None
    preprocessing: str | None = None

    @property
    def source(self) -> str:
        return f"SDK {MODEL_ZOO_VERSION} download: {self.archive}"


@dataclass(frozen=True)
class ModelPackage:
    path: Path


PROFILES = {
    profile.name: profile
    for profile in (
        Profile(
            "detect",
            "YOLO26n object detection",
            "detection",
            "yolo_26n_mpk.tar.gz",
            decode_type="YoloV26",
        ),
        Profile(
            "pose",
            "YOLO26n pose estimation",
            "pose",
            "yolo_26n_pose_mpk.tar.gz",
            decode_type="YoloV26Pose",
        ),
        Profile(
            "segment",
            "YOLO26n instance segmentation",
            "segmentation",
            "yolo_26n_seg_mpk.tar.gz",
            decode_type="YoloV26Seg",
        ),
        Profile(
            "ssd",
            "SSD-MobileNet V2 object detection",
            "detection",
            "ssd_mobilenet_v2_modalix_int8_tess_mla_mpk.tar.gz",
            decode_type="Ssd",
            preprocessing="tensorflow_ssd",
        ),
        Profile(
            "classify",
            "ResNet-50 image classification",
            "classification",
            "resnet_50_mpk.tar.gz",
        ),
        Profile(
            "depth",
            "MiDaS v2.1 Small depth estimation",
            "depth",
            "midas_v21_small_256_mpk.tar.gz",
        ),
    )
}


def profile_named(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(PROFILES)
        raise ModelPackageError(f"unknown profile {name!r}; choose one of: {choices}") from exc


def inspect_package(path: Path, profile: Profile) -> ModelPackage:
    """Resolve a model package path; Neat owns package parsing and compatibility checks."""
    del profile
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ModelPackageError(f"model package does not exist: {path}")
    if not path.name.endswith(".tar.gz"):
        raise ModelPackageError(f"model must be a compiled .tar.gz package: {path}")

    return ModelPackage(path)
