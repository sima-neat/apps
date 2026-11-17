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
    registry_variant: str | None = None

    @property
    def source(self) -> str:
        if self.registry_variant:
            model_id = self.archive.split("/", 1)[0]
            return f"Model Registry {model_id} {self.registry_variant}: {self.archive}"
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
            "yolo26n/w8a8/yolo26n_w8a8_mpk.tar.gz",
            decode_type="YoloV26",
            registry_variant="w8a8",
        ),
        Profile(
            "pose",
            "YOLO26n pose estimation",
            "pose",
            "yolo26n_pose/w16a16/yolo26n_pose_w16a16_mpk.tar.gz",
            decode_type="YoloV26Pose",
            registry_variant="w16a16",
        ),
        Profile(
            "segment",
            "YOLO26n instance segmentation",
            "segmentation",
            "yolo26n_seg/w16a16/yolo26n_seg_w16a16_mpk.tar.gz",
            decode_type="YoloV26Seg",
            registry_variant="w16a16",
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
            "resnet_50/w8a8/resnet_50_w8a8_mpk.tar.gz",
            registry_variant="w8a8",
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
