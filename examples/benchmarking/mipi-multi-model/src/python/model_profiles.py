"""Model profiles and package handling for the MIPI multi-model example."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

MODEL_ZOO_VERSION = "2.1.3"
BOARD_TYPE = "modalix"


class ModelPackageError(RuntimeError):
    """A model profile or compiled package is invalid."""


@dataclass(frozen=True)
class Profile:
    name: str
    title: str
    task: str
    archive: str
    decode_type: str | None = None
    modelzoo_id: str | None = None
    registry_target: str | None = None
    preprocessing: str | None = None

    @property
    def source(self) -> str:
        if self.modelzoo_id:
            return f"Model Zoo {MODEL_ZOO_VERSION}: {self.modelzoo_id}"
        return f"sima-neat/models: {self.registry_target}"


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
            modelzoo_id="yolo_26n",
        ),
        Profile(
            "pose",
            "YOLO26n pose estimation",
            "pose",
            "yolo_26n_pose_mpk.tar.gz",
            decode_type="YoloV26Pose",
            modelzoo_id="yolo_26n_pose",
        ),
        Profile(
            "segment",
            "YOLO26n instance segmentation",
            "segmentation",
            "yolo_26n_seg_mpk.tar.gz",
            decode_type="YoloV26Seg",
            modelzoo_id="yolo_26n_seg",
        ),
        Profile(
            "ssd",
            "SSD-MobileNet V3 object detection",
            "detection",
            "ssd_mobilenet_v3_modalix_int8_tess_mla_mpk.tar.gz",
            decode_type="Ssd",
            registry_target="models/ssd_mobilenet_v3@develop:latest",
            preprocessing="torchvision_ssdlite",
        ),
        Profile(
            "classify",
            "ResNet-50 image classification",
            "classification",
            "resnet_50_mpk.tar.gz",
            modelzoo_id="resnet_50",
        ),
        Profile(
            "depth",
            "MiDaS v2.1 Small depth estimation",
            "depth",
            "midas_v21_small_256_mpk.tar.gz",
            modelzoo_id="midas_v21_small_256",
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
    """Resolve an MPK path; Neat owns package parsing and compatibility checks."""
    del profile
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ModelPackageError(f"model package does not exist: {path}")
    if not path.name.endswith("_mpk.tar.gz"):
        raise ModelPackageError(f"model must be a directly consumable *_mpk.tar.gz: {path}")

    return ModelPackage(path)


def fetch_profile(profile: Profile, models_dir: Path) -> ModelPackage:
    """Fetch one canonical tar.gz through sima-cli and validate it."""
    models_dir = models_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    destination = models_dir / profile.archive

    try:
        if profile.modelzoo_id:
            subprocess.run(
                [
                    "sima-cli",
                    "modelzoo",
                    "-v",
                    MODEL_ZOO_VERSION,
                    "--boardtype",
                    BOARD_TYPE,
                    "get",
                    profile.modelzoo_id,
                ],
                cwd=models_dir,
                check=True,
            )
        elif profile.registry_target:
            with tempfile.TemporaryDirectory(prefix="mipi-model-") as temporary:
                staging = Path(temporary)
                subprocess.run(
                    [
                        "sima-cli",
                        "neat",
                        "install",
                        "--stg",
                        profile.registry_target,
                        "--install-dir",
                        str(staging),
                    ],
                    check=True,
                )
                selected = sorted(staging.glob(f"**/{profile.archive}"))
                if len(selected) != 1:
                    raise ModelPackageError(
                        f"registry install produced {len(selected)} copies of "
                        f"{profile.archive}"
                    )
                shutil.copy2(selected[0], destination)
        else:
            raise ModelPackageError(f"profile {profile.name} has no package source")
    except FileNotFoundError as exc:
        raise ModelPackageError("sima-cli is not installed or is not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise ModelPackageError(f"sima-cli model fetch failed with exit code {exc.returncode}") from exc

    return inspect_package(destination, profile)
