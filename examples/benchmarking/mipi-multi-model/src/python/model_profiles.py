"""Model profiles and package handling for the MIPI multi-model example."""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
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
    decoder_policy_required: bool = False
    preprocessing: str | None = None

    @property
    def source(self) -> str:
        if self.modelzoo_id:
            return f"Model Zoo {MODEL_ZOO_VERSION}: {self.modelzoo_id}"
        return f"sima-neat/models: {self.registry_target}"


@dataclass(frozen=True)
class DecoderPolicy:
    score_threshold: float
    nms_iou_threshold: float
    top_k: int


@dataclass(frozen=True)
class ModelPackage:
    path: Path
    name: str
    sdk_version: str
    decoder_policy: DecoderPolicy | None


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
            decoder_policy_required=True,
        ),
        Profile(
            "pose",
            "YOLO26n pose estimation",
            "pose",
            "yolo_26n_pose_mpk.tar.gz",
            decode_type="YoloV26Pose",
            modelzoo_id="yolo_26n_pose",
            decoder_policy_required=True,
        ),
        Profile(
            "segment",
            "YOLO26n instance segmentation",
            "segmentation",
            "yolo_26n_seg_mpk.tar.gz",
            decode_type="YoloV26Seg",
            modelzoo_id="yolo_26n_seg",
            decoder_policy_required=True,
        ),
        Profile(
            "ssd",
            "SSD-MobileNet V3 object detection",
            "detection",
            "ssd_mobilenet_v3_mpk.tar.gz",
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


def _json_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> dict:
    stream = archive.extractfile(member)
    if stream is None:
        raise ModelPackageError(f"cannot read {member.name}")
    value = json.load(stream)
    if not isinstance(value, dict):
        raise ModelPackageError(f"{member.name} must contain a JSON object")
    return value


def inspect_package(path: Path, profile: Profile) -> ModelPackage:
    """Validate a directly consumable Model SDK 2.1.3 MPK archive."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ModelPackageError(f"model package does not exist: {path}")
    if not path.name.endswith("_mpk.tar.gz"):
        raise ModelPackageError(f"model must be a directly consumable *_mpk.tar.gz: {path}")

    try:
        with tarfile.open(path, "r:gz") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            metadata = [member for member in members if member.name.endswith("_mpk.json")]
            executables = [member for member in members if member.name.endswith("_mla.elf")]
            decoders = [member for member in members if member.name.endswith("boxdecoder.json")]
            if len(metadata) != 1:
                raise ModelPackageError(
                    f"package must contain one *_mpk.json; found {len(metadata)}"
                )
            if not executables:
                raise ModelPackageError("package contains no MLA executable (*_mla.elf)")
            if len(decoders) > 1:
                raise ModelPackageError("package contains multiple boxdecoder.json files")
            manifest = _json_member(archive, metadata[0])
            decoder = _json_member(archive, decoders[0]) if decoders else None
    except ModelPackageError:
        raise
    except (OSError, tarfile.TarError, json.JSONDecodeError) as exc:
        raise ModelPackageError(f"invalid model package {path}: {exc}") from exc

    name = manifest.get("name")
    sdk_version = manifest.get("model_sdk_version")
    if not isinstance(name, str) or not name:
        raise ModelPackageError(f"{metadata[0].name} has no model name")
    if sdk_version != MODEL_ZOO_VERSION:
        raise ModelPackageError(
            f"{path.name} uses Model SDK {sdk_version}; expected {MODEL_ZOO_VERSION}"
        )

    policy = None
    if decoder is not None:
        try:
            policy = DecoderPolicy(
                score_threshold=float(decoder["detection_threshold"]),
                nms_iou_threshold=float(decoder["nms_iou_threshold"]),
                top_k=int(decoder["topk"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelPackageError(f"invalid decoder policy in {path.name}: {exc}") from exc
    if profile.decoder_policy_required and policy is None:
        raise ModelPackageError(f"{path.name} has no packaged decoder policy")

    return ModelPackage(path, name, sdk_version, policy)


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
                candidates = sorted(staging.glob("**/*_mpk.tar.gz"))
                preferred = [
                    path for path in candidates if "modalix_int8_tess_mla" in path.parts
                ]
                selected = preferred or candidates
                if len(selected) != 1:
                    raise ModelPackageError(
                        f"registry install produced {len(selected)} candidate MPKs"
                    )
                shutil.copy2(selected[0], destination)
        else:
            raise ModelPackageError(f"profile {profile.name} has no package source")
    except FileNotFoundError as exc:
        raise ModelPackageError("sima-cli is not installed or is not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise ModelPackageError(f"sima-cli model fetch failed with exit code {exc.returncode}") from exc

    return inspect_package(destination, profile)
