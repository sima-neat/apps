"""End-to-end RF-DETR detection, segmentation, and throughput tests."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
PERFORMANCE_FRAMES = 300
MIN_OUTPUT_FPS = {"small": 65.0, "medium": 45.0, "segmentation": 35.0}


def _runtime_ready() -> bool:
    return all(
        importlib.util.find_spec(name) is not None for name in ("numpy", "pyneat")
    )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _output_fps(stdout: str) -> float:
    match = re.search(r"output_fps=([0-9]+(?:\.[0-9]+)?)", stdout)
    assert match, f"missing output_fps in application output:\n{stdout}"
    return float(match.group(1))


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("task", "variant", "codec", "stream_fixture"),
    [
        ("detection", variant, codec, f"rtsp_{codec}_url")
        for codec in ("h264", "h265", "mjpeg")
        for variant in ("small", "medium")
    ]
    + [
        ("segmentation", "medium", codec, f"rtsp_{codec}_url")
        for codec in ("h264", "h265", "mjpeg")
    ],
)
def test_publishes_insight_metadata(
    task,
    variant,
    codec,
    stream_fixture,
    models_dir,
    test_timeout_ms,
    skip_unless_e2e_ready,
    e2e_config_writer,
    request,
):
    skip_unless_e2e_ready(_runtime_ready(), "numpy and pyneat are required")
    model_name = f"rfdetr-{'seg-' if task == 'segmentation' else ''}{variant}"
    paths = {
        "backbone": models_dir / f"{model_name}-backbone.tar.gz",
        "transformer": models_dir / f"{model_name}-transformer.tar.gz",
    }
    for path in paths.values():
        skip_unless_e2e_ready(path.exists(), f"missing RF-DETR artifact: {path}")
    rtsp_url = request.getfixturevalue(stream_fixture)

    model_paths = {key: str(value) for key, value in paths.items()}
    if task == "detection":
        model_config = {"task": task, task: {"variant": variant, variant: model_paths}}
    else:
        model_config = {"task": task, task: model_paths}

    metadata_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
    config_path = e2e_config_writer(
        {
            "model": model_config,
            "source": {"rtsp_url": rtsp_url, "codec": codec},
            "inference": {
                "frames": PERFORMANCE_FRAMES if codec in ("h265", "mjpeg") else 20,
                task: {"min_score": 0.2},
            },
            "output": {
                "insight": {
                    "host": "127.0.0.1",
                    "video_port": _env_int(
                        "SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000
                    ),
                    "metadata_port": metadata_port,
                }
            },
        }
    )

    metadata_type = "object-detection" if task == "detection" else "segmentation"
    data_key = "objects" if task == "detection" else "segments"
    with MetadataJsonListener(
        "127.0.0.1",
        metadata_port,
        num_ports=1,
        metadata_type=metadata_type,
        data_array_key=data_key,
        require_all_ports=True,
        min_object_count=1,
    ) as listener:
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
            check=False,
        )
        metadata = listener.wait_for_messages(5.0)

    assert result.returncode == 0, (
        f"RF-DETR {task} {variant} exited with {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert metadata.success, metadata.error
    entries = json.loads(metadata.messages[-1].payload)["data"][data_key]
    for entry in entries:
        assert entry.get("label")
        assert 0.0 <= entry.get("confidence", -1.0) <= 1.0
        assert len(entry.get("bbox", [])) == 4
        assert all(value >= 0.0 for value in entry["bbox"])
        if task == "segmentation":
            assert entry.get("mask_format") == "polygon"
            assert len(entry.get("mask", [])) >= 3
            assert all(
                len(point) == 2 and all(value >= 0.0 for value in point)
                for point in entry["mask"]
            )
    if codec in ("h265", "mjpeg"):
        threshold = MIN_OUTPUT_FPS[
            "segmentation" if task == "segmentation" else variant
        ]
        measured_fps = _output_fps(result.stdout)
        print(
            f"[perf] task={task} model={variant} codec={codec} "
            f"frames={PERFORMANCE_FRAMES} output_fps={measured_fps:.1f} minimum={threshold:.1f}"
        )
        assert measured_fps >= threshold
