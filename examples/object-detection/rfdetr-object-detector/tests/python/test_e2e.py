"""End-to-end tests for both RF-DETR variants."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


def _runtime_ready() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("numpy", "pyneat"))


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


@pytest.mark.e2e
@pytest.mark.parametrize("variant", ["small", "medium"])
@pytest.mark.parametrize(
    ("codec", "stream_fixture"),
    [("h264", "rtsp_h264_url"), ("h265", "rtsp_h265_url")],
)
def test_variant_publishes_insight_metadata(
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
    paths = {
        "small": {
            "backbone": models_dir / "rfdetr-small-backbone.tar.gz",
            "transformer": models_dir / "rfdetr-small-transformer.tar.gz",
        },
        "medium": {
            "backbone": models_dir / "rfdetr-medium-backbone.tar.gz",
            "transformer": models_dir / "rfdetr-medium-transformer.tar.gz",
        },
    }
    for path in paths[variant].values():
        skip_unless_e2e_ready(path.exists(), f"missing RF-DETR artifact: {path}")
    rtsp_url = request.getfixturevalue(stream_fixture)

    metadata_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
    config_path = e2e_config_writer(
        {
            "model": {
                "variant": variant,
                variant: {key: str(value) for key, value in paths[variant].items()},
            },
            "source": {"rtsp_url": rtsp_url, "codec": codec},
            "inference": {"frames": 20, "min_score": 0.2},
            "output": {
                "insight": {
                    "host": "127.0.0.1",
                    "video_port": _env_int("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000),
                    "metadata_port": metadata_port,
                }
            },
        }
    )

    with MetadataJsonListener(
        "127.0.0.1",
        metadata_port,
        num_ports=1,
        metadata_type="object-detection",
        data_array_key="objects",
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
        f"RF-DETR {variant} exited with {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert metadata.success, metadata.error
    objects = json.loads(metadata.messages[-1].payload)["data"]["objects"]
    assert all(
        obj.get("label")
        and 0.0 <= obj.get("confidence", -1.0) <= 1.0
        and len(obj.get("bbox", [])) == 4
        and all(value >= 0.0 for value in obj["bbox"])
        for obj in objects
    )
