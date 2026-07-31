"""Unit tests for model-benchmark (Python)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"

sys.path.insert(0, str(EXAMPLE_DIR / "scripts"))
import refresh_results  # noqa: E402

# The fake traces the options reaching pyneat.Model so a test can assert the exact Core
# enum and top-K, and resolves the postprocess the way Core does.
FAKE_PYNEAT = '''
import os
from pathlib import Path
from types import SimpleNamespace


class BoxDecodeType:
    YoloV26 = "YoloV26"
    YoloV26Seg = "YoloV26Seg"


class ModelOptions:
    decode_type = None
    top_k = 0


class Report:
    latency_ms = 1.25
    fps = 800.0
    avg_power_watts = 2.5
    energy_joules = 0.75


class Model:
    def __init__(self, path, options=None):
        trace = "none" if options is None else f"{options.decode_type}:{options.top_k}"
        Path(os.environ["FAKE_PYNEAT_TRACE"]).write_text(trace, encoding="utf-8")
        self.outputs = 1 if options else 6
        self.post_kind = "boxdecode" if options else "detessdequant"

    def info(self):
        return SimpleNamespace(
            selection=SimpleNamespace(selected_post_kind=self.post_kind),
            output_topology=SimpleNamespace(
                physical_outputs=1, logical_outputs=1, packed_outputs=False
            ),
        )

    def input_specs(self):
        return ["input:uint8[1,3,640,640]"]

    def output_specs(self):
        return [f"out{i}:float32[1,80,80]" for i in range(self.outputs)]

    def benchmark(self, frames):
        assert frames == 7
        return Report()
'''


def run_main(tmp_path: Path, report: Path, *extra: str) -> tuple[subprocess.CompletedProcess, Path]:
    (tmp_path / "pyneat.py").write_text(FAKE_PYNEAT, encoding="utf-8")
    model = tmp_path / "model.tar.gz"
    model.write_text("fake model", encoding="utf-8")
    trace = tmp_path / "trace.txt"

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{tmp_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["FAKE_PYNEAT_TRACE"] = str(trace)
    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--model", str(model), "--frames", "7",
         "--output-json", str(report), *extra],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )
    return result, trace


@pytest.mark.unit
def test_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
    assert "yolo26-det" in result.stdout
    assert "yolo26-seg" in result.stdout


@pytest.mark.unit
def test_missing_model_fails_before_pyneat_import(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
model:
  path: missing.tar.gz
benchmark:
  frames: 1
output:
  report_json: report.json
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "model file does not exist" in result.stderr


@pytest.mark.unit
def test_zero_frames_fails(tmp_path: Path) -> None:
    model = tmp_path / "model.tar.gz"
    model.write_text("fake model", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--model", str(model), "--frames", "0"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "benchmark.frames must be > 0" in result.stderr


@pytest.mark.unit
def test_writes_json_report_with_benchmark_metrics(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    result, trace = run_main(tmp_path, report)

    assert result.returncode == 0, result.stderr
    assert trace.read_text(encoding="utf-8") == "none"
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["benchmark"]["type"] == "model.synthetic"
    assert data["benchmark"]["frames"] == 7
    assert data["model"]["file"] == "model.tar.gz"
    assert data["model"]["requested_decode_type"] is None
    assert data["model"]["resolved_postprocess"] == "detessdequant"
    assert len(data["model"]["output_specs"]) == 6
    assert data["metrics"] == {
        "latency_ms": 1.25,
        "fps": 800.0,
        "avg_power_watts": 2.5,
        "energy_joules": 0.75,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("flag", "enum_name"), [("yolo26-det", "YoloV26"), ("yolo26-seg", "YoloV26Seg")]
)
def test_decode_type_selects_core_enum(tmp_path: Path, flag: str, enum_name: str) -> None:
    report = tmp_path / "report.json"
    result, trace = run_main(tmp_path, report, "--decode-type", flag)

    assert result.returncode == 0, result.stderr
    traced_enum, traced_top_k = trace.read_text(encoding="utf-8").split(":")
    assert traced_enum == enum_name
    # neatobjectdecode rejects a zero top-K, so any BoxDecode run must carry a cap.
    assert int(traced_top_k) > 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["model"]["requested_decode_type"] == flag
    assert data["model"]["resolved_postprocess"] == "boxdecode"
    assert len(data["model"]["output_specs"]) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("default_id", "boxdecode_id", "flag"),
    [
        ("yolo26m-det-int8-b1", "yolo26m-det-int8-b1-boxdecode", "yolo26-det"),
        ("yolo26m-seg-int8-b1", "yolo26m-seg-int8-b1-boxdecode", "yolo26-seg"),
    ],
)
def test_boxdecode_row_shares_package_with_its_default(
    tmp_path: Path, default_id: str, boxdecode_id: str, flag: str
) -> None:
    rows = {row.model_id: row for row in refresh_results.all_rows()}
    default_row, boxdecode_row = rows[default_id], rows[boxdecode_id]
    assert default_row.package == boxdecode_row.package
    assert default_row.decode_type is None

    model = tmp_path / boxdecode_row.package
    default_command = refresh_results.benchmark_command(
        default_row, model, 5, tmp_path / f"{default_id}.json"
    )
    boxdecode_command = refresh_results.benchmark_command(
        boxdecode_row, model, 5, tmp_path / f"{boxdecode_id}.json"
    )
    assert "--decode-type" not in default_command
    assert boxdecode_command == default_command[:-1] + [
        str(tmp_path / f"{boxdecode_id}.json"),
        "--decode-type",
        flag,
    ]


@pytest.mark.unit
def test_render_markdown_reports_each_route(tmp_path: Path) -> None:
    for row in refresh_results.all_rows():
        boxdecode = row.decode_type is not None
        (tmp_path / f"{row.model_id}.json").write_text(
            json.dumps(
                {
                    "benchmark": {"timestamp_utc": "2026-07-31T00:00:00+00:00"},
                    "model": {
                        "resolved_postprocess": "boxdecode" if boxdecode else "detessdequant",
                        "output_specs": ["o"] if boxdecode else ["o"] * 6,
                    },
                    "metrics": {"latency_ms": 1.0, "fps": 2.0},
                }
            ),
            encoding="utf-8",
        )

    markdown = refresh_results.render_markdown(7, "2.1.3", "2026-07-31", tmp_path)

    assert "| Model ID | Package | Used By | Postprocess | Outputs | Latency / FPS |" in markdown
    for model_id in ("yolo26m-det-int8-b1-boxdecode", "yolo26m-seg-int8-b1-boxdecode"):
        assert f"| `{model_id}` |" in markdown
    assert markdown.count("`boxdecode`") == 2
    assert markdown.count("`detessdequant`") == len(refresh_results.all_rows()) - 2
