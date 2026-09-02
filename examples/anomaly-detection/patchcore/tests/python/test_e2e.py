"""E2E test for the PatchCore example (Python), hardware-gated per Apps conventions.

Runs `--calibrate` against the bundled test image set to produce a fresh
memory bank and bank_meta.json, then runs the scoring pipeline against the
same images and checks it produces annotated overlay output. The bundled
COCO test images are not a real defect-free inspection set -- this only
proves the pipeline runs end to end on real hardware, not that the resulting
scores are meaningful; see the README for how to calibrate against a real
nominal image set.
"""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


@pytest.mark.e2e
class TestE2E:
    def test_calibrate_then_score(
        self,
        e2e_model_path,
        tmp_output_dir,
        test_images_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        skip_unless_e2e_ready(
            test_images_dir.exists() and any(test_images_dir.iterdir()),
            "test_images_dir is missing or empty",
        )

        bank_path = tmp_output_dir.parent / "memory_bank.npy"
        meta_path = tmp_output_dir.parent / "bank_meta.json"
        config_path = e2e_config_writer(
            {
                "source": {"type": "image_dir", "image_dir": str(test_images_dir)},
                "calibration": {
                    "nominal_images_dir": str(test_images_dir),
                    "threshold_images_dir": str(test_images_dir),
                },
                "memory_bank": {"path": str(bank_path), "meta_path": str(meta_path)},
                "output": {"dir": str(tmp_output_dir)},
            }
        )
        timeout_s = test_timeout_ms / 1000

        calibrate = subprocess.run(
            [sys.executable, str(MAIN_PY), "--calibrate", "--config", str(config_path)],
            capture_output=True, text=True, timeout=timeout_s, cwd=str(EXAMPLE_DIR),
        )
        assert calibrate.returncode == 0, (
            f"--calibrate exited with code {calibrate.returncode}\n"
            f"stdout:\n{calibrate.stdout}\nstderr:\n{calibrate.stderr}"
        )
        assert bank_path.is_file() and bank_path.stat().st_size > 0
        assert meta_path.is_file() and meta_path.stat().st_size > 0

        score = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True, text=True, timeout=timeout_s, cwd=str(EXAMPLE_DIR),
        )
        assert score.returncode == 0, (
            f"main.py exited with code {score.returncode}\n"
            f"stdout:\n{score.stdout}\nstderr:\n{score.stderr}"
        )

        output_files = [
            path for path in tmp_output_dir.iterdir()
            if path.is_file() and path.name != "config.yaml"
        ]
        assert output_files, "Expected overlay output files but output directory is empty"
        for f in output_files:
            assert f.stat().st_size > 0, f"Output file is empty: {f.name}"

    def test_bank_model_mismatch_fails_at_load(
        self,
        e2e_model_path,
        tmp_output_dir,
        test_images_dir,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        """A bank_meta.json pinned to a different model hash must fail at load,
        not silently score against a mismatched bank."""
        skip_unless_e2e_ready(
            test_images_dir.exists() and any(test_images_dir.iterdir()),
            "test_images_dir is missing or empty",
        )

        bank_path = tmp_output_dir.parent / "memory_bank.npy"
        meta_path = tmp_output_dir.parent / "bank_meta.json"
        config_path = e2e_config_writer(
            {
                "source": {"type": "image_dir", "image_dir": str(test_images_dir)},
                "memory_bank": {"path": str(bank_path), "meta_path": str(meta_path)},
                "output": {"dir": str(tmp_output_dir)},
            }
        )

        import numpy as np

        np.save(bank_path, np.zeros((4, 1536), dtype=np.float32))
        meta_path.write_text(
            '{"model_sha256": "' + "0" * 64 + '", "threshold": {"value": 1.0}}',
            encoding="utf-8",
        )

        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True, text=True, timeout=30, cwd=str(EXAMPLE_DIR),
        )
        assert result.returncode != 0
        assert "different model package" in result.stderr
