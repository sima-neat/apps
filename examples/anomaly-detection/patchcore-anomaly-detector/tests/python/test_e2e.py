"""E2E test for the PatchCore example (Python), hardware-gated per Apps
conventions. Runs `--calibrate` against the bundled nominal set, then scores
the bundled test images and checks verdicts/overlays."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
APPS_ROOT = EXAMPLE_DIR.parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
REAL_IMAGES_DIR = APPS_ROOT / "assets" / "datasets" / "patchcore" / "images"
NOMINAL_DIR = APPS_ROOT / "assets" / "datasets" / "patchcore" / "nominal"
HELD_OUT_NORMAL_DIR = APPS_ROOT / "assets" / "datasets" / "patchcore" / "held_out_normal"
NOMINAL_IMAGE = "plain_0.png"
DEFECT_IMAGE = "scratch_0.png"

SCORE_RE = re.compile(r"^(?P<path>.+): score=(?P<score>[-\d.]+) threshold=(?P<threshold>[-\d.]+) "
                      r"verdict=(?P<verdict>\w+)", re.MULTILINE)


def _find_cpp_binary() -> Path | None:
    """Resolve the patchcore C++ binary for the cross-language regression
    test below (no shared fixture for this -- the only test here needing it)."""
    raw = os.environ.get("SIMANEAT_APPS_TEST_CPP_BINARY", "").strip()
    if raw:
        return Path(raw)
    candidates = (
        # Installed/packaged layout (CI, `sima-cli neat install`).
        EXAMPLE_DIR / "src" / "cpp" / "pre-built" / "patchcore-anomaly-detector",
        # Raw monorepo CMake build tree (local/DevKit development).
        APPS_ROOT / "build" / "examples" / "anomaly-detection" / "patchcore-anomaly-detector"
        / "patchcore-anomaly-detector",
    )
    return next((c for c in candidates if c.is_file()), None)


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

    def test_partial_write_failure_fails_the_run(
        self,
        e2e_model_path,
        tmp_output_dir,
        test_images_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        """A run that writes some overlays and fails to write others must
        exit nonzero -- succeeding as long as at least one write went through
        would silently under-report incomplete output."""
        images = sorted(p for p in test_images_dir.iterdir() if p.is_file()) \
            if test_images_dir.exists() else []
        skip_unless_e2e_ready(len(images) >= 2, "test_images_dir needs at least 2 images")

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

        # Pre-create one output path as read-only so its overlay write fails
        # while every other image's write should still succeed.
        blocked_name = images[0].name
        blocked_path = tmp_output_dir / blocked_name
        blocked_path.write_bytes(b"")
        blocked_path.chmod(0o444)
        try:
            score = subprocess.run(
                [sys.executable, str(MAIN_PY), "--config", str(config_path)],
                capture_output=True, text=True, timeout=timeout_s, cwd=str(EXAMPLE_DIR),
            )
        finally:
            blocked_path.chmod(0o644)

        assert score.returncode != 0, (
            f"expected a nonzero exit code for a partial write failure, got 0\n"
            f"stdout:\n{score.stdout}\nstderr:\n{score.stderr}"
        )
        assert blocked_name in score.stderr, (
            f"expected the failed path ({blocked_name}) named in stderr:\n{score.stderr}"
        )
        other_outputs = [
            path for path in tmp_output_dir.iterdir()
            if path.is_file() and path.name not in ("config.yaml", blocked_name)
        ]
        assert other_outputs, "the one blocked write should not have stopped the rest from running"

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

    def test_held_out_normal_passes_and_defect_fails(
        self,
        e2e_model_path,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        """Scores every held-out normal image and every defect image, and
        asserts the actual pass/fail verdict on each -- not just relative
        ordering, and not just one cherry-picked passing image."""
        skip_unless_e2e_ready(
            NOMINAL_DIR.is_dir() and any(NOMINAL_DIR.iterdir()),
            f"nominal calibration set missing under {NOMINAL_DIR}",
        )
        held_out_paths = sorted(HELD_OUT_NORMAL_DIR.glob("*.png")) if HELD_OUT_NORMAL_DIR.is_dir() else []
        defect_paths = sorted(p for p in REAL_IMAGES_DIR.glob("scratch_*.png"))
        skip_unless_e2e_ready(
            bool(held_out_paths) and bool(defect_paths),
            f"held-out normal or defect images missing ({HELD_OUT_NORMAL_DIR}, {REAL_IMAGES_DIR})",
        )

        score_dir = tmp_output_dir.parent / "score_inputs"
        score_dir.mkdir(parents=True, exist_ok=True)
        for path in held_out_paths + defect_paths:
            (score_dir / path.name).write_bytes(path.read_bytes())

        bank_path = tmp_output_dir.parent / "memory_bank.npy"
        meta_path = tmp_output_dir.parent / "bank_meta.json"
        config_path = e2e_config_writer(
            {
                "source": {"type": "image_dir", "image_dir": str(score_dir)},
                "calibration": {
                    "nominal_images_dir": str(NOMINAL_DIR),
                    "threshold_images_dir": str(NOMINAL_DIR),
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

        score = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True, text=True, timeout=timeout_s, cwd=str(EXAMPLE_DIR),
        )
        assert score.returncode == 0, (
            f"main.py exited with code {score.returncode}\n"
            f"stdout:\n{score.stdout}\nstderr:\n{score.stderr}"
        )

        verdicts = {
            Path(m.group("path")).name: (m.group("verdict"), m.group("score"))
            for m in SCORE_RE.finditer(score.stdout)
        }

        held_out_names = [p.name for p in held_out_paths]
        defect_names = [p.name for p in defect_paths]
        for name in held_out_names + defect_names:
            assert name in verdicts, f"no verdict for {name} in stdout:\n{score.stdout}"

        # Report every image's outcome (not just the first failure) so a
        # regression here shows the full pass/fail picture in one run.
        held_out_failures = [
            f"{name} scored {verdicts[name][1]} (ANOMALOUS)"
            for name in held_out_names
            if verdicts[name][0] != "normal"
        ]
        defect_failures = [
            f"{name} scored {verdicts[name][1]} (normal)"
            for name in defect_names
            if verdicts[name][0] != "ANOMALOUS"
        ]
        assert not held_out_failures, (
            f"{len(held_out_failures)}/{len(held_out_names)} held-out normal images were "
            f"flagged anomalous: {held_out_failures}"
        )
        assert not defect_failures, (
            f"{len(defect_failures)}/{len(defect_names)} defect images were not flagged "
            f"anomalous: {defect_failures}"
        )

    def test_cpp_built_bank_scores_correctly_in_python(
        self,
        e2e_model_path,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        """Cross-language regression: a memory bank calibrated by the C++
        implementation must load and separate defect from nominal correctly
        when scored by Python. Both implementations share the same on-disk
        bank/meta format and previously used a numerically unstable distance
        formula that could disagree near ties between the two languages' own
        accumulation orders (see patchcore_scoring.py's _pairwise_l2
        docstring) -- this proves the fix holds across a bank the other
        language actually built, not just within one language's own round
        trip."""
        cpp_binary = _find_cpp_binary()
        skip_unless_e2e_ready(
            cpp_binary is not None,
            "C++ patchcore binary not found (set SIMANEAT_APPS_TEST_CPP_BINARY or build it under build/)",
        )
        nominal_path = REAL_IMAGES_DIR / NOMINAL_IMAGE
        defect_path = REAL_IMAGES_DIR / DEFECT_IMAGE
        skip_unless_e2e_ready(
            nominal_path.is_file() and defect_path.is_file(),
            f"real nominal/defect images missing under {REAL_IMAGES_DIR}",
        )

        nominal_only_dir = tmp_output_dir.parent / "nominal"
        nominal_only_dir.mkdir(parents=True, exist_ok=True)
        (nominal_only_dir / NOMINAL_IMAGE).write_bytes(nominal_path.read_bytes())

        score_dir = tmp_output_dir.parent / "score_inputs"
        score_dir.mkdir(parents=True, exist_ok=True)
        (score_dir / NOMINAL_IMAGE).write_bytes(nominal_path.read_bytes())
        (score_dir / DEFECT_IMAGE).write_bytes(defect_path.read_bytes())

        bank_path = tmp_output_dir.parent / "memory_bank.npy"
        meta_path = tmp_output_dir.parent / "bank_meta.json"
        config_path = e2e_config_writer(
            {
                "source": {"type": "image_dir", "image_dir": str(score_dir)},
                "calibration": {
                    "nominal_images_dir": str(nominal_only_dir),
                    "threshold_images_dir": str(nominal_only_dir),
                },
                "memory_bank": {"path": str(bank_path), "meta_path": str(meta_path)},
                "output": {"dir": str(tmp_output_dir)},
            }
        )
        timeout_s = test_timeout_ms / 1000

        calibrate = subprocess.run(
            [str(cpp_binary), "--calibrate", "--config", str(config_path)],
            capture_output=True, text=True, timeout=timeout_s, cwd=str(APPS_ROOT),
        )
        assert calibrate.returncode == 0, (
            f"C++ --calibrate exited with code {calibrate.returncode}\n"
            f"stdout:\n{calibrate.stdout}\nstderr:\n{calibrate.stderr}"
        )

        py_score = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True, text=True, timeout=timeout_s, cwd=str(EXAMPLE_DIR),
        )
        assert py_score.returncode == 0, (
            f"Python scoring against a C++-built bank exited with code {py_score.returncode}\n"
            f"stdout:\n{py_score.stdout}\nstderr:\n{py_score.stderr}"
        )

        # Also score the same bank with the C++ binary that built it, so
        # this compares actual scores/verdicts across languages -- not just
        # that each language's own score ordering looks sensible.
        cpp_score = subprocess.run(
            [str(cpp_binary), "--config", str(config_path)],
            capture_output=True, text=True, timeout=timeout_s, cwd=str(APPS_ROOT),
        )
        assert cpp_score.returncode == 0, (
            f"C++ scoring its own bank exited with code {cpp_score.returncode}\n"
            f"stdout:\n{cpp_score.stdout}\nstderr:\n{cpp_score.stderr}"
        )

        py_scores = {
            Path(m.group("path")).name: (m.group("score"), m.group("verdict"))
            for m in SCORE_RE.finditer(py_score.stdout)
        }
        cpp_scores = {
            Path(m.group("path")).name: (m.group("score"), m.group("verdict"))
            for m in SCORE_RE.finditer(cpp_score.stdout)
        }
        for image in (NOMINAL_IMAGE, DEFECT_IMAGE):
            assert image in py_scores, f"Python: no score for {image} in:\n{py_score.stdout}"
            assert image in cpp_scores, f"C++: no score for {image} in:\n{cpp_score.stdout}"
            py_value, py_verdict = float(py_scores[image][0]), py_scores[image][1]
            cpp_value, cpp_verdict = float(cpp_scores[image][0]), cpp_scores[image][1]
            assert py_verdict == cpp_verdict, (
                f"{image}: Python verdict={py_verdict} but C++ verdict={cpp_verdict} "
                f"(scores {py_value:.6f} vs {cpp_value:.6f})"
            )
            assert abs(py_value - cpp_value) < 0.01, (
                f"{image}: Python and C++ scores disagree by more than expected floating-point "
                f"noise: {py_value:.6f} vs {cpp_value:.6f}"
            )

        assert float(py_scores[DEFECT_IMAGE][0]) > float(py_scores[NOMINAL_IMAGE][0]), (
            "Python scoring a C++-built bank must still separate defect from nominal: "
            f"nominal={py_scores[NOMINAL_IMAGE][0]} defect={py_scores[DEFECT_IMAGE][0]}"
        )
