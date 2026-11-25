"""E2E tests for pcb-defect-detector (Python)."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.output_assertions import assert_saved_frames_are_usable

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
# Mirrors main.OUTPUT_TAG; the app is run as a subprocess, not imported.
OUTPUT_TAG = "_pcb"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Annotated output keeps the source extension, so a JPEG board is re-encoded on
# write. That perturbs most pixels slightly, which a plain "any difference" test
# would mistake for drawing. Box outlines and labels instead replace pixels
# outright, so only a large per-channel delta counts as drawn.
ANNOTATION_DELTA = 60
MIN_ANNOTATED_PIXELS = 200


def classes_per_image(stdout: str) -> dict[str, set[str]]:
    """Map each image name to the classes its progress line reports.

    Progress lines read "[i/N] <image> [-> <output>] (<n> defects) {class: n, ...}".
    """
    lines = re.findall(
        r"^\[\d+/\d+\] (\S+)(?: -> \S+)? \(\d+ defects\) (\{[^}]*\})$", stdout, re.MULTILINE
    )
    return {name: set(re.findall(r"([a-z_]+):", counts)) for name, counts in lines}


def assert_each_board_reports_its_defect(fixtures: list[Path], stdout: str) -> None:
    """Each fixture is named after the defect it contains, so that board's own
    result must report it. Checking per image rather than the run's totals
    catches class outputs swapped or misattributed between boards."""
    named = {path.name: path.stem.split("_", 2)[2] for path in fixtures
             if path.stem.count("_") >= 2}
    # Without this the loop below asserts nothing and the test passes vacuously.
    # A fixture renamed to fewer than two underscores would otherwise drop out.
    assert named, "no defect class could be derived from the fixture names"
    reported = classes_per_image(stdout)
    for name, defect in sorted(named.items()):
        assert name in reported, f"no progress line for {name}:\n{stdout}"
        assert defect in reported[name], (
            f"{name} is named for {defect}, but its result reports "
            f"{sorted(reported[name])}:\n{stdout}"
        )


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline(
        self,
        e2e_model_path,
        apps_root,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        # PCB defects are not present in the shared COCO fixtures, so this
        # example uses its own test images instead of test_images_dir.
        input_dir = apps_root / "assets" / "datasets-test" / "pcb"
        skip_unless_e2e_ready(
            input_dir.is_dir() and any(input_dir.iterdir()),
            f"PCB test images are missing or empty: {input_dir}",
        )

        config_path = e2e_config_writer(
            {
                "input": {"source": str(input_dir)}, "output": {"dir": str(tmp_output_dir)},
            }
        )

        result = subprocess.run(
            [
                sys.executable, str(MAIN_PY),
                "--config", str(config_path),
            ],
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        expected = len([path for path in input_dir.iterdir() if path.is_file()])
        output_files = assert_saved_frames_are_usable(tmp_output_dir, expected)
        assert len(output_files) == expected, (
            f"Expected {expected} annotated images, found {len(output_files)}"
        )
        assert all(path.suffix.lower() in IMAGE_EXTENSIONS for path in output_files)

        # The checks above pass even when every detection list is empty, because
        # the application writes an annotated copy either way. Assert the actual
        # inspection result so a broken decoder, an incompatible model package or
        # a missing BBOX payload cannot pass as a clean board.
        stdout = result.stdout

        summary = re.search(r"images_with_defects=(\d+) total_defects=(\d+) failed=(\d+)", stdout)
        assert summary, f"summary line missing from stdout:\n{stdout}"
        images_with_defects, total_defects, failed = (int(g) for g in summary.groups())

        assert failed == 0, f"{failed} image(s) failed to process:\n{stdout}"
        assert images_with_defects == expected, (
            f"every bundled board carries defects, but only {images_with_defects}"
            f" of {expected} produced any:\n{stdout}"
        )
        assert total_defects >= expected, (
            f"expected at least one defect per board, got {total_defects}:\n{stdout}"
        )

        # This is what actually proves PCB detection works rather than that the
        # process merely ran.
        assert "Per-class totals:" in stdout, f"per-class totals missing from stdout:\n{stdout}"
        assert_each_board_reports_its_defect(
            [path for path in input_dir.iterdir() if path.is_file()], stdout
        )

        # Everything above reads stdout or file metadata. An application that
        # reported detections but wrote an untouched copy of its input would
        # still pass all of it -- exactly the "saves unchanged images" failure
        # this test exists to catch. Compare decoded pixels so the overlay has
        # to be present in the file that ships.
        import cv2
        import numpy as np

        for source in sorted(path for path in input_dir.iterdir() if path.is_file()):
            written = tmp_output_dir / f"{source.stem}{OUTPUT_TAG}{source.suffix}"
            assert written.is_file(), f"no annotated image written for {source.name}"
            before = cv2.imread(str(source))
            after = cv2.imread(str(written))
            assert before is not None, f"could not read source {source}"
            assert after is not None, f"could not read annotated output {written}"
            assert before.shape == after.shape, (
                f"{written.name} is {after.shape[1]}x{after.shape[0]} but its source is "
                f"{before.shape[1]}x{before.shape[0]}"
            )
            delta = np.abs(before.astype(np.int16) - after.astype(np.int16)).max(axis=2)
            changed = int(np.count_nonzero(delta > ANNOTATION_DELTA))
            assert changed >= MIN_ANNOTATED_PIXELS, (
                f"{written.name} has only {changed} strongly-changed pixel(s); the "
                f"board was reported as defective but the saved image carries no visible "
                f"overlay:\n{stdout}"
            )

    def test_mixed_resolutions_go_through_core_preprocessing(
        self, e2e_model_path, apps_root, tmp_output_dir, test_timeout_ms,
        skip_unless_e2e_ready, e2e_config_writer,
    ):
        """Core letterboxes and maps coordinates for input that is not 640x640.

        Every bundled fixture is exactly the model's own 640x640 input, so the
        letterbox is an identity operation. Each fixture is scaled *up* and then
        padded into a landscape, portrait, or square frame, so Core's on-device
        letterbox must resize and pad along either axis through the single
        graph seeded at the configured ingress capacity.

        The scaled boards keep their defects at full detail, so the same
        classes must still be reported. This does not check box positions: the
        application saves the source frame, so the output size only shows that
        the annotated image is that frame.
        """
        import cv2

        source_dir = apps_root / "assets" / "datasets-test" / "pcb"
        skip_unless_e2e_ready(
            source_dir.is_dir() and any(source_dir.iterdir()),
            f"PCB test images are missing or empty: {source_dir}",
        )

        # (scale, extra width, extra height) as fractions of the scaled board:
        # landscape, portrait, and square frames, all larger than the model
        # input, so every image needs a real letterbox through the single graph.
        frames = [(2.0, 0.25, 0.0), (1.5, 0.0, 0.25), (1.25, 0.0, 0.0)]
        mixed_dir = tmp_output_dir.parent / "mixed-input"
        mixed_dir.mkdir(parents=True, exist_ok=True)
        expected_sizes = {}
        fixtures = sorted(path for path in source_dir.iterdir() if path.is_file())
        for index, path in enumerate(fixtures):
            image = cv2.imread(str(path))
            assert image is not None, f"could not read fixture {path}"
            scale, extra_width, extra_height = frames[index % len(frames)]
            height, width = image.shape[:2]
            scaled = cv2.resize(
                image, (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
            resized = cv2.copyMakeBorder(
                scaled, 0, int(scaled.shape[0] * extra_height),
                0, int(scaled.shape[1] * extra_width),
                cv2.BORDER_CONSTANT, value=(114, 114, 114),
            )
            cv2.imwrite(str(mixed_dir / path.name), resized)
            expected_sizes[f"{path.stem}{OUTPUT_TAG}{path.suffix}"] = (
                resized.shape[1], resized.shape[0]
            )

        assert len(set(expected_sizes.values())) > 1, (
            "the rescaled fixtures must span more than one size or this test "
            "does not exercise mixed resolutions at all"
        )
        assert any(w > h for w, h in expected_sizes.values()) and any(
            w < h for w, h in expected_sizes.values()
        ), "the rescaled fixtures must include landscape and portrait frames"

        mixed_output = tmp_output_dir.parent / "mixed-output"
        mixed_output.mkdir(parents=True, exist_ok=True)
        config_path = e2e_config_writer({
            "input": {"source": str(mixed_dir)}, "output": {"dir": str(mixed_output)},
        })
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True, text=True, timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode == 0, (
            f"mixed-resolution input must run cleanly through Core's preprocessing\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        summary = re.search(
            r"images_with_defects=(\d+) total_defects=(\d+) failed=(\d+)", result.stdout,
        )
        assert summary, f"summary line missing from stdout:\n{result.stdout}"
        images_with_defects, total_defects, failed = (int(g) for g in summary.groups())
        assert failed == 0, (
            f"{failed} rescaled image(s) failed to process:\n{result.stdout}"
        )
        assert images_with_defects == len(fixtures), (
            f"every board still carries its defects after rescaling, but only "
            f"{images_with_defects} of {len(fixtures)} produced any:\n{result.stdout}"
        )
        assert total_defects >= len(fixtures), (
            f"expected at least one defect per board, got {total_defects}:\n{result.stdout}"
        )

        # Boxes are drawn on the source frame, so a correct mapping keeps the
        # annotated image at the source size. A 640x640 output here would mean
        # the letterboxed canvas leaked into the result.
        for name, (width, height) in expected_sizes.items():
            written = mixed_output / name
            assert written.is_file(), (
                f"no annotated image for {name}:\n{result.stdout}"
            )
            annotated = cv2.imread(str(written))
            assert annotated is not None, f"could not read annotated output {written}"
            assert (annotated.shape[1], annotated.shape[0]) == (width, height), (
                f"{name} was written at {annotated.shape[1]}x{annotated.shape[0]} but its "
                f"source is {width}x{height}; the output is not the source frame"
            )

        # Same content, only larger, so each board must still report its defect.
        assert_each_board_reports_its_defect(fixtures, result.stdout)

    def test_unreadable_input_reports_failure_and_exits_nonzero(
        self, e2e_model_path, apps_root, tmp_output_dir, test_timeout_ms,
        skip_unless_e2e_ready, e2e_config_writer,
    ):
        """An input that cannot be processed must not report success.

        Asserting failed == 0 on the happy path does not prove the failure path
        reports anything, so drive it with a file that passes the extension
        filter but decodes to nothing.
        """
        source_dir = apps_root / "assets" / "datasets-test" / "pcb"
        skip_unless_e2e_ready(
            source_dir.is_dir() and any(source_dir.iterdir()),
            f"PCB test images are missing or empty: {source_dir}",
        )

        broken_input = tmp_output_dir.parent / "broken-input"
        broken_input.mkdir(parents=True, exist_ok=True)
        for path in source_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, broken_input / path.name)
        (broken_input / "corrupt.jpg").write_bytes(b"not a jpeg")

        broken_output = tmp_output_dir.parent / "broken-output"
        broken_output.mkdir(parents=True, exist_ok=True)
        config_path = e2e_config_writer(
            {"input": {"source": str(broken_input)}, "output": {"dir": str(broken_output)}},
        )

        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True, text=True, timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode != 0, (
            f"an unreadable input must not exit 0\nstdout:\n{result.stdout}"
        )
        summary = re.search(r"failed=(\d+)", result.stdout)
        assert summary and int(summary.group(1)) >= 1, (
            f"an unreadable input must be counted in failed=\nstdout:\n{result.stdout}"
        )

    def test_min_score_of_one_reports_no_defects(
        self, e2e_model_path, apps_root, tmp_output_dir, test_timeout_ms,
        skip_unless_e2e_ready, e2e_config_writer,
    ):
        """inference.min_score takes effect only through Core's decode; nothing on the host
        filters the scores again. No score is above 1, so at 1.0 the run must report no defect,
        while Core's default threshold of 0 would report every box. This is also the
        no-detection case."""
        input_dir = apps_root / "assets" / "datasets-test" / "pcb"
        skip_unless_e2e_ready(
            input_dir.is_dir() and any(input_dir.iterdir()),
            f"PCB test images are missing or empty: {input_dir}",
        )

        config_path = e2e_config_writer({
            "input": {"source": str(input_dir)}, "output": {"dir": str(tmp_output_dir)},
            "inference": {"min_score": 1.0},
        })
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True, text=True, timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR), check=False,
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        summary = re.search(r"total_defects=(\d+)", result.stdout)
        assert summary, f"summary line missing from stdout:\n{result.stdout}"
        assert int(summary.group(1)) == 0, (
            f"inference.min_score: 1.0 must report total_defects=0:\n{result.stdout}"
        )
