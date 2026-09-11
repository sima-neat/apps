"""E2E tests for pcb-defect-detector (Python)."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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
                "io": {"input_dir": str(input_dir), "output_dir": str(tmp_output_dir)},
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
        output_files = [
            path
            for path in tmp_output_dir.iterdir()
            if path.is_file() and path.name != "config.yaml"
        ]
        assert len(output_files) == expected, (
            f"Expected {expected} annotated images, found {len(output_files)}"
        )
        assert all(path.stat().st_size > 0 for path in output_files)
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

        # Each fixture is named after the defect it contains, so the class the
        # model reports must include it. This is what actually proves PCB
        # detection works rather than that the process merely ran.
        per_class = re.search(r"Per-class totals: (\{.*\})", stdout)
        assert per_class, f"per-class totals missing from stdout:\n{stdout}"
        reported_classes = set(re.findall(r"([a-z_]+):", per_class.group(1)))

        expected_classes = {
            path.stem.split("_", 2)[2]
            for path in input_dir.iterdir()
            if path.is_file() and path.stem.count("_") >= 2
        }
        # Without this the set difference below asserts nothing at all and the
        # test passes vacuously -- exactly the failure mode this test exists to
        # prevent. A fixture renamed to fewer than two underscores would
        # otherwise drop out silently.
        assert expected_classes, (
            "no defect class could be derived from the fixture names in "
            f"{input_dir}; the class assertion would pass vacuously"
        )

        missing = expected_classes - reported_classes
        assert not missing, (
            f"fixtures are named for the defects they contain, but "
            f"{sorted(missing)} were never detected (reported: {sorted(reported_classes)}):\n{stdout}"
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
        letterbox is an identity operation and the happy-path test would pass
        even if the application did the geometry itself. Rescaling the fixtures
        to three different sizes exercises Core's on-device letterbox and its
        mapping of boxes back to source coordinates, through the single graph
        seeded at the configured ingress capacity.

        Each fixture is scaled *up*, so the defects are still present at full
        detail and the same classes must still be reported. A wrong mapping
        would place boxes in letterbox space instead of source space, and the
        annotated image would no longer match the source dimensions.
        """
        import cv2

        source_dir = apps_root / "assets" / "datasets-test" / "pcb"
        skip_unless_e2e_ready(
            source_dir.is_dir() and any(source_dir.iterdir()),
            f"PCB test images are missing or empty: {source_dir}",
        )

        # Distinct sizes, all larger than the model input, so every image needs a
        # real letterbox through the single graph.
        scales = [2.0, 1.5, 1.25]
        mixed_dir = tmp_output_dir.parent / "mixed-input"
        mixed_dir.mkdir(parents=True, exist_ok=True)
        expected_sizes = {}
        fixtures = sorted(path for path in source_dir.iterdir() if path.is_file())
        for index, path in enumerate(fixtures):
            image = cv2.imread(str(path))
            assert image is not None, f"could not read fixture {path}"
            scale = scales[index % len(scales)]
            height, width = image.shape[:2]
            resized = cv2.resize(
                image, (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
            cv2.imwrite(str(mixed_dir / path.name), resized)
            expected_sizes[f"{path.stem}{OUTPUT_TAG}{path.suffix}"] = (
                resized.shape[1], resized.shape[0]
            )

        assert len(set(expected_sizes.values())) > 1, (
            "the rescaled fixtures must span more than one size or this test "
            "does not exercise mixed resolutions at all"
        )

        mixed_output = tmp_output_dir.parent / "mixed-output"
        mixed_output.mkdir(parents=True, exist_ok=True)
        config_path = e2e_config_writer({
            "io": {"input_dir": str(mixed_dir), "output_dir": str(mixed_output)},
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
                f"source is {width}x{height}; detections were not mapped back to source "
                f"coordinates"
            )

        # Same content, only larger, so the model must still name the same defects.
        per_class = re.search(r"Per-class totals: (\{.*\})", result.stdout)
        assert per_class, f"per-class totals missing from stdout:\n{result.stdout}"
        reported_classes = set(re.findall(r"([a-z_]+):", per_class.group(1)))
        expected_classes = {
            path.stem.split("_", 2)[2] for path in fixtures if path.stem.count("_") >= 2
        }
        assert expected_classes, (
            f"no defect class could be derived from the fixture names in {source_dir}; "
            f"the class assertion would pass vacuously"
        )
        missing = expected_classes - reported_classes
        assert not missing, (
            f"after rescaling, {sorted(missing)} were no longer detected "
            f"(reported: {sorted(reported_classes)}); Core's letterbox or its "
            f"coordinate mapping is not handling non-native input sizes:\n{result.stdout}"
        )

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
            {"io": {"input_dir": str(broken_input), "output_dir": str(broken_output)}},
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

    def test_unwritable_output_reports_failure_and_exits_nonzero(
        self, e2e_model_path, apps_root, tmp_output_dir, test_timeout_ms,
        skip_unless_e2e_ready, e2e_config_writer,
    ):
        """A save that fails must be reported, not counted as a processed image.

        The application checks the image writer's return value; nothing else in
        this suite drives that branch.
        """
        input_dir = apps_root / "assets" / "datasets-test" / "pcb"
        skip_unless_e2e_ready(
            input_dir.is_dir() and any(input_dir.iterdir()),
            f"PCB test images are missing or empty: {input_dir}",
        )

        readonly = tmp_output_dir.parent / "readonly-output"
        readonly.mkdir(parents=True, exist_ok=True)
        # Written before the directory is sealed; the config itself does not live
        # in the output directory.
        config_path = e2e_config_writer({
            "io": {"input_dir": str(input_dir), "output_dir": str(readonly)},
        })
        readonly.chmod(0o555)
        try:
            # root ignores the mode bits, which would let every assertion below
            # pass without a single write ever failing.
            probe = readonly / ".probe"
            try:
                probe.write_text("x")
                probe.unlink()
                sealed = False
            except OSError:
                sealed = True
            skip_unless_e2e_ready(
                sealed,
                f"{readonly} is still writable, so the write-failure path cannot be "
                f"exercised (running as root?)",
            )

            result = subprocess.run(
                [sys.executable, str(MAIN_PY), "--config", str(config_path)],
                capture_output=True, text=True, timeout=test_timeout_ms / 1000,
                cwd=str(EXAMPLE_DIR),
            )
        finally:
            readonly.chmod(0o755)

        assert result.returncode != 0, (
            f"images that could not be saved must not exit 0\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        summary = re.search(r"Done: (\d+)/(\d+) images .*?failed=(\d+)", result.stdout)
        assert summary, f"no summary line found\nstdout:\n{result.stdout}"
        processed, total, failed = (int(value) for value in summary.groups())
        assert failed == total and processed == 0, (
            f"no image could be saved, so all {total} must be counted as failed and none "
            f"as processed; got {processed} processed, {failed} failed\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_timeout_aborts_the_batch(
        self, e2e_model_path, apps_root, tmp_output_dir, test_timeout_ms,
        skip_unless_e2e_ready, e2e_config_writer,
    ):
        """A timeout stops the run instead of being attributed to another image.

        A frame that times out is still in the pipeline, so continuing would hand
        its result to the next image's pull. A 1 ms timeout is far below the
        ~200 ms inference, so nothing can complete and the run must abort with no
        image reported as processed.
        """
        input_dir = apps_root / "assets" / "datasets-test" / "pcb"
        skip_unless_e2e_ready(
            input_dir.is_dir() and any(input_dir.iterdir()),
            f"PCB test images are missing or empty: {input_dir}",
        )

        config_path = e2e_config_writer({
            "io": {"input_dir": str(input_dir), "output_dir": str(tmp_output_dir)},
            "runtime": {"timeout_ms": 1},
        })
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True, text=True, timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode != 0, (
            f"a run where inference timed out must not exit 0\nstdout:\n{result.stdout}"
        )
        assert "timeout" in result.stderr.lower(), (
            f"the timeout must be named on stderr\nstderr:\n{result.stderr}"
        )
        progressed = re.search(r"Aborted after (\d+)/", result.stderr)
        assert progressed and int(progressed.group(1)) == 0, (
            f"no image can complete under a 1 ms timeout, so none may be reported as "
            f"processed\nstderr:\n{result.stderr}"
        )

    def test_missing_input_directory_fails(self, apps_root, tmp_output_dir, e2e_config_writer,
                                           test_timeout_ms, skip_unless_e2e_ready):
        """A batch that cannot run must report failure, not success."""
        config_path = e2e_config_writer({
            "io": {
                "input_dir": str(tmp_output_dir / "does-not-exist"),
                "output_dir": str(tmp_output_dir),
            },
        })
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True, text=True, timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )
        assert result.returncode != 0, (
            f"a missing input directory must not exit 0\nstdout:\n{result.stdout}"
        )
