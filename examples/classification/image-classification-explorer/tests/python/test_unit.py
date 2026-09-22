"""Focused unit tests for image-classification-explorer (Python)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"

SPEC = importlib.util.spec_from_file_location("image_classification_explorer_main", MAIN_PY)
main = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = main  # dataclasses need the module registered before exec
SPEC.loader.exec_module(main)


def _make_image(path, color=(0, 0, 255), size=32):
    import cv2
    import numpy as np

    img = np.full((size, size, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_fake_pyneat(num_classes=5, fail_message=None, fail_run_for_paths=None):
    """A minimal in-process stand-in for the real pyneat SDK.

    This lets error paths that otherwise only run on real hardware (missing
    model archives, corrupted images, unsupported formats) be exercised
    automatically in CI without a devkit. Model.run() returns deterministic,
    seeded-random scores so results are reproducible across test runs.

    fail_message: every Model(path, opt) construction raises (simulates a bad
    model archive, e.g. a missing file).
    fail_run_for_paths: Model.run() raises only when the model was built from
    one of these paths, so other models on the same image still succeed
    (simulates one model failing inference while others are fine).
    """
    fail_run_for_paths = fail_run_for_paths or ()
    import hashlib
    import types

    import numpy as np

    mod = types.ModuleType("pyneat")
    # Live-instance accounting so tests can prove models are released one at a
    # time (CPython refcounting makes __del__ run as soon as the last ref drops).
    mod.live_models = 0
    mod.max_live_models = 0

    class InputKind:
        Image = "image"

    class PreprocessColorFormat:
        RGB = "rgb"

    class NormalizePreset:
        ImageNet = "imagenet"

    class PixelFormat:
        RGB = "rgb"

    class TensorDType:
        UInt8, Int8, UInt16, Int16, Int32, Float32, Float64 = range(7)

    class _Tensor:
        def __init__(self, arr):
            self._arr = arr
            self.dtype = TensorDType.Float32
            self.shape = arr.shape

        @staticmethod
        def from_numpy(arr, copy=True, image_format=None):
            return _Tensor(arr)

        def copy_dense_bytes_tight(self):
            return self._arr.astype(np.float32).tobytes()

    class ModelOptions:
        def __init__(self):
            self.preprocess = types.SimpleNamespace(
                kind=None,
                color_convert=types.SimpleNamespace(input_format=None),
                input_max_width=0,
                input_max_height=0,
                input_max_depth=0,
                preset=None,
            )

    class Model:
        def __init__(self, path, opt):
            if fail_message is not None:
                raise RuntimeError(fail_message)
            self.path = path
            mod.live_models += 1
            mod.max_live_models = max(mod.max_live_models, mod.live_models)

        def __del__(self):
            mod.live_models -= 1

        def run(self, tensors, timeout_ms=0):
            if self.path in fail_run_for_paths:
                raise RuntimeError(f"simulated inference failure for {self.path}")
            # hashlib (not the builtin hash()) so results are reproducible across
            # processes: hash() is salted per-process by PYTHONHASHSEED.
            material = self.path.encode() + tensors[0]._arr.tobytes()[:16]
            seed = int(hashlib.sha256(material).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            return [_Tensor(rng.normal(size=num_classes).astype(np.float32))]

    mod.InputKind = InputKind
    mod.PreprocessColorFormat = PreprocessColorFormat
    mod.NormalizePreset = NormalizePreset
    mod.PixelFormat = PixelFormat
    mod.TensorDType = TensorDType
    mod.ModelOptions = ModelOptions
    mod.Model = Model
    mod.Tensor = _Tensor
    return mod


def _write_config(path, input_path, output_dir, model_path="fake_model.tar.gz",
                  extensions=".jpg,.jpeg,.png,.bmp", num_classes=5):
    path.write_text(f"""
io:
  input: {input_path}
  output_dir: {output_dir}
  extensions: {extensions}
models:
  m:
    path: {model_path}
    num_classes: {num_classes}
    top_k: 3
""")


class TestDiscoverImages:
    def test_missing_input_uses_fallback(self, tmp_path):
        dest = tmp_path / "fallback.jpeg"
        url = "http://example.invalid/x.jpg"
        dest.write_bytes(b"fake")
        dest.with_name(f"{dest.name}.source-url").write_text(url)
        images, skipped = main.discover_images(None, (".jpg",), url, dest)
        assert images == [dest]
        assert skipped == []

    def test_fallback_refreshes_when_url_changes(self, tmp_path, monkeypatch):
        dest = tmp_path / "fallback.jpeg"
        old_url = "https://example.test/old.jpeg"
        new_url = "https://example.test/new.jpeg"
        _make_image(dest, color=(0, 0, 255))
        old_bytes = dest.read_bytes()
        source_path = dest.with_name(f"{dest.name}.source-url")
        source_path.write_text(old_url)
        downloaded = []

        def _download(url, target):
            downloaded.append(url)
            # urlretrieve writes to a ".tmp" path; render a real JPEG elsewhere
            # and copy its bytes there since cv2.imwrite keys off the extension.
            rendered = tmp_path / "rendered.jpeg"
            _make_image(rendered, color=(0, 255, 0))
            Path(target).write_bytes(rendered.read_bytes())

        monkeypatch.setattr(main.urllib.request, "urlretrieve", _download)

        assert main.download_image(new_url, dest) == dest
        assert downloaded == [new_url]
        assert dest.read_bytes() != old_bytes
        assert main.is_decodable_image(dest)
        assert source_path.read_text() == new_url

    def test_single_file(self, tmp_path):
        img = tmp_path / "a.jpg"
        _make_image(img)
        images, skipped = main.discover_images(str(img), (".jpg",), "", tmp_path / "fb.jpg")
        assert images == [img]
        assert skipped == []

    def test_single_file_unsupported_extension(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("nope")
        images, skipped = main.discover_images(str(f), (".jpg",), "", tmp_path / "fb.jpg")
        assert images == []
        assert len(skipped) == 1

    def test_directory_deterministic_order_and_skips(self, tmp_path):
        _make_image(tmp_path / "b.jpg")
        _make_image(tmp_path / "a.jpg")
        (tmp_path / "notes.txt").write_text("skip me")
        images, skipped = main.discover_images(str(tmp_path), (".jpg",), "", tmp_path / "fb.jpg")
        assert [p.name for p in images] == ["a.jpg", "b.jpg"]
        assert len(skipped) == 1
        assert "notes.txt" in skipped[0]

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main.discover_images(str(tmp_path / "nope"), (".jpg",), "", tmp_path / "fb.jpg")

    def test_unreadable_directory_fails_cleanly(self, tmp_path, monkeypatch, capsys):
        """Regression: a directory that exists but cannot be enumerated must
        surface as the concise input error (exit 3), not a traceback."""
        monkeypatch.setitem(sys.modules, "pyneat", _make_fake_pyneat())
        input_dir = tmp_path / "locked"
        input_dir.mkdir()

        def _denied(self):
            raise PermissionError(13, "Permission denied", str(self))

        monkeypatch.setattr(Path, "iterdir", _denied)
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, input_dir, tmp_path / "out")
        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_path)])

        rc = main.main()

        assert rc == 3
        assert "Permission denied" in capsys.readouterr().err


class TestLoadProfiles:
    def test_requires_models_section(self):
        with pytest.raises(ValueError, match="models"):
            main.load_profiles({})

    def test_requires_path(self):
        with pytest.raises(ValueError, match="path"):
            main.load_profiles({"models": {"a": {}}})

    def test_builds_profile_list_in_order(self):
        raw = {
            "models": {
                "resnet_50": {"path": "models/r50.tar.gz", "top_k": 3},
                "resnet_18": {"path": "models/r18.tar.gz"},
            }
        }
        profiles = main.load_profiles(raw)
        assert [p.name for p in profiles] == ["resnet_50", "resnet_18"]
        assert profiles[0].top_k == 3
        assert profiles[1].top_k == 5  # default
        assert profiles[0].output == "softmax"  # default

    def test_rejects_dotted_profile_name(self):
        raw = {"models": {"resnet.v2": {"path": "m.tar.gz"}}}
        with pytest.raises(ValueError, match="must not contain '.'"):
            main.load_profiles(raw)

    def test_rejects_unsupported_output_interpretation(self):
        raw = {"models": {"a": {"path": "m.tar.gz", "output": "raw_logits"}}}
        with pytest.raises(ValueError, match="output"):
            main.load_profiles(raw)

    @pytest.mark.parametrize("top_k", [0, -1, -5])
    def test_rejects_nonpositive_top_k(self, top_k):
        raw = {"models": {"a": {"path": "m.tar.gz", "top_k": top_k}}}
        with pytest.raises(ValueError, match="top_k"):
            main.load_profiles(raw)

    @pytest.mark.parametrize("num_classes", [0, -1])
    def test_rejects_nonpositive_num_classes(self, num_classes):
        raw = {"models": {"a": {"path": "m.tar.gz", "num_classes": num_classes}}}
        with pytest.raises(ValueError, match="num_classes"):
            main.load_profiles(raw)

    @pytest.mark.parametrize(("width", "height"), [(0, 224), (224, 0), (-1, 224)])
    def test_rejects_nonpositive_dimensions(self, width, height):
        raw = {
            "models": {"a": {"path": "m.tar.gz", "input_width": width, "input_height": height}}
        }
        with pytest.raises(ValueError, match="input_width/input_height"):
            main.load_profiles(raw)


class TestLoadLabelMap:
    def test_default_numeric_labels(self):
        assert main.load_label_map(None, 3) == ["0", "1", "2"]

    def test_custom_file(self, tmp_path):
        label_file = tmp_path / "labels.txt"
        label_file.write_text("cat\ndog\nbird\n")
        assert main.load_label_map(str(label_file), 3) == ["cat", "dog", "bird"]

    def test_too_few_labels_raises(self, tmp_path):
        label_file = tmp_path / "labels.txt"
        label_file.write_text("cat\n")
        with pytest.raises(ValueError):
            main.load_label_map(str(label_file), 3)

    def test_missing_file_raises_value_error_not_os_error(self, tmp_path):
        """Regression: a missing label_map path must surface as ValueError (the
        type main() catches for config problems), not an uncaught OSError."""
        missing = tmp_path / "does_not_exist" / "labels.txt"
        with pytest.raises(ValueError, match="failed to open label map"):
            main.load_label_map(str(missing), 3)


class TestDownloadImage:
    def test_network_failure_raises_file_not_found_error(self, tmp_path, monkeypatch):
        """Regression: a download failure must surface as FileNotFoundError (the
        type main() catches for input problems), not an uncaught URLError."""
        import urllib.error

        def _raise(*args, **kwargs):
            raise urllib.error.URLError("simulated network failure")

        monkeypatch.setattr(main.urllib.request, "urlretrieve", _raise)
        dest = tmp_path / "fallback.jpg"
        with pytest.raises(FileNotFoundError, match="failed to download"):
            main.download_image("http://example.invalid/x.jpg", dest)

    def test_undecodable_download_is_not_cached(self, tmp_path, monkeypatch):
        """Regression: an HTTP 200 response whose body is not an image (e.g. a
        proxy error page) must fail the run instead of being cached and reused
        by every later fallback run."""
        url = "http://example.invalid/x.jpg"

        def _download(_url, target):
            Path(target).write_bytes(b"<html>not an image</html>")

        monkeypatch.setattr(main.urllib.request, "urlretrieve", _download)
        dest = tmp_path / "fallback.jpg"
        source_path = dest.with_name(f"{dest.name}.source-url")
        with pytest.raises(FileNotFoundError, match="not a decodable image"):
            main.download_image(url, dest)
        assert not dest.exists()
        assert not source_path.exists()
        assert not dest.with_name(f"{dest.name}.tmp").exists()

    def test_undecodable_download_keeps_previous_cache(self, tmp_path, monkeypatch):
        """A failed refresh must leave the previously cached image and its
        recorded source URL untouched."""
        old_url = "https://example.test/old.jpeg"
        new_url = "https://example.test/new.jpeg"
        dest = tmp_path / "fallback.jpg"
        _make_image(dest)
        old_bytes = dest.read_bytes()
        source_path = dest.with_name(f"{dest.name}.source-url")
        source_path.write_text(old_url)

        def _download(_url, target):
            Path(target).write_bytes(b"<html>not an image</html>")

        monkeypatch.setattr(main.urllib.request, "urlretrieve", _download)
        with pytest.raises(FileNotFoundError, match="not a decodable image"):
            main.download_image(new_url, dest)
        assert dest.read_bytes() == old_bytes
        assert source_path.read_text() == old_url


class TestAgreement:
    def test_none_with_single_model(self):
        result = main.ImageResult(image_path=Path("x.jpg"))
        result.predictions["a"] = main.Prediction(top_k=[(1, "cat", 0.9)], inference_ms=1.0)
        assert main.agreement(result, ["a"]) is None

    def test_true_when_top1_matches(self):
        result = main.ImageResult(image_path=Path("x.jpg"))
        result.predictions["a"] = main.Prediction(top_k=[(1, "cat", 0.9)], inference_ms=1.0)
        result.predictions["b"] = main.Prediction(top_k=[(1, "cat", 0.7)], inference_ms=1.0)
        assert main.agreement(result, ["a", "b"]) is True

    def test_false_when_top1_differs(self):
        result = main.ImageResult(image_path=Path("x.jpg"))
        result.predictions["a"] = main.Prediction(top_k=[(1, "cat", 0.9)], inference_ms=1.0)
        result.predictions["b"] = main.Prediction(top_k=[(2, "dog", 0.7)], inference_ms=1.0)
        assert main.agreement(result, ["a", "b"]) is False

    def test_none_when_one_of_three_selected_models_has_no_result(self):
        """Regression: two of three selected models agreeing must not be reported
        as agreement while the third selected model has no result at all."""
        result = main.ImageResult(image_path=Path("x.jpg"))
        result.predictions["a"] = main.Prediction(top_k=[(1, "cat", 0.9)], inference_ms=1.0)
        result.predictions["b"] = main.Prediction(top_k=[(1, "cat", 0.8)], inference_ms=1.0)
        result.errors["c"] = "simulated failure"
        assert main.agreement(result, ["a", "b", "c"]) is None


class TestReports:
    def _sample_results(self):
        r1 = main.ImageResult(image_path=Path("img1.jpg"))
        r1.predictions["a"] = main.Prediction(top_k=[(1, "cat", 0.9), (2, "dog", 0.05)], inference_ms=2.0)
        r2 = main.ImageResult(image_path=Path("img2.jpg"), errors={"a": "failed to read image"})
        return [r1, r2]

    def _sample_profile(self):
        return main.ModelProfile(
            name="a", path="m.tar.gz", input_width=224, input_height=224,
            preprocess="imagenet", output="softmax", num_classes=1000, label_map=None, top_k=5,
        )

    def test_json_report_shape(self, tmp_path):
        results = self._sample_results()
        profiles = [self._sample_profile()]
        summary = main.build_class_summary(results, profiles)
        out = tmp_path / "report.json"
        main.write_json_report(out, results, profiles, ["skipped.txt: bad ext"], summary,
                               {"total_ms": 5.0, "image_count": 2, "model_count": 1})
        payload = json.loads(out.read_text())
        assert payload["models"] == ["a"]
        assert payload["skipped"] == ["skipped.txt: bad ext"]
        assert len(payload["images"]) == 2
        assert payload["images"][0]["predictions"]["a"]["top_k"][0]["label"] == "cat"
        assert payload["images"][1]["errors"] == {"a": "failed to read image"}
        assert "predictions" not in payload["images"][1]

    def test_csv_report_shape(self, tmp_path):
        results = self._sample_results()
        profiles = [self._sample_profile()]
        out = tmp_path / "report.csv"
        main.write_csv_report(out, results, profiles)
        lines = out.read_text().splitlines()
        assert lines[0].startswith("image,model,status")
        assert any("cat" in line for line in lines)
        assert any("error" in line for line in lines)

    def test_build_class_summary(self):
        results = self._sample_results()
        profiles = [self._sample_profile()]
        summary = main.build_class_summary(results, profiles)
        assert summary["a"]["cat"] == 1


class TestFakeHardwarePipeline:
    """End-to-end runs of main() against a fake pyneat SDK (see _make_fake_pyneat).

    Covers behavior that the issue requires but that otherwise only gets
    verified manually against real hardware: missing model files, corrupted
    images, supported-format coverage, and leaving originals untouched.
    """

    def test_missing_model_fails_cleanly(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setitem(sys.modules, "pyneat", _make_fake_pyneat(fail_message="archive not found"))
        img = tmp_path / "a.jpg"
        _make_image(img)
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, img, tmp_path / "out", model_path="models/does_not_exist.tar.gz")
        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_path)])

        rc = main.main()

        assert rc == 6
        assert "archive not found" in capsys.readouterr().err

    def test_unwritable_output_dir_fails_cleanly(self, tmp_path, monkeypatch, capsys):
        """Regression: a report directory that cannot be created must surface as
        a concise nonzero failure (as the C++ implementation does), not a traceback."""
        monkeypatch.setitem(sys.modules, "pyneat", _make_fake_pyneat())
        img = tmp_path / "a.jpg"
        _make_image(img)
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("file in the way")
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, img, blocker / "report")
        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_path)])

        rc = main.main()

        assert rc == 6
        assert "failed to write report" in capsys.readouterr().err

    def test_corrupted_image_reported_not_fatal(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyneat", _make_fake_pyneat())
        _make_image(tmp_path / "good.jpg")
        (tmp_path / "broken.jpg").write_bytes(b"not a jpeg")
        out_dir = tmp_path / "out"
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, tmp_path, out_dir)
        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_path)])

        rc = main.main()

        assert rc == 0
        payload = json.loads((out_dir / "report.json").read_text())
        by_name = {Path(img["path"]).name: img for img in payload["images"]}
        assert "m" in by_name["broken.jpg"]["errors"]
        assert "failed to read image" in by_name["broken.jpg"]["errors"]["m"]
        assert "predictions" not in by_name["broken.jpg"]
        assert "errors" not in by_name["good.jpg"]
        assert by_name["good.jpg"]["predictions"]["m"]["top_k"]

    def test_all_documented_formats_load(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyneat", _make_fake_pyneat())
        for ext, color in ((".jpg", (0, 0, 255)), (".jpeg", (0, 255, 255)),
                          (".png", (0, 255, 0)), (".bmp", (255, 0, 0))):
            _make_image(tmp_path / f"img{ext}", color=color)
        out_dir = tmp_path / "out"
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, tmp_path, out_dir)
        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_path)])

        rc = main.main()

        assert rc == 0
        payload = json.loads((out_dir / "report.json").read_text())
        assert len(payload["images"]) == 4
        for img in payload["images"]:
            assert "errors" not in img, f"{img['path']} had errors: {img.get('errors')}"
            assert img["predictions"]["m"]["top_k"], f"{img['path']} produced no predictions"

    def test_original_files_left_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyneat", _make_fake_pyneat())
        img = tmp_path / "a.jpg"
        _make_image(img)
        original_bytes = img.read_bytes()
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, img, tmp_path / "out")
        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_path)])

        rc = main.main()

        assert rc == 0
        assert img.read_bytes() == original_bytes

    def test_multi_model_agreement_and_disagreement_surface_in_report(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyneat", _make_fake_pyneat())
        for i in range(10):
            _make_image(tmp_path / f"img{i}.jpg", color=(i * 20 % 256, i * 37 % 256, i * 53 % 256))
        out_dir = tmp_path / "out"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"""
io:
  input: {tmp_path}
  output_dir: {out_dir}
models:
  a:
    path: fake_model_a.tar.gz
    num_classes: 5
    top_k: 3
  b:
    path: fake_model_b.tar.gz
    num_classes: 5
    top_k: 3
""")
        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_path)])

        rc = main.main()

        assert rc == 0
        payload = json.loads((out_dir / "report.json").read_text())
        assert set(payload["models"]) == {"a", "b"}
        agreements = {img["agreement"] for img in payload["images"]}
        # Two independently (fake-)randomized models: expect to see both
        # agreement and disagreement represented, exercising both code paths.
        assert agreements == {True, False}

    def test_one_model_failing_does_not_discard_another_models_prediction(
        self, tmp_path, monkeypatch
    ):
        """Regression: model B failing on an image must not hide model A's
        already-computed prediction for that same image from the report, and
        class_summary must stay consistent with what the report actually shows."""
        good_model_path = "fake_model_a.tar.gz"
        bad_model_path = "fake_model_b.tar.gz"
        monkeypatch.setitem(
            sys.modules,
            "pyneat",
            _make_fake_pyneat(fail_run_for_paths={bad_model_path}),
        )
        _make_image(tmp_path / "img.jpg")
        out_dir = tmp_path / "out"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"""
io:
  input: {tmp_path}
  output_dir: {out_dir}
models:
  a:
    path: {good_model_path}
    num_classes: 5
    top_k: 3
  b:
    path: {bad_model_path}
    num_classes: 5
    top_k: 3
""")
        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_path)])

        rc = main.main()

        assert rc == 0
        payload = json.loads((out_dir / "report.json").read_text())
        image_entry = payload["images"][0]
        # Model a's prediction must survive despite model b's failure on the same image.
        assert "a" in image_entry["predictions"]
        assert image_entry["predictions"]["a"]["top_k"]
        assert "b" in image_entry["errors"]
        assert "simulated inference failure" in image_entry["errors"]["b"]
        assert "b" not in image_entry["predictions"]
        # class_summary must only count what the report actually shows.
        top1_label = image_entry["predictions"]["a"]["top_k"][0]["label"]
        assert payload["class_summary"]["a"].get(top1_label) == 1
        assert payload["class_summary"].get("b", {}) == {}


    def test_models_are_loaded_one_at_a_time(self, tmp_path, monkeypatch):
        """Regression: the previous profile's model must be released before the
        next one is constructed, so two models never coexist on the accelerator."""
        fake = _make_fake_pyneat()
        monkeypatch.setitem(sys.modules, "pyneat", fake)
        _make_image(tmp_path / "img.jpg")
        out_dir = tmp_path / "out"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"""
io:
  input: {tmp_path}
  output_dir: {out_dir}
models:
  a:
    path: fake_model_a.tar.gz
    num_classes: 5
  b:
    path: fake_model_b.tar.gz
    num_classes: 5
  c:
    path: fake_model_c.tar.gz
    num_classes: 5
""")
        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_path)])

        rc = main.main()

        assert rc == 0
        assert fake.max_live_models == 1
        assert fake.live_models == 0
        payload = json.loads((out_dir / "report.json").read_text())
        assert set(payload["images"][0]["predictions"]) == {"a", "b", "c"}

    def test_thumbnail_write_failure_fails_run(self, tmp_path, monkeypatch, capsys):
        """Regression: an ignored cv2.imwrite failure let the report reference a
        thumbnail that was never written while still reporting success."""
        import cv2

        img = tmp_path / "a.jpg"
        _make_image(img)  # before imwrite is stubbed out
        monkeypatch.setitem(sys.modules, "pyneat", _make_fake_pyneat())
        monkeypatch.setattr(cv2, "imwrite", lambda *args, **kwargs: False)
        out_dir = tmp_path / "out"
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, img, out_dir)
        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_path)])

        rc = main.main()

        assert rc == 6
        assert "failed to write thumbnail" in capsys.readouterr().err
        assert not (out_dir / "report.html").exists()
        assert not list(tmp_path.glob(".out.*")), "staging/previous directories left behind"

    def test_failed_rerun_leaves_previous_report_intact(self, tmp_path, monkeypatch):
        """Regression: when a rerun into the same output_dir fails part-way, the
        directory must still hold the complete previous report rather than a
        mixture of new report.json/report.csv and old report.html/thumbnails."""
        import cv2

        monkeypatch.setitem(sys.modules, "pyneat", _make_fake_pyneat())
        img = tmp_path / "a.jpg"
        _make_image(img)
        out_dir = tmp_path / "out"
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, img, out_dir)
        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_path)])

        assert main.main() == 0
        before = {name: (out_dir / name).read_bytes()
                  for name in ("report.json", "report.csv", "report.html")}
        thumbs_before = sorted(p.name for p in (out_dir / "thumbnails").iterdir())
        assert thumbs_before

        # Second run: JSON/CSV would succeed, thumbnail encoding fails.
        _make_image(tmp_path / "b.jpg")
        _write_config(config_path, tmp_path, out_dir)
        monkeypatch.setattr(cv2, "imwrite", lambda *args, **kwargs: False)

        assert main.main() == 6
        for name, content in before.items():
            assert (out_dir / name).read_bytes() == content, f"{name} was clobbered"
        assert sorted(p.name for p in (out_dir / "thumbnails").iterdir()) == thumbs_before
        assert not list(tmp_path.glob(".out.*")), "staging/previous directories left behind"

    def test_failed_swap_rolls_back_previous_report(self, tmp_path, monkeypatch):
        """If the final rename of the staged report fails, the previous report
        must be restored at output_dir and nothing left behind."""
        monkeypatch.setitem(sys.modules, "pyneat", _make_fake_pyneat())
        img = tmp_path / "a.jpg"
        _make_image(img)
        out_dir = tmp_path / "out"
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, img, out_dir)
        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_path)])
        assert main.main() == 0
        before = (out_dir / "report.json").read_bytes()

        real_rename = Path.rename

        def flaky_rename(self, target):
            if ".staging-" in self.name:
                raise OSError(5, "simulated rename failure")
            return real_rename(self, target)

        monkeypatch.setattr(Path, "rename", flaky_rename)
        assert main.main() == 6
        assert (out_dir / "report.json").read_bytes() == before
        assert (out_dir / "report.html").is_file()
        assert not list(tmp_path.glob(".out.*")), "staging/previous directories left behind"

    def test_output_dir_with_foreign_entries_is_refused(self, tmp_path, monkeypatch, capsys):
        """output_dir is replaced as a whole, so a directory holding anything
        other than a previous report must be refused rather than swapped away."""
        monkeypatch.setitem(sys.modules, "pyneat", _make_fake_pyneat())
        img = tmp_path / "a.jpg"
        _make_image(img)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "notes.txt").write_text("customer data")
        config_path = tmp_path / "config.yaml"
        _write_config(config_path, img, out_dir)
        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_path)])

        rc = main.main()

        assert rc == 6
        assert "not part of a previous report" in capsys.readouterr().err
        assert (out_dir / "notes.txt").read_text() == "customer data"
        assert not (out_dir / "report.json").exists()

    def test_successful_rerun_replaces_stale_outputs(self, tmp_path, monkeypatch):
        """A successful rerun must not leave thumbnails from a previous run behind."""
        monkeypatch.setitem(sys.modules, "pyneat", _make_fake_pyneat())
        out_dir = tmp_path / "out"
        config_path = tmp_path / "config.yaml"
        first = tmp_path / "first"
        first.mkdir()
        _make_image(first / "a.jpg")
        _make_image(first / "b.jpg")
        _write_config(config_path, first, out_dir)
        monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config_path)])
        assert main.main() == 0
        assert len(list((out_dir / "thumbnails").iterdir())) == 2

        second = tmp_path / "second"
        second.mkdir()
        _make_image(second / "c.jpg")
        _write_config(config_path, second, out_dir)
        assert main.main() == 0
        assert len(list((out_dir / "thumbnails").iterdir())) == 1
        payload = json.loads((out_dir / "report.json").read_text())
        assert [Path(i["path"]).name for i in payload["images"]] == ["c.jpg"]


class TestHtmlReportContent:
    def _write(self, tmp_path, profiles):
        r1 = main.ImageResult(image_path=Path("img1.jpg"))
        for p in profiles:
            r1.predictions[p.name] = main.Prediction(top_k=[(1, "cat", 0.9)], inference_ms=1.0)
        r2 = main.ImageResult(image_path=Path("img2.jpg"),
                             errors={p.name: "failed to read image" for p in profiles})
        out = tmp_path / "report.html"
        main.write_html_report(out, [r1, r2], profiles, ["skipped.txt: bad ext"], {}, tmp_path)
        return out.read_text()

    def test_contains_model_checkboxes_and_controls(self, tmp_path):
        profiles = [
            main.ModelProfile(name="resnet_50", path="m.tar.gz", input_width=224, input_height=224,
                              preprocess="imagenet", output="softmax", num_classes=1000,
                              label_map=None, top_k=5),
            main.ModelProfile(name="resnet_18", path="m.tar.gz", input_width=224, input_height=224,
                              preprocess="imagenet", output="softmax", num_classes=1000,
                              label_map=None, top_k=5),
        ]
        html = self._write(tmp_path, profiles)

        assert 'value="resnet_50"' in html
        assert 'value="resnet_18"' in html
        assert 'id="modelDropdownBtn"' in html
        assert 'id="filterResult"' in html
        assert 'id="minConfidence"' in html
        assert 'id="sortBy"' in html
        assert '"label": "cat"' in html or '"label":"cat"' in html or '&quot;label&quot;: &quot;cat&quot;' in html
        assert "error: failed to read image" in html


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the classification pipeline."""

    def test_help(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 0
        assert "usage" in r.stdout.lower()

    def test_bad_config_path(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", "/nonexistent/config.yaml"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode != 0

    def test_unknown_flag(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--bogus"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()
