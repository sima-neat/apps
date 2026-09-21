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


def _make_fake_pyneat(num_classes=5, fail_message=None):
    """A minimal in-process stand-in for the real pyneat SDK.

    This lets error paths that otherwise only run on real hardware (missing
    model archives, corrupted images, unsupported formats) be exercised
    automatically in CI without a devkit. Model.run() returns deterministic,
    seeded-random scores so results are reproducible across test runs.
    """
    import hashlib
    import types

    import numpy as np

    mod = types.ModuleType("pyneat")

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

        def run(self, tensors, timeout_ms=0):
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
        dest.write_bytes(b"fake")  # already present, so no network call
        images, skipped = main.discover_images(None, (".jpg",), "http://example.invalid/x.jpg", dest)
        assert images == [dest]
        assert skipped == []

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

    def test_rejects_unsupported_output_interpretation(self):
        raw = {"models": {"a": {"path": "m.tar.gz", "output": "raw_logits"}}}
        with pytest.raises(ValueError, match="output"):
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


class TestReports:
    def _sample_results(self):
        r1 = main.ImageResult(image_path=Path("img1.jpg"))
        r1.predictions["a"] = main.Prediction(top_k=[(1, "cat", 0.9), (2, "dog", 0.05)], inference_ms=2.0)
        r2 = main.ImageResult(image_path=Path("img2.jpg"), error="failed to read image")
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
        assert "error" in payload["images"][1]

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
        assert "error" in by_name["broken.jpg"]
        assert "error" not in by_name["good.jpg"]

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
        assert all("error" not in img for img in payload["images"])

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


class TestHtmlReportContent:
    def _write(self, tmp_path, profiles):
        r1 = main.ImageResult(image_path=Path("img1.jpg"))
        for p in profiles:
            r1.predictions[p.name] = main.Prediction(top_k=[(1, "cat", 0.9)], inference_ms=1.0)
        r2 = main.ImageResult(image_path=Path("img2.jpg"), error="failed to read image")
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
