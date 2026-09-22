"""Classify a single image or a directory of images with one or more models
and generate a browsable HTML report plus JSON/CSV results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class ModelProfile:
    name: str
    path: str
    input_width: int
    input_height: int
    preprocess: str
    output: str
    num_classes: int
    label_map: str | None
    top_k: int
    labels: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    top_k: list[tuple[int, str, float]]
    inference_ms: float


@dataclass
class ImageResult:
    image_path: Path
    predictions: dict[str, Prediction] = field(default_factory=dict)  # model name -> Prediction
    errors: dict[str, str] = field(default_factory=dict)  # model name -> error message


def download_image(url: str, dest: Path) -> Path:
    """Download an image if it does not already exist."""
    if not dest.exists():
        print(f"Downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, dest)
        except (urllib.error.URLError, OSError) as exc:
            raise FileNotFoundError(f"failed to download {url}: {exc}") from exc
    return dest


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_profiles(raw: dict[str, Any]) -> list[ModelProfile]:
    models_cfg = raw.get("models") or {}
    if not isinstance(models_cfg, dict) or not models_cfg:
        raise ValueError("config.yaml must define at least one entry under `models`")

    profiles = []
    for name, cfg in models_cfg.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"models.{name} must be a mapping")
        profile = ModelProfile(
            name=name,
            path=str(cfg.get("path", "")),
            input_width=int(cfg.get("input_width", 224)),
            input_height=int(cfg.get("input_height", 224)),
            preprocess=str(cfg.get("preprocess", "imagenet")),
            output=str(cfg.get("output", "softmax")),
            num_classes=int(cfg.get("num_classes", 1000)),
            label_map=cfg.get("label_map"),
            top_k=int(cfg.get("top_k", 5)),
        )
        if not profile.path:
            raise ValueError(f"models.{name}.path is required")
        if profile.output != "softmax":
            raise ValueError(
                f"models.{name}.output={profile.output!r} is not supported; "
                "only 'softmax' is implemented (raw per-class scores, softmax applied, "
                "index i maps to label_map[i])"
            )
        if profile.top_k <= 0:
            raise ValueError(f"models.{name}.top_k must be positive, got {profile.top_k}")
        if profile.num_classes <= 0:
            raise ValueError(
                f"models.{name}.num_classes must be positive, got {profile.num_classes}"
            )
        if profile.input_width <= 0 or profile.input_height <= 0:
            raise ValueError(
                f"models.{name}.input_width/input_height must be positive, got "
                f"{profile.input_width}x{profile.input_height}"
            )
        profiles.append(profile)
    return profiles


def load_label_map(path: str | None, num_classes: int) -> list[str]:
    if not path:
        return [str(i) for i in range(num_classes)]
    label_path = Path(path)
    if not label_path.exists():
        # Bundled label maps live next to this script regardless of the caller's cwd
        # (model.path stays cwd-relative since it points at a user-downloaded file).
        bundled = Path(__file__).resolve().parents[1] / "common" / label_path.name
        if bundled.exists():
            label_path = bundled
    try:
        with label_path.open("r", encoding="utf-8") as handle:
            labels = [line.strip() for line in handle if line.strip()]
    except OSError as exc:
        raise ValueError(f"failed to open label map {label_path}: {exc}") from exc
    if len(labels) < num_classes:
        raise ValueError(
            f"label map {label_path} has {len(labels)} entries, expected at least {num_classes}"
        )
    return labels


def discover_images(input_path: str | None, extensions: tuple[str, ...],
                     fallback_url: str, fallback_dest: Path) -> tuple[list[Path], list[str]]:
    """Return (images, skipped_descriptions) in deterministic order."""
    if not input_path:
        download_image(fallback_url, fallback_dest)
        return [fallback_dest], []

    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in extensions:
            return [], [f"{path}: unsupported extension {path.suffix or '(none)'}"]
        return [path], []

    if not path.is_dir():
        raise FileNotFoundError(f"input path does not exist: {path}")

    images: list[Path] = []
    skipped: list[str] = []
    for entry in sorted(path.iterdir(), key=lambda p: p.name):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in extensions:
            skipped.append(f"{entry}: unsupported extension {entry.suffix or '(none)'}")
            continue
        images.append(entry)

    if not images and not skipped:
        raise FileNotFoundError(f"no image files found under {path}")
    return images, skipped


def load_rgb_resized(path: str, width: int, height: int):
    """Load an image with OpenCV and return an RGB uint8 HWC array."""
    import cv2

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to read image: {path}")
    # INTER_AREA matches sima_examples::load_rgb_resized (support/runtime/example_utils.cpp)
    # used by the C++ implementation, so both languages preprocess images identically.
    bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def softmax(x):
    import numpy as np

    e = np.exp(x - np.max(x))
    return e / e.sum()


def tensor_to_numpy_dense(tensor) -> Any:
    """Convert a dense pyneat tensor to a NumPy array without DLPack."""
    import numpy as np
    import pyneat

    dtype_map = {
        pyneat.TensorDType.UInt8: np.uint8,
        pyneat.TensorDType.Int8: np.int8,
        pyneat.TensorDType.UInt16: np.uint16,
        pyneat.TensorDType.Int16: np.int16,
        pyneat.TensorDType.Int32: np.int32,
        pyneat.TensorDType.Float32: np.float32,
        pyneat.TensorDType.Float64: np.float64,
    }
    np_dtype = dtype_map.get(tensor.dtype)
    if np_dtype is None:
        raise TypeError(f"Unsupported tensor dtype for NumPy conversion: {tensor.dtype}")

    shape = tuple(int(x) for x in tensor.shape)
    arr = np.frombuffer(tensor.copy_dense_bytes_tight(), dtype=np_dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr


def build_model(profile: ModelProfile):
    import pyneat

    if profile.preprocess != "imagenet":
        raise ValueError(
            f"models.{profile.name}.preprocess={profile.preprocess!r} is not supported; "
            "only 'imagenet' is implemented"
        )

    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.input_max_width = profile.input_width
    opt.preprocess.input_max_height = profile.input_height
    opt.preprocess.input_max_depth = 3
    opt.preprocess.preset = pyneat.NormalizePreset.ImageNet
    return pyneat.Model(profile.path, opt)


def classify(model, profile: ModelProfile, image_path: Path, timeout_ms: int) -> Prediction:
    import numpy as np
    import pyneat

    rgb = load_rgb_resized(str(image_path), profile.input_width, profile.input_height)
    tensor = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)

    start = time.monotonic()
    outputs = model.run([tensor], timeout_ms=timeout_ms)
    inference_ms = (time.monotonic() - start) * 1000.0

    if not outputs:
        raise RuntimeError("model run returned empty output")
    scores = tensor_to_numpy_dense(outputs[0]).flatten().astype(np.float32)
    if scores.size < profile.num_classes:
        raise RuntimeError(f"expected at least {profile.num_classes} scores, got {scores.size}")
    scores = scores[: profile.num_classes]

    probs = softmax(scores)
    # Tie-break by ascending class id so equal scores resolve deterministically
    # and identically to the C++ implementation's topk_with_softmax.
    top_indices = np.lexsort((np.arange(scores.size), -scores))[: profile.top_k]
    top_k = [
        (int(i), profile.labels[int(i)], float(probs[int(i)]))
        for i in top_indices
    ]
    return Prediction(top_k=top_k, inference_ms=inference_ms)


def run_all(profiles: list[ModelProfile], images: list[Path], timeout_ms: int) -> list[ImageResult]:
    results = [ImageResult(image_path=p) for p in images]

    for profile in profiles:
        print(f"Loading model '{profile.name}': {profile.path}")
        model = build_model(profile)
        for result in results:
            try:
                result.predictions[profile.name] = classify(model, profile, result.image_path, timeout_ms)
            except Exception as exc:  # noqa: BLE001 - per-image, per-model failures must not abort the run
                print(f"  {result.image_path}: {profile.name} failed: {exc}", file=sys.stderr)
                result.errors[profile.name] = str(exc)

    return results


def agreement(result: ImageResult, profile_names: list[str]) -> bool | None:
    """True/False only when every named model has a top-1 result; otherwise
    indeterminate (None) rather than silently agreeing/disagreeing over a
    partial subset."""
    if len(profile_names) < 2:
        return None
    labels = []
    for name in profile_names:
        pred = result.predictions.get(name)
        if pred is None or not pred.top_k:
            return None
        labels.append(pred.top_k[0][1])
    return len(set(labels)) == 1


def write_json_report(path: Path, results: list[ImageResult], profiles: list[ModelProfile],
                       skipped: list[str], class_summary: dict[str, dict[str, int]],
                       timing: dict[str, Any]) -> None:
    payload = {
        "models": [p.name for p in profiles],
        "images": [],
        "skipped": skipped,
        "class_summary": class_summary,
        "timing": timing,
    }
    for result in results:
        entry: dict[str, Any] = {"path": str(result.image_path)}
        if result.predictions:
            entry["predictions"] = {
                name: {
                    "top_k": [{"class_id": c, "label": lbl, "probability": p} for c, lbl, p in pred.top_k],
                    "inference_ms": pred.inference_ms,
                }
                for name, pred in result.predictions.items()
            }
            entry["agreement"] = agreement(result, [p.name for p in profiles])
        if result.errors:
            entry["errors"] = result.errors
        payload["images"].append(entry)

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv_report(path: Path, results: list[ImageResult], profiles: list[ModelProfile]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "model", "status", "top1_class_id", "top1_label",
                          "top1_probability", "inference_ms", "top_k"])
        for result in results:
            for profile in profiles:
                pred = result.predictions.get(profile.name)
                if pred is not None and pred.top_k:
                    top1 = pred.top_k[0]
                    top_k_str = ";".join(f"{lbl}:{p:.4f}" for _, lbl, p in pred.top_k)
                    writer.writerow([
                        result.image_path, profile.name, "ok", top1[0], top1[1],
                        f"{top1[2]:.4f}", f"{pred.inference_ms:.2f}", top_k_str,
                    ])
                elif profile.name in result.errors:
                    writer.writerow([
                        result.image_path, profile.name, "error", "", "", "", "",
                        result.errors[profile.name],
                    ])
                else:
                    writer.writerow([result.image_path, profile.name, "no_result", "", "", "", "", ""])


def make_thumbnail(image_path: Path, thumb_dir: Path, max_side: int = 160) -> str | None:
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_name = f"{abs(hash(str(image_path)))}.jpg"
    cv2.imwrite(str(thumb_dir / thumb_name), resized)
    return f"thumbnails/{thumb_name}"


def html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def write_html_report(path: Path, results: list[ImageResult], profiles: list[ModelProfile],
                       skipped: list[str], class_summary: dict[str, dict[str, int]],
                       output_dir: Path) -> None:
    profile_names = [p.name for p in profiles]
    rows = []
    for idx, result in enumerate(results):
        thumb = make_thumbnail(result.image_path, output_dir / "thumbnails")

        top1_by_model = {
            name: {"label": pred.top_k[0][1], "prob": pred.top_k[0][2]}
            for name, pred in result.predictions.items() if pred.top_k
        }
        top1_json = html_escape(json.dumps(top1_by_model))
        cells = []
        for name in profile_names:
            pred = result.predictions.get(name)
            if pred is not None and pred.top_k:
                top_str = "<br>".join(f"{html_escape(lbl)} ({p:.2%})" for _, lbl, p in pred.top_k)
                cells.append(
                    f'<td data-model-col="{html_escape(name)}">{top_str}<br>'
                    f'<span class="timing">{pred.inference_ms:.1f} ms</span></td>'
                )
            elif name in result.errors:
                cells.append(
                    f'<td data-model-col="{html_escape(name)}" class="error">'
                    f'error: {html_escape(result.errors[name])}</td>'
                )
            else:
                cells.append(f'<td data-model-col="{html_escape(name)}">no result</td>')

        has_error = "1" if result.errors else "0"
        rows.append(f"""
        <tr class="row" data-has-error="{has_error}" data-idx="{idx}" data-top1="{top1_json}">
          <td>{f'<img src="{thumb}">' if thumb else ''}</td>
          <td>{html_escape(str(result.image_path))}</td>
          {''.join(cells)}
          <td class="agree-cell">&mdash;</td>
        </tr>""")

    header_cols = "".join(
        f'<th data-model-col="{html_escape(name)}">{html_escape(name)}</th>' for name in profile_names
    )
    model_checkboxes = "".join(
        f'<label><input type="checkbox" class="model-checkbox" value="{html_escape(name)}" checked> '
        f'{html_escape(name)}</label>'
        for name in profile_names
    )
    summary_rows = "".join(
        f"<tr><td>{html_escape(name)}</td><td>{html_escape(cls)}</td><td>{count}</td></tr>"
        for name, per_class in class_summary.items()
        for cls, count in sorted(per_class.items(), key=lambda kv: -kv[1])
    )
    skipped_rows = "".join(f"<li>{html_escape(s)}</li>" for s in skipped)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Image Classification Explorer Report</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; margin: 24px; color: #1a1a1a; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; font-size: 13px; }}
  th {{ background: #f4f4f4; position: sticky; top: 0; }}
  img {{ max-width: 100px; max-height: 100px; }}
  .timing {{ color: #888; font-size: 11px; }}
  .error {{ color: #b00020; }}
  tr[data-has-error="1"] {{ background: #fff6e5; }}
  #controls {{ margin: 12px 0; display: flex; gap: 12px; align-items: flex-start; flex-wrap: wrap; }}
  #controls input, #controls select {{ padding: 4px; }}
  .dropdown {{ position: relative; display: inline-block; }}
  .dropdown-btn {{
    padding: 5px 10px; border: 1px solid #ccc; border-radius: 4px; background: #fff; cursor: pointer;
    font-size: 13px;
  }}
  .dropdown-panel {{
    display: none; position: absolute; top: 100%; left: 0; margin-top: 4px; padding: 8px;
    background: #fff; border: 1px solid #ccc; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    z-index: 10; min-width: 160px; max-height: 240px; overflow-y: auto;
  }}
  .dropdown-panel.open {{ display: block; }}
  .dropdown-panel label {{ display: block; font-size: 13px; padding: 2px 0; white-space: nowrap; }}
  .dropdown-panel .dropdown-actions {{ margin-top: 6px; padding-top: 6px; border-top: 1px solid #eee; }}
  .dropdown-panel .dropdown-actions button {{
    font-size: 12px; padding: 2px 6px; margin-right: 6px; cursor: pointer;
  }}
</style>
</head>
<body>
<h1>Image Classification Explorer Report</h1>
<p>Models: {html_escape(', '.join(profile_names))} &middot; Images: {len(results)} &middot; Skipped: {len(skipped)}</p>

<div id="controls">
  <div class="dropdown" id="modelDropdown">
    <button type="button" class="dropdown-btn" id="modelDropdownBtn">All models &#9662;</button>
    <div class="dropdown-panel" id="modelDropdownPanel">
      {model_checkboxes}
      <div class="dropdown-actions">
        <button type="button" id="modelSelectAll">All</button>
        <button type="button" id="modelSelectNone">None</button>
      </div>
    </div>
  </div>
  <select id="filterResult">
    <option value="">All results</option>
    <option value="agree">Agreement</option>
    <option value="disagree">Disagreement</option>
    <option value="error">Errors</option>
  </select>
  <input id="filterClass" placeholder="Filter by predicted class...">
  <input id="minConfidence" type="number" min="0" max="100" step="1" placeholder="Min confidence %">
  <select id="sortBy">
    <option value="">Sort: default order</option>
    <option value="confidence">Sort: confidence (high to low)</option>
    <option value="class">Sort: predicted class (A-Z)</option>
    <option value="result">Sort: result</option>
  </select>
</div>

<table id="reportTable">
  <thead>
    <tr><th>Image</th><th>Path</th>{header_cols}<th>Models Agree?</th></tr>
  </thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>

<h2>Per-class summary</h2>
<table>
  <thead><tr><th>Model</th><th>Predicted class</th><th>Count</th></tr></thead>
  <tbody>{summary_rows}</tbody>
</table>

<h2>Skipped files</h2>
<ul>{skipped_rows if skipped_rows else '<li>None</li>'}</ul>

<script>
  const classInput = document.getElementById('filterClass');
  const resultSelect = document.getElementById('filterResult');
  const confidenceInput = document.getElementById('minConfidence');
  const sortSelect = document.getElementById('sortBy');
  const tbody = document.querySelector('#reportTable tbody');
  const rows = Array.from(document.querySelectorAll('#reportTable tbody tr'));
  const modelBtn = document.getElementById('modelDropdownBtn');
  const modelPanel = document.getElementById('modelDropdownPanel');
  const modelChecks = Array.from(document.querySelectorAll('.model-checkbox'));

  function selectedModels() {{
    return modelChecks.filter((c) => c.checked).map((c) => c.value);
  }}

  function updateModelBtnLabel() {{
    const selected = modelChecks.filter((c) => c.checked);
    let label;
    if (selected.length === 0) {{
      label = 'No models';
    }} else if (selected.length === modelChecks.length) {{
      label = 'All models';
    }} else {{
      label = selected.length + ' model' + (selected.length > 1 ? 's' : '');
    }}
    modelBtn.textContent = label + ' ▾';
  }}

  modelBtn.addEventListener('click', (e) => {{
    e.stopPropagation();
    modelPanel.classList.toggle('open');
  }});
  document.addEventListener('click', () => modelPanel.classList.remove('open'));
  modelPanel.addEventListener('click', (e) => e.stopPropagation());
  document.getElementById('modelSelectAll').addEventListener('click', () => {{
    modelChecks.forEach((c) => {{ c.checked = true; }});
    updateModelBtnLabel();
    applyFilters();
  }});
  document.getElementById('modelSelectNone').addEventListener('click', () => {{
    modelChecks.forEach((c) => {{ c.checked = false; }});
    updateModelBtnLabel();
    applyFilters();
  }});
  modelChecks.forEach((c) => c.addEventListener('change', () => {{
    updateModelBtnLabel();
    applyFilters();
  }}));

  function rowTop1(row) {{
    try {{ return JSON.parse(row.dataset.top1 || '{{}}'); }} catch (e) {{ return {{}}; }}
  }}

  function rowAgreement(row, models) {{
    // True/False only when every selected model has a top-1 result; otherwise
    // indeterminate, rather than silently agreeing/disagreeing over a subset.
    if (models.length < 2) return null;
    const top1 = rowTop1(row);
    const labels = models.map((m) => top1[m] && top1[m].label);
    if (labels.some((l) => l === undefined)) return null;
    return labels.every((l) => l === labels[0]);
  }}

  function rowMaxConfidence(row, models) {{
    const top1 = rowTop1(row);
    const probs = models.map((m) => top1[m] && top1[m].prob).filter((v) => v !== undefined);
    return probs.length ? Math.max(...probs) : null;
  }}

  function rowClassText(row, models) {{
    const top1 = rowTop1(row);
    return models.map((m) => (top1[m] && top1[m].label) || '').join(' ');
  }}

  function applyFilters() {{
    const models = selectedModels();
    const classQuery = classInput.value.trim().toLowerCase();
    const resultQuery = resultSelect.value;
    const minConfidence = confidenceInput.value === '' ? null : parseFloat(confidenceInput.value) / 100;

    document.querySelectorAll('[data-model-col]').forEach((cell) => {{
      cell.style.display = models.includes(cell.dataset.modelCol) ? '' : 'none';
    }});

    for (const row of rows) {{
      const hasError = row.dataset.hasError === '1';
      let visible;

      if (resultQuery === 'error') {{
        visible = hasError;
      }} else {{
        const agree = rowAgreement(row, models);
        if (resultQuery === 'agree') visible = agree === true;
        else if (resultQuery === 'disagree') visible = agree === false;
        else visible = true;
      }}

      if (visible && classQuery) {{
        visible = rowClassText(row, models).toLowerCase().includes(classQuery);
      }}

      if (visible && minConfidence !== null) {{
        const maxConf = rowMaxConfidence(row, models);
        visible = maxConf !== null && maxConf >= minConfidence;
      }}

      const agreeCell = row.querySelector('.agree-cell');
      if (agreeCell) {{
        const agree = rowAgreement(row, models);
        agreeCell.textContent = agree === null ? '—' : (agree ? 'agree' : 'disagree');
      }}

      row.style.display = visible ? '' : 'none';
    }}

    applySort(models);
  }}

  function applySort(models) {{
    const sortKey = sortSelect.value;
    const sorted = rows.slice();
    if (sortKey === 'confidence') {{
      sorted.sort((a, b) => {{
        const av = rowMaxConfidence(a, models);
        const bv = rowMaxConfidence(b, models);
        return (bv === null ? -1 : bv) - (av === null ? -1 : av);
      }});
    }} else if (sortKey === 'class') {{
      sorted.sort((a, b) => rowClassText(a, models).localeCompare(rowClassText(b, models)));
    }} else if (sortKey === 'result') {{
      sorted.sort((a, b) => {{
        const ra = a.dataset.hasError === '1' ? 2 : (rowAgreement(a, models) === false ? 1 : 0);
        const rb = b.dataset.hasError === '1' ? 2 : (rowAgreement(b, models) === false ? 1 : 0);
        return ra - rb;
      }});
    }} else {{
      sorted.sort((a, b) => Number(a.dataset.idx) - Number(b.dataset.idx));
    }}
    for (const row of sorted) tbody.appendChild(row);
  }}

  updateModelBtnLabel();
  classInput.addEventListener('input', applyFilters);
  resultSelect.addEventListener('change', applyFilters);
  confidenceInput.addEventListener('input', applyFilters);
  sortSelect.addEventListener('change', applyFilters);
  applyFilters();
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def build_class_summary(results: list[ImageResult], profiles: list[ModelProfile]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {p.name: {} for p in profiles}
    for result in results:
        for name, pred in result.predictions.items():
            if not pred.top_k:
                continue
            label = pred.top_k[0][1]
            summary.setdefault(name, {})
            summary[name][label] = summary[name].get(label, 0) + 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Image classification explorer")
    default_config = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
    parser.add_argument("--config", type=Path, default=default_config, help="Path to YAML configuration")
    args = parser.parse_args()

    raw = load_config(args.config)

    # Relative paths in config.yaml (model artifacts, label maps, output dir) resolve
    # against the current working directory, matching every other example in this repo:
    # customers run commands from the installed `prebuilt-apps/` root, not from here.
    io_cfg = raw.get("io", {})
    runtime = raw.get("runtime", {})
    timeout_ms = int(runtime.get("timeout_ms", 20000))
    extensions = tuple(
        e.strip().lower() for e in str(io_cfg.get("extensions", ",".join(DEFAULT_EXTENSIONS))).split(",")
        if e.strip()
    ) or DEFAULT_EXTENSIONS
    fallback_url = io_cfg.get(
        "fallback_image_url",
        "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
        "n01443537_goldfish.JPEG",
    )
    output_dir = Path(io_cfg.get("output_dir", "report"))

    try:
        profiles = load_profiles(raw)
        for profile in profiles:
            profile.labels = load_label_map(profile.label_map, profile.num_classes)
    except ValueError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2

    try:
        images, skipped = discover_images(
            io_cfg.get("input"), extensions, fallback_url, Path("/tmp/goldfish.jpeg")
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    for entry in skipped:
        print(f"Skipping {entry}")

    results: list[ImageResult] = []
    total_ms = 0.0
    if not images:
        print("No images to classify.", file=sys.stderr)
    else:
        print(f"Classifying {len(images)} image(s) with {len(profiles)} model(s): "
              f"{', '.join(p.name for p in profiles)}")

        start = time.monotonic()
        try:
            results = run_all(profiles, images, timeout_ms)
        except Exception as exc:  # noqa: BLE001 - a whole-model failure (e.g. bad path) is fatal
            print(f"Error: {exc}", file=sys.stderr)
            return 6
        total_ms = (time.monotonic() - start) * 1000.0

    output_dir.mkdir(parents=True, exist_ok=True)
    class_summary = build_class_summary(results, profiles)
    timing = {
        "total_ms": total_ms,
        "image_count": len(images),
        "model_count": len(profiles),
    }

    write_json_report(output_dir / "report.json", results, profiles, skipped, class_summary, timing)
    write_csv_report(output_dir / "report.csv", results, profiles)
    write_html_report(output_dir / "report.html", results, profiles, skipped, class_summary, output_dir)

    images_with_errors = sum(1 for r in results if r.errors)
    print(f"Done in {total_ms:.1f} ms. Report written to {output_dir}")
    print(f"  report.html, report.json, report.csv")
    if images_with_errors:
        print(f"  {images_with_errors} image(s) had at least one model failure (see report.json)")

    validation = raw.get("validation") or {}
    expected_class_id = validation.get("expected_class_id")
    used_fallback_sample = not io_cfg.get("input")
    if expected_class_id is not None and used_fallback_sample and len(images) == 1 and profiles:
        first = results[0]
        pred = first.predictions.get(profiles[0].name)
        if pred and pred.top_k:
            top1_id = pred.top_k[0][0]
            min_probability = float(validation.get("min_probability", 0.0))
            top1_prob = pred.top_k[0][2]
            if top1_id != int(expected_class_id) or top1_prob < min_probability:
                print(
                    f"Note: {profiles[0].name} top1={top1_id} ({top1_prob:.4f}) did not match "
                    f"expected_class_id={expected_class_id} (min_probability={min_probability}); "
                    "see report for details.",
                    file=sys.stderr,
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
