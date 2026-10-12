# Vendored from https://github.com/florianvoss-commit/supertonic-sima
# (app/supertonic_sima/engine.py) at commit 3b837b3e1b6a378ab8c24c3c04b079429b67e237,
# included in Neat GenAI Studio at the upstream author's request. See README.md
# in this directory for how to refresh it and THIRD_PARTY_TTS_MODELS.md for
# attribution. Local changes: DEFAULT_ASSET_ROOT / DEFAULT_OUTPUT_ROOT removed (unused; the worker passes every path explicitly).
#!/usr/bin/env python3
"""Persistent hybrid Supertonic 3 engine for a SiMa Modalix DevKit.

The duration predictor and text encoder use persistent ONNX Runtime sessions on
A65. Persistent pyneat ModelRunners execute the vector field and vocoder on MLA.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import onnxruntime as ort
import pyneat

from .inputs import (
    CONDITIONAL_STYLE_KEY,
    DEFAULT_STEPS,
    MAX_SEQUENCE_LENGTH,
    ROPE_TABLE_KEY,
    RUNTIME_CONSTANT_KEYS,
    TIME_TABLE_KEY,
    UNCONDITIONAL_STYLE_KEY,
    UNCONDITIONAL_STYLE_VALUE,
    UNCONDITIONAL_TEXT,
    build_time_table,
    pack_rope_input,
    validate_rope_bank,
)
from .text import MAX_SPEED, MIN_SPEED, preprocess_text


UPSTREAM_REVISION = "724fb5abbf5502583fb520898d45929e62f02c0b"
BASE_PHYSICAL_INPUTS = (
    ("noisy_latent", (1, 144, 1, 192)),
    ("text_emb", (1, 256, 1, 192)),
    ("style_ttl", (1, 256, 1, 50)),
    ("style_key", (1, 256, 1, 50)),
    ("latent_mask", (1, 1, 1, 192)),
    ("text_mask", (1, 192, 1, 1)),
    ("time_sinusoidal", (1, 64, 1, 1)),
    ("rope_tables", (1, 128, 1, 192)),
)
BASE_PHYSICAL_OUTPUT = ("velocity", (1, 144, 1, 192))
VOCODER_PHYSICAL_INPUT = ("latent", (1, 144, 1, 192))
VOCODER_PHYSICAL_OUTPUT = ("wav_frames", (1, 512, 1, 1152))


@dataclass(frozen=True)
class SynthesisResult:
    waveform: np.ndarray
    sample_rate: int
    normalized_text: str
    text_length: int
    latent_length: int
    valid_latent_length: int
    predicted_duration_seconds: float
    generation_seconds: float
    timings: dict[str, float]

    @property
    def audio_seconds(self) -> float:
        return float(self.waveform.size / self.sample_rate)

    @property
    def real_time_factor(self) -> float:
        if self.audio_seconds <= 0.0:
            return float("inf")
        return self.generation_seconds / self.audio_seconds


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pad_last(value: np.ndarray, width: int = MAX_SEQUENCE_LENGTH) -> np.ndarray:
    if value.shape[-1] > width:
        raise ValueError(f"input width {value.shape[-1]} exceeds {width}")
    padding = [(0, 0)] * value.ndim
    padding[-1] = (0, width - value.shape[-1])
    return np.ascontiguousarray(np.pad(value, padding), dtype=np.float32)


def _style_nchw(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value.transpose(0, 2, 1)[:, :, None, :])


def _shape_from_spec(spec: Any) -> tuple[int, ...] | None:
    shape = getattr(spec, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(value) for value in shape)
    except (TypeError, ValueError):
        return None


def _public_shape(physical_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Convert the batch-one compiler NCHW contract to public HWC."""
    _, channels, height, width = physical_shape
    return (height, width, channels)


def _tensor_from_public(value: np.ndarray) -> Any:
    # Neat uses HWC as the axis semantic for both HWC and batched NHWC tensors.
    return pyneat.Tensor.from_numpy(
        value,
        copy=True,
        layout=pyneat.TensorLayout.HWC,
    )


def _first_public_output_array(
    output: Any,
    *,
    expected_public_shape: tuple[int, ...],
) -> np.ndarray:
    tensors: list[Any] = []
    if isinstance(output, (list, tuple)):
        tensors = list(output)
    elif getattr(output, "tensor", None) is not None:
        tensors = [output.tensor]
    elif getattr(output, "tensors", None):
        tensors = list(output.tensors)
    elif getattr(output, "fields", None):
        tensors = [field.tensor for field in output.fields if field.tensor is not None]
    elif hasattr(output, "to_numpy"):
        tensors = [output]
    if not tensors:
        raise RuntimeError("MLA runner returned no output tensor")
    value = np.asarray(tensors[0].to_numpy(copy=False), dtype=np.float32)
    if value.shape != expected_public_shape:
        raise RuntimeError(
            f"MLA output shape {value.shape} != expected {expected_public_shape}"
        )
    if not np.isfinite(value).all():
        raise RuntimeError("MLA output contains a non-finite value")
    return value


def _first_output_array(
    output: Any,
    *,
    expected_public_shape: tuple[int, ...],
) -> np.ndarray:
    value = _first_public_output_array(
        output, expected_public_shape=expected_public_shape
    )
    return np.ascontiguousarray(value.transpose(2, 0, 1)[None, ...])


def _build_model_runner(model: Any, seed_values: list[np.ndarray]) -> Any:
    route_options = pyneat.ModelRouteOptions()
    route_options.include_input = True
    route_options.include_output = True
    run_options = pyneat.RunOptions()
    run_options.queue_depth = 2
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    run_options.output_memory = pyneat.OutputMemory.Owned
    return model.build(
        [_tensor_from_public(value) for value in seed_values],
        route_options=route_options,
        run_options=run_options,
    )


def _timing_statistics(values_seconds: list[float]) -> dict[str, float]:
    values = np.asarray(values_seconds, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("timing statistics require at least one sample")
    return {
        "mean_ms": float(values.mean() * 1000.0),
        "median_ms": float(np.median(values) * 1000.0),
        "min_ms": float(values.min() * 1000.0),
        "max_ms": float(values.max() * 1000.0),
        "p95_ms": float(np.percentile(values, 95) * 1000.0),
        "stddev_ms": float(values.std(ddof=1) * 1000.0) if values.size > 1 else 0.0,
    }


def benchmark_summary(results: list[SynthesisResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("benchmark summary requires at least one synthesis result")
    generation_values = [result.generation_seconds for result in results]
    audio_values = [result.audio_seconds for result in results]
    rtf_values = [result.real_time_factor for result in results]
    generation_mean = float(np.mean(generation_values))
    stage_names = tuple(results[0].timings)
    stages: dict[str, dict[str, float]] = {}
    accounted_per_run = [0.0] * len(results)
    for stage_name in stage_names:
        values = [result.timings[stage_name] for result in results]
        for index, value in enumerate(values):
            accounted_per_run[index] += value
        statistics = _timing_statistics(values)
        statistics["mean_percent_of_generation"] = (
            float(np.mean(values)) / generation_mean * 100.0
        )
        stages[stage_name] = statistics
    overhead_values = [
        max(0.0, total - accounted)
        for total, accounted in zip(
            generation_values,
            accounted_per_run,
            strict=True,
        )
    ]
    overhead_statistics = _timing_statistics(overhead_values)
    overhead_statistics["mean_percent_of_generation"] = (
        float(np.mean(overhead_values)) / generation_mean * 100.0
    )
    stages["runtime_overhead_seconds"] = overhead_statistics
    rtf = np.asarray(rtf_values, dtype=np.float64)
    return {
        "runs": len(results),
        "generation": _timing_statistics(generation_values),
        "audio_length": _timing_statistics(audio_values),
        "real_time_factor": {
            "mean": float(rtf.mean()),
            "median": float(np.median(rtf)),
            "min": float(rtf.min()),
            "max": float(rtf.max()),
            "p95": float(np.percentile(rtf, 95)),
            "stddev": float(rtf.std(ddof=1)) if rtf.size > 1 else 0.0,
        },
        "stages": stages,
    }


class SupertonicModalix:
    """Persistent hybrid TTS engine for repeated utterances."""

    def __init__(
        self,
        *,
        model_dir: Path,
        mpk_path: Path,
        runtime_data_path: Path,
        artifact_manifest_path: Path,
        vocoder_backend: str,
        vocoder_mpk_path: Path,
        vocoder_manifest_path: Path,
        threads: int,
        timeout_ms: int,
        verify_hashes: bool,
        steps: int = DEFAULT_STEPS,
    ) -> None:
        self.model_dir = model_dir.resolve()
        self.mpk_path = mpk_path.resolve()
        self.runtime_data_path = runtime_data_path.resolve()
        self.artifact_manifest_path = artifact_manifest_path.resolve()
        self.vocoder_backend = vocoder_backend
        self.vocoder_mpk_path = vocoder_mpk_path.resolve()
        self.vocoder_manifest_path = vocoder_manifest_path.resolve()
        if steps < 1:
            raise ValueError("steps must be positive")
        self.steps = steps
        self.timeout_ms = timeout_ms
        self.runner = None
        self.vocoder_runner = None
        self.vocoder_session = None
        self._style_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if self.vocoder_backend not in ("mla", "cpu"):
            raise ValueError(f"unsupported vocoder backend: {self.vocoder_backend}")

        self._validate_paths()
        if verify_hashes:
            self._verify_artifacts()

        onnx_dir = self.model_dir / "onnx"
        self.config = json.loads((onnx_dir / "tts.json").read_text())
        self.indexer = json.loads((onnx_dir / "unicode_indexer.json").read_text())
        if not isinstance(self.indexer, list) or not self.indexer:
            raise ValueError("unicode_indexer.json must contain a non-empty list")

        self.sample_rate = int(self.config["ae"]["sample_rate"])
        base_chunk_size = int(self.config["ae"]["base_chunk_size"])
        self.compress = int(self.config["ttl"]["chunk_compress_factor"])
        self.latent_dim = int(self.config["ttl"]["latent_dim"]) * self.compress
        self.chunk_size = base_chunk_size * self.compress
        if self.latent_dim != BASE_PHYSICAL_INPUTS[0][1][1]:
            raise RuntimeError(
                f"configured latent dimension {self.latent_dim} does not match MPK contract"
            )

        with np.load(self.runtime_data_path) as archive:
            archived_time_table = (
                np.asarray(archive[TIME_TABLE_KEY], dtype=np.float32).copy()
                if TIME_TABLE_KEY in archive
                else None
            )
            self.rope_table = validate_rope_bank(archive[ROPE_TABLE_KEY]).copy()
            self.constants = {
                name: np.asarray(archive[name], dtype=np.float32).copy()
                for name in RUNTIME_CONSTANT_KEYS
            }
        if archived_time_table is not None:
            if (
                archived_time_table.ndim != 5
                or archived_time_table.shape[1:] != BASE_PHYSICAL_INPUTS[6][1]
            ):
                raise ValueError(
                    f"time table shape {archived_time_table.shape} has an invalid contract"
                )
            expected_archived_table = build_time_table(archived_time_table.shape[0])
            if not np.allclose(
                archived_time_table, expected_archived_table, rtol=0.0, atol=1e-7
            ):
                raise ValueError("archived time table does not match the pinned model")
        self.time_table = build_time_table(self.steps)

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = threads
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_pattern = True
        providers = ["CPUExecutionProvider"]
        self.duration_session = ort.InferenceSession(
            str(onnx_dir / "duration_predictor.onnx"),
            sess_options=session_options,
            providers=providers,
        )
        self.text_session = ort.InferenceSession(
            str(onnx_dir / "text_encoder.onnx"),
            sess_options=session_options,
            providers=providers,
        )
        if self.vocoder_backend == "cpu":
            self.vocoder_session = ort.InferenceSession(
                str(onnx_dir / "vocoder.onnx"),
                sess_options=session_options,
                providers=providers,
            )

        self.model = pyneat.Model(str(self.mpk_path))
        compiled_batch_size = int(self.model.compiled_batch_size())
        if compiled_batch_size != 1:
            raise RuntimeError(
                f"compiled vector estimator has batch size {compiled_batch_size}; expected 1"
            )
        self.physical_inputs = BASE_PHYSICAL_INPUTS
        self.physical_output = BASE_PHYSICAL_OUTPUT
        self.public_inputs = tuple(
            (name, _public_shape(shape))
            for name, shape in BASE_PHYSICAL_INPUTS
        )
        self.public_output = (
            BASE_PHYSICAL_OUTPUT[0],
            _public_shape(BASE_PHYSICAL_OUTPUT[1]),
        )
        self._validate_model_contract()
        self._input_buffers = [
            np.zeros(shape, dtype=np.float32) for _, shape in self.public_inputs
        ]
        self.runner = _build_model_runner(self.model, self._input_buffers)

        self.vocoder_model = None
        self.vocoder_public_input = _public_shape(VOCODER_PHYSICAL_INPUT[1])
        self.vocoder_public_output = _public_shape(VOCODER_PHYSICAL_OUTPUT[1])
        if self.vocoder_backend == "mla":
            self.vocoder_model = pyneat.Model(str(self.vocoder_mpk_path))
            self._validate_vocoder_contract()
            self._vocoder_input_buffer = np.zeros(
                self.vocoder_public_input, dtype=np.float32
            )
            self.vocoder_runner = _build_model_runner(
                self.vocoder_model, [self._vocoder_input_buffer]
            )

    def _validate_paths(self) -> None:
        required = [
            self.mpk_path,
            self.runtime_data_path,
            self.artifact_manifest_path,
            self.model_dir / "onnx" / "duration_predictor.onnx",
            self.model_dir / "onnx" / "text_encoder.onnx",
            self.model_dir / "onnx" / "tts.json",
            self.model_dir / "onnx" / "unicode_indexer.json",
        ]
        if self.vocoder_backend == "mla":
            required.extend((self.vocoder_mpk_path, self.vocoder_manifest_path))
        else:
            required.append(self.model_dir / "onnx" / "vocoder.onnx")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing required assets:\n  " + "\n  ".join(missing))

    def _verify_artifacts(self) -> None:
        manifest = json.loads(self.artifact_manifest_path.read_text())
        try:
            expected_mpk = manifest["compiled"]["archive"]["sha256"]
            expected_runtime = manifest["graph_surgery"]["runtime_data"]["sha256"]
        except KeyError as error:
            raise ValueError(
                "artifact manifest does not describe the supported batch-one release"
            ) from error
        for label, path, expected in (
            ("MPK", self.mpk_path, expected_mpk),
            ("runtime data", self.runtime_data_path, expected_runtime),
        ):
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(
                    f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
                )
        if self.vocoder_backend == "mla":
            vocoder_manifest = json.loads(self.vocoder_manifest_path.read_text())
            expected_vocoder = vocoder_manifest["compiled"]["archive"]["sha256"]
            actual_vocoder = _sha256(self.vocoder_mpk_path)
            if actual_vocoder != expected_vocoder:
                raise ValueError(
                    "vocoder MPK SHA-256 mismatch: "
                    f"expected {expected_vocoder}, got {actual_vocoder}"
                )

    def _validate_model_contract(self) -> None:
        input_specs = list(self.model.input_specs())
        output_specs = list(self.model.output_specs())
        if len(input_specs) != len(self.public_inputs):
            raise RuntimeError(
                f"MPK reports {len(input_specs)} inputs; expected {len(self.public_inputs)}"
            )
        if len(output_specs) != 1:
            raise RuntimeError(f"MPK reports {len(output_specs)} outputs; expected 1")
        for index, (spec, (name, expected_shape)) in enumerate(
            zip(input_specs, self.public_inputs, strict=True)
        ):
            actual_shape = _shape_from_spec(spec)
            if actual_shape is not None and actual_shape != expected_shape:
                raise RuntimeError(
                    f"MPK input {index} ({name}) shape {actual_shape} != {expected_shape}"
                )
        output_shape = _shape_from_spec(output_specs[0])
        if output_shape is not None and output_shape != self.public_output[1]:
            raise RuntimeError(
                f"MPK output shape {output_shape} != {self.public_output[1]}"
            )

    def _validate_vocoder_contract(self) -> None:
        if self.vocoder_model is None:
            raise RuntimeError("MLA vocoder model is not loaded")
        if int(self.vocoder_model.compiled_batch_size()) != 1:
            raise RuntimeError("compiled vocoder must use batch size 1")
        input_specs = list(self.vocoder_model.input_specs())
        output_specs = list(self.vocoder_model.output_specs())
        if len(input_specs) != 1 or len(output_specs) != 1:
            raise RuntimeError(
                "compiled vocoder must report exactly one input and one output"
            )
        input_shape = _shape_from_spec(input_specs[0])
        output_shape = _shape_from_spec(output_specs[0])
        if input_shape is not None and input_shape != self.vocoder_public_input:
            raise RuntimeError(
                f"vocoder input shape {input_shape} != {self.vocoder_public_input}"
            )
        if output_shape is not None and output_shape != self.vocoder_public_output:
            raise RuntimeError(
                f"vocoder output shape {output_shape} != {self.vocoder_public_output}"
            )

    def _set_input(self, index: int, physical_nchw: np.ndarray) -> None:
        expected = self.physical_inputs[index][1]
        value = np.asarray(physical_nchw, dtype=np.float32)
        if value.shape != expected:
            raise RuntimeError(
                f"{self.physical_inputs[index][0]} physical shape {value.shape} != {expected}"
            )
        public_value = value[0].transpose(1, 2, 0)
        np.copyto(self._input_buffers[index], public_value)

    def _encode_text(self, text: str, language: str) -> tuple[np.ndarray, np.ndarray, str]:
        normalized = preprocess_text(text, language)
        ids: list[int] = []
        unsupported: list[str] = []
        for character in normalized:
            codepoint = ord(character)
            index = self.indexer[codepoint] if codepoint < len(self.indexer) else -1
            if index < 0:
                unsupported.append(character)
            ids.append(index)
        if unsupported:
            raise ValueError(
                f"unsupported characters after preprocessing: {sorted(set(unsupported))}"
            )
        if len(ids) > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"processed text length {len(ids)} exceeds static limit "
                f"{MAX_SEQUENCE_LENGTH}"
            )
        text_ids = np.asarray([ids], dtype=np.int64)
        text_mask = np.ones((1, 1, len(ids)), dtype=np.float32)
        return text_ids, text_mask, normalized

    def _load_style(self, voice: str) -> tuple[np.ndarray, np.ndarray]:
        if voice in self._style_cache:
            return self._style_cache[voice]
        path = self.model_dir / "voice_styles" / f"{voice}.json"
        if not path.is_file():
            raise FileNotFoundError(f"voice style does not exist: {path}")
        payload = json.loads(path.read_text())
        values: list[np.ndarray] = []
        for name, expected in (("style_ttl", (1, 50, 256)), ("style_dp", (1, 8, 16))):
            entry = payload[name]
            value = np.asarray(entry["data"], dtype=np.float32).reshape(entry["dims"])
            if value.shape != expected:
                raise ValueError(f"{name} shape {value.shape} != {expected}")
            values.append(np.ascontiguousarray(value))
        style = (values[0], values[1])
        self._style_cache[voice] = style
        return style

    def _run_vector_field(self) -> np.ndarray:
        if self.runner is None:
            raise RuntimeError("MLA runner is closed")
        tensors = [_tensor_from_public(value) for value in self._input_buffers]
        output = self.runner.run(tensors, timeout_ms=self.timeout_ms)
        return _first_output_array(
            output,
            expected_public_shape=self.public_output[1],
        )

    def _run_vocoder(
        self, padded_latent: np.ndarray, natural_latent_length: int
    ) -> np.ndarray:
        decoded_samples = natural_latent_length * self.chunk_size
        if self.vocoder_backend == "cpu":
            if self.vocoder_session is None:
                raise RuntimeError("CPU vocoder session is closed")
            natural_latent = np.ascontiguousarray(
                padded_latent[..., :natural_latent_length]
            )
            output = self.vocoder_session.run(None, {"latent": natural_latent})[0]
            waveform = np.asarray(output, dtype=np.float32).reshape(-1)
            if waveform.size != decoded_samples:
                raise RuntimeError(
                    f"CPU vocoder returned {waveform.size} samples; "
                    f"expected {decoded_samples}"
                )
            return waveform

        if self.vocoder_runner is None:
            raise RuntimeError("MLA vocoder runner is closed")
        np.copyto(
            self._vocoder_input_buffer,
            padded_latent.transpose(0, 2, 1),
        )
        output = self.vocoder_runner.run(
            [_tensor_from_public(self._vocoder_input_buffer)],
            timeout_ms=self.timeout_ms,
        )
        frames = _first_public_output_array(
            output, expected_public_shape=self.vocoder_public_output
        )
        if decoded_samples > frames.size:
            raise RuntimeError(
                f"requested {decoded_samples} vocoder samples from {frames.size}"
            )
        # The public HWC buffer is already time-major: [frame][512 samples].
        # Reshape is only a view; copy the cropped region so the unused static
        # profile tail is not retained by the synthesis result.
        return frames.reshape(-1)[:decoded_samples].copy()

    def synthesize(
        self,
        text: str,
        *,
        voice: str,
        language: str,
        speed: float,
        seed: int,
    ) -> SynthesisResult:
        if not MIN_SPEED <= speed <= MAX_SPEED:
            raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
        total_started = perf_counter()
        timings: dict[str, float] = {}

        started = perf_counter()
        text_ids, text_mask, normalized = self._encode_text(text, language)
        style_ttl, style_dp = self._load_style(voice)
        timings["preprocess_seconds"] = perf_counter() - started

        started = perf_counter()
        duration = self.duration_session.run(
            None,
            {"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask},
        )[0].astype(np.float32) / speed
        timings["duration_predictor_seconds"] = perf_counter() - started

        started = perf_counter()
        text_emb = self.text_session.run(
            None,
            {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask},
        )[0].astype(np.float32)
        timings["text_encoder_seconds"] = perf_counter() - started

        started = perf_counter()
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        latent_length = int(
            (float(duration.max()) * self.sample_rate + self.chunk_size - 1)
            / self.chunk_size
        )
        valid_latent_length = int(
            (int(wav_lengths[0]) + self.chunk_size - 1) // self.chunk_size
        )
        if latent_length > MAX_SEQUENCE_LENGTH or valid_latent_length > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"predicted latent length {max(latent_length, valid_latent_length)} "
                f"exceeds static limit {MAX_SEQUENCE_LENGTH}; shorten the text or increase speed"
            )
        latent_mask = np.zeros((1, 1, latent_length), dtype=np.float32)
        latent_mask[..., :valid_latent_length] = 1.0
        latent = (
            np.random.RandomState(seed)
            .randn(1, self.latent_dim, latent_length)
            .astype(np.float32)
            * latent_mask
        )
        padded_text = _pad_last(text_emb)
        padded_text_mask = _pad_last(text_mask)
        padded_mask = _pad_last(latent_mask)
        padded_latent = _pad_last(latent)
        rope_input = pack_rope_input(self.rope_table, padded_mask, padded_text_mask)
        self._set_input(4, padded_mask[:, :, None, :])
        self._set_input(5, padded_text_mask.transpose(0, 2, 1)[:, :, :, None])
        self._set_input(7, rope_input)
        timings["host_preparation_seconds"] = perf_counter() - started

        started = perf_counter()
        for step in range(self.steps):
            branch_outputs: list[np.ndarray] = []
            for conditional in (True, False):
                if conditional:
                    branch_text = padded_text
                    style_key = self.constants[CONDITIONAL_STYLE_KEY]
                    style_value = style_ttl
                else:
                    branch_text = np.broadcast_to(
                        self.constants[UNCONDITIONAL_TEXT], padded_text.shape
                    )
                    style_key = self.constants[UNCONDITIONAL_STYLE_KEY]
                    style_value = self.constants[UNCONDITIONAL_STYLE_VALUE]
                self._set_input(0, padded_latent[:, :, None, :])
                self._set_input(1, branch_text[:, :, None, :])
                self._set_input(2, _style_nchw(style_value))
                self._set_input(3, _style_nchw(style_key))
                self._set_input(6, self.time_table[step])
                branch_outputs.append(self._run_vector_field()[:, :, 0, :])
            conditional_output, unconditional_output = branch_outputs
            padded_latent = (
                (
                    padded_latent
                    + (4.0 * conditional_output - 3.0 * unconditional_output)
                    / float(self.steps)
                )
                * padded_mask
            ).astype(np.float32)
        timings["vector_field_seconds"] = perf_counter() - started

        started = perf_counter()
        waveform = self._run_vocoder(padded_latent, latent_length)
        waveform_samples = int(wav_lengths[0])
        if waveform_samples > waveform.size:
            raise RuntimeError(
                f"predicted waveform length {waveform_samples} exceeds decoded "
                f"length {waveform.size}"
            )
        waveform = waveform[:waveform_samples].copy()
        timings["vocoder_seconds"] = perf_counter() - started
        generation_seconds = perf_counter() - total_started

        return SynthesisResult(
            waveform=waveform,
            sample_rate=self.sample_rate,
            normalized_text=normalized,
            text_length=int(text_ids.shape[-1]),
            latent_length=latent_length,
            valid_latent_length=valid_latent_length,
            predicted_duration_seconds=float(np.asarray(duration).reshape(-1)[0]),
            generation_seconds=generation_seconds,
            timings=timings,
        )

    def close(self) -> None:
        if self.vocoder_runner is not None:
            self.vocoder_runner.close()
            self.vocoder_runner = None
        if self.runner is not None:
            self.runner.close()
            self.runner = None

    def __enter__(self) -> "SupertonicModalix":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
