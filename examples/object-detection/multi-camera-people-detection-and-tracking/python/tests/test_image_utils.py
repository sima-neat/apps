"""Image helper tests for the multi-camera people tracking example."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


@pytest.mark.unit
class TestImageUtils:
    def test_cpu_quanttess_input_letterboxes_rgb_frame_into_rgb_fp32_tensor(self):
        from utils.image_utils import build_cpu_quanttess_preproc_state, cpu_quanttess_input
        from utils.pipeline import QuantTessCpuPreproc, RuntimeModules

        class FakeImage:
            def __init__(
                self,
                *,
                shape,
                channels="rgb",
                dtype="uint8",
                source=None,
                scale=None,
            ):
                self.shape = shape
                self.channels = channels
                self.dtype = dtype
                self.source = source
                self.scale = scale

            def astype(self, dtype):
                return FakeImage(
                    shape=self.shape,
                    channels=self.channels,
                    dtype=dtype,
                    source=self.source or self,
                    scale=self.scale,
                )

        class FakeCanvasRegion:
            def __init__(self, canvas, key):
                self.canvas = canvas
                self.key = key

        class FakeCanvas:
            def __init__(self, shape, dtype):
                self.shape = shape
                self.dtype = dtype
                self.assignments = []
                self.fill_calls = []

            def __getitem__(self, key):
                return FakeCanvasRegion(self, key)

            def fill(self, value):
                self.fill_calls.append(value)

        class FakeCV2:
            INTER_LINEAR = "linear"

            @staticmethod
            def resize(image, size, interpolation=None):
                assert interpolation == FakeCV2.INTER_LINEAR
                dst_w, dst_h = size
                return FakeImage(
                    shape=(dst_h, dst_w, 3),
                    channels=image.channels,
                    dtype=image.dtype,
                    source=image.source or image,
                )

        class FakeNp:
            float32 = "float32"

            @staticmethod
            def zeros(shape, dtype=None):
                return FakeCanvas(shape, dtype)

            @staticmethod
            def multiply(image, scale, out=None, casting=None):
                assert casting == "unsafe"
                assert isinstance(out, FakeCanvasRegion)
                out.canvas.assignments.append(
                    (
                        out.key,
                        FakeImage(
                            shape=image.shape,
                            channels=image.channels,
                            dtype=FakeNp.float32,
                            source=image.source or image,
                            scale=scale,
                        ),
                    )
                )
                return out.canvas

        runtime = RuntimeModules(cv2=FakeCV2(), np=FakeNp(), pyneat=None)
        contract = QuantTessCpuPreproc(width=8, height=8, aspect_ratio=True, padding_type="CENTER")
        frame_rgb = FakeImage(shape=(2, 4, 3), channels="rgb")
        state = build_cpu_quanttess_preproc_state(runtime, contract, src_width=4, src_height=2)

        quant_input = cpu_quanttess_input(runtime, frame_rgb, state)

        assert quant_input.shape == (8, 8, 3)
        assert quant_input.dtype == runtime.np.float32
        key, value = quant_input.assignments[0]
        assert key == (slice(2, 6, None), slice(0, 8, None))
        assert value.shape == (4, 8, 3)
        assert value.channels == "rgb"
        assert value.dtype == runtime.np.float32
        assert value.scale == pytest.approx(1.0 / 255.0)
        assert quant_input.fill_calls == [0.0]

    def test_cpu_quanttess_input_reuses_preallocated_output_buffer_when_state_is_supplied(self):
        from utils.image_utils import build_cpu_quanttess_preproc_state, cpu_quanttess_input
        from utils.pipeline import QuantTessCpuPreproc, RuntimeModules

        class FakeImage:
            def __init__(self, *, shape, channels="rgb", dtype="uint8", source=None, scale=None):
                self.shape = shape
                self.channels = channels
                self.dtype = dtype
                self.source = source
                self.scale = scale

        class FakeCanvasRegion:
            def __init__(self, canvas, key):
                self.canvas = canvas
                self.key = key

        class FakeCanvas:
            def __init__(self, shape, dtype):
                self.shape = shape
                self.dtype = dtype
                self.assignments = []
                self.fill_calls = []

            def __getitem__(self, key):
                return FakeCanvasRegion(self, key)

            def fill(self, value):
                self.fill_calls.append(value)

        class FakeCV2:
            INTER_LINEAR = "linear"

            @staticmethod
            def resize(image, size, interpolation=None):
                assert interpolation == FakeCV2.INTER_LINEAR
                dst_w, dst_h = size
                return FakeImage(
                    shape=(dst_h, dst_w, 3),
                    channels=image.channels,
                    dtype=image.dtype,
                    source=image.source or image,
                )

        class FakeNp:
            float32 = "float32"

            @staticmethod
            def zeros(shape, dtype=None):
                return FakeCanvas(shape, dtype)

            @staticmethod
            def multiply(image, scale, out=None, casting=None):
                assert casting == "unsafe"
                assert isinstance(out, FakeCanvasRegion)
                out.canvas.assignments.append(
                    (
                        out.key,
                        FakeImage(
                            shape=image.shape,
                            channels=image.channels,
                            dtype=FakeNp.float32,
                            source=image.source or image,
                            scale=scale,
                        ),
                    )
                )
                return out.canvas

        runtime = RuntimeModules(cv2=FakeCV2(), np=FakeNp(), pyneat=None)
        contract = QuantTessCpuPreproc(width=8, height=8, aspect_ratio=True, padding_type="CENTER")
        state = build_cpu_quanttess_preproc_state(runtime, contract, src_width=4, src_height=2)

        first = cpu_quanttess_input(
            runtime,
            FakeImage(shape=(2, 4, 3), channels="rgb"),
            state,
        )
        second = cpu_quanttess_input(
            runtime,
            FakeImage(shape=(2, 4, 3), channels="rgb"),
            state,
        )

        assert first is state.quant_input
        assert second is state.quant_input
        assert state.quant_input.fill_calls == [0.0, 0.0]
        assert len(state.quant_input.assignments) == 2
