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
    def test_cpu_quanttess_input_letterboxes_bgr_frame_into_rgb_fp32_tensor(self):
        from utils.image_utils import cpu_quanttess_input
        from utils.pipeline import QuantTessCpuPreproc, RuntimeModules

        class FakeImage:
            def __init__(
                self,
                *,
                shape,
                channels="bgr",
                dtype="uint8",
                source=None,
                scale=None,
            ):
                self.shape = shape
                self.channels = channels
                self.dtype = dtype
                self.source = source
                self.scale = scale

            def __getitem__(self, item):
                if item == (Ellipsis, slice(None, None, -1)):
                    return FakeImage(
                        shape=self.shape,
                        channels="rgb",
                        dtype=self.dtype,
                        source=self.source or self,
                    )
                raise AssertionError(f"unexpected image slice: {item}")

            def astype(self, dtype):
                return FakeImage(
                    shape=self.shape,
                    channels=self.channels,
                    dtype=dtype,
                    source=self.source or self,
                    scale=self.scale,
                )

            def __truediv__(self, value):
                return FakeImage(
                    shape=self.shape,
                    channels=self.channels,
                    dtype=self.dtype,
                    source=self.source or self,
                    scale=value,
                )

        class FakeCanvas:
            def __init__(self, shape, dtype):
                self.shape = shape
                self.dtype = dtype
                self.assignments = []

            def __setitem__(self, key, value):
                self.assignments.append((key, value))

        class FakeCV2:
            COLOR_BGR2RGB = "bgr2rgb"
            INTER_LINEAR = "linear"

            @staticmethod
            def cvtColor(frame, code):
                assert code == FakeCV2.COLOR_BGR2RGB
                return frame[..., ::-1]

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
            def ascontiguousarray(value):
                return value

        runtime = RuntimeModules(cv2=FakeCV2(), np=FakeNp(), pyneat=None)
        contract = QuantTessCpuPreproc(width=8, height=8, aspect_ratio=True, padding_type="CENTER")
        frame_bgr = FakeImage(shape=(2, 4, 3), channels="bgr")

        quant_input = cpu_quanttess_input(runtime, frame_bgr, contract)

        assert quant_input.shape == (8, 8, 3)
        assert quant_input.dtype == runtime.np.float32
        key, value = quant_input.assignments[0]
        assert key == (slice(2, 6, None), slice(0, 8, None))
        assert value.shape == (4, 8, 3)
        assert value.channels == "rgb"
        assert value.dtype == runtime.np.float32
        assert value.scale == 255.0
