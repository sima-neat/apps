"""Pipeline builder tests for the multi-camera people tracking example."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


@pytest.mark.unit
class TestPipelineBuilders:
    def test_build_detection_run_uses_quanttess_mla_and_boxdecode_tensor_contract(self):
        from utils.config import AppConfig
        from utils.pipeline import QuantTessCpuPreproc, RtspProbe, RuntimeModules, build_detection_run

        class FakeSession:
            def __init__(self):
                self.added = []
                self.build_calls = []

            def add(self, node):
                self.added.append(node)

            def build(self, *args, **kwargs):
                self.build_calls.append((args, kwargs))
                return "fake-run"

        class FakeQuantTessOptions:
            def __init__(self, model):
                self.model = model

        class FakePyneat:
            def __init__(self):
                self.last_session = None
                self.nodes = SimpleNamespace(
                    input=lambda opt=None: ("input", opt),
                    quant_tess=lambda opt: ("quant_tess", opt),
                    sima_box_decode=lambda model, **kwargs: ("boxdecode", model, kwargs),
                    output=lambda: ("output",),
                )
                self.groups = SimpleNamespace(mla=lambda model: ("mla", model))

            def Session(self):
                self.last_session = FakeSession()
                return self.last_session

            def QuantTessOptions(self, model):
                return FakeQuantTessOptions(model)

        class FakeModel:
            def input_appsrc_options(self, copy):
                return SimpleNamespace(
                    copy=copy,
                    media_type="application/vnd.simaai.tensor",
                    format="FP32",
                )

        class FakeNp:
            float32 = "float32"

            @staticmethod
            def zeros(shape, dtype=None):
                return {"shape": shape, "dtype": dtype}

        cfg = AppConfig(
            model="model.tar.gz",
            rtsp_urls=["rtsp://camera-0"],
            output_dir=None,
            frames=0,
            optiview_host="127.0.0.1",
            optiview_video_port_base=9000,
            optiview_json_port_base=9100,
            fps=0,
            bitrate_kbps=2500,
            save_every=30,
            profile=False,
            person_class_id=0,
            detection_threshold=None,
            nms_iou_threshold=None,
            top_k=None,
            tracker_iou_threshold=0.3,
            tracker_max_missing=15,
            latency_ms=200,
            tcp=False,
        )
        probe = RtspProbe(width=1280, height=720, fps=16)
        preproc = QuantTessCpuPreproc(width=640, height=640, aspect_ratio=True, padding_type="CENTER")
        runtime = RuntimeModules(cv2=None, np=FakeNp(), pyneat=FakePyneat())

        _, run = build_detection_run(runtime, cfg, FakeModel(), probe, preproc)

        added_kinds = [node[0] for node in runtime.pyneat.last_session.added]
        assert run == "fake-run"
        assert added_kinds == ["input", "quant_tess", "mla", "boxdecode", "output"]

    def test_build_optiview_json_output_uses_per_stream_ports(self):
        from utils.config import AppConfig
        from utils.pipeline import RuntimeModules, build_optiview_json_output

        class FakeChannelOptions:
            def __init__(self):
                self.host = ""
                self.channel = 0
                self.video_port_base = 0
                self.json_port_base = 0

        class FakeJsonOutput:
            def __init__(self, opt):
                self.opt = opt

        class FakePyneat:
            def OptiViewChannelOptions(self):
                return FakeChannelOptions()

            def OptiViewJsonOutput(self, opt):
                return FakeJsonOutput(opt)

        cfg = AppConfig(
            model="model.tar.gz",
            rtsp_urls=["rtsp://camera-0"],
            output_dir=None,
            frames=0,
            optiview_host="192.168.0.107",
            optiview_video_port_base=9000,
            optiview_json_port_base=9100,
            fps=0,
            bitrate_kbps=2500,
            save_every=30,
            profile=False,
            person_class_id=0,
            detection_threshold=0.25,
            nms_iou_threshold=0.55,
            top_k=120,
            tracker_iou_threshold=0.3,
            tracker_max_missing=15,
            latency_ms=120,
            tcp=True,
        )
        runtime = RuntimeModules(cv2=None, np=None, pyneat=FakePyneat())

        sender = build_optiview_json_output(runtime, cfg, stream_index=2)

        assert sender.opt.host == cfg.optiview_host
        assert sender.opt.channel == 2
        assert sender.opt.video_port_base == cfg.optiview_video_port_base
        assert sender.opt.json_port_base == cfg.optiview_json_port_base
