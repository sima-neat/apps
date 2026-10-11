"""What every model folder provides to main.py."""

from __future__ import annotations

import pyneat


class Model:
    """A model sets ``neat_model`` and turns its results into Insight metadata.

    main.py feeds the newest decoded frames to ``neat_model`` and reads its output "result".
    """

    neat_model: pyneat.Model

    def results(self, run, stop):
        """Yield result samples until ``stop`` is set. Multi-stage models override this."""
        while not stop.is_set():
            sample = run.pull("result", 500)
            if sample is not None:
                yield sample
            elif not run.running():
                raise RuntimeError(f"result output closed: {run.last_error()}")

    def postprocess(self, sample) -> list[tuple[str, dict]]:
        """Return ``(metadata_type, data)`` pairs to send to Insight for this frame."""
        raise NotImplementedError


def image_options(width: int, height: int, resize_mode, preset) -> pyneat.ModelOptions:
    """Preprocess decoded NV12 frames on EV74: resize, convert to RGB, and normalize."""
    opt = pyneat.ModelOptions()
    pre = opt.preprocess
    pre.kind = pyneat.InputKind.Image
    pre.enable = pyneat.AutoFlag.On
    pre.input_max_width = width
    pre.input_max_height = height
    pre.input_max_depth = 3
    pre.resize.enable = pyneat.AutoFlag.On
    pre.resize.mode = resize_mode
    pre.color_convert.enable = pyneat.AutoFlag.On
    pre.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    pre.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    pre.preset = preset
    return opt


def tensors(sample) -> list:
    """Every tensor in a sample, in output order."""
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        found = [sample.tensor]
    elif sample.kind == pyneat.SampleKind.TensorSet:
        found = list(sample.tensors)
    else:
        found = []
    for field in sample.fields:
        found.extend(tensors(field))
    return found
