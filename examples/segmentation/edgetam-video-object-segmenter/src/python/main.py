"""Follow one object through a video file with EdgeTAM and save its mask in every frame, using pyneat."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pyneat
import yaml

from utils.tracker import (
    EMBEDDING_SHAPE,
    HIGH_RES_SHAPES,
    IMAGE_SIZE,
    MEMORY_DIM,
    MEMORY_TOKENS,
    POINTER_DIM,
    MemoryBank,
    choose_mask,
    locate_object,
    memory_mask,
    prompt_arrays,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
RUN_TIMEOUT_MS = 10000
# The mask's tint and outline, and the ring around the prompt point, in BGR.
MASK_COLOR = (255, 170, 0)
PROMPT_COLOR = (255, 255, 255)
MASK_OPACITY = 0.45


@dataclass(frozen=True)
class Config:
    image_encoder: str
    memory_attention: str
    mask_decoder: str
    memory_encoder: str
    video_path: str
    prompt_x: float
    prompt_y: float
    frames: int
    output_dir: Path
    save_every: int


@dataclass(frozen=True)
class Runners:
    image_encoder: pyneat.ModelRunner
    memory_attention: pyneat.ModelRunner
    mask_decoder: pyneat.ModelRunner
    memory_encoder: pyneat.ModelRunner


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    model, prompt, output = raw["model"], raw["prompt"], raw["output"]
    cfg = Config(
        image_encoder=model["image_encoder"],
        memory_attention=model["memory_attention"],
        mask_decoder=model["mask_decoder"],
        memory_encoder=model["memory_encoder"],
        video_path=raw["source"]["video_path"],
        prompt_x=float(prompt["x"]),
        prompt_y=float(prompt["y"]),
        frames=raw["inference"]["frames"],
        output_dir=Path(output["dir"]),
        save_every=output["save_every"],
    )
    for name in ("image_encoder", "memory_attention", "mask_decoder", "memory_encoder"):
        if not Path(getattr(cfg, name)).is_file():
            raise ValueError(f"model.{name} {getattr(cfg, name)} does not exist; download the EdgeTAM model and "
                             "point it at that package")
    if not (0 <= cfg.prompt_x <= 1 and 0 <= cfg.prompt_y <= 1):
        raise ValueError(f"prompt.x {cfg.prompt_x} and prompt.y {cfg.prompt_y} must be fractions of the frame, "
                         "from 0 to 1")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be 0 or more")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be 0 or more")
    return cfg


def open_video(path: str) -> cv2.VideoCapture:
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        raise ValueError(f"could not open source.video_path {path}; point it at a video file")
    return video


def image_encoder_options(width: int, height: int) -> pyneat.ModelOptions:
    # Neat stretches the RGB frame bicubically to the model's 1024x1024 input and applies ImageNet's mean and std on
    # the board.
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.input_max_width = width
    options.preprocess.input_max_height = height
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.resize.width = options.preprocess.resize.height = IMAGE_SIZE
    options.preprocess.resize.interpolation = pyneat.ResizeInterpolation.Bicubic
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    options.preprocess.preset = pyneat.NormalizePreset.ImageNet
    # The output stage runs on the EV74 and writes memory the other models' input stages read, so the embeddings
    # and features pass on without a copy.
    options.processcvu.post_run_target = "EV74"
    return options


def tensor_options() -> pyneat.ModelOptions:
    # The inputs are features, memories and prompts, not images.
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    return options


def mask_decoder_options() -> pyneat.ModelOptions:
    # The input stage runs on the CPU, which reads the memory attention's output where its CPU output stage wrote
    # it and the image encoder's features where the EV74 wrote them.
    options = tensor_options()
    options.processcvu.pre_run_target = "A65"
    return options


def ev74_tensor(array: np.ndarray) -> pyneat.Tensor:
    # The array already has the package's shape, so the tensor declares no layout: Neat would read a 3D array as
    # channels first and reject it.
    return pyneat.Tensor.from_numpy(np.ascontiguousarray(array, np.float32), copy=True,
                                    memory=pyneat.TensorMemory.EV74, layout=pyneat.TensorLayout.Unknown)


def frame_tensor(frame_bgr: np.ndarray) -> pyneat.Tensor:
    return pyneat.Tensor.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), copy=True,
                                    image_format=pyneat.PixelFormat.RGB, memory=pyneat.TensorMemory.EV74)


def build_runners(cfg: Config, width: int, height: int) -> Runners:
    """Loads the four models and builds one runner for each, from inputs of the shapes it takes."""
    frame = frame_tensor(np.zeros((height, width, 3), np.uint8))
    embeddings = ev74_tensor(np.zeros(EMBEDDING_SHAPE))
    high_res = [ev74_tensor(np.zeros(shape)) for shape in HIGH_RES_SHAPES]
    bank = MemoryBank()
    bank.add(np.zeros((MEMORY_TOKENS, MEMORY_DIM), np.float32), np.zeros(POINTER_DIM, np.float32))
    memories = [ev74_tensor(array) for array in bank.arrays()]
    prompt = [ev74_tensor(array) for array in prompt_arrays(None)]
    mask = ev74_tensor(np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 1)))

    image_encoder = pyneat.Model(cfg.image_encoder, image_encoder_options(width, height))
    memory_attention = pyneat.Model(cfg.memory_attention, tensor_options())
    mask_decoder = pyneat.Model(cfg.mask_decoder, mask_decoder_options())
    memory_encoder = pyneat.Model(cfg.memory_encoder, tensor_options())
    return Runners(
        image_encoder=image_encoder.build([frame]),
        memory_attention=memory_attention.build([embeddings, *memories]),
        mask_decoder=mask_decoder.build([embeddings, *high_res, *prompt]),
        memory_encoder=memory_encoder.build([embeddings, mask]),
    )


def track_frame(runners: Runners, bank: MemoryBank, frame_bgr: np.ndarray,
                prompt: tuple[float, float] | None) -> np.ndarray:
    """Segments the object in one frame and adds the frame to the memory bank. The first frame takes the prompt
    point; a later frame takes None and finds the object from the bank. Returns the [256, 256] mask logits."""
    embeddings, *high_res = runners.image_encoder.run([frame_tensor(frame_bgr)], timeout_ms=RUN_TIMEOUT_MS)
    conditioned = embeddings
    if prompt is None:
        memories = [ev74_tensor(array) for array in bank.arrays()]
        (conditioned,) = runners.memory_attention.run([embeddings, *memories], timeout_ms=RUN_TIMEOUT_MS)
    points = [ev74_tensor(array) for array in prompt_arrays(prompt)]
    decoded = runners.mask_decoder.run([conditioned, *high_res, *points], timeout_ms=RUN_TIMEOUT_MS)
    mask, pointer = choose_mask(*(tensor.to_numpy(copy=False) for tensor in decoded))
    remembered = ev74_tensor(memory_mask(mask, prompted=prompt is not None))
    (memory,) = runners.memory_encoder.run([embeddings, remembered], timeout_ms=RUN_TIMEOUT_MS)
    bank.add(memory.to_numpy(copy=True), pointer)
    return mask


def draw_mask(frame_bgr: np.ndarray, mask: np.ndarray, prompt: tuple[float, float] | None) -> None:
    """Tints and outlines the object in the frame, in place, and rings the prompt point on the prompted frame."""
    height, width = frame_bgr.shape[:2]
    covered = (cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR) > 0).astype(np.uint8)
    tinted = cv2.addWeighted(frame_bgr, 1 - MASK_OPACITY, np.full_like(frame_bgr, MASK_COLOR), MASK_OPACITY, 0)
    cv2.copyTo(tinted, covered, frame_bgr)
    contours, _ = cv2.findContours(covered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame_bgr, contours, -1, MASK_COLOR, 2)
    if prompt is not None:
        cv2.circle(frame_bgr, (int(prompt[0] * width), int(prompt[1] * height)), 6, PROMPT_COLOR, 2)


def run(cfg: Config) -> None:
    video = open_video(cfg.video_path)
    width, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    runners = build_runners(cfg, width, height)
    print(f"video={cfg.video_path} frame={width}x{height} prompt=({cfg.prompt_x}, {cfg.prompt_y})", flush=True)

    bank = MemoryBank()
    results = []
    started = time.perf_counter()
    try:
        while cfg.frames == 0 or len(results) < cfg.frames:
            read, frame_bgr = video.read()
            if not read:
                break
            index = len(results)
            prompt = (cfg.prompt_x, cfg.prompt_y) if index == 0 else None
            mask = track_frame(runners, bank, frame_bgr, prompt)
            location = locate_object(mask)
            results.append({"frame": index, **asdict(location)})
            if cfg.save_every > 0 and index % cfg.save_every == 0:
                draw_mask(frame_bgr, mask, prompt)
                image_path = cfg.output_dir / f"frame_{index:06d}.jpg"
                if not cv2.imwrite(str(image_path), frame_bgr):
                    raise RuntimeError(f"could not write {image_path}; check that output.dir is writable")
                where = (f"object at ({location.center_x:.2f}, {location.center_y:.2f}), {location.area:.1%} of "
                         "the frame" if location.visible else "no object")
                print(f"frame {index}: {where} -> {image_path}", flush=True)
    finally:
        for runner in (runners.image_encoder, runners.memory_attention, runners.mask_decoder,
                       runners.memory_encoder):
            runner.close()
        video.release()
    if not results:
        raise RuntimeError(f"{cfg.video_path} has no frame to read")

    seconds = time.perf_counter() - started
    results_path = cfg.output_dir / "results.json"
    results_path.write_text(json.dumps({"frames": results}, indent=2) + "\n", encoding="utf-8")
    visible = sum(result["visible"] for result in results)
    print(f"tracked {len(results)} frames in {seconds:.1f} s ({len(results) / seconds:.1f} frames/s), object visible "
          f"in {visible}; results in {results_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Follow one object through a video file with EdgeTAM, from its position on the first frame, and "
                    "save its mask in every frame")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
