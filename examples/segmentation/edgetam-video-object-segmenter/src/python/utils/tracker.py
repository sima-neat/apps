"""The CPU steps of EdgeTAM tracking: the prompt encoding, the mask choice and the memory bank."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

import cv2
import numpy as np

# The model's interfaces, channels last as the packages take and return them.
IMAGE_SIZE = 1024
EMBEDDING_SHAPE = (1, 64, 64, 256)
HIGH_RES_SHAPES = ((1, 256, 256, 32), (1, 128, 128, 64))
MASKS, MASK_SIZE = 4, 256
POINTER_DIM, MEMORY_DIM = 256, 64
MEMORY_TOKENS, MEMORY_SLOTS, OBJECT_POINTERS = 512, 7, 16
PROMPT_POINTS = 3
POINT_TYPES = ["padding", "background", "foreground", "box_top_left", "box_bottom_right"]
# The mask logits of a frame where the model's object score says the object is absent.
NO_OBJECT_LOGIT = -1024.0


@dataclass(frozen=True)
class ObjectLocation:
    """Where the object is in one frame, in fractions of the frame rounded to four decimals."""

    visible: bool
    # The share of the frame the mask covers.
    area: float
    # The mask's centre of mass, None when the object is not visible.
    center_x: float | None
    center_y: float | None


def prompt_arrays(point: tuple[float, float] | None) -> list[np.ndarray]:
    """The mask decoder's point_coords and point_types: one foreground point at (x, y) frame fractions on the
    prompted frame, and only padding on a tracked frame, where point is None.

    A coordinate is the centre of its pixel of the 1024x1024 model input, normalized to [-1, 1].
    """
    coords = np.zeros((1, 1, PROMPT_POINTS, 2), np.float32)
    types = np.zeros((1, 1, PROMPT_POINTS, len(POINT_TYPES)), np.float32)
    types[..., POINT_TYPES.index("padding")] = 1
    if point is not None:
        coords[0, 0, 0] = (np.asarray(point, np.float32) * IMAGE_SIZE + 0.5) / IMAGE_SIZE * 2 - 1
        types[0, 0, 0] = np.eye(len(POINT_TYPES), dtype=np.float32)[POINT_TYPES.index("foreground")]
    return [coords, types]


def choose_mask(
    masks: np.ndarray, iou_scores: np.ndarray, object_score: np.ndarray, pointers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The object's [256, 256] mask logits and its object pointer, from the mask decoder's outputs.

    The decoder returns four candidate masks. With one point or none the model's authors keep the best of masks 1
    to 3 by predicted IoU. A frame whose object score is not positive has no object.
    """
    best = 1 + int(np.argmax(iou_scores.reshape(MASKS)[1:]))
    mask = masks.reshape(MASK_SIZE, MASK_SIZE, MASKS)[..., best].copy()
    if float(object_score.reshape(())) <= 0:
        mask[:] = NO_OBJECT_LOGIT
    return mask, pointers.reshape(MASKS, POINTER_DIM)[best].copy()


def memory_mask(mask: np.ndarray, prompted: bool) -> np.ndarray:
    """The memory encoder's [1, 1024, 1024, 1] mask: the logits upsampled bilinearly, thresholded on the prompted
    frame and as probabilities on a tracked one."""
    values = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    if prompted:
        values = (values > 0).astype(np.float32)
    else:
        # 1 / (1 + exp(-logit)). OpenCV's exp overflows to infinity for the no-object logit without a warning,
        # which gives the probability 0 it should.
        values = 1 / (1 + cv2.exp(-values))
    return values.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)


def locate_object(mask: np.ndarray) -> ObjectLocation:
    """The area and centre of the mask's positive logits."""
    rows, columns = np.nonzero(mask > 0)
    if rows.size == 0:
        return ObjectLocation(False, 0.0, None, None)
    return ObjectLocation(
        True,
        round(rows.size / mask.size, 4),
        round(float(columns.mean() + 0.5) / mask.shape[1], 4),
        round(float(rows.mean() + 0.5) / mask.shape[0], 4),
    )


@dataclass
class MemoryBank:
    """The frames the tracker remembers: the prompted frame, then the latest frames, newest first."""

    memories: collections.deque = field(default_factory=lambda: collections.deque(maxlen=MEMORY_SLOTS - 1))
    pointers: collections.deque = field(default_factory=lambda: collections.deque(maxlen=OBJECT_POINTERS - 1))
    prompted_memory: np.ndarray | None = None
    prompted_pointer: np.ndarray | None = None

    def add(self, memory: np.ndarray, pointer: np.ndarray) -> None:
        """Remember one frame: the memory encoder's memory and the chosen mask's object pointer."""
        memory = memory.reshape(MEMORY_TOKENS, MEMORY_DIM)
        if self.prompted_memory is None:
            self.prompted_memory, self.prompted_pointer = memory, pointer
        else:
            self.memories.appendleft(memory)
            self.pointers.appendleft(pointer)

    def arrays(self) -> list[np.ndarray]:
        """The memory attention's memories, memory_mask, object_pointers and object_pointer_mask. An unused slot
        stays zero with a mask of 0."""
        slots = [self.prompted_memory, *self.memories]
        memories = np.zeros((1, MEMORY_SLOTS * MEMORY_TOKENS, MEMORY_DIM), np.float32)
        memories[0, : len(slots) * MEMORY_TOKENS] = np.concatenate(slots)
        used_memories = np.zeros((1, MEMORY_SLOTS), np.float32)
        used_memories[0, : len(slots)] = 1
        kept = [self.prompted_pointer, *self.pointers]
        pointers = np.zeros((1, OBJECT_POINTERS, POINTER_DIM), np.float32)
        pointers[0, : len(kept)] = np.stack(kept)
        used_pointers = np.zeros((1, OBJECT_POINTERS), np.float32)
        used_pointers[0, : len(kept)] = 1
        return [memories, used_memories, pointers, used_pointers]
