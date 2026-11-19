"""Unit tests for clip-zero-shot-classifier (Python)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
CONFIG = EXAMPLE_DIR / "src" / "common" / "config.yaml"
PROMPTS = EXAMPLE_DIR / "src" / "common" / "prompts.txt"

sys.path.insert(0, str(MAIN_PY.parent))
_SPEC = importlib.util.spec_from_file_location("clip_zero_shot_main", MAIN_PY)
assert _SPEC is not None and _SPEC.loader is not None
main = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = main
_SPEC.loader.exec_module(main)

from clip_tokenizer import ClipTokenizer  # noqa: E402

# The model package's tokenizer.json is not in the repository, so the tests build a small one with the same
# structure: printable ASCII characters at their code points, the same characters ending a word at code point + 100,
# each merge's result at 300 + its rank, and the start and end tokens at 400 and 401. "é" is the bytes C3 A9, which
# the byte-level vocabulary spells "Ã" and "©", so those two get the same scheme. The C++ unit test builds the same
# file and asserts the same ids.
# "e l" ranks after "l l", so "elle" shows that the lowest-ranked pair merges first, not the leftmost.
MERGES = ["h e", "l l", "he ll", "hell o</w>", "' s</w>", "c a", "ca t</w>", "l o</w>", "e l"]
START, END = 400, 401
TOKENS = {
    "elle": [101, 301, 201],
    "hello": [303],
    "Hello,  CAT's!": [303, 144, 306, 304, 133],
    "hell 12": [300, 108, 208, 149, 150],
    "\tlo  hello\n": [307, 303],
    "hellohello": [302, 111, 303],
    "cats": [305, 116, 215],
    "It's 7.": [105, 216, 304, 155, 146],
    # "é" is a letter, so "café" is one word and "f" does not end it.
    "café": [305, 102, 195, 269],
}
# Prompt vectors whose cosine with the image is exactly 0.28, 0.30, 0.20 and 0.29. The image is not unit length.
COSINES = [0.28, 0.30, 0.20, 0.29]
TOP_3 = [
    {"label": "b", "confidence": 0.6652, "similarity": 0.3},
    {"label": "d", "confidence": 0.2447, "similarity": 0.29},
    {"label": "a", "confidence": 0.09, "similarity": 0.28},
]


@pytest.fixture
def tokenizer_json(tmp_path) -> Path:
    vocab = {chr(c): c for c in range(33, 127)} | {chr(c) + "</w>": c + 100 for c in range(33, 127)}
    vocab |= {"Ã": 195, "©</w>": 269}
    vocab |= {merge.replace(" ", ""): 300 + rank for rank, merge in enumerate(MERGES)}
    vocab |= {"<|startoftext|>": START, "<|endoftext|>": END}
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"model": {"vocab": vocab, "merges": MERGES}}), encoding="utf-8")
    return path


def write_config(tmp_path, section, key, value) -> Path:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw[section][key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def prompt_vectors() -> tuple[np.ndarray, np.ndarray]:
    vectors = np.zeros((len(COSINES), main.EMBED_DIM), np.float32)
    for i, cosine in enumerate(COSINES):
        vectors[i, 0], vectors[i, i + 1] = cosine, np.sqrt(1 - cosine * cosine)
    image = np.zeros(main.EMBED_DIM, np.float32)
    image[0] = 2.0
    return vectors, image


def tensor(array: np.ndarray):
    return SimpleNamespace(shape=list(array.shape), to_numpy=lambda copy: array)


@pytest.mark.unit
def test_packaged_config_loads():
    cfg = main.load_config(CONFIG)
    assert (cfg.codec, cfg.tcp, cfg.latency_ms, cfg.frames) == ("h264", True, 100, 0)
    assert (cfg.profile, cfg.profile_interval) == (False, 100)
    assert (cfg.video_port, cfg.metadata_port) == (9000, 9100)
    assert cfg.prompts_path == "examples/classification/clip-zero-shot-classifier/src/common/prompts.txt"
    assert cfg.tokenizer == "<tokenizer-path>" and cfg.token_embeddings == "<token-embeddings-path>"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("runtime", "profile_interval", 0, "runtime.profile_interval must be > 0"),
        ("source", "codec", "vp9", "source.codec must be h264, h265 or mjpeg"),
    ],
)
def test_config_rejects_invalid_values(tmp_path, section, key, value, message):
    with pytest.raises(ValueError, match=message):
        main.load_config(write_config(tmp_path, section, key, value))


@pytest.mark.unit
def test_packaged_prompts_load():
    prompts = main.load_prompts(str(PROMPTS))
    assert len(prompts) >= 2 and all(prompt == prompt.strip() and prompt for prompt in prompts)


@pytest.mark.unit
def test_prompts_skip_blank_lines_and_trim(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("\n  a dog \n\n\ta cat\n   \n", encoding="utf-8")
    assert main.load_prompts(str(path)) == ["a dog", "a cat"]


@pytest.mark.unit
def test_missing_prompts_file_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing.txt does not exist"):
        main.load_prompts(str(tmp_path / "missing.txt"))


@pytest.mark.unit
@pytest.mark.parametrize(("content", "count"), [("", 0), ("\n \n", 0), ("a dog\n", 1)])
def test_fewer_than_two_prompts_are_rejected(tmp_path, content, count):
    path = tmp_path / "prompts.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=f"has {count} prompts; write at least two"):
        main.load_prompts(str(path))


@pytest.mark.unit
@pytest.mark.parametrize("prompt", TOKENS)
def test_tokenizer_ids(tokenizer_json, prompt):
    expected = [START, *TOKENS[prompt], END]
    assert ClipTokenizer(tokenizer_json).encode(prompt, 12).tolist() == expected + [END] * (12 - len(expected))


@pytest.mark.unit
def test_tokenizer_truncates_and_keeps_the_end_token(tokenizer_json):
    tokens = ClipTokenizer(tokenizer_json).encode("hello " * 20, 8).tolist()
    assert tokens == [START] + [303] * 6 + [END]


@pytest.mark.unit
def test_prompt_tokens_fill_the_text_encoder_input(tokenizer_json):
    tokens = main.prompt_tokens(str(tokenizer_json), ["hello", "cats"])
    assert tokens.shape == (2, main.TEXT_TOKENS)
    assert tokens[1, :5].tolist() == [START, 305, 116, 215, END] and (tokens[1, 5:] == END).all()


@pytest.mark.unit
def test_prompt_longer_than_the_text_encoder_is_rejected(tokenizer_json):
    fits = "hello " * 75
    assert main.prompt_tokens(str(tokenizer_json), [fits, "cats"])[0, 76] == END
    with pytest.raises(ValueError, match=r"prompt 'hello .* hello' is 76 tokens; the text encoder reads at most 75"):
        main.prompt_tokens(str(tokenizer_json), ["cats", fits + "hello"])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda path: main.prompt_tokens(path, ["a dog", "a cat"]),
         "model.tokenizer .* does not exist; point it at text_encoder/tokenizer.json in the model package"),
        (main.load_token_embeddings,
         "model.token_embeddings .* does not exist; point it at text_encoder/token_embeddings.npy in the model package"),
    ],
)
def test_missing_model_file_is_rejected(tmp_path, call, message):
    with pytest.raises(ValueError, match=message):
        call(str(tmp_path / "missing"))


@pytest.mark.unit
def test_token_table_is_memory_mapped(tmp_path):
    path = tmp_path / "token_embeddings.npy"
    table = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(main.VOCAB_SIZE, main.EMBED_DIM))
    table[49407, :3] = [1.0, 2.0, 3.0]
    table.flush()
    loaded = main.load_token_embeddings(str(path))
    assert isinstance(loaded, np.memmap)
    assert loaded[[49406, 49407]][1, :3].tolist() == [1.0, 2.0, 3.0]


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(10, 512), (49408, 256), (49408,)])
def test_token_table_of_the_wrong_shape_is_rejected(tmp_path, shape):
    path = tmp_path / "token_embeddings.npy"
    np.save(path, np.zeros(shape, np.float32))
    with pytest.raises(ValueError, match=rf"has shape \({shape[0]},.*not \(49408, 512\)"):
        main.load_token_embeddings(str(path))


@pytest.mark.unit
def test_token_table_that_is_not_float32_is_rejected(tmp_path):
    path = tmp_path / "token_embeddings.npy"
    np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(main.VOCAB_SIZE, main.EMBED_DIM)).flush()
    with pytest.raises(ValueError, match="is not little-endian float32 in C order"):
        main.load_token_embeddings(str(path))


@pytest.mark.unit
def test_text_vector_is_the_unit_row_at_the_first_end_token():
    rows = np.arange(main.TEXT_TOKENS * main.EMBED_DIM, dtype=np.float32).reshape(1, main.TEXT_TOKENS, main.EMBED_DIM)
    ids = np.array([49406, 320, 1125, 49407] + [49407] * 73)
    vector = main.end_of_text_row(rows, ids)
    np.testing.assert_allclose(vector, rows[0, 3] / np.linalg.norm(rows[0, 3]), rtol=1e-6)
    assert abs(np.linalg.norm(vector) - 1.0) < 1e-6


@pytest.mark.unit
def test_image_embedding_is_found_by_size():
    embedding = np.arange(main.EMBED_DIM, dtype=np.float32).reshape(1, main.EMBED_DIM)
    logits = np.zeros((1, 1000), np.float32)
    for tensors in ([tensor(embedding)], [tensor(logits), tensor(embedding)], [tensor(embedding), tensor(logits)]):
        sample = SimpleNamespace(kind=main.pyneat.SampleKind.TensorSet, tensor=None, tensors=tensors)
        assert main.image_embedding(sample).tolist() == embedding[0].tolist()
    # A package with only the embedding output may arrive as a single-tensor sample.
    single = SimpleNamespace(kind=main.pyneat.SampleKind.Tensor, tensor=tensor(embedding), tensors=[])
    assert main.image_embedding(single).tolist() == embedding[0].tolist()


@pytest.mark.unit
def test_missing_image_embedding_is_rejected():
    sample = SimpleNamespace(kind=main.pyneat.SampleKind.TensorSet, tensor=None,
                             tensors=[tensor(np.zeros((1, 1000), np.float32))])
    with pytest.raises(RuntimeError, match=r"outputs of sizes \[1000\], none of 512 values"):
        main.image_embedding(sample)


@pytest.mark.unit
def test_top_prompts_rank_by_softmax_of_100_times_cosine():
    vectors, image = prompt_vectors()
    assert main.top_prompts(image, vectors, ["a", "b", "c", "d"]) == TOP_3


@pytest.mark.unit
def test_two_prompts_give_two_top_prompts():
    vectors, image = prompt_vectors()
    assert main.top_prompts(image, vectors[:2], ["a", "b"]) == [
        {"label": "b", "confidence": 0.8808, "similarity": 0.3},
        {"label": "a", "confidence": 0.1192, "similarity": 0.28},
    ]


@pytest.mark.unit
def test_equal_scores_keep_prompt_order():
    vectors, image = prompt_vectors()
    assert main.top_prompts(image, vectors[[1, 1, 0]], ["first", "second", "third"]) == [
        {"label": "first", "confidence": 0.4683, "similarity": 0.3},
        {"label": "second", "confidence": 0.4683, "similarity": 0.3},
        {"label": "third", "confidence": 0.0634, "similarity": 0.28},
    ]


@pytest.mark.unit
def test_top_prompts_serialize_as_classification_metadata():
    vectors, image = prompt_vectors()
    top = main.top_prompts(image, vectors, ["a", "b", "c", "d"])
    assert json.loads(main.top_classes_json(top)) == {"top_classes": TOP_3}


# The result panel is drawn on a flat frame: luma 200, chroma U 90 and V 160, away from neutral gray.
PANEL_TOP = [{"label": "a photo of a dog", "confidence": 0.5}, {"label": "a photo of a cat", "confidence": 0.3},
             {"label": "a photo of a bird", "confidence": 0.2}]
LONG_PROMPT = "a photo of " + "a very " * 20 + "long prompt"
# The same prompts as PANEL_TOP in another order, so "a photo of a cat" moves to the best row.
PANEL_TOP_NEXT = [{"label": "a photo of a cat", "confidence": 0.7}, {"label": "a photo of a bird", "confidence": 0.2},
                  {"label": "a photo of a dog", "confidence": 0.1}]


def flat_nv12(width: int, height: int) -> np.ndarray:
    nv12 = np.full((height * 3 // 2, width), 200, np.uint8)
    nv12[height:, 0::2], nv12[height:, 1::2] = 90, 160
    return nv12


def drawn_panel(width: int, height: int, top: list[dict],
                texts: dict | None = None) -> tuple[np.ndarray, np.ndarray, "main.PanelLayout"]:
    before = flat_nv12(width, height)
    after = before.copy()
    layout = main.panel_layout(width, height, len(top))
    main.draw_result_panel(after, top, layout, {} if texts is None else texts)
    return before, after, layout


@pytest.mark.unit
@pytest.mark.parametrize(("width", "height", "top"), [(1280, 720, PANEL_TOP), (640, 360, PANEL_TOP[:2])])
def test_panel_changes_only_its_box(width, height, top):
    before, after, layout = drawn_panel(width, height, top)
    x, y, w, h = layout.box
    assert len(layout.rows) == len(top) and x > 0 and y > 0 and x + w < width and y + h < height
    changed = np.argwhere(before != after)
    luma, chroma = changed[changed[:, 0] < height], changed[changed[:, 0] >= height] - [height, 0]
    assert len(luma) and len(chroma)
    assert (luma.min(axis=0) >= [y, x]).all() and (luma.max(axis=0) < [y + h, x + w]).all()
    assert (chroma.min(axis=0) >= [y // 2, x]).all() and (chroma.max(axis=0) < [(y + h) // 2, x + w]).all()


@pytest.mark.unit
def test_panel_scales_with_the_frame_height():
    assert main.panel_layout(640, 360, 3).box[2] * 2 == main.panel_layout(1280, 720, 3).box[2] == 640


@pytest.mark.unit
def test_panel_darkens_its_box_and_pulls_chroma_toward_gray():
    _, after, layout = drawn_panel(1280, 720, PANEL_TOP)
    x, y, _, _ = layout.box
    # The box's top left corner holds no text or bar.
    assert after[y + 1, x + 1] == 60
    assert after[720 + y // 2, x:x + 2].tolist() == [117, 138]


@pytest.mark.unit
def test_panel_bars_follow_the_probabilities():
    _, after, layout = drawn_panel(1280, 720, PANEL_TOP)
    for i, (row, prompt) in enumerate(zip(layout.rows, PANEL_TOP)):
        bar_x, bar_y, bar_width, bar_height = row.bar
        line = after[bar_y + bar_height // 2, bar_x:bar_x + bar_width]
        fill = main.even(bar_width * prompt["confidence"])
        assert (line[:fill] == (main.ACCENT_YUV[0] if i == 0 else main.BAR_LUMA)).all()
        assert (line[fill:] == main.TRACK_LUMA).all()


@pytest.mark.unit
def test_panel_best_row_stands_out():
    _, after, layout = drawn_panel(1280, 720, PANEL_TOP)
    best, second = layout.rows[:2]
    assert best.font_scale > second.font_scale and best.bar[3] > second.bar[3]
    for row, accent in ((best, True), (second, False)):
        bar_x, bar_y, _, _ = row.bar
        uv = after[720 + bar_y // 2, bar_x:bar_x + 2].tolist()
        assert (uv == list(main.ACCENT_YUV[1:])) == accent
        text = after[row.baseline_y - 10:row.baseline_y, bar_x:row.bar[0] + row.label_width]
        assert text.max() == (main.BEST_TEXT_LUMA if accent else main.TEXT_LUMA)


@pytest.mark.unit
def test_panel_percentage_ends_at_the_bar_end():
    _, after, layout = drawn_panel(1280, 720, PANEL_TOP)
    for row in layout.rows:
        bar_x, _, bar_width, _ = row.bar
        text = after[row.baseline_y - 10:row.baseline_y, bar_x:bar_x + bar_width]
        lit = np.flatnonzero((text > 100).any(axis=0))
        # The percentage's last lit column lies at the bar's right end, and a gap separates it from the label.
        assert bar_width - 2 * row.thickness <= lit[-1] < bar_width + row.thickness
        assert row.label_width < lit[-1] and not (text[:, row.label_width - 1] > 100).any()


@pytest.mark.unit
def test_panel_text_cache_draws_like_a_new_cache():
    texts = {}
    first, second = drawn_panel(1280, 720, PANEL_TOP, texts)[1], drawn_panel(1280, 720, PANEL_TOP_NEXT, texts)[1]
    assert (first == drawn_panel(1280, 720, PANEL_TOP)[1]).all()
    assert (second == drawn_panel(1280, 720, PANEL_TOP_NEXT)[1]).all()


@pytest.mark.unit
def test_long_prompt_is_cut_once_per_text_and_width():
    texts = {}
    _, _, layout = drawn_panel(1280, 720, [{"label": LONG_PROMPT, "confidence": 1.0}], texts)
    row = layout.rows[0]
    label = [image for key, (image, _) in texts.items() if key[0] == LONG_PROMPT]
    assert [key[4] for key in texts if key[0] == LONG_PROMPT] == [row.label_width]
    assert label[0].shape[1] <= row.label_width + 2 * row.thickness


@pytest.mark.unit
def test_long_prompt_is_cut_to_its_column():
    row = main.panel_layout(1280, 720, 3).rows[0]
    cut = main.fit_text(LONG_PROMPT, row.font_scale, row.thickness, row.label_width)
    assert cut.startswith("a photo of a very") and cut.endswith("...") and len(cut) < len(LONG_PROMPT)
    assert main.text_width(cut, row.font_scale, row.thickness) <= row.label_width
    assert main.fit_text("a photo of a dog", row.font_scale, row.thickness, row.label_width) == "a photo of a dog"
    # The panel font draws ASCII only.
    assert main.fit_text("caf\u00e9", row.font_scale, row.thickness, row.label_width) == "caf?"


def padded_nv12_frame(width: int, height: int, stride: int, uv_offset: int) -> tuple["main.pyneat.Sample", np.ndarray]:
    """A joined-frame field holding an NV12 tensor with padded rows, its UV plane apart from the Y plane and its last
    row as long as the width, and the frame it holds."""
    size = uv_offset + (height // 2 - 1) * stride + width
    buffer = (np.arange(size) % 251).astype(np.uint8)
    tensor = main.pyneat.Tensor.from_numpy(buffer, copy=False, image_format=main.pyneat.PixelFormat.NV12,
                                           memory=main.pyneat.TensorMemory.CPU)
    planes = []
    roles = main.pyneat.PlaneRole
    for role, rows, offset in ((roles.Y, height, 0), (roles.UV, height // 2, uv_offset)):
        plane = main.pyneat.Plane()
        plane.role, plane.shape, plane.strides_bytes, plane.byte_offset = role, [rows, width], [stride, 1], offset
        planes.append(plane)
    tensor.shape, tensor.strides_bytes, tensor.planes = [height, width], [], planes
    expected = np.vstack([buffer[offset + row * stride:offset + row * stride + width][None]
                          for offset, rows in ((0, height), (uv_offset, height // 2)) for row in range(rows)])
    return main.pyneat.make_tensor_sample("frame", tensor), expected


@pytest.mark.unit
def test_frame_copy_reads_padded_rows():
    frame, expected = padded_nv12_frame(64, 32, 80, 80 * 32 + 32)
    nv12 = np.empty((48, 64), np.uint8)
    main.copy_frame(frame, nv12)
    assert (nv12 == expected).all()


@pytest.mark.unit
def test_frame_of_another_size_is_rejected():
    frame, _ = padded_nv12_frame(64, 32, 80, 80 * 32)
    with pytest.raises(RuntimeError, match="the decoded frame is not 32x16 NV12"):
        main.copy_frame(frame, np.empty((24, 32), np.uint8))


@pytest.mark.unit
def test_pushed_frame_keeps_its_planes_and_source_timestamp():
    nv12 = (np.arange(48 * 64) % 253).astype(np.uint8).reshape(48, 64)
    sample = main.nv12_sample(nv12, SimpleNamespace(pts_ns=123_000_000, frame_id=7))
    assert (sample.pts_ns, sample.frame_id) == (123_000_000, 7)
    planes = [(plane.role, list(plane.shape), list(plane.strides_bytes), plane.byte_offset)
              for plane in main.sample_tensors(sample)[0].planes]
    assert planes == [(main.pyneat.PlaneRole.Y, [32, 64], [64, 1], 0),
                      (main.pyneat.PlaneRole.UV, [16, 64], [64, 1], 32 * 64)]
    copied = np.empty_like(nv12)
    main.copy_frame(sample, copied)
    assert (copied == nv12).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("tag", "codec"),
    [("h264", "h264"), ("avc1", "h264"), ("hevc", "h265"), ("hvc1", "h265"), ("MJPG", "mjpeg"), ("VP90", None)],
)
def test_stream_codec_comes_from_the_fourcc(tag, codec):
    assert main.fourcc_codec(float(sum(ord(c) << 8 * i for i, c in enumerate(tag)))) == codec
    assert main.fourcc_codec(0.0) is None


@pytest.mark.unit
def test_cli_rejects_unknown_flags():
    r = subprocess.run([sys.executable, str(MAIN_PY), "--bogus"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert "unrecognized arguments" in r.stderr
