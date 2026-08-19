# Adaptive Resolution Policy

**Scope: `--mode adaptive` only.** The example ships two topologies behind one
entry point (see the [README](README.md)). Everything below applies to
`--mode adaptive`, which builds one graph per stream and can therefore resize a
single stream without touching the others. `--mode fused` has neither axis: it
builds one graph for all streams into a shared detector, forwards the source
H.264 untouched, and changes resolution only if you change the sources.

The policy is pure and unit-tested in `src/python/adaptive_policy.py`
(`src/cpp/adaptive_policy.h` for C++); `src/python/adaptive_app.py` and
`src/cpp/adaptive_app.h` call it. `main.py` / `main.cpp` only dispatch on
`--mode`.

`--mode adaptive` adapts resolution on **two independent axes**:

- **Delivered output resolution** (default, always on) — the H.264 video sent to
  Insight, driven by a shared *bandwidth* budget and the active stream count. See
  [The output-resolution budget](#the-output-resolution-budget-delivered-video).
- **Model input tier** (optional, off by default) — the compiled YOLO26n input
  size, driven by per-stream *scene content* under a shared *compute* budget. The
  rest of this document describes this axis; it activates only when
  `adaptive.resolutions` lists more than one size.

Both axes rebuild a stream's pipeline through the same rate-limited, serialized,
per-channel-staggered machinery (Step 4 below applies to both).

---

## The output-resolution budget (delivered video)

Neat exposes **no hardware decoder-bandwidth number** (only a per-stream 4K
ceiling), so the shared limit is a **configured total output pixel-rate**,
`output.adaptive.budget_megapixels_per_s` (default **280**), fair-shared across
active streams. It bounds *encode/deliver* load, not raw decode (single-profile
sources always decode at native).

1. **Candidates** (`output_candidates`) — for a source, the native size plus each
   `output.adaptive.heights` entry **strictly below native** (never upscaled),
   each with an aspect-preserving even width and pixel-rate `w × h × fps`. Sorted
   highest-area first.
2. **Fair share per stream** = `budget_megapixels_per_s / active_streams`.
3. **Selected resolution** (`select_output_index`) = the highest-area candidate
   whose pixel-rate ≤ fair share (clamped to the smallest candidate if none fit).

Because the share depends only on the active stream count, the delivered
resolution changes **only on stream add/remove**, not per frame. With
`budget_megapixels_per_s: 280` and 30 fps 16:9 sources:

| Active streams | Fair share (MP/s) | Delivered per stream | Total (MP/s) |
|---|---|---|---|
| 1 | 280 | **2160p (4K)** — 248.8 | 249 |
| 2 | 140 | 1080p — 62.2 | 124 |
| 4 | 70 | **1080p** — 62.2 | 249 |
| 5 | 56 | 720p — 27.6 | 138 |
| 10 | 28 | **720p** — 27.6 | 276 |
| 16 | 17.5 | 480p — 12.3 | 197 |

Config:

```yaml
output:
  adaptive:
    heights: [2160, 1080, 720, 480]  # candidate delivered heights (clamped to source native)
    budget_megapixels_per_s: 280     # total output pixel-rate fair-shared across streams
```

fps is folded in automatically (a 60 fps source costs 2× and lands a tier lower).
Tune the budget to your platform's sustainable encode/deliver capacity.

---

## The model-input tier (optional axis)

The rest of this document describes the model-tier axis, active only when
`adaptive.resolutions` has more than one size. It has two layers:
1. **Content policy** — what tier a stream's *scene* wants (per stream).
2. **Shared budget** — what tier it's *allowed*, given how many streams are active.

`effective_tier() = min(content_wants, budget_allows)`.

---

## Step 1 — what each frame measures

After parsing detections (only boxes above `inference.min_score`, default 0.30),
it reduces the frame to three numbers:

| Value | Meaning | If no objects |
|---|---|---|
| `object_count` | how many objects | 0 |
| `min_object_px` | the smallest object's *smaller side*, in pixels | ∞ |
| `min_confidence` | the lowest detection score | 1.0 |

---

## Step 2 — which tier the *content* wants (`desired_tier`)

The tier moves **at most one step per decision** (320 ↔ 640 ↔ 960) — never a jump.

**Step UP** if *any* of these is true:

| Trigger | Condition | Default |
|---|---|---|
| small / distant object | `min_object_px < min_object_px` | **< 24 px** |
| low confidence | `object_count > 0` and `min_confidence < confidence_low` | **< 0.40** |
| crowded | `object_count >= density_high` | **≥ 20 objects** |

**Step DOWN** only if it is *not* stepping up **and** all of these hold (a
comfortably easy scene):

| Condition | Default |
|---|---|
| few objects: `object_count <= density_low` | **≤ 5** |
| objects comfortably large: `min_object_px >= min_object_px × down_size_factor` | **≥ 48 px** (24 × 2.0) |
| confident: `min_confidence >= confidence_low + confidence_margin` (or no objects) | **≥ 0.50** (0.40 + 0.10) |

Otherwise the tier stays put. A "medium" scene — a handful of large, confident
objects — matches neither rule and sits still (this is why a people clip with
~14 confident detections parks at 640 rather than climbing or dropping).

---

## Step 3 — hysteresis (anti-thrash)

A desired change **does not commit until it has persisted for
`hysteresis_frames` consecutive frames** (default **15**). A single noisy frame
never moves the tier; if the desire flips before the count is reached, the
counter resets. This is the main knob for how twitchy vs. stable the adaptation
is.

---

## Step 4 — rate limit (protects the MLA runtime)

Even once hysteresis says "switch," a stream **rebuilds its pipeline at most once
every 2.5 s** (a wall-clock floor in the run loop). Rebuilding an MLA pipeline
every frame would thrash the runtime, so switches physically happen no more
often than ~every few seconds regardless of how fast the scene changes.

> A tier switch **reloads that tier's compiled model** (~6–10 s on Modalix), so
> adaptation tracks scenes that change over *tens of seconds to minutes*, not
> sub-10-second flips. For instant switching you would pre-load all three tier
> models (≈3× memory) — a planned enhancement.

---

## The shared budget — how it caps everything

Step 2 gives what the *content* wants. The budget decides what's *allowed*:

1. **Tier cost ∝ pixel area**, normalised so the smallest tier costs 1:
   `cost = (size / 320)²` → **320 = 1, 640 = 4, 960 = 9**.
2. **Fair share per stream** = `budget_units / active_streams`.
3. **Allowed tier** = the highest tier whose cost ≤ that fair share (always ≥ 320).
4. **Final tier = min(content_wants, budget_allows).**

With `budget_units = 12`:

| Active streams | Share (12 ÷ N) | Highest cost ≤ share | Ceiling |
|---|---|---|---|
| 1 | 12 | 9 (960) | **960** |
| 2 | 6 | 4 (640) | **640** |
| 3 | 4 | 4 (640) | **640** |
| 4 | 3 | 1 (320) | **320** |
| 8 | 1.5 | 1 (320) | **320** |

So the budget both **caps** a hot stream and **degrades gracefully**: as you add
streams the ceiling drops automatically, keeping total MLA cost ≈ `budget_units`
so no stream is starved. Example: 2 streams at `budget_units: 12` → ceiling 640
(a stream wanting more is held at 640); raise to `budget_units: 30` → share 15 →
ceiling 960, and a dense stream is free to climb while a sparse one still drops
to 320.

---

## All the knobs

In `src/common/config.yaml`:

```yaml
adaptive:
  resolutions: [320, 640, 960]   # the tiers themselves (must match model.tiers)
  confidence_low: 0.40           # below this → step up
  min_object_px: 24              # smaller than this → step up; ≥ 2× this → eligible to step down
  hysteresis_frames: 15          # frames a change must persist before committing
  density_high: 20               # this many objects → step up
  budget_units: 12               # shared compute budget (cost units)
inference:
  min_score: 0.30                # detection threshold — defines what counts as an "object"
```

Internal constants (not in the YAML — ask if you want them exposed):

| Constant | Value | Role |
|---|---|---|
| `density_low` | 5 | at most this many objects to be eligible to step down |
| `down_size_factor` | 2.0 | objects must be ≥ this × `min_object_px` to step down |
| `confidence_margin` | 0.10 | extra confidence headroom required to step down |
| min switch interval | 2.5 s | wall-clock floor between rebuilds per stream |

---

## Where it is in code

| Function | File | Role |
|---|---|---|
| `frame_stats()` | `adaptive_policy.h/.py` | frame → `object_count` / `min_object_px` / `min_confidence` |
| `desired_tier()` | `adaptive_policy.h/.py` | the step-up / step-down rules above |
| `select_tier()` | `adaptive_policy.h/.py` | applies the hysteresis state machine |
| `tier_cost()` / `budget_allowed_index()` | `adaptive_policy.h/.py` | the model-tier budget math |
| `effective_tier()` | `adaptive_policy.h/.py` | `min(content, budget)` — the tier actually used |
| `output_candidates()` | `adaptive_policy.h/.py` | per-source delivered-resolution candidates (w, h, pixels/s) |
| `select_output_index()` | `adaptive_policy.h/.py` | highest delivered resolution fitting the bandwidth fair share |

The three tiers compile to genuinely different MLA work — measured
**0.32M / 0.67M / 1.24M MLA cycles** for 320 / 640 / 960 on Modalix.
