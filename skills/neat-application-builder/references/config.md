# Application Configuration

The Apps examples read one `config.yaml` with the same key names, so a setting learned in one
example means the same in the next. New applications follow the same convention.

## Key names

Read an example's `src/common/config.yaml` for the full set. The shared names include:

| Key | Meaning |
|---|---|
| `model.path`, `model.labels` | Model package (`.tar.gz`) and class-label file |
| `input.source` | One RTSP URL, or a list with one URL per stream |
| `input.codec`, `input.width`, `input.height`, `input.fps` | Format of every input stream; `fps: 0` probes each stream |
| `input.rtsp.tcp`, `input.rtsp.latency_ms` | RTSP transport and jitter buffer |
| `inference.min_score`, `inference.nms_iou`, `inference.max_detections` | Detection thresholds and box count |
| `inference.max_inflight_per_stream`, `inference.max_inflight_total` | Frames a shared model holds per stream and in total |
| `output.video_enabled`, `output.insight.host` | Whether video goes to Insight, and Insight's address |
| `output.insight.video_port_base`, `output.insight.metadata_port_base` | Stream `i` uses `port base + i` |
| `app.*` | Settings only this application has |

## Reading it

- Put every value a user may change in the config, and give the code no default for it: a
  missing key is an error that names the key.
- Read the config once into one typed object, and pass that object to the functions that
  need it.
- Reject keys the application never reads, so a renamed or misspelled key fails at startup
  instead of being ignored.
- A value that never changes between configurations is not a setting: set it in the code
  where the option is configured.
