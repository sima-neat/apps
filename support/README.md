# Shared Example Library

`support/` holds the helpers that more than one example uses, such as the config reader and the
RTSP probe. Python examples import it as `sima_examples`, and the C++ examples include it as
`sima_examples/<file>.h`.

The helpers may read and convert Neat data types, such as a `Tensor`, a `Sample`, or a
`PullStatus`. They never create a model, build a graph, or start a run. Each example's
`main.py` and `main.cpp` do that themselves, so you can read the whole Neat flow in one place.

## Where code goes

| Code | Place |
| --- | --- |
| Creating models, building and connecting the graph, running it, and closing it | The example's `src/python/main.py` and `src/cpp/main.cpp` |
| Logic only one example needs, such as RF-DETR's postprocessing or a tracker | That example's `src/python/utils/` and `src/cpp/utils/` |
| Logic two or more examples need | `support/python/sima_examples/` and `support/cpp/sima_examples/` |

Code moves from an example's `utils/` to `support/` when a second example needs it.

## Modules

A module in both columns has the same file name, function names, and behavior in each language.

| Module | What it does | Python | C++ |
| --- | --- | --- | --- |
| `config` | Reads `config.yaml` by dotted key, such as `input.rtsp.tcp`. A missing key or a value of the wrong type raises with the key's name. `choice(key, table)` returns the table entry the key's value names, and `lookup(key, value, table)` does the same for a value `main` derives; an unknown value raises with the key, the value, and the choices. Keys the example does not read are ignored | `config.py` | `config.h` |
| `labels` | Reads a class-label file with one label per line | `labels.py` | `labels.h` |
| `rtsp_probe` | Reads an RTSP stream's frame size and frame rate before the graph is built, for each value the config leaves at 0 or below | `rtsp_probe.py` | `rtsp_probe.h` |
| `stop_signals` | Sets a stop flag on Ctrl-C, SIGTERM, or a dropped ssh session (SIGHUP), so the pull loop ends and the run closes | `stop_signals.py` | `stop_signals.h` |
| `detection_egress` | Builds the metadata messages that Insight draws over the video: `serialize` the object-detection message, and `serialize_tracks` the tracking message, which carries each box's track ID. `track_entries` builds only the tracking message's `data.tracks` entries, for an application that sends its own message. Each box is clipped to the frame before its size is measured. Until the multi-stream tracker is converted (#789), only the tiny-drone tracker calls `serialize_tracks`, and the multi-stream tracker calls `track_entries`. `frame_metadata` copies a pulled sample's frame identity into either | `detection_egress.py` | `detection_egress.h` |
| `pull_status` | Reads the status of one `Run::pull`: a sample arrived, or the pull timed out and the loop tries again. A closed output or a pull error throws with the reason | | `pull_status.h` |
| `example_utils` | Converts tensors to OpenCV images and float vectors, and other small C++ helpers | | `example_utils.h` |
| `obj_detection_utils` | Parses and draws bounding boxes from a box tensor | | `obj_detection_utils.h` |
| `asset_utils` | Downloads sample files | | `asset_utils.h` |

Test helpers live in `support/cpp/sima_examples/testing/` for C++ and `tests/utils/` for
Python. `validate_config_body` runs an example with `--validate-config-only`, so only the
examples that still read their config into an `AppConfig` use it. It goes away with the last
of them.

## Changing the library

A pull request that adds, removes, or changes a module, function, or behavior here updates
this page in the same pull request, and lists the change in its description under `Shared
library`. Each example that calls the changed function is updated in that pull request too.

The C++ source ships in the installed package as reference. The packaged C++ executables are
already built with it.
