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
| `config` | Reads `config.yaml` by dotted key, such as `input.rtsp.tcp`, and rejects keys the example never read | `config.py` | `config.h` |
| `labels` | Reads a class-label file with one label per line | `labels.py` | `labels.h` |
| `rtsp_probe` | Reads an RTSP stream's frame size and frame rate before the graph is built | `rtsp_probe.py` | `rtsp_probe.h` |
| `detection_egress` | Builds the object-detection metadata message that Insight draws over the video | `detection_egress.py` | `detection_egress.h` |
| `pull_status` | Reads the status of one `Run::pull`: a sample arrived, or the pull timed out and the loop tries again. A closed output or a pull error throws with the reason | | `pull_status.h` |
| `example_utils` | Converts tensors to OpenCV images and float vectors, and other small C++ helpers | | `example_utils.h` |
| `obj_detection_utils` | Parses and draws bounding boxes from a box tensor | | `obj_detection_utils.h` |
| `asset_utils` | Downloads sample files | | `asset_utils.h` |

The C++ source ships in the installed package as reference. The packaged C++ executables are
already built with it.
