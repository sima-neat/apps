# Coding Style for Examples

A customer reads an example to learn how a Neat application is built. Write the code so that
reader follows it in one pass. `examples/object-detection/high-density-multi-stream-object-detector`
is the reference: read its `main.py` and `main.cpp` before writing or reviewing an entrypoint.

## Flow

- `main` reads top to bottom as the Neat flow: parse the command line, load the config,
  create the `Model`, build the graph, build the run, process the outputs, close the run.
- Each step outside the graph is one line or one call to a function named for what it does. A
  reader understands the application from `main` without opening the functions.
- The C++ and Python entrypoints have the same Neat-flow functions, with the same names, in the
  same order. A helper only one language needs exists only there. In C++, `main` only reports errors and calls `run_app`. `run_app` is the function
  these rules call `main`.
- Use the simplest implementation that meets the requirement. Each extra helper,
  abstraction, dependency, process, and compatibility path needs a reason a reader can see.
- Python entrypoints import the Neat Library at the top: `import pyneat`.
- Placement of code outside `main` follows `agents/rules.md` "Where code goes".

## Graph topology

- Build the whole graph in `main`. Every `Graph`, node group, and `connect` call is visible
  there, so a reader sees which part feeds which without searching.
- A function that returns an option struct or a small subgraph is named for what it returns:
  `rtsp_options`, `decoder_graph`, `detector_link`, `run_options`. It sets one node's options
  and connects nothing outside the subgraph it returns.
- Write each node and each link once. Create a subgraph in a variable and connect that
  variable, so two `connect` calls cannot build two copies.
- Set only the options whose values differ from Core's defaults. Check the default in the
  installed header before setting an option, and delete assignments that restate it.

## Names

- A function name says what the function returns or does. A noun names a value or an option
  struct (`video_options`). A verb names work that takes time or touches a device
  (`send_detections`, `probe_rtsp_stream_info`).
- A variable name states its meaning and, when it has one, its unit: `latency_ms`,
  `fps_by_url`, `metadata_senders`.
- Use one name per concept, the same name in both languages and in the config key:
  `input.rtsp.latency_ms` becomes `rtsp_latency_ms`.

## Configuration

- `config.yaml` owns every value a customer may change, with its key name from
  `agents/rules.md` "Config keys". The code reads every key as required, and a missing key is
  an error that names it.
- Each key carries a short comment saying what it does. Values are scalars, lists of scalars,
  or lists of mappings, which the C++ and Python readers parse the same way. A list of
  mappings holds one entry per item, such as one `model` entry per stream, and `main` reads it
  with `mapping_list(key)`. Each field of an entry means what the registered key of that name
  means, such as `path` for `model.path`.
- `main` loads the file once with `sima_examples.config.Config` in Python or
  `sima_examples::Config` in C++ and passes that object to the functions that build Neat
  options. Each function reads the keys it sets, where it sets them, such as
  `options.latency_ms = config.integer("input.rtsp.latency_ms")`, so `main` stays short. Two
  functions may read the same key, such as `output.insight.host` for the video and metadata
  senders. A value derived from several keys, such as the probed stream size, is read once in
  `main` and passed in, and so is a value two parts must agree on, such as
  `inference.max_detections` for the model and the box decode.
- A value that names a table entry, such as `input.codec`, is read with
  `config.choice(key, table)`, or `lookup(key, value, table)` when `main` derives the value. An
  unknown value fails with the key, the value, and the choices:
  `input.codec: unknown value 'vp8' (expected h264, h265, mjpeg)`.
- Paths are used as written, with no `expanduser`, so both languages read them the same way.
- The application reads only the keys it uses: a missing or mistyped key fails with its name,
  and an extra key in a user's own config is ignored. Range and supported-value checks of the
  shipped configs live in the unit tests, as `agents/rules.md` "Tests" describes.
- `config.yaml` sets one configuration: one input type with only the keys it reads. When the
  example supports several input types, `input.type` names the one set, and comments under it
  list the keys each other type takes:

  ```yaml
  input:
    # rtsp, http, or usb. For http, delete codec and rtsp, set source to the MJPEG URL and
    # width, height, and fps to the stream's, and add "app:" with "ssl_strict: true". For usb,
    # delete codec and rtsp, set source to a camera ID, and set width, height, and fps to one
    # of its modes.
    type: rtsp
    source: rtsp://<host>:<port>/<stream>
    codec: h264       # h264, h265, or mjpeg.
    width: 0          # 0 reads the stream's frame size and rate with a probe.
    height: 0         # Set them only for a stream that does not report them.
    fps: 0
    rtsp:
      tcp: true       # Receive RTSP over TCP.
      latency_ms: 100 # RTSP jitter buffer.
  ```

  The code reads only the keys of the configured type.
- An RTSP stream's frame size and rate come from `probe_rtsp_stream_info`, which fills each
  of `input.width`, `input.height`, and `input.fps` the config leaves at 0. A customer then
  switches to a stream of another resolution without editing the config.
- A value a customer has no reason to change, such as an internal buffer count that is the
  same in every shipped profile, is not a setting: write it in the code where the option is
  set.

## Functions and state

- Each function does one job. Split a function when a part is reused or tested on its own.
- In the example's own functions, give a behavior that a boolean parameter would switch its
  own function. A flag that mirrors a Core API, such as `strict` in `decode_bbox_tensor`,
  stays a parameter.
- Keep state in `main` and pass it down. Stop on Ctrl-C, SIGTERM, and SIGHUP with
  `stop_on_signals()` from the shared library: `main` passes its flag to every pull loop and
  closes the run when the loop ends.
- Delete every function, option, branch, and config key that no entry point reaches.

## Errors, output, and comments

- An error is one line that names the cause and the key or value involved:
  `input.codec must be h264 or h265`. Handle the failures a customer following the README can
  hit. Platform tooling owns hardening against concurrent runs or hostile file systems.
- A failure the application tolerates by design, such as a nonblocking send that drops a
  message, carries a one-line comment saying so, and the application reports the count when
  it stops.
- Print only what the customer needs to see the run start, progress, and stop. The result
  the reader came for comes first, and every number names what it measures and its unit:
  `inference 3.0 ms`.
- Comment only what the code cannot say: why it does something, a constraint the reader would
  otherwise break, or a Core gap. Never restate what the code does or what a name says.
- A comment is one line, and two at most. A function or file gets a comment only when its
  name and signature leave something unclear.
- Measurements, Core file and line numbers, and how the code got this way go in the pull
  request description, the cheat sheet, or the skills. A comment may name the issue instead.
  It describes the code as it is.
