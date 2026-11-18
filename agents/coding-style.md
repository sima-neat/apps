# Coding Style for Examples

A customer reads an example to learn how a Neat application is built. Write the code so that
reader follows it in one pass. `examples/object-detection/high-density-multi-stream-object-detector`
is the reference: read its `main.py` and `main.cpp` before writing or reviewing an entrypoint.

## Flow

- `main` reads top to bottom as the application: parse the command line, load the config,
  create the `Model`, build the graph, build the run, process the outputs, close the run.
- Each step is one line or one call to a function named for what it does. A reader understands
  the application from `main` without opening the functions.
- The C++ and Python entrypoints have the same functions, with the same names, in the same
  order.

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
  struct (`video_options`); a verb names work that takes time or touches a device
  (`send_detections`, `probe_rtsp_stream_info`).
- A variable name states its meaning and, when it has one, its unit: `latency_ms`,
  `fps_by_url`, `metadata_senders`.
- Use one name per concept, the same name in both languages and in the config key:
  `input.rtsp.latency_ms` becomes `rtsp_latency_ms`.

## Configuration

- `config.yaml` owns every value a customer may change. The code has no default for a config
  value; a missing key is an error that names the key.
- Read the config once, at the start, into one typed `AppConfig`, and pass that object.
- Read it with `sima_examples.config.Config` in Python and `sima_examples::ScalarConfig` in
  C++, and call `reject_unread_keys()` after the last read, so an old or misspelled key fails
  at startup.
- A value the same in every shipped profile is not a setting: write it in the code where the
  option is set.

## Functions and state

- Each function does one job. A function described with "and" becomes two functions.
- Give a behavior that a boolean parameter would switch its own function.
- Keep state in `main` and pass it down. Use no globals except the C++ stop flag a signal
  handler sets.
- Delete every function, option, branch, and config key that no entry point reaches.

## Errors, output, and comments

- An error is one line that names the cause and the key or value involved:
  `input.codec must be h264 or h265`.
- A failure the application tolerates by design, such as a nonblocking send that drops a
  message, carries a one-line comment saying so, and the application reports the count when
  it stops.
- Print only what the customer needs to see the run start, progress, and stop.
- A comment explains a choice the code cannot show, with its evidence. It describes the code
  as it is, not its history.
