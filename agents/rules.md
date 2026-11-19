# Repository Rules for Examples

Where each part of an application goes, and what every change keeps true.
`CONTRIBUTING.md` holds the layout, README contract, test registration, and commands.

## Where code goes

Place each piece of code by the first question it answers yes to:

1. Does it create a `Model`, add or connect a graph node, build a `Run`, or push to or pull
   from one? It goes in the example's entrypoint, `src/python/main.py` and `src/cpp/main.cpp`.
2. Would every example need it around a Neat API call? It is a Core gap: open a Core issue
   with a reproducer instead of adding it to Apps.
3. Do two or more examples need it? It goes in `support/`, the shared library.
4. Otherwise it belongs to one example: in its entrypoint, or in its `src/python/utils/` and
   `src/cpp/utils/` when it would bury the application flow.

| Place | Holds | Examples |
|---|---|---|
| Entrypoint | The Neat flow: models, the whole graph, the run, and its teardown | `main.py`, `main.cpp` |
| `src/*/utils/` | Host-side logic one example needs, such as postprocessing or tracking | RF-DETR `postprocess.*`, the trackers' `tracker.*` |
| `support/python/sima_examples/`, `support/cpp/sima_examples/` | Code two or more examples need. It may read and convert Neat data types, such as `Tensor`, `Sample`, and `PullStatus`, and never creates a model, graph, or run | `config`, `labels`, `rtsp_probe`, `detection_egress` |
| `src/common/` | Files one example's C++ and Python share | `config.yaml`, label files |
| Model Registry | Reusable model preparation. Temporary preparation stays in the example's `src/common/` as one development-only flow, never a packaged runtime dependency | |
| `tests/utils/`, `support/cpp/sima_examples/testing/` | Test helpers, in Python and C++ | the metadata listener |

- When a second example needs code from one example's `utils/`, move it to `support/` in the
  same pull request, in both languages, and delete the copy.
- `support/` mirrors across languages: a helper both languages need has the same file name,
  function names, and behavior in each, such as `Config.load` in `config.py` and
  `Config::load` in `config.h`, and each names the other in its header comment. A helper only
  one language needs, such as C++ `pull_status.h`, exists only there. `support/README.md`
  lists every module.
- `support/` has no example-specific names, keys, or defaults. It is tested once, under
  `tests/scripts/` for Python, not again in each example.
- The C++ `utils/` files join the example's target through `target_sources`. A Python
  entrypoint imports its `utils/` modules with `from utils import <module>`.
- A Python entrypoint finds `support/python` in a folder above itself, as the high-density
  example does. `build.sh` and `scripts/get-example.sh` ship that folder, and
  `support/cpp/sima_examples/` without `testing/` as reference source.
- Keep system provisioning, recovery, and setup outside the examples. Platform tooling owns
  the host, network, mounts, devices, the SDK, and unrelated processes.
- Treat a missing runtime capability as Core work rather than an Apps workaround. A comment
  that claims a runtime limitation links the Core issue that reproduces it.

## Config keys

- `examples/config-keys.yaml` lists the key names examples share. An example uses those names
  for the settings listed there and puts every setting only it has under `app.`.
- When a converted example needs a shared setting the file does not list, add the key to the
  file in the same pull request. A new key, shared or under `app.`, names a setting that no
  listed key already covers, even under another name or unit.
- `tests/scripts/test_config_keys.py` checks every converted example's config files. Remove an
  example from its `PENDING_EXAMPLES` list in the pull request that converts it.
- Changing a key name, CLI argument, output file, or metadata field changes the customer
  surface: update the README and tests in the same pull request, and state the change in the
  pull request description.

## Behavior

- Keep the C++ and Python implementations aligned in models, thresholds, outputs, and
  metadata.
- Measure before changing topology, queue policy, copies, buffer counts, or worker counts.
  Faster output never excuses wrong output, and a performance claim names what was measured.
  Keeping up with a source's frame rate is not a capacity measurement, and a latency number
  follows an untimed warm-up.
- Real-time visualization uses Insight, with video and metadata sent through the Neat APIs.
  Offline and batch examples may read files and save or display results.
- Output files default to a path under `sandbox/`, which git ignores. The application writes
  only the outputs the config selects.
- New examples normally include `src/common/config.yaml`, C++ and Python implementations,
  and unit and end-to-end tests. The pull request explains every omission.

## Tests

- Each example's `tests/test-scope.yaml` decides which of its tests run and with which
  models. Register enabled C++ tests with CMake. `build.sh`, `tests/test.sh`, and workflow
  filters name no individual test, so one registration serves local runs and CI.
- An end-to-end test proves useful output, such as boxes received in metadata, not only that
  the process started. Unit tests download nothing and need no live service.
- A broken test prerequisite fails clearly. A test never skips silently or falls back to
  other inputs.
- Unit tests cover the example's own logic. `support/` code is tested once, as
  "Where code goes" says.

## Documentation and publication

- README commands, models, paths, configuration, and expected results match the installed
  package. Run documented commands exactly as written.
- The demonstration is complete and small: every prerequisite is shipped, obtainable with a
  documented command, provided by the customer as the README states, or produced by an
  earlier documented step.
- README steps for live sample streams use the Insight web page: install or pick the videos
  there, start the streams, and copy their URLs into the config.
- A change to a rule every README follows accounts for every example, and updates the
  template, the validator, and the contract tests that can check it.
- Keep credentials, customer information, internal material, agent plans, scratch notes,
  and build or runtime artifacts out of this public repository.
