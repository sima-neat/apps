# Repository Rules for Examples

Where each part of an application goes, and what every change keeps true.
`CONTRIBUTING.md` holds the layout, README contract, test registration, and commands.

## Where code goes

| Code | Place |
|---|---|
| Model, graph, run, and every other Neat API call | The example's entrypoint, `src/python/main.py` and `src/cpp/main.cpp` |
| Settings a customer may change | The example's `src/common/config.yaml` |
| Files the example's C++ and Python share | The example's `src/common/` |
| Code without a Neat API call that two or more examples need | `support/python/sima_examples/` and `support/cpp/sima_examples/`, included as `sima_examples/<file>.h` |
| Code only one example needs | That example's entrypoint, or its `src/python/utils/` and `src/cpp/utils/` when the code would bury the application flow. The C++ `utils/` files join the example's target through `target_sources` |
| Model preparation | The Model Registry when it is reusable. Temporary preparation stays in the example's `src/common` as one development-only flow, never a packaged runtime dependency |
| A helper every example would need around a Neat API call | Core: open a Core issue with a reproducer instead of adding it to Apps |
| Test helpers | `tests/utils/` in Python and `support/cpp/sima_examples/testing/` in C++ |

- `support/` code mirrors across languages: a helper both languages need has the same file
  name, function names, and behavior in each, such as `Config.load` in `config.py` and
  `Config::load` in `config.h`, and each names the other in its header comment. A helper only one language needs, such as C++
  `pull_status.h`, exists only there.
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
- Unit tests cover the example's own logic. Code moved to `support/` is tested once there,
  under `tests/scripts/` for Python.

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
