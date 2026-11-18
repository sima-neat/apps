# Repository Rules for Examples

Where each part of an application goes, and what every change keeps true.
`CONTRIBUTING.md` holds the layout, README contract, test registration, and commands.

## Where code goes

| Code | Place |
|---|---|
| Model, graph, run, and every other Neat API call | The example's entrypoint, `src/python/main.py` and `src/cpp/main.cpp` |
| Settings a customer may change | The example's `src/common/config.yaml` |
| Files the example's C++ and Python share | The example's `src/common/` |
| Code without a Neat API call that two or more examples need | `support/python/sima_examples/` and `support/runtime/` or `support/object_detection/` in C++ |
| Code only one example needs | That example's entrypoint |
| A helper every example would need around a Neat API call | Core: open a Core issue with a reproducer instead of adding it to Apps |
| Test helpers | `tests/utils/` and `support/testing/` |

- `support/` code mirrors across languages: the Python and C++ versions of a helper have the
  same name and behavior, and each names the other in its header comment.
- A Python entrypoint finds `support/python` in a folder above itself, as the high-density
  example does. `build.sh` and `scripts/get-example.sh` ship that folder.
- Keep system provisioning, recovery, and setup outside the examples. Platform tooling owns
  the host, network, mounts, devices, the SDK, and unrelated processes.
- Treat a missing runtime capability as Core work rather than an Apps workaround.

## Config keys

- `examples/config-keys.yaml` lists the key names examples share. An example uses those names
  for the settings listed there and puts every setting only it has under `app.`.
- When a converted example needs a shared setting the file does not list, add the key to the
  file in the same pull request.
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
- Real-time visualization uses Insight, with video and metadata sent through the Neat APIs.
  Offline and batch examples may read files and save or display results.
- New examples normally include `src/common/config.yaml`, C++ and Python implementations,
  and unit and end-to-end tests. The pull request explains every omission.

## Tests

- `tests/test-scope.yaml` and each example's `tests/test-scope.yaml` decide which tests run
  and with which models. Register enabled C++ tests with CMake.
- An end-to-end test proves useful output, such as boxes received in metadata, not only that
  the process started. Unit tests download nothing and need no live service.
- A broken test prerequisite fails clearly. A test never skips silently or falls back to
  other inputs.
- Unit tests cover the example's own logic. Code moved to `support/` is tested once there,
  under `tests/scripts/` for Python.

## Documentation and publication

- README commands, models, paths, configuration, and expected results match the installed
  package. Run documented commands exactly as written.
- Keep credentials, customer information, internal material, agent plans, scratch notes,
  and build or runtime artifacts out of this public repository.
