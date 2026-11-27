# Repository Rules for Examples

Where each part of an example goes, and what every change keeps true. `agents/coding-style.md`
owns how the code reads. `CONTRIBUTING.md` holds the layout, README contract, test
registration, and commands.

## Where code goes

Place each piece of code by the first question it answers yes to:

1. Does it create a `Model`, add or connect a graph node, build a `Run`, or push to or pull
   from one? It is part of the Neat flow and goes in the entrypoint, `src/python/main.py` and
   `src/cpp/main.cpp`.
2. Would every example need it around a Neat API call? It is a Core gap: write the smallest
   workaround in the example with a one-line comment naming what Core lacks. The review
   reports it as a Core candidate, and the maintainer decides which become Core issues.
3. Do two or more examples need it, counting the examples still in `PENDING_EXAMPLES`? It
   goes in the shared library, `support/`.
4. Otherwise it is the example's own logic: in its entrypoint, or in its `src/python/utils/`
   and `src/cpp/utils/` when it would bury the Neat flow.

| Place | Holds | Examples |
|---|---|---|
| Entrypoint | The Neat flow: models, the whole graph, the run, and its teardown | `main.py`, `main.cpp` |
| `src/*/utils/` | The example's own logic, such as postprocessing or tracking | RF-DETR `postprocess.*`, the trackers' `tracker.*` |
| `support/python/sima_examples/`, `support/cpp/sima_examples/` | The shared library: code two or more examples need. It may read and convert Neat data types, such as `Tensor`, `Sample`, and `PullStatus`, and never creates a model, graph, or run | `config`, `labels`, `rtsp_probe`, `stop_signals`, `detection_egress` |
| `support/labels/` | Class-label files two or more examples' configs name, one label per line in the model's class-ID order | `coco80.txt`, `coco91.txt` |
| `src/common/` | Files one example's C++ and Python share, such as the label file of a model only it uses | `config.yaml`, `pcb_label.txt` |
| Model Registry | Reusable model preparation. Temporary preparation stays in the example's `src/common/` as one development-only flow, outside the packaged runtime | |
| `tests/utils/`, `support/cpp/sima_examples/testing/` | Test helpers two or more examples' tests use, in Python and C++ | the metadata listener |
| The example's `tests/python/`, `tests/cpp/` | Test helpers only that example's tests use | high-density `received_metadata_tracker.*` |

- When a second example needs code from one example's `utils/`, move it to `support/` in the
  same pull request, in both languages, and delete the copy.
- The shared library mirrors across languages: a helper both languages need has the same file
  name, function names, and behavior in each, such as `Config.load` in `config.py` and
  `Config::load` in `config.h`, and each names the other in its header comment. A helper only
  one language needs, such as C++ `pull_status.h`, exists only there. `support/README.md`
  lists every packaged module.
- A change to the shared library updates `support/README.md` in the same pull request: the
  module table, and any note on a helper the change leaves only some examples using. The pull
  request description lists the change under `Shared library`, and every caller is updated in
  that pull request.
- The shared library holds only generic names, keys, and defaults, and is tested once, under
  `tests/scripts/` for Python.
- The C++ `utils/` files join the example's target through `target_sources`. A Python
  entrypoint imports them with `from utils import <module>`.
- A Python entrypoint finds `support/python` in a folder above itself, as the high-density
  example does. `scripts/ci/package-apps.sh` ships that folder, `support/README.md`,
  `support/labels/`, and `support/cpp/sima_examples/` without `testing/` as reference source.
  `scripts/get-example.sh` copies the bundle's whole `support/`.
- When a pull request adds a `support/` folder that configs or code read, it adds that folder
  to `scripts/ci/package-apps.sh`. The installed bundle and a fetched example hold only the
  folders that script ships.
- When `scripts/ci/package-apps.sh` ships a new top-level folder outside `support/` that
  examples read, the same pull request adds it to the folder list in `scripts/get-example.sh`.
  A fetched example holds only the top-level folders that list names.
- An example names each dataset it reads by its path, `assets/datasets/<folder>`, in its
  config, source, or README. `scripts/get-example.sh` copies only the dataset folders an
  example's files name, so a dataset path built in code is missing from a fetched example.
- When an example runs another example, it names that example as `<category>/<name>` in a
  file other than its `README.md`, tests and docs included. `scripts/get-example.sh` then
  refuses it and points to the whole bundle, because a fetched folder holds one example. A
  mention in `README.md` does not count.
- A label file moves from an example's `src/common/` to `support/labels/` when a second
  example's model uses the same classes in the same class-ID order. An example that keeps a
  copy deletes it and names the shared file in its own conversion pull request. A config
  names it as `model.labels: support/labels/<file>`, read from the working directory. A label
  set with another order or count, such as COCO's 80 classes and its 91 class IDs, is its own
  file. A class ID that names no class has the line `N/A`.
- Platform tooling owns provisioning, recovery, and setup of the host, network, mounts,
  devices, and the SDK. Examples start from a prepared system.

## Config keys

- `examples/config-keys.yaml` lists the key names examples share, each with a type and a
  meaning. An example uses those names for the settings listed there and puts every setting
  only it has under `app.`.
- A converted example that needs an unlisted shared setting adds the key to the file in the
  same pull request. A new key, shared or under `app.`, names a setting no listed key covers,
  even under another name or unit.
- `tests/scripts/test_config_keys.py` checks every converted example's config files. The pull
  request that converts an example removes it from `PENDING_EXAMPLES`.
- Each shipped config holds only keys the example reads. Tests write their own values with
  the end-to-end config writer, so `config.yaml` has no test section.
- A changed key name or meaning, CLI argument, output file, or metadata field changes the
  customer surface, such as a relative `model.path` that now starts at the config file. The same pull request updates the README and tests and states the change in its
  description.

## Behavior

- The C++ and Python implementations match in models, thresholds, outputs, and metadata.
- A change to topology, queue policy, copies, buffer counts, or worker counts starts from a
  measurement, as `skills/sima-application-optimizer/SKILL.md` describes, and keeps the
  output correct.
- Real-time visualization uses Insight, with video and metadata sent through the Neat APIs.
  Offline and batch examples may read files and save or display results.
- Output files default to a path under `sandbox/`, which git ignores. The application writes
  only the outputs the config selects.
- A new example includes `src/common/config.yaml`, C++ and Python implementations, and unit
  and end-to-end tests. The pull request explains every omission.

## Tests

- `agents/skills/apps-write-example-tests/SKILL.md` decides which tests an example has.
- Each example's `tests/test-scope.yaml` decides which of its tests run and with which models.
  Register enabled C++ tests with CMake. `build.sh`, `tests/test.sh`, and workflow filters
  name no individual test, so one registration serves local runs and CI.
- Unit tests check what the application leaves to its config: every shipped config loads
  with the shared `Config`, its codec, model, and label file are ones the application
  supports, and its numbers are in range. `tests/scripts/test_config_keys.py` checks that
  every key is a shared name. The application itself does none of these checks.
- Unit tests download nothing and need no live service. A broken prerequisite fails with an
  error that names it. A test runs with the inputs it names or fails.
- `support/` has no C++ test program yet, so a shared C++ helper's test lives in the first
  converted example that uses it, as C++ `detection_egress` lives in the high-density example.

## Documentation and publication

- README commands, models, paths, configuration, and expected results match the installed
  package. Run documented commands exactly as written.
- The demonstration is complete and small: every prerequisite is shipped, obtainable with a
  documented command, provided by the customer as the README states, or produced by an
  earlier documented step.
- README steps for live sample streams use the Insight web page: install or pick the videos
  there, start the streams, and copy their URLs into the config.
- A change to a rule every README follows accounts for every example and updates the
  template, the validator, and the contract tests that can check it.
- Repository files name no Core commit. Name the Core release, such as `Neat Core 0.6.0 or
  newer`, or the Core issue, such as `core#1081`, because a commit hash tells a reader nothing
  and goes stale.
- Files, pull request text, and review comments name no specific machine: no board, host,
  or Insight IP address or hostname, no lab or site name, and no home directory or local
  path of one person's machine, such as `/Users/<name>` or `~/py-extra`. Code and configs
  read these values from the config or the command line, and docs and logs write a
  placeholder, such as `<board-ip>`, `<insight-host>`, or `<path>`, because each reader's
  values differ and the repository is public.
- The repository is public. It holds source, docs, and shipped assets only: credentials,
  customer information, internal material, agent plans, scratch notes, and build or runtime
  artifacts stay elsewhere.

## Skills and cheat sheet

`skills/sima-application-builder` and `skills/sima-application-optimizer` teach customers'
agents the public Neat APIs for any project: inputs, graph topologies, runs, options, and
tuning. sima-cli ships only `skills/`, so `skills/` links Core paths and public URLs, never
`agents/`, `docs/neat-api-cheat-sheet/`, or other repository files.

`docs/neat-api-cheat-sheet/` holds one page per Core API, with its options, defaults, and C++
and Python differences, for contributors' agents. This repository's conventions stay in `agents/`
and `CONTRIBUTING.md`.

Every pull request feeds what its work taught back into these files:

- **Check:** before an example uses, sets, or changes a Core API, read its cheat sheet page
  and compare it with the header and `python/src/module.cpp` at the Core that
  `deps/manifest.json` names as `neat-core`. A wrong or missing option, default, or language
  difference is fixed in the same pull request.
- **Lesson:** a finding the work established, by reading Core source or by a measurement on a
  board, goes where it applies:

  | The finding is about | It goes in |
  |---|---|
  | An API option, default, error, or behavior | Its page in `docs/neat-api-cheat-sheet/`, following `docs/neat-api-cheat-sheet/TEMPLATE.md` |
  | A node, link, or topology pattern, and when to choose it | `skills/sima-application-builder/references/` |
  | A tuning choice, with its measurement | `skills/sima-application-optimizer/references/tuning.md` |
  | How this repository's examples are written, tested, or reviewed | `agents/` |

- **Lift:** a cheat sheet pitfall or behavior that changes how any application should be
  built or tuned, such as `GraphLinkOptions::queue_depth` adding no buffering, also goes into
  the matching skill reference, written in full there.
- **Generalize:** write a lesson brief and to the point, as general as it holds, as a rule
  with its condition: when X, do Y, because Z. Write it by [the language rules](references/writing.md#language). It names the condition, not the example where it
  was found, and holds for any application that meets the condition. A fact that holds only for one example stays in that example's README.
  A fact about one API, such as a `BoxDecodeType` value or an option default, is stated
  exactly, because it holds wherever that API is used.
- **Correct:** a statement the work proves wrong is fixed in the same pull request, in
  `skills/` and in the cheat sheet, even when the pull request is about something else.
- **Merge:** extend or generalize an existing statement before adding a new one, so each fact
  lives in one place within `skills/` and once in the cheat sheet.

Keep routing in `SKILL.md` and detail in `references/`. `version` in `playbook.yml` stays
unchanged. The pull request description lists each change under
`Rule, skill, and knowledge updates`, or writes `Rule, skill, and knowledge updates: none` with the
reason.
