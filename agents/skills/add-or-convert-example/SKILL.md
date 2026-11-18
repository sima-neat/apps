---
name: apps-add-or-convert-example
description: Add a new example to sima-neat/apps, or convert an existing one to the shared structure, config keys, and shared library.
---

# Add or Convert an Example

The reference is `examples/object-detection/high-density-multi-stream-object-detector`. Read its
`main.py`, `main.cpp`, `src/common/config.yaml`, and tests before you start, and follow
`agents/rules.md` and `agents/coding-style.md`.

## Steps

1. Read the issue and the nearest shipped example. For a conversion, list what the example
   advertises in its README: inputs, outputs, CLI arguments, config keys, and printed results.
   The conversion keeps each one or the pull request states why it goes.
2. Rename the config keys to the names in `examples/config-keys.yaml`, and move every setting
   only this example has under `app.`. Write a value the same in every shipped profile into the
   code instead. Remove the example from `PENDING_EXAMPLES` in
   `tests/scripts/test_config_keys.py`. The step is done when that test passes.
3. Write `load_app_config` with the shared config reader: one typed `AppConfig`, every key
   required, and `reject_unread_keys()` at the end.
4. Write `main` as the application flow with the whole graph in it, and one small function per
   option struct or subgraph. Check each option against the installed Core header and delete
   assignments that restate a default.
5. Move code without a Neat API call that another example also needs into `support/`, in both
   languages. Replace hand-written code that a public Neat API already provides, such as box
   payload parsing with `decode_bbox_tensor`. When a missing Neat API forces a workaround, open
   a Core issue and link it in a comment.
6. Delete everything nothing reaches: options, branches, config keys, environment variables,
   profiling, and validation that only repeats what the reader already checks.
7. Shrink the unit tests to the example's own logic: help, missing config, every shipped
   profile loads, unknown and missing keys. Keep the end-to-end tests' assertions and change
   only the keys they override.
8. Update the README for every changed key, argument, and output.
9. Compare the result with the customer skills under `skills/`. Add what this example teaches
   that they lack, or record `Skill updates: none` with the reason.
10. Build in the Neat SDK, run the unit and end-to-end tests in both languages on a Modalix
    DevKit, install the branch package with `sima-cli neat install apps@<branch>`, and follow
    the installed README. Record each result in the pull request, and mark anything not run
    `not verified`.
