---
name: apps-add-or-convert-example
description: Add or convert a sima-neat/apps example. Use to create a new example, and to convert an existing one to the shared config keys, shared library, and Neat flow.
---

# Add or Convert an Example

The reference is `examples/object-detection/high-density-multi-stream-object-detector`, with
`examples/object-detection/rfdetr-detection-segmentation` for two models with host work
between them. Read the reference's `main.py`, `main.cpp`, `src/common/config.yaml`, and tests,
then `agents/coding-style.md` and `agents/rules.md`.

## Steps

1. Read the issue the pull request will close, and the nearest shipped example. For a
   conversion, list what the README advertises: inputs, outputs, CLI arguments, config keys,
   and printed results. The step is done when each item is marked kept, or dropped with the reason the pull request will state.
2. Write `config.yaml` with the key names `agents/rules.md` "Config keys" sets, and remove the
   example from `PENDING_EXAMPLES` in `tests/scripts/test_config_keys.py`. The step is done when
   that test passes and every key left in `config.yaml` names a setting the README or the
   code needs.
3. Write `load_app_config` as `agents/coding-style.md` "Configuration" describes. The step is
   done when every key in every shipped config is read and `--validate-config-only` passes on
   each.
4. Write `main` as `agents/coding-style.md` "Flow" and "Graph topology" describe. The step is
   done when every option was checked against the installed Core header, located as
   `skills/neat-application-builder/SKILL.md` step 1 says, and no assignment restates a
   default.
5. Place every other piece of code as `agents/rules.md` "Where code goes" decides. Use a public
   Neat API where one does the job, such as `decode_bbox_tensor` for box payloads. The step is
   done when a search of the other examples, converted and pending, finds no copy of a helper
   this example keeps to itself.
6. Delete what `agents/coding-style.md` "Functions and state" says no entrypoint reaches,
   including environment variables, profiling paths, and validation that repeats a check the
   config reader makes.
7. Write the tests with `agents/skills/write-example-tests/SKILL.md`.
8. Update the README for every changed key, argument, and output, in the sections and order
   `CONTRIBUTING.md` "README Contract" sets. Change only the lines the change affects.
9. Compare the result with the customer skills as `agents/rules.md` "Customer skills"
   describes, and record the skill text to add or `Skill updates: none` with the reason.
10. Build in the Neat SDK, run the unit and end-to-end tests in both languages on a Modalix
    DevKit, install the branch package with `sima-cli neat install apps@<branch>`, and follow
    the installed README. The step is done when the pull request records each result and marks
    anything not run `not verified`.
