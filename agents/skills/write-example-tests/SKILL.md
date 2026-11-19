---
name: apps-write-example-tests
description: Write, trim, or review an example's unit and end-to-end tests. Use when adding or converting an example, when changing what an example outputs, and when reviewing its tests.
---

# Write Example Tests

A test earns its place by running what the example does for a customer and failing when that
breaks. A few tests of the example's own logic and its end-to-end output cover more than many
checks of flags and messages.

## Steps

1. List the behaviors the README promises: what the example outputs, where, how fast, and for
   which inputs. List the example's own logic: code in its entrypoint or `utils/` that turns
   model output into results, such as postprocessing, tracking, or routing.
2. Write one end-to-end test per README mode, in both languages. A mode is each input type
   and each output choice the README documents, such as saving on or off. It runs the example against a
   real source and asserts values the example decides: each result reaches its stream's
   channel, labels come from the label file, scores pass `inference.min_score`, and the
   throughput floor the README states holds. For saved frames, assert they decode with
   `assert_saved_frames_are_usable` in `tests/utils/output_assertions.py` and
   `saved_frames_problem` in `support/cpp/sima_examples/testing/test_process.h`, and for a
   saved stream, that it advances, with `assert_streamed_frames_are_usable` and
   `streamed_frames_problem`. Assert values the code can get wrong,
   not ones the serializer guarantees, such as a box clamped to the frame.
3. Write unit tests for the example's own logic, with real inputs and exact expected values,
   such as RF-DETR's proposal ranking and mask polygons. Add one test that every shipped config
   loads, because a broken shipped config stops the customer at the first step.
4. Break each behavior once in a throwaway copy, such as a wrong box scale or an ignored
   `min_score`, and confirm a test fails. Unit tests break on any host that runs them, and
   end-to-end tests on the DevKit run that validates the change. Rewrite or delete a test that
   still passes, and record each break and the failing test in the pull request.
5. Give the C++ and Python tests the same names, inputs, and assertions.

The step is done when every README behavior maps to an end-to-end assertion, every function of
the example's own logic maps to a unit test, and every test was seen failing.

## Where other checks live

- The shared config reader's missing, unknown, and mistyped keys are tested once, in
  `tests/scripts/test_config.py`.
- Other shared library code is tested once, as `agents/rules.md` "Tests" says.
- Argument parsing, `--help`, and exit codes for a missing file need no test of their own: an
  end-to-end test runs them.
- Test helpers are exercised by the tests that use them.
