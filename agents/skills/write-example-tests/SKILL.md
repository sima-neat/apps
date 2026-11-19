---
name: apps-write-example-tests
description: Write, trim, or review an example's unit and end-to-end tests. Use when adding or converting an example, when changing what an example outputs, and when reviewing its tests.
---

# Write Example Tests

A test earns its place by exercising what the example does for a customer and failing when
that breaks. Few tests that run the real logic beat many that check a flag or a message.

## Steps

1. List the behaviors the README promises: what the example outputs, where, how fast, and
   for which inputs. List the example's own logic: code in its entrypoint or `utils/` that
   turns model output into results, such as postprocessing, tracking, or routing.
2. Write one end-to-end test per README mode, in both languages. It runs the installed
   example against a real source and asserts the result is right: messages reach the right
   channel, boxes and polygons lie inside the frame, labels come from the label file, and
   any throughput floor the README states holds.
3. Write unit tests only for the example's own logic, with real inputs and exact expected
   values, such as RF-DETR's proposal ranking and mask polygons. Add one test that every
   shipped config loads, since a broken shipped config stops the customer at the first step.
4. Break each behavior once in a throwaway copy, such as a wrong box scale or a swapped
   channel, and confirm a test fails. A test that still passes is deleted or rewritten.
5. Keep the C++ and Python tests the same: same names, same inputs, same assertions.

The step is done when every README behavior maps to an end-to-end assertion, every function
of the example's own logic maps to a unit test, and every test was seen failing.

## What stays out

- Argument parsing, `--help`, and exit codes for a missing file.
- Config reader behavior: missing, unknown, and mistyped keys. `tests/scripts/test_config.py`
  tests the shared reader once.
- Any other `support/` code, tested once outside the examples.
- Assertions that a field exists without checking its value, and tests of the test helpers.
