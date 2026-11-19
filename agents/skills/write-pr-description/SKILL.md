---
name: apps-write-pr-description
description: Write or update the description of a sima-neat/apps pull request. Use when opening a pull request, after pushing commits that change what an existing description says, or when asked to describe a change.
---

# Write a Pull Request Description

A reviewer reads the description once and must come away knowing what changed, why, how the
example's graph looks now, what it produces, and what each test proves. Write it plain,
precise, and brief: short sentences, the project's own names, one fact per line, no
adjectives that do not carry information.

## Steps

1. Read the linked issue, every commit, and the full diff. List each changed example and each
   changed shared file (`support/`, `build.sh`, CI, `skills/`, `agents/`).
2. Write the sections below in order. Leave a section out only where this skill says so.
3. For each changed example, draw its graph topology with
   [references/flowgraph.md](references/flowgraph.md), from its `main.py` and `main.cpp` at the
   pushed commit.
4. For each changed example, capture one screenshot of its result: the Insight view for a live
   example, or the annotated output image for an offline one. Run it from the installed
   package or the branch build on a Modalix DevKit. GitHub's CLI cannot upload images, so save
   each image outside the repository, put `<!-- screenshot: <example> -->` where it goes, and
   ask the author to drag the file into that spot.
5. Check every claim against the source or a command you ran. Remove what you cannot check, or
   mark it `not verified`.
6. Publish with `gh pr edit <number> --body-file <file>`, then read the published body back.

The step is done when a reviewer who reads only the description could explain what changed
and why, and every test and number in it traces to a file or a run.

## Sections

```markdown
## Why

<The problem a customer or contributor hits, in two or three sentences. Name the issue.>

## What Changed

- <One line per change a reviewer must know, with the file or key it touches.>

### <example name>

<Flowgraph of the example's graph topology, with change markers.>

- <One note line per part or path worth attention.>

<!-- screenshot: <example name> -->

## Tests

| Test | Checks | Why |
| --- | --- | --- |
| `<file>::<test>` | <the behavior it asserts> | <the failure it would catch> |

## Validation

- <Venue: what ran and what it showed. Host, SDK build, DevKit, and CI are separate lines.>
- Not verified: <what did not run, and why>.

## Skill updates

<The customer skill files changed and the lesson each teaches, or `Skill updates: none` with
the reason.>

## Notes

<Breaking changes, follow-ups, and known issues, one line each. Omit when empty.>
```

## Rules

- **Why** says what was wrong or missing, not what the diff does. Bad: "Refactored the config
  loader." Good: "Each example parsed its own config keys, so a key learned in one example
  meant nothing in the next."
- **What Changed** names the user-visible effect first and the mechanism only when a reviewer
  needs it to judge correctness. Give before and after for a renamed key, argument, or output.
- **Flowgraph:** one graph per changed example, showing the Neat parts the entrypoint
  connects: inputs, decode, models, links and their policies, outputs, and senders. Mark parts
  `(new)`, `(changed)`, or `(removed)`. A pull request that changes no example's graph draws
  the graph of the files it rewires instead, or omits the graph.
- **Tests** lists every test the pull request adds or changes, and each existing test it relies
  on as evidence. One row per test. "Checks" is the assertion, such as "every channel gets at
  least 27 detection messages per second"; "Why" is the regression it guards against.
- **Validation** keeps venues apart. A host check is not board evidence, and a run on an earlier
  commit says which commit. Report measured numbers with the venue, the MLA clock, and the
  input that produced them.
- Each list item and table row stays on one source line. No headings beyond the ones above.
- Omit AI attribution, internal hostnames, private links, board addresses, and customer
  information. The repository is public.
