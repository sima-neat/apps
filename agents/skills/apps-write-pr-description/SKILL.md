---
name: apps-write-pr-description
description: Write or update a sima-neat/apps pull request description. Use when opening a pull request, and after pushing commits that change what an existing description says.
---

# Write a Pull Request Description

Write it executive-first: a reviewer who reads only the Summary and Risk sections knows what
was done and what could break. Everything below them is evidence for those two sections. Write
it as [Writing](#writing) says, so the reviewer finds each point without reading the rest.

## Steps

1. Read the linked issue, every commit, and the full diff. List each changed example and each
   changed shared file (`support/`, `build.sh`, CI, `skills/`, `agents/`).
2. Write the sections below in order. Leave a section out only where this skill says so.
3. For each changed example, draw its graph topology with
   `agents/references/flowgraph.md`: copy the graph from the example's README `Concept` and
   mark what changed, or draw it from its `main.py` and `main.cpp` at the
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
## Summary

<What the pull request does and for whom, in one or two sentences. Name the issue.>

- **<Outcome>.** <One line per main result a reviewer must know.>

## Risk

- **<Risk>.** <Breaking changes, dependency or pin changes, code owner files, flaky or tight
  thresholds, and known issues, one line each, with what breaks and for whom.>

## What Changed

- <One line per change the Summary does not name, with the file or key it touches.>

### <example name>

<Flowgraph of the example's graph topology, with change markers.>

- <One or two note lines per part or path worth attention.>

<!-- screenshot: <example name> -->

## Tests

| Test | Checks |
| --- | --- |
| `<file>` | <the behavior it asserts> |

## Validation

- <Venue: what ran and what it showed. Host, SDK build, DevKit, and CI are separate lines.>
- Not verified: <what did not run, and why>.

## Rule, skill, and knowledge updates

<The rule, skill, and cheat sheet files changed and the lesson each teaches, as
`agents/rules.md` "Skills and cheat sheet" defines, or
`Rule, skill, and knowledge updates: none` with the reason.>

Refs #<issue>, #<issue>
```

## Rules

- **Summary** states outcomes, not mechanisms. Bad: "Refactored the config loader." Good:
  "Every example reads the same config keys, and CI rejects a key outside the shared list."
- **Risk** comes before any detail and names every failure the change can cause or exposes,
  including test failures seen during validation. Write `Risk: none` only when nothing applies.
- **What Changed** adds only what the Summary leaves out. It names the user-visible effect first and the mechanism only when a reviewer
  needs it to judge correctness. Give before and after for a renamed key, argument, or output.
- **Flowgraph:** one graph per changed example, showing the Neat parts the entrypoint
  connects: inputs, decode, models, links and their policies, outputs, and senders. Mark parts
  `(new)`, `(changed)`, or `(removed)`. A pull request that changes no example's graph draws
  the graph of the files it rewires instead, or omits the graph.
- **Tests** has one row per test file or group the pull request adds, changes, or relies on as
  evidence. "Checks" is the assertion, such as "every channel gets at least 27 detection
  messages per second".
- **Validation** keeps venues apart. A host check is not board evidence, and a run on an earlier
  commit says which commit. Report measured numbers with the venue, the MLA clock, and the
  input that produced them, in one table.
- Each list item and table row stays on one source line. No headings beyond the ones above.
- The last line names every linked issue as `Refs #<issue>`, or `Closes #<issue>` for an issue
  the merge finishes. PR sanitization fails a description without one, and a bare `(#770)`
  does not count.
- Omit AI attribution, private links, and customer information, and name no specific
  machine, as `agents/rules.md` "Documentation and publication" says. The repository is
  public.

## Writing

- **Keep only what the reviewer needs to judge the change:** what changed, what it gives the
  customer or the example, and what can break. History per commit, flakes on earlier commits,
  investigations of other issues, Core file and line numbers, and line counts stay in commit
  messages or issue comments.
- **State each fact once.** A fact in the Summary does not come back in What Changed or
  Validation.
- **Use precise, plain words.** Use the project's name for each thing and keep it the same in
  every section. Explain a term the first time when a reviewer outside the work may not know
  it.
- **Write short sentences with one fact each,** in active voice. Do not shorten a sentence so
  far that the reader has to guess what it means.
- **Before publishing, read it as the reviewer.** Delete each line whose removal would not
  change the review.
