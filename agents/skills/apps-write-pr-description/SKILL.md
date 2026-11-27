---
name: apps-write-pr-description
description: Write or update a sima-neat/apps pull request description. Use when opening a pull request, and after pushing commits that change what an existing description says.
---

# Write a Pull Request Description

Write it executive-first: a reviewer who reads only the Summary and Risk sections knows what
was done and what could break. Everything below them is evidence for those two sections. Write
it plain and brief: short sentences, the project's own names, one fact per line, no adjectives
that do not carry information. Detail a reviewer does not need to judge the change, such as
history per commit, line counts, or before and after numbers, stays in the commit messages.

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
6. Before you ask for review, review the full diff against the base branch with
   `agents/skills/apps-review-pr/SKILL.md`, fix the findings that apply, and record the result
   under Review outcome.
7. Publish with `gh pr edit <number> --body-file <file>`, then read the published body back.

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

## Review outcome

<One sentence on what the author's own review found and fixed, or `No findings`.>

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
- **Review outcome** names what the self-review changed, not that it ran. Bad: "Reviewed
  the diff." Good: "Moved the RTSP probe into `main` and removed an unused config key." Write
  `No findings` when the review changed nothing.
- Each list item and table row stays on one source line. No headings beyond the ones above.
- The last line names every linked issue as `Refs #<issue>`, or `Closes #<issue>` for an issue
  the merge finishes. PR sanitization fails a description without one, and a bare `(#770)`
  does not count.
- Omit AI attribution, internal hostnames, private links, board addresses, and customer
  information. The repository is public.
