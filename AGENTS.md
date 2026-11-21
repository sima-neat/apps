# Neat Apps

Each example teaches a customer the Neat flow: create the `Model`, build the graph, run it,
and send the results to Insight. Every rule here serves that reader.

## Read first

| When you | Read |
|---|---|
| Write or review example code | `agents/coding-style.md` |
| Place code, name config keys, register tests, package, or publish | `agents/rules.md` |
| Add an example, or convert one to the shared structure | `agents/skills/add-or-convert-example/SKILL.md` |
| Write, trim, or review an example's tests | `agents/skills/write-example-tests/SKILL.md` |
| Review a pull request | `agents/skills/review-pr/SKILL.md` |
| Write or update a pull request description | `agents/skills/write-pr-description/SKILL.md` |
| Use, set, or change a Core API in an example | Its page in `docs/neat-api-cheat-sheet/`, checked as `agents/rules.md` "Skills and cheat sheet" says |
| Change `skills/` or `docs/neat-api-cheat-sheet/`, or judge whether a change should | `agents/rules.md`, section "Skills and cheat sheet" |
| Judge how an example uses inputs, graphs, runs, and options | `skills/neat-application-builder/SKILL.md` |
| Judge copies, queues, buffers, and throughput | `skills/neat-application-optimizer/SKILL.md` |

`CONTRIBUTING.md` holds the example layout, README contract, test registration, and commands.
`agents/` serves contributors' agents and the automatic review. sima-cli installs only
`skills/`.

## Every review reports

1. **Shape:** whether the pull request follows `agents/coding-style.md` and `agents/rules.md`,
   and whether the packaged customer workflow works as the README says.
2. **Skill and cheat sheet updates:** the customer skill and cheat sheet text the change
   should add or correct, as `agents/rules.md` "Skills and cheat sheet" defines, or
   `Skill and cheat sheet updates: none` with the reason.

Report one finding per root cause with the condition, the breakage, and its impact. Report
anything not run or observed as `not verified`.
