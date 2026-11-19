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
| Change `skills/`, or judge whether a change should | `agents/rules.md`, section "Customer skills" |
| Judge how an example uses inputs, graphs, runs, and options | `skills/neat-application-builder/SKILL.md` |
| Judge copies, queues, buffers, and throughput | `skills/neat-application-optimizer/SKILL.md` |

`CONTRIBUTING.md` holds the example layout, README contract, test registration, and commands.
`agents/` serves contributors' agents and the automatic review. sima-cli installs only
`skills/`.

## Every review reports

1. **Shape:** whether the pull request follows `agents/coding-style.md` and `agents/rules.md`,
   and whether the packaged customer workflow works as the README says.
2. **Skill updates:** the customer skill text the change should add or correct, as
   `agents/rules.md` "Customer skills" defines, or `Skill updates: none` with the reason.

Report one finding per root cause with the condition, the breakage, and its impact. Report
anything not run or observed as `not verified`.
