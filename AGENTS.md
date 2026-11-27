# Neat Apps

Each example teaches a customer the Neat flow: create the `Model`, build the graph, run it,
and send the results to Insight. Every rule here serves that reader.

Read [agents/rules.md](agents/rules.md) before any change. It holds where code goes, the
config keys, behavior, tests, publication, and where a lesson goes.

## Trees and readers

| Tree | Reader | Ships | It holds |
|---|---|---|---|
| `CONTRIBUTING.md` | people who add or change an example | never | the example layout, README contract, test registration, and commands |
| `docs/neat-api-cheat-sheet/` | contributors' agents | never | one page per Core API, with its options, defaults, and C++ and Python differences |
| `agents/` | agents that maintain this repository | never | the rules, the coding style, repository skills in `agents/skills/`, and references in `agents/references/` |
| `skills/` | customers' agents | alone, through sima-cli | customer skills for building and tuning a Neat application |

`skills/` stands alone for its reader: it links Core paths and public URLs, never `agents/`,
`docs/neat-api-cheat-sheet/`, or another file of this repository.

## Skills

Claude Code and Codex discover every skill through `.claude/skills/` and `.agents/skills/`,
which hold one link per skill folder and no copies. A new skill goes in its tree and gets a
link in both folders. A repository skill's name starts with `apps-` and a customer skill's
with `sima-`, because Claude Code lets a personal skill of the same name hide a project skill.
`tests/scripts/test_skill_links_contract.py` checks the links and names.

- Repository skills:
  [`apps-add-or-convert-example`](agents/skills/apps-add-or-convert-example/SKILL.md) to add
  an example or convert one to the shared structure,
  [`apps-write-example-tests`](agents/skills/apps-write-example-tests/SKILL.md) to write,
  trim, or review an example's tests, [`apps-review-pr`](agents/skills/apps-review-pr/SKILL.md)
  for every review, and
  [`apps-write-pr-description`](agents/skills/apps-write-pr-description/SKILL.md) for a pull
  request description.
- Customer skills, used here too:
  [`sima-application-builder`](skills/sima-application-builder/SKILL.md) to judge how an
  example uses inputs, graphs, runs, and options, and
  [`sima-application-optimizer`](skills/sima-application-optimizer/SKILL.md) to judge copies,
  queues, buffers, and throughput.

## References

| When you | Read |
|---|---|
| write or review example code | [agents/coding-style.md](agents/coding-style.md) |
| place code, name config keys, register tests, package, or publish | [agents/rules.md](agents/rules.md) |
| write or check the wording of any text a person or agent reads | [agents/references/writing.md](agents/references/writing.md) |
| draw or review an example's graph topology | [agents/references/flowgraph.md](agents/references/flowgraph.md) |
| use, set, or change a Core API in an example | its page in the [cheat sheet](docs/neat-api-cheat-sheet/README.md), checked as [Skills and cheat sheet](agents/rules.md#skills-and-cheat-sheet) says |
| change `skills/` or the cheat sheet, or judge whether a change should | [Skills and cheat sheet](agents/rules.md#skills-and-cheat-sheet) |
| change an example's layout, README, test registration, or build | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Keep these files true

Every piece of work, whether or not it ends in a pull request, leaves `AGENTS.md`, `agents/`,
`skills/`, and `docs/neat-api-cheat-sheet/` true. It corrects each statement that Core source,
a build, or a measurement on a DevKit proves wrong, and writes each lesson where
[Skills and cheat sheet](agents/rules.md#skills-and-cheat-sheet) places it. When a statement
looks wrong and nothing proves it, report it to the user and leave the text.

This repository is public, and `skills/` ships to customers. Write only what a reader outside
SiMa could check: this repository, a public SiMa repository or documentation page, or a
measurement on a DevKit. A lesson never contains:

- a customer's name, application, data, or request
- runtime or compiler source that is not public, including its file paths and function names
- internal hosts, addresses, runners by name, credentials, or private links
- unreleased products, versions, or plans
- a person's name

When a lesson cannot be written without one of these, leave it out and tell the user.

## Code Review Rules

Every review, the automatic one included, follows
[`apps-review-pr`](agents/skills/apps-review-pr/SKILL.md) and opens with two verdicts:

1. **Shape:** whether the pull request follows [agents/coding-style.md](agents/coding-style.md)
   and [agents/rules.md](agents/rules.md), and whether the packaged customer workflow works as
   the README says.
2. **Rule, skill, and knowledge updates:** the text the change should add or correct in
   `agents/`, `skills/`, or the cheat sheet, as
   [Skills and cheat sheet](agents/rules.md#skills-and-cheat-sheet) defines, or
   `Rule, skill, and knowledge updates: none` with the reason.

Report each of these as P1, because the next agent acts on the stale text:

- a pull request description without its `Rule, skill, and knowledge updates` section
- a statement in `AGENTS.md`, `agents/`, `skills/`, or `docs/neat-api-cheat-sheet/` that the
  change makes wrong
- a lesson the change established that no file holds

Report one finding per root cause with the condition, the breakage, and its impact. Report
anything not run or observed as `not verified`.
