# Neat Apps

The examples here teach customers how to build an application with the Neat APIs: create the
model, build the graph, run it, and send the results to Insight. Every rule below serves that
reader. `CONTRIBUTING.md` holds the layout, README contract, test registration, and commands.

## Rules

1. An example's entrypoint shows the Neat flow. One function creates the `Model`, builds the
   whole graph with every `connect`, runs it, and closes it.
2. Code without a Neat API call that two or more examples need lives in `support/`. Code only
   one example needs stays in that example, in its entrypoint or its `utils/` folder.
3. A helper every example would need around a Neat API call is a Core gap. Open a Core issue
   instead of adding an Apps workaround.
4. `config.yaml` uses the key names in `examples/config-keys.yaml` and puts example-only
   settings under `app.`. The application rejects keys it does not read.
5. Code follows `agents/coding-style.md`: a straightforward flow, a minimal graph topology, and
   names that say what each function returns or does.

## Read first

| When you | Read |
|---|---|
| Add an example, or convert one to the shared structure | `agents/skills/add-or-convert-example/SKILL.md` |
| Review a pull request | `agents/skills/review-pr/SKILL.md` |
| Write or review any example code | `agents/coding-style.md` |
| Place code, change config keys, tests, or packaging | `agents/rules.md` |
| Judge how an example uses nodes, graphs, and the Neat APIs | `skills/neat-application-builder/SKILL.md` |
| Judge copies, queues, buffers, and throughput | `skills/neat-application-optimizer/SKILL.md` |

## Every review reports

1. **Shape:** whether the pull request follows the rules above, `agents/coding-style.md`, and
   `agents/rules.md`, and whether the packaged customer workflow works as the README says.
2. **Skill updates:** whether the change teaches something the customer skills under `skills/`
   lack or contradict: a node or topology pattern, a tuning choice with its measurement, or a
   pitfall. Name the skill file and the text to add, or write `Skill updates: none` with the
   reason.

Report one finding per root cause with the condition, the breakage, and its impact. Report
anything not run or observed as `not verified`.

## Customer skills

`skills/` holds the skills that sima-cli installs for customers' agents.
`neat-application-builder` covers building an application, and `neat-application-optimizer`
covers making it fast. Customers use them in their own projects, so they teach the Neat APIs
only: inputs, graph topologies, runs, and options. This repository's conventions, such as the
config key names, the `support/` layout, and the test rules, stay in `agents/` and
`CONTRIBUTING.md` and never go into `skills/`. Update them in the pull request that teaches the lesson. Keep routing in
`SKILL.md` and details in `references/`, and point to Core docs, installed headers, and examples
instead of copying them. Do not bump `version` in `playbook.yml`. The pull request description
lists the changes under `Skill updates`, or writes `Skill updates: none` with the reason.

`agents/` is for contributors' agents and the automatic review. sima-cli does not install it.
