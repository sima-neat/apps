---
name: apps-review-pr
description: Review a sima-neat/apps pull request for merge readiness and for the customer skill updates it should make.
---

# Review an Apps Pull Request

Review the complete affected application, including its tests, configuration, README, and
preview. A changed line can break code it does not touch.

## Steps

1. Read the linked issue and the pull request description. Note what the change claims.
2. Read `agents/rules.md`, `agents/coding-style.md`, and both customer skills:
   `skills/neat-application-builder/` and `skills/neat-application-optimizer/`.
3. Check each item under Shape against the change. Report every item that fails.
4. Compare the change with the customer skills, as Skill updates describes.
5. Report the two verdicts that `AGENTS.md` requires, then the findings.

## Shape

- The implementation solves the issue's problem, delivers every claimed behavior, and adds no
  unjustified scope.
- The entrypoint follows `agents/coding-style.md`: `main` reads as the application flow and
  builds the whole graph, functions are named for what they return or do, and options restate
  no Core default.
- Code sits where `agents/rules.md` places it. Report a Neat-free helper copied into an example
  that `support/` already has or that another example needs, and a Neat API call hidden in
  `support/`.
- The config uses the shared key names, keeps example-only settings under `app.`, and the
  application rejects unread keys.
- The design adds no duplicate entrypoint, controller, UI, process, or setup path compared
  with the nearest shipped example.
- One customer workflow works from the package: install with
  `sima-cli neat install apps@<branch>`, configure, run, and inspect the result. Report
  source-only paths, missing packaged files, and customer options spread across config, CLI
  arguments, and environment variables. The pull request description records this evidence.
- Real-time visualization uses Insight through the Neat video and metadata senders.
- Each advertised behavior maps to an end-to-end assertion in both languages. A test that only
  starts the process is no evidence.
- Stateful applications test failure, replacement, stop, cleanup, and restart.
- Changes to dependency pins, `build.sh`, or `.github/workflows/vulcan-ci.yml` are flagged for
  code owner review.

## Skill updates

The customer skills teach application developers' agents. A pull request updates them when it
shows something they lack or contradict:

- a node, node group, or link policy used in a way the builder skill does not describe;
- a topology pattern, such as a shared model across streams, that no reference covers;
- a tuning choice with its measurement, such as a queue depth or buffer count, that the
  optimizer skill does not cover;
- a default, error, or pitfall the change uncovered;
- a skill statement the change proves wrong.

For each, name the skill file and the text to add or change. When nothing applies, the
verdict is `Skill updates: none` with the reason. Do not ask for a `playbook.yml` version bump.

## Findings

- Report one finding per root cause, anchored to the smallest changed line range, with the
  condition, the breakage, and its impact.
- Separate correctness, structure, packaging, and customer-workflow blockers from optional
  performance follow-ups.
- Skip style preferences outside `agents/coding-style.md`, vague risks, and test requests that
  name no behavior that could regress.
- Write in plain English and state what should change.
- Check the latest Vulcan CI run for the branch. Report anything not run or observed as
  `not verified`.
