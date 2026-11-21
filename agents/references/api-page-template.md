# API Reference Page Template

Each page of `skills/neat-application-builder/references/api/` follows this template. The
folder ships to customers, so a page links examples by their public URL on `main` and names
Core paths, never `agents/` or other repository files.

Rules for a page:

- Keep the template's section order.
- Write for an application developer who has not read the Core source.
- Give every option a default and a reason to change it.
- Take the C++ column from the header and the Python column from `python/src/module.cpp`, at
  the Core that `deps/manifest.json` names as `neat-core`.
- Keep each code block to 10 lines or fewer.
- Link a shipped example for the complete flow.
- Leave out environment variables such as `SIMA_*`. They are internal Core switches, so give
  the API-level fix instead.

Copy everything below the line to `<area>/<api-name>.md`.

---

# <API name>

<!--
Replace every placeholder.
Keep the section order. Delete these comments before you open the pull request.
Use the C++ name in the title, for example "GraphLinkOptions".
-->

## What it is

<!--
Two or three plain-English sentences. Say what the API does and which problem
it solves. Define each Neat term the first time you use it.
-->

## When to use it

<!--
Say when an application needs this API and when it does not. Name the
alternative for the cases where it does not.
-->

## Minimal example

<!--
One C++ block and one Python block, 10 lines or fewer each.
Show only this API. Leave out includes, argument parsing, error handling, and
setup that belongs to another page. Link to that page instead.
Take the code from a shipped example and link its file by public URL for the complete flow.
-->

C++

```cpp
```

Python

```python
```

Complete flow: [<example name>](https://github.com/sima-neat/apps/tree/main/examples/<category>/<example>)

## Options

<!--
One row per option, in the order of the C++ header.
- C++ and Python hold the option's name in that language, or "not available".
- Take the C++ column from the header and the Python column from
  python/src/module.cpp in Core. Do not take either from the examples.
- Default is the value you get when you do not set the option. If the two
  languages have different defaults, give both.
- "Why and when to change it" names a real situation, for example
  "a live camera where a late frame is worthless". "For performance" is not
  an answer.
For an enum, add a second table with one row per value.
-->

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| | | | | | |

## Language differences

<!--
List behavior that differs between C++ and Python: an API that exists in only
one language, a different return value, or a different way to report an error.
Names and missing options belong in the options table, not here.
Write "None" when there is no difference.
-->

## Pitfalls

<!--
Mistakes that are easy to make with this API, each with its symptom and the
fix. Include only pitfalls you reproduced or found in the source.
-->

## Used in

<!-- Link the examples that use this API at https://github.com/sima-neat/apps/tree/main/examples/<category>/<example>. Write "Not used in Apps" if none do. -->

## Source

<!-- Use the short commit hash of the Core checkout you read. -->

- C++ header: `include/<path>.h`
- Python binding: `python/src/module.cpp`, `<bound class or function>`
- Core commit: `<hash>`
