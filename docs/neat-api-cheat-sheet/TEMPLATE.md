# <API name>

<!--
Copy this file to <area>/<api-name>.md and replace every placeholder.
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
Take the code from a shipped example and link to the file for the complete flow.
-->

C++

```cpp
```

Python

```python
```

Complete flow: [<example name>](../../../examples/<category>/<example>/)

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

<!-- Link the examples that use this API. Write "Not used in Apps" if none do. -->

## Source

- C++ header: `include/<path>.h`
- Python binding: `python/src/module.cpp`, `<bound class or function>`
