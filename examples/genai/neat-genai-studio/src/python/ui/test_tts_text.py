#!/usr/bin/env python3
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_text import sanitize_for_tts as s

FORBIDDEN = set("*`#$~|{}\\^＊")
fails = 0


def check(inp, *needles, absent=()):
    global fails
    out = s(inp)
    bad = sorted({c for c in out if c in FORBIDDEN})
    if bad:
        print(f"ARTIFACT {bad} survived: {inp!r} -> {out!r}")
        fails += 1
    for nd in needles:
        if nd not in out:
            print(f"MISSING {nd!r}: {inp!r} -> {out!r}")
            fails += 1
    for nd in absent:
        if nd in out:
            print(f"UNWANTED {nd!r}: {inp!r} -> {out!r}")
            fails += 1
    return out


# --- Original battery (must still hold) ---
check("**bold** and *italic*", "bold", "italic")
check("Use `pip install foo` now", "pip install foo")
check("# Heading one\nBody text.", "Heading one", "Body text.")
check("- one\n- two\n- three", "one", "two", "three")
check("See [the docs](https://example.com/x) here", "the docs", "here")
check("![a cat](cat.png) look", "a cat", "look")
check("Visit https://example.com/page for more", "Visit", "for more")
check("> quoted line\ntext", "quoted line", "text")
check("The formula $E = mc^2$ is famous", "equals", "squared", "famous")
check(r"Compute $\frac{a}{b}$ please", "over", "please")
check(r"$\alpha \leq \beta$", "alpha", "less than or equal to", "beta")
check("It costs $5.99 today", "5.99 dollars", "today")
check("~~gone~~ kept", "gone", "kept")
check("snake_case_name here", "snake case name", "here")
check("don't can't won't", "don't", "can't", "won't")
check("state-of-the-art model", "state-of-the-art")
check("正しい日本語のテキスト。", "正しい日本語のテキスト。")

# --- All 29 adversarial findings ---
check("Let me compute $x + 1", "compute", "x")                      # stray $
check("See issue #42 for the full details.", "number 42", "issue")  # #42
check("Streaming chunk: $$x^2 + y^2", "squared")                    # $$ open
check("The languages C# and F# both run on .NET.",
      "C sharp", "F sharp", absent=("on.NET",))                     # C#/F# + .NET space
check("The result is $$\\frac{a}{b}", "over")                       # $$ + frac open
check("The temperature is $37^\\circ C$", "37", "degrees",
      absent=("dollars",))                                          # currency/math desync
check("Sum: $a_1 x^1 + a_2 x^2$", "squared")                        # sub/sup
check("Costs $x and $y and $z", "Costs")                            # odd $ count
check("Set the $PATH environment variable correctly.", "PATH")      # env var
check("He said the cost was 100$ total.", "100", "total")           # trailing $
check("Here is the formula: $", "formula")                          # trailing lone $
check("This item is only $.99 today!", "99 dollars", "today")       # $.99
check("Match the pattern ^abc$ in your editor.", "abc", "editor")   # regex anchors
check("We have $a \\# b$", "a", "b")                                # \# in math
check("We're ranked #1 in the region.", "number 1", "region")       # #1
check("Loving this! #Winning #100DaysOfCode", "Winning")            # hashtags
check("彼は C# を勉強しています。", "C sharp", "勉強")                  # CJK + C#
check("Range from $5 to $x dollars", "5 dollars", "x")              # mixed currency/var
check("The integral $\\int_a^b f(x)\\,dx$", "integral", "from a to b")  # bounds
check("$$\\begin{align*} a &= b \\end{align*}$$", "a", "equals", "b",
      absent=("begin", "align"))                                    # environment
check("It costs US$5 for shipping.", "5 dollars", "shipping")       # US$5
check("Prices: $5,$10,$15 range.", "5 dollars", "10 dollars", "15 dollars")
check("Items are 5$ and 10$ each.", "Items", "each")               # trailing $ pair
check("The kids' toys are everywhere.", "kids' toys")              # possessive apostrophe
check("Text <!-- hidden comment --> visible.", "Text", "visible.",
      absent=("hidden",))                                           # html comment
check("Reference [link][1] here.\n\n[1]: https://example.com",
      "link", "here.", absent=("example.com",))                    # ref link + def
check("- [ ] Buy milk\n- [x] Walk the dog", "Buy milk", "Walk the dog")  # task list
check("Text with a note.[^1]\n\n[^1]: The footnote body.", "Text with a note.")

# --- 13 regressions caught by the v2 verification round ---
check("[Warning]: do not touch the wires.", "Warning", "wires")          # [1] prose kept
check("[Note]: results are preliminary.", "Note", "preliminary")
check("You owe $1,000.", "1,000 dollars")                                # [2] grouped currency
check("The deal is worth $1,000,000.", "1,000,000 dollars")
check("The shirt was $5. A-B pricing made it $3.",
      "5 dollars", "3 dollars", "A-B", absent=("A minus B",))            # [3] no spurious math span
check("It cost $5. Read a/b then paid $9.", "a/b", absent=("a over b",))
check("It costs $5.99.", "5.99 dollars")                                 # [4] price ends sentence
check("The item is $10.", "10 dollars")
check("#42 wins the race", "number 42")                                  # [5] #42 at line start
check("#7 needs review now", "number 7")
check("We support pre- and post-processing steps.",
      "post-processing", absent=("pre-and",))                           # [6] hanging hyphen
check("left- or right-handed people", "right-handed", absent=("left-or",))
check("Use `git commit -m` to save.", "commit", absent=("commit-m",))
check(r"x &= a + b \\ &= c", "a", "b", absent=("&",))                    # [7] alignment &
check(r"$\int_0^\infty e^{-x}\,dx$", "from 0 to infinity")              # [8] command bound
check(r"$\sum_{n=1}^N n^2$", "the sum from n equals 1 to N of")         # [9] grammatical bounds
check(r"\label{eq:main}", absent=("label",))                            # [10] metadata dropped
check(r"See equation \eqref{eq:1} for details.", "See equation", "for details",
      absent=("eqref",))
check(r"$\frac{1}{1+\frac{1}{1+\frac{1}{1+\frac{1}{1+\frac{1}{1+\frac{1}{1+\frac{1}{1+"
      r"\frac{1}{2}}}}}}}}$", absent=("frac",))                          # [11] deep nesting
check("Match the /^foo$/ pattern here.", "foo", "pattern",
      absent=("to the power of",))                                       # [12] regex ^ in prose

# --- 9 regressions caught by the v3 convergence round ---
check("We have $5 + 3 = 8$ as a fact.", "5 plus 3 equals 8", absent=("dollars",))  # [0]
check("Compute $2 \\times 2 = 4$ now.", "2 times 2 equals 4", absent=("dollars",))
check("The value $5$ is prime.", "5", absent=("5 dollars",))
check("Then $10 - 4$ is six.", "10 minus 4", absent=("dollars",))
check("I have $5 and $10 in cash.", "5 dollars", "10 dollars")     # money, not math span
check("I went home - it was late - and slept.",
      "home", "late", "slept", absent=("home-it", "late-and"))     # [1] em-dash
check("The plan - which failed - cost millions.", absent=("plan-which",))
check("if x<y and a>b then swap the values.", "y and a", "swap")   # [3] inequalities kept
check("[^0-9]: matches any non-digit character here.",
      "matches", "non-digit")                                      # [4] regex class kept
check("See PR#42 for the fix.", "number 42", absent=("PRnumber",))  # [5] glued hash
check("This closes issue#123 finally.", "number 123", absent=("issuenumber",))
check("Use List<int> and Map<String, Integer> here.",
      "List", "int", "Map")                                        # [6] type params kept
check("Text and notice &copy; 2020 here.", "Text", "2020", absent=("copy;",))  # [7] entity
check("She said ' hello ' to me quietly.", "hello", absent=("said'hello",))    # [8] quotes
check("don ' t stop and can ' t wait", "don't", "can't")          # contraction still repaired

# --- 6 issues caught by the v4 final round ---
check("Our R&D; the budget for it was cut this quarter.",
      "R and D", "budget")                                         # [v4-0] entity delete
check("Companies like AT&T; Verizon are big.", "AT and T", "Verizon")
check(r"The roots are $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$ exactly.",
      "over", "square root", "plus or minus", absent=("frac",))    # [v4-1] nested frac
check(r"$\frac{x^{2}}{y}$", "x squared over y", absent=("frac",))
check(r"$\frac{a}{\sqrt{b}}$", "a over the square root of b", absent=("frac",))
check("The company raised $1.5M in funding.", "1.5 million dollars")  # [v4-2] magnitude
check("The deal was worth $2.5B to the firm.", "2.5 billion dollars")
check("The role pays $120K plus equity.", "120 thousand dollars")
check("Revenue grew to $5M last year and $100K in Q1.",
      "5 million dollars", "100 thousand dollars")                 # [v4-5]
check("Tickets are $20-$30 for the show tonight.",
      "20 dollars", "30 dollars", absent=("20 minus 30",))         # [v4-3] dash range
check("Buy 3 - 2 = 1 apples", "3", "apples")                       # sanity: prose math ok
check(r"$$\begin{align} a &= b \\ c &= d \end{align}$$",
      "a equals b", "c equals d", absent=("and equals",))          # [v4-4] align &

# --- Perf / robustness (backtracking guard) ---
for label, payload in [("20k$", "$" * 20000), ("8k[", "\\[" * 8000),
                       ("2k-", "-" * 2000 + "z"), ("dashline", "-" * 5000 + " Total: 5"),
                       ("unclosed[", "See [ref" * 6000), ("bracket", "[" * 30000),
                       ("truncjson", "[1, " * 8000),
                       ("mixed", ("$x^2$ " * 2000)), ("longline", "a" * 50000),
                       ("tabs", "\t" * 50000 + "done"),
                       ("comments", "<!--" * 10000 + "-->"),
                       ("punct", " " * 50000 + ".")]:
    t0 = time.perf_counter()
    out = s(payload)
    dt = time.perf_counter() - t0
    bad = sorted({c for c in out if c in FORBIDDEN})
    if bad:
        print(f"ARTIFACT {bad} survived perf case {label}")
        fails += 1
    if dt > 1.0:
        print(f"SLOW {label}: {dt:.3f}s")
        fails += 1
    print(f"  perf {label}: {dt*1000:.1f} ms")

# --- Edge: empty / whitespace / no double spaces ---
assert s("") == "" and s("   ") == ""
assert "  " not in s("a\n\n\nb   c")

if fails:
    print(f"\n{fails} FAILURES")
    sys.exit(1)
print("\nALL PASS — no artifacts, needles present, perf bounded")
