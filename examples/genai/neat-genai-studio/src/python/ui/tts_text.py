"""Turn a model's Markdown/LaTeX answer into clean, speakable text for TTS.

Models reply in Markdown (headings, lists, ``code``, **bold**, links, tables) and
often LaTeX (``$x^2$``, ``\\frac{a}{b}``). Speaking that verbatim makes Piper utter
"star star", "dollar", "backslash frac", "hash" and read out raw URLs. This module
strips the formatting and converts common math to words so only the prose is
spoken. It uses the standard library only (no pyneat), so it is importable and
unit-testable on any host.

It is called per streaming *sentence chunk*, so inputs are frequently partial or
unbalanced (an opening ``$`` with no close, half a ``\\frac``, a trailing
backslash). The hard guarantee is that **no formatting artifact character**
(``* ` # $ ~ | { } \\ ^``) ever survives into the output — a final sweep enforces
it — while common constructs are converted to natural words first.

The single entry point is :func:`sanitize_for_tts`.
"""
from __future__ import annotations

import re

# Named LaTeX commands → spoken words (Greek letters, operators, relations, funcs).
_LATEX_WORDS = {
    # Greek
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta",
    "epsilon": "epsilon", "varepsilon": "epsilon", "zeta": "zeta", "eta": "eta",
    "theta": "theta", "vartheta": "theta", "iota": "iota", "kappa": "kappa",
    "lambda": "lambda", "mu": "mu", "nu": "nu", "xi": "xi", "omicron": "omicron",
    "pi": "pi", "varpi": "pi", "rho": "rho", "sigma": "sigma", "tau": "tau",
    "upsilon": "upsilon", "phi": "phi", "varphi": "phi", "chi": "chi", "psi": "psi",
    "omega": "omega",
    "Gamma": "gamma", "Delta": "delta", "Theta": "theta", "Lambda": "lambda",
    "Xi": "xi", "Pi": "pi", "Sigma": "sigma", "Phi": "phi", "Psi": "psi",
    "Omega": "omega",
    # Binary operators
    "times": "times", "cdot": "times", "div": "divided by", "ast": "times",
    "pm": "plus or minus", "mp": "minus or plus", "oplus": "plus", "otimes": "times",
    # Relations
    "leq": "less than or equal to", "le": "less than or equal to",
    "geq": "greater than or equal to", "ge": "greater than or equal to",
    "neq": "not equal to", "ne": "not equal to", "equiv": "equivalent to",
    "approx": "approximately", "sim": "similar to", "propto": "proportional to",
    "ll": "much less than", "gg": "much greater than", "cong": "congruent to",
    # Set / logic
    "in": "in", "notin": "not in", "subset": "subset of", "subseteq": "subset of",
    "supset": "superset of", "supseteq": "superset of", "cup": "union",
    "cap": "intersection", "emptyset": "the empty set", "varnothing": "the empty set",
    "forall": "for all", "exists": "there exists", "neg": "not", "land": "and",
    "lor": "or",
    # Arrows
    "rightarrow": "goes to", "Rightarrow": "implies", "to": "to",
    "leftarrow": "from", "Leftarrow": "is implied by", "leftrightarrow": "if and only if",
    "Leftrightarrow": "if and only if", "iff": "if and only if", "mapsto": "maps to",
    # Big operators / calculus
    "sum": "the sum of", "prod": "the product of", "int": "the integral of",
    "oint": "the contour integral of", "lim": "the limit of", "partial": "partial",
    "nabla": "del", "infty": "infinity",
    # Functions
    "sin": "sine", "cos": "cosine", "tan": "tangent", "cot": "cotangent",
    "sec": "secant", "csc": "cosecant", "arcsin": "arc sine", "arccos": "arc cosine",
    "arctan": "arc tangent", "sinh": "hyperbolic sine", "cosh": "hyperbolic cosine",
    "tanh": "hyperbolic tangent", "log": "log", "ln": "natural log",
    "exp": "the exponential of", "det": "the determinant of", "gcd": "gcd",
    "min": "the minimum of", "max": "the maximum of",
    # Misc symbols
    "prime": "prime", "circ": "degrees", "deg": "degrees", "angle": "angle",
    "perp": "perpendicular to", "parallel": "parallel to", "cdots": "and so on",
    "ldots": "and so on", "dots": "and so on", "vdots": "and so on",
    "ddots": "and so on", "quad": " ", "qquad": " ", "hbar": "h bar",
}

# Spacing / formatting commands that carry no sound — drop them.
_LATEX_SPACING = (
    "left", "right", "big", "Big", "bigg", "Bigg", "displaystyle", "textstyle",
    "scriptstyle", "limits", "nolimits", "notag", "nonumber",
)
_SPACING_RE = re.compile(r"\\(?:" + "|".join(_LATEX_SPACING) + r")\b")

# Big operators spoken naturally with bounds: "\sum_a^b" → "the sum from a to b of".
_BIG_OP = {
    "int": "the integral", "iint": "the double integral", "iiint": "the triple integral",
    "oint": "the contour integral", "sum": "the sum", "prod": "the product",
    "coprod": "the coproduct",
}
# Metadata commands that carry no speech (name + braced argument dropped whole).
_LATEX_META = re.compile(r"\\(?:label|ref|eqref|cite[a-z]*|tag|footnote|caption)\s*\{[^{}]*\}")
_LATEX_BOUND = r"(\{[^{}]*\}|\\[A-Za-z]+|[A-Za-z0-9]+)"   # a sub/superscript limit token


def _superscript(expr: str) -> str:
    """Render a superscript exponent as words: ^2 → squared, ^3 → cubed, else
    'to the power of ...'."""
    e = expr.strip().strip("{}").strip()
    if e == "2":
        return " squared "
    if e == "3":
        return " cubed "
    return " to the power of " + e + " "


def _read_group(s: str, i: int):
    """Given s[i] == '{', return (inner_text, index_after_closing_brace) with full
    brace matching (handles nesting). On an unbalanced group, take the rest."""
    depth = 0
    for k in range(i, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:k], k + 1
    return s[i + 1:], len(s)


def _expand_frac(s: str) -> str:
    """Rewrite \\frac{A}{B} → " A over B " with brace matching, so nested numerators
    or denominators (\\sqrt{...}, x^{2}, another \\frac, the quadratic formula) are
    handled — something the flat `\\{[^{}]*\\}` regex cannot do."""
    if "frac" not in s:
        return s
    pat = re.compile(r"\\[dt]?frac\s*(?=\{)")
    for _ in range(40):                       # bound: one frac resolved per pass
        m = pat.search(s)
        if not m:
            break
        b1 = s.index("{", m.end() - 1)
        num, j = _read_group(s, b1)
        while j < len(s) and s[j] in " \t":
            j += 1
        if j < len(s) and s[j] == "{":
            den, k = _read_group(s, j)
            s = s[:m.start()] + " " + num + " over " + den + " " + s[k:]
        else:                                  # malformed (no denominator) — drop cmd
            s = s[:m.start()] + " " + num + " " + s[j:]
    return s


def _latex_commands(s: str) -> str:
    """Convert LaTeX *commands* (``\\frac``, ``\\sqrt``, ``\\alpha`` …) and
    superscripts to spoken words. Safe to run on ordinary prose: it only touches
    backslash commands and ``^`` (it leaves ``_`` alone so ``snake_case`` is not
    mangled). Used both inside detected math spans and as a global pass that mops
    up un-delimited LaTeX from partial streaming chunks."""
    # \begin{env}/\end{env} and metadata (\label, \tag, \cite…) carry no speech.
    s = re.sub(r"\\(?:begin|end)\s*\{[^{}]*\}", " ", s)
    s = _LATEX_META.sub(" ", s)
    # \frac{a}{b} → "a over b" (brace-aware, so nested fracs/sqrt/exponents work).
    s = _expand_frac(s)
    # Roots.
    s = re.sub(r"\\sqrt\s*\[([^\[\]]*)\]\s*\{([^{}]*)\}", r" \1 root of \2 ", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r" the square root of \1 ", s)
    # Text-ish wrappers and accents: keep the contents, drop the command.
    s = re.sub(
        r"\\(?:text|textrm|textbf|textit|mathrm|mathbf|mathit|mathcal|mathbb|"
        r"mathsf|mathtt|operatorname|boldsymbol|bm)\s*\{([^{}]*)\}", r" \1 ", s)
    s = re.sub(r"\\(?:hat|bar|vec|tilde|dot|ddot|overline|underline|widehat|"
               r"widetilde)\s*\{([^{}]*)\}", r" \1 ", s)
    # Big-operator bounds: \int_a^b → "the integral from a to b of" (limits may be
    # braced, bare tokens, or a command like \infty). Run before generic ^/_.
    s = re.sub(
        r"\\(int|iint|iiint|oint|sum|prod|coprod)\s*_\s*" + _LATEX_BOUND +
        r"\s*\^\s*" + _LATEX_BOUND,
        lambda m: " " + _BIG_OP[m.group(1)] + " from " + m.group(2).strip("{}")
                  + " to " + m.group(3).strip("{}") + " of ", s)
    s = re.sub(r"\\lim\s*_\s*(\{[^{}]*\}|[^\s{}]+)",
               lambda m: " the limit as " + m.group(1).strip("{}") + " of ", s)
    # Superscripts (braced or single token), anchored to a base so a regex '^'
    # anchor in prose ("/^foo$/") is not mistaken for an exponent. Subscripts are
    # handled only inside math spans (see _latex_math) to protect identifiers.
    s = re.sub(r"(?<=[\w)\]}])\^\s*\{([^{}]*)\}", lambda m: _superscript(m.group(1)), s)
    s = re.sub(r"(?<=[\w)\]}])\^\s*([A-Za-z0-9]+)", lambda m: _superscript(m.group(1)), s)
    # Spacing commands and explicit line breaks → nothing.
    s = _SPACING_RE.sub(" ", s)
    s = re.sub(r"\\[,;:!>()\[\] ]", " ", s)
    s = re.sub(r"\\\\", " ", s)
    # Remaining named commands → words (unknown ones keep their name, sans slash).
    s = re.sub(r"\\([A-Za-z]+)",
               lambda m: " " + _LATEX_WORDS.get(m.group(1), m.group(1)) + " ", s)
    return s


def _latex_math(inner: str) -> str:
    """Convert the inside of a *delimited* math span to spoken words: commands,
    subscripts, and operators/relations."""
    s = _latex_commands(inner)
    # Subscripts (safe here — we are between math delimiters).
    s = re.sub(r"_\s*\{([^{}]*)\}", r" sub \1 ", s)
    s = re.sub(r"_\s*([A-Za-z0-9]+)", r" sub \1 ", s)
    # Operators and relations to words.
    s = s.replace("=", " equals ").replace("+", " plus ")
    s = re.sub(r"(?<=[\w)\]])\s*-\s*(?=[\w(\[])", " minus ", s)   # binary minus
    s = s.replace("-", " minus ")                                  # unary/leading
    s = s.replace("*", " times ").replace("/", " over ")
    s = s.replace("<", " less than ").replace(">", " greater than ")
    # Drop any leftover math grouping/markup so nothing leaks from the span.
    s = re.sub(r"[{}\[\]|&\\^_~#$]", " ", s)
    return s


def _math_repl(m: "re.Match") -> str:
    return " " + _latex_math(m.group(1)) + " "


# A "$...$" span is only treated as math when its inner text actually looks like
# math. This keeps a real inline formula ("$5 + 3 = 8$") from being read as money
# while leaving two separate prices in one chunk ("$5 and $10", or "was $5. A-B
# … $3.") for the currency pass — the two used to fight over the "$".
_STRONG_HINT = re.compile(r"[\\^_=<>]")                       # reliable math markers
_WEAK_HINT = re.compile(r"\d[A-Za-z]|[A-Za-z]\d")            # 5x, x5 (also 5M money)
_ENGLISH_WORD = re.compile(r"[A-Za-z]{3,}")                  # a real word → prose
# Arithmetic only when an operator sits BETWEEN two numbers ("10 - 4"), so a price
# range "$20-$30" (inner "20-") is not misread as subtraction.
_ARITHMETIC = re.compile(r"(?=.*\d\s*[-+*/]\s*\d)[\d\s+\-*/().,]+")
_LONE = re.compile(r"[A-Za-z0-9]{1,4}")                       # x, n, 5, x0

_MAGNITUDE = {"k": "thousand", "m": "million", "b": "billion", "bn": "billion",
              "thousand": "thousand", "million": "million", "billion": "billion",
              "trillion": "trillion"}


def _currency_repl(m: "re.Match") -> str:
    amount = m.group(1)
    word = _MAGNITUDE.get((m.group(2) or "").lower(), "")
    return " " + amount + (" " + word if word else "") + " dollars "


def _mathy(inner: str) -> bool:
    """Whether a "$…$" inner should be spoken as math. Strong markers (\\ ^ _ = < >)
    always qualify; the weak markers (a bare "5M", pure arithmetic, a lone token)
    only qualify when the span has no real word, so a number next to ordinary prose
    ("$5M last year and $10") is left for the currency pass instead."""
    s = inner.strip()
    if not s:
        return False
    if _STRONG_HINT.search(s):                      # \  ^  _  =  <  >
        return True
    if _ENGLISH_WORD.search(s):                     # contains a word → prose, not math
        return False
    if _WEAK_HINT.search(s):                         # 5x / x5
        return True
    if len(s) <= 20 and _ARITHMETIC.fullmatch(s):   # arithmetic w/ operator: "10 - 4"
        return True
    return bool(_LONE.fullmatch(s))                 # a lone token: x, n, 5, x0


def _inline_dollar_repl(m: "re.Match") -> str:
    if _mathy(m.group(1)):
        return " " + _latex_math(m.group(1)) + " "
    return m.group(0)                              # leave "$…$" for the currency pass


# Pre-compiled patterns used on every call.
_FENCE = re.compile(r"```+[ \t]*[A-Za-z0-9_+\-]*")
_IMAGE = re.compile(r"!\[([^\]]{0,300})\]\([^)]{0,600}\)")
_FOOTNOTE_REF = re.compile(r"\[\^\w{1,12}\]")
_INLINE_LINK = re.compile(r"\[([^\]]{0,300})\]\([^)]{0,600}\)")
_REF_LINK = re.compile(r"\[([^\]]{0,300})\]\[[^\]]{0,120}\]")
_AUTO_LINK = re.compile(r"<((?:https?|mailto):[^>]{0,600})>")
_BARE_URL = re.compile(r"\b(?:https?|ftp)://\S+")
# Only strip *real* HTML tags (known element names) so prose inequalities
# ("x<y and a>b") and generic type parameters ("List<int>") are not swallowed.
_HTML_TAG = re.compile(
    r"</?(?:a|abbr|b|blockquote|br|button|caption|center|cite|code|col|colgroup|"
    r"dd|del|details|div|dl|dt|em|figcaption|figure|font|footer|form|h[1-6]|header|"
    r"hr|i|iframe|img|input|ins|kbd|label|li|mark|nav|ol|p|pre|q|s|samp|section|"
    r"select|small|source|span|strong|sub|summary|sup|table|tbody|td|textarea|"
    r"tfoot|th|thead|tr|u|ul|video|audio|style|script)\b[^>]{0,400}?>", re.IGNORECASE)
# Currency: "$" then a well-formed amount, with an optional magnitude suffix
# ("$1.5M" → "1.5 million dollars"), only when it is not the start of a math token.
# A trailing "." is allowed so a price ending a sentence ("$5.99.") still matches.
_CURRENCY = re.compile(
    r"\$\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"\s?(bn|BN|[KkMmBb]|thousand|million|billion|trillion)?(?![\w^])")
_MATH_DISPLAY_DOLLAR = re.compile(r"\$\$(.{1,2000}?)\$\$", re.DOTALL)
_MATH_DISPLAY_BRACKET = re.compile(r"\\\[(.{1,2000}?)\\\]", re.DOTALL)
_MATH_INLINE_PAREN = re.compile(r"\\\((.{1,2000}?)\\\)", re.DOTALL)
_MATH_INLINE_DOLLAR = re.compile(r"\$([^$\n]{1,800}?)\$")
_HR = re.compile(r"(?m)^[ \t]{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
# ATX heading: the '#'s must be followed by a space or end of line, so "#42" (a
# reference, not a heading) is left for _HASH_NUM.
_HEADING = re.compile(r"(?m)^[ \t]{0,3}#{1,6}(?:[ \t]+|$)")
_BLOCKQUOTE = re.compile(r"(?m)^[ \t]{0,3}>+[ \t]?")
_LIST_MARKER = re.compile(r"(?m)^[ \t]{0,3}(?:[-*+]|\d+[.)])[ \t]+")
_TASK_BOX = re.compile(r"(?m)^[ \t]*\[[ xX]\][ \t]*")
_HASH_SHARP = re.compile(r"(?<![A-Za-z0-9])([A-Za-z])#")      # C#, F#, A# → "C sharp"
_HASH_NUM = re.compile(r"#(\d)")                              # #42, #1 → "number 42"
# Only numeric character references are stripped generically; named entities go
# through the table below. (A blanket "&\w+;" would eat "R&D;", "AT&T;" etc.)
_GENERIC_ENTITY = re.compile(r"&#(?:\d{1,6}|x[0-9a-fA-F]{1,6});")
_ARTIFACTS = re.compile(r"[$#`~|{}\\^*＊]")                   # final safety sweep
# Only repair a space-split contraction ("don ' t" → "don't"); do not weld a
# spaced quotation mark ("said ' hello '") onto its neighbours.
_APOS = re.compile(r"(\w)\s+['’‘]\s*(t|s|d|m|re|ll|ve)\b")

_HTML_ENTITIES = {
    "&amp;": " and ", "&lt;": " less than ", "&gt;": " greater than ",
    "&nbsp;": " ", "&quot;": '"', "&#39;": "'", "&apos;": "'", "&hellip;": "...",
    "&mdash;": " - ", "&ndash;": " - ",
    "&copy;": " ", "&reg;": " ", "&trade;": " ", "&deg;": " degrees ",
    "&bull;": " ", "&middot;": " ", "&sect;": " section ", "&para;": " ",
    "&times;": " times ", "&divide;": " divided by ", "&plusmn;": " plus or minus ",
    "&micro;": " micro ", "&pound;": " pounds ", "&euro;": " euros ", "&yen;": " yen ",
    "&cent;": " cents ", "&laquo;": '"', "&raquo;": '"', "&ldquo;": '"', "&rdquo;": '"',
    "&lsquo;": "'", "&rsquo;": "'", "&infin;": " infinity ", "&rarr;": " to ",
}


def _strip_auxiliary_definition_lines(text: str) -> str:
    """Drop Markdown footnote/reference definitions with a linear line scan."""
    def is_definition(line: str) -> bool:
        value = line.lstrip(" \t")
        if not value.startswith("["):
            return False
        close = value.find("]:", 1)
        if close < 0:
            return False
        label = value[1:close]
        remainder = value[close + 2:].lstrip(" \t")
        if label.startswith("^"):
            ident = label[1:]
            return 1 <= len(ident) <= 12 and all(c == "_" or c.isalnum() for c in ident)
        if not 1 <= len(label) <= 120:
            return False
        if remainder.startswith("<"):
            remainder = remainder[1:]
        return remainder.startswith(("http://", "https://", "ftp://", "mailto:", "www."))

    return "\n".join(" " if is_definition(line) else line for line in text.split("\n"))


def _strip_html_comments(text: str) -> str:
    """Remove complete HTML comments without regular-expression backtracking."""
    pieces = []
    cursor = 0
    while True:
        start = text.find("<!--", cursor)
        if start < 0:
            pieces.append(text[cursor:])
            break
        end = text.find("-->", start + 4)
        if end < 0:
            pieces.append(text[cursor:])
            break
        pieces.append(text[cursor:start])
        pieces.append(" ")
        cursor = end + 3
    return "".join(pieces)


def _strip_table_separator_lines(text: str) -> str:
    """Drop Markdown table separators using a bounded linear line scan."""
    def is_separator(line: str) -> bool:
        value = line.strip(" \t")
        return len(value) >= 2 and "-" in value and all(c in "-:| \t" for c in value)

    return "\n".join(" " if is_separator(line) else line for line in text.split("\n"))


def _normalize_spaces(text: str) -> str:
    """Collapse whitespace and remove spaces before terminal punctuation."""
    value = " ".join(text.split())
    punctuation = ".,;:!?。！？"
    out = []
    for index, char in enumerate(value):
        if (
            char == " "
            and index + 1 < len(value)
            and value[index + 1] in punctuation
            and (index + 2 == len(value) or value[index + 2] == " ")
        ):
            continue
        out.append(char)
    return "".join(out)


def sanitize_for_tts(text: str) -> str:
    """Strip Markdown and LaTeX from ``text`` and return clean, speakable prose."""
    if not text:
        return ""

    # 1. Code fences and inline code — drop the markers, keep any inner words.
    text = _FENCE.sub(" ", text)
    text = text.replace("`", "")

    # 2. Footnote / reference-link definition lines — drop them whole (auxiliary).
    text = _strip_auxiliary_definition_lines(text)

    # 3. Images, links, footnote refs, autolinks, bare URLs — keep visible text.
    #    Bracket forms only run when a "]" exists (a partial chunk of unclosed "["
    #    can't backtrack, and the bounded inner groups keep them linear).
    if "]" in text:
        text = _IMAGE.sub(r"\1", text)
        text = _FOOTNOTE_REF.sub("", text)
        text = _INLINE_LINK.sub(r"\1", text)
        text = _REF_LINK.sub(r"\1", text)
    text = _AUTO_LINK.sub(r"\1", text)
    text = _BARE_URL.sub(" ", text)

    # 4. HTML comments, real tags, and named/numeric entities.
    text = _strip_html_comments(text)
    text = _HTML_TAG.sub(" ", text)
    for ent, rep in _HTML_ENTITIES.items():
        text = text.replace(ent, rep)
    text = _GENERIC_ENTITY.sub(" ", text)    # &#8212; &#x2014;

    # 5. LaTeX math spans first, then currency on what remains. Display math and
    #    "\\(..\\)" are always math; an inline "$…$" is math only when its inner
    #    text looks mathy (see _mathy) so "$5 and $10" stays money, not a span.
    if "$" in text:
        text = _MATH_DISPLAY_DOLLAR.sub(_math_repl, text)
        text = _MATH_INLINE_DOLLAR.sub(_inline_dollar_repl, text)
        text = _CURRENCY.sub(_currency_repl, text)
    if "\\[" in text:
        text = _MATH_DISPLAY_BRACKET.sub(_math_repl, text)
    if "\\(" in text:
        text = _MATH_INLINE_PAREN.sub(_math_repl, text)

    # A bare "&" (R&D, or a LaTeX alignment column that survived) → "and". Done
    # after math so an align/matrix "&" is dropped inside its span, not verbalized.
    text = text.replace("&", " and ")

    # 6. Block-level Markdown (still multi-line at this point).
    text = _HR.sub(" ", text)
    text = _strip_table_separator_lines(text)
    text = _HEADING.sub("", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _LIST_MARKER.sub("", text)
    text = _TASK_BOX.sub("", text)           # "[ ]"/"[x]" checkboxes
    text = text.replace("|", " ")            # table cell separators

    # 7. Un-delimited LaTeX left over from partial streaming chunks.
    if "\\" in text or "^" in text:
        text = _latex_commands(text)

    # 8. Hashes: "C#"/"F#" → "C sharp", "#42"/"PR#42" → "number 42"; rest are swept.
    #    The leading space keeps "PR#42" from fusing into "PRnumber".
    if "#" in text:
        text = _HASH_SHARP.sub(r"\1 sharp ", text)
        text = _HASH_NUM.sub(r" number \1", text)

    # 9. Emphasis markers and identifier underscores.
    text = text.replace("_", " ")            # snake_case → "snake case"

    # 10. Final safety sweep: no formatting artifact character may survive.
    text = _ARTIFACTS.sub(" ", text)

    # 11. Whitespace and legibility cleanup. (No hyphen-joining: a spaced hyphen is
    #     usually an em-dash / clause break, so fusing "home - it" into "home-it"
    #     garbled more prose than the rare "state - of - the - art" it helped.)
    text = text.replace("\n", " ")
    text = _normalize_spaces(text)
    text = _APOS.sub(r"\1'\2", text)                 # don ' t → don't
    text = _normalize_spaces(text)                     # "word ." → "word." (keeps ".NET")
    return text
