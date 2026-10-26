"""Canonical answer normalization & equivalence for short VQA answers.

`_norm` is a wide string normalizer (lowercases, strips articles / quotes /
brackets / whitespace / commas / periods). `norm_match` first tries string
equality under `_norm`; if both sides look numeric it falls back to
`mathruler.grader.grade_answer` so "16" / "sixteen" / "16.0" all match.
"""
import re
import string

try:
    from mathruler.grader import grade_answer as _mathruler_grade
except Exception:  # pragma: no cover
    _mathruler_grade = None


_ARTICLES = re.compile(r"^(?:a|an|the)\s+(?=\S)", re.IGNORECASE)
_LEAD_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"^(?:the\s+)?(?:final\s+)?answer(?:'s|\s+is|\s+was|\s+should\s+be|\s+to\s+the\s+question\s+is)?\s*[:：]?\s*",
        r"^(?:the\s+)?(?:final\s+|correct\s+)?(?:answer|solution|result|value|number|count|total)\s+is\s*[:：]?\s*",
        r"^(?:so|thus|hence|therefore|finally|then|consequently|in\s+conclusion)[：:,，]?\s*",
        r"^(?:所以|因此|综上|结论是|最终答案[是为：:]?|答案是|答案为)\s*[:：]?\s*",
        r"^=\s*",
        r"^[:：=]\s*",
    )
]


def _strip_leadins(text: str) -> str:
    c = text
    for _ in range(5):
        before = c
        for pat in _LEAD_PATTERNS:
            c = pat.sub("", c).strip()
        c = c.strip("`'\"" + "“”‘’()[]{}")
        c = _ARTICLES.sub("", c).strip()
        if c == before:
            break
    return c


_THOUSAND_SEP = re.compile(r"(\d),(\d)")
# Preserve Unicode word chars (CJK / Hiragana / Katakana / etc.) so non-ASCII
# short answers like "家"/"堂" don't get stripped to empty. \w in Python 3
# matches [a-zA-Z0-9_] plus Unicode letters/digits by default. . - / kept for
# numeric expressions like "0.5" / "1/2".
_FRACTION_SAFE = re.compile(r"[^\w\.\-/_\s]", re.UNICODE)
# Sentence-ending punctuation. Period is excluded when followed by a digit so
# decimals like "0.5" survive ("." preceded by digit AND followed by digit).
_SENTENCE_END = re.compile(r"[!?\n;；。！？]+|\.(?!\d)")

# Trailing "boxed answer" suffix tokens that should be stripped so "magenta
# box" matches "magenta". At this point whitespace has already been removed,
# so patterns match the end of the compressed string directly. Order matters:
# longer suffixes first.
_BOX_SUFFIX_PATTERNS = [
    re.compile(r"outlined$", re.IGNORECASE),
    re.compile(r"outline$", re.IGNORECASE),
    re.compile(r"region$", re.IGNORECASE),
    re.compile(r"area$", re.IGNORECASE),
    re.compile(r"box$", re.IGNORECASE),
    re.compile(r"框$"),
    re.compile(r"区域$"),
]


def _strip_box_suffix(text: str) -> str:
    """Repeatedly strip trailing 'box'/'region'/'框' etc. so color-only and
    color+box answers normalize to the same token. Stops when no suffix
    matches or the remaining text would become empty. Refuses to strip if the
    remaining prefix is shorter than 2 characters (so "box" alone stays).
    """
    t = text
    for _ in range(3):
        if not t:
            break
        before = t
        for pat in _BOX_SUFFIX_PATTERNS:
            new = pat.sub("", t)
            if len(new) >= 2 and new != t:
                t = new
                break
        if t == before:
            break
    return t


def _norm(text: str) -> str:
    """Wide string normalizer: case-insensitive, ignores whitespace/articles,
    preserves digits / decimal points / minus signs / fraction slashes so
    numeric expressions like "1,600", "0.5", "1/2" survive.

    Comma is NOT a separator here — it's treated as a thousands separator
    when it sits between digits and stripped otherwise.
    """
    if not text:
        return ""
    t = str(text).strip()
    t = _strip_leadins(t)
    # Drop thousand separators inside digit groups: "1,600" -> "1600".
    while _THOUSAND_SEP.search(t):
        t = _THOUSAND_SEP.sub(r"\1\2", t)
    # Cut at first sentence-ending punctuation (NOT comma/colon, and NOT a
    # period immediately followed by a digit so "0.5" is preserved).
    t = _SENTENCE_END.split(t, 1)[0]
    # Strip surrounding quotes/brackets.
    t = t.strip("`'\"" + "“”‘’()[]{}")
    # Remove every character NOT in [alnum . - / _ whitespace].
    t = _FRACTION_SAFE.sub(" ", t)
    t = re.sub(r"\s+", "", t)
    # Strip CJK punctuation that might have slipped through.
    t = re.sub(r"[，。！？；：、（）【】《》「」『』]", "", t)
    t = t.lower()
    # Strip trailing " box"/" region"/"框" etc. so "magenta box" == "magenta".
    t = _strip_box_suffix(t)
    return t


_WORD_NUMS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion",
}


# A token "looks numeric" if it is a number/fraction/decimal or a spelled-out
# English number. Used to decide whether to fall back to mathruler.
_NUMERIC_TOKEN_RE = re.compile(r"^-?\d+(\.\d+)?(/-?\d+(\.\d+)?)?$")


def _looks_numeric(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    if not t:
        return False
    if _NUMERIC_TOKEN_RE.match(t):
        return True
    # English spelled-out numbers ("sixteen", "one hundred and three")
    tokens = re.sub(r"[^a-z\s]", " ", t).split()
    if tokens and all(tok in _WORD_NUMS or tok in {"and"} for tok in tokens):
        return True
    return False


def norm_match(candidate: str, ground_truth: str) -> bool:
    """True if `candidate` matches `ground_truth` after normalization.

    Uses _norm string equality first. If both sides look numeric, also falls
    back to mathruler's sympy-backed grade_answer, so "16" / "16.0" /
    "sixteen" all match.
    """
    a = _norm(candidate)
    b = _norm(ground_truth)
    if not a or not b:
        return False
    if a == b:
        return True
    if _looks_numeric(a) and _looks_numeric(b) and _mathruler_grade is not None:
        try:
            return bool(_mathruler_grade(a, b))
        except Exception:
            return False
    return False