# Selected definitions from the project-adapted reference source.
# Not a relicense of the original. See MANIFEST.json and THIRD_PARTY_NOTICES.md.
from __future__ import annotations
import re
import json
from decimal import Decimal, ROUND_HALF_UP
_NUM = r"[-+]?\d+(?:\.\d+)?"

_RANGE_RE = re.compile(
    rf"(?:acceptable(?:\s+range)?|range|±|between)\s*:?\s*({_NUM})\s*(?:\s+to\s+|–|—)\s*({_NUM})",
    re.IGNORECASE,
)

def _to_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None

def _parse_ranges(text: str):
    out = []
    for m in _RANGE_RE.finditer(text):
        lo, hi = _to_float(m.group(1)), _to_float(m.group(2))
        if lo is None or hi is None:
            continue
        lo, hi = min(lo, hi), max(lo, hi)
        prefix = text[: m.start()]
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", prefix)
        nominal = _to_float(nums[-1]) if nums else (lo + hi) / 2
        out.append((nominal, lo, hi))
    return out

def _extract_numbers(text: str):
    return [float(m.group(0)) for m in re.finditer(r"[-+]?\d+(?:\.\d+)?", text)]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _extract_final_answer(text: str) -> str:
    '\u4ece\u6a21\u578b\u8f93\u51fa\u91cc\u62a0 final answer\uff0c\u907f\u5f00\u4e2d\u95f4\u63a8\u7406\u6570\u5b57\u3002\n    \u4f18\u5148\u7ea7\uff1a\n      1. \u627e "Answer:"/"Final answer:"/"The answer is"/"Final:" \u7b49\u6807\u8bb0\u540e\u7684\u5185\u5bb9\n      2. \u5982\u679c\u6709 </think> \u6807\u7b7e\uff08Qwen3 thinking \u6a21\u5f0f\uff09\uff0c\u53d6 </think> \u4e4b\u540e\u7684\n      3. \u5426\u5219\u53d6\u6700\u540e ~600 \u5b57\u7b26\uff08final answer \u901a\u5e38\u5728\u672b\u5c3e\uff09\n    '
    if not text:
        return ""
    # Preserved upstream logic; see the corresponding protocol documentation.
    m = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()
    # Preserved upstream logic; see the corresponding protocol documentation.
    for pat in [r"(?:final answer|answer|final|result|the answer is)\s*[:：]\s*(.*?)(?:\n\n|\Z)",
                r"(?:the answer is|answer is)\s+(.*?)(?:\n\n|\Z)"]:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m and m.group(1).strip():
            return m.group(1).strip()
    # Preserved upstream logic; see the corresponding protocol documentation.
    return text[-600:].strip()

def _tokenize_set(s: str) -> set[str]:
    '\u7b80\u6613 token \u96c6\u5408\uff0c\u7528\u4e8e"\u5217\u8868\u5143\u7d20\u96c6\u5408\u5339\u914d"\u3002\n    \u628a\u5b57\u7b26\u4e32\u6309\u9017\u53f7/\u6362\u884c/\u5206\u53f7\u5207\uff0c\u518d\u89c4\u8303\u5316\uff08\u53bb\u6807\u70b9\u3001\u5c0f\u5199\u3001\u538b\u7a7a\u767d\uff09\u3002'
    parts = re.split(r"[,;\n]+", s.lower())
    tokens = set()
    for p in parts:
        p = re.sub(r"[\s\.\(\)\[\]\{\}]+", " ", p).strip(" :-")
        p = re.sub(r"\s+", " ", p)
        # Preserved upstream logic; see the corresponding protocol documentation.
        p = re.sub(r"^(the|a|an)\s+", "", p)
        if p:
            tokens.add(p)
    return tokens

def _normalize_answer(s: str) -> str:
    '\u5f3a\u89c4\u8303\u5316\uff1a\u53bb\u6807\u70b9 / \u5355\u4f4d\u7a7a\u767d / \u5927\u5c0f\u5199\uff0c\u7528\u4e8e"\u7b80\u77ed\u5b57\u7b26\u4e32\u9898"\u5339\u914d\u3002'
    s = (s or "").lower().strip()
    s = re.sub(r"[\s\.,;:!\?\(\)\[\]\{\}'\"]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def score_one(pred: str, golden: str):
    '\u5b98\u65b9 judge \u89c4\u5219\uff08README\uff09\uff1a\n      - golden \u5e26 `acceptable range`\uff1apred \u6570\u503c\u843d\u5728\u4efb\u4e00\u533a\u95f4\u5373\u5bf9\uff08\u591a\u533a\u95f4\u89c6\u4e3a\u5404\u72ec\u7acb\u9879\uff0c\u5168\u90e8\u547d\u4e2d\u624d\u5bf9\uff09\n      - golden \u65e0 range\uff1a\u5b57\u7b26\u4e32\u5339\u914d\uff08\u987a\u5e8f\u654f\u611f\u65f6\u6309 token \u96c6\u5408\uff0c\u7b49\u4ef7\u6027\u4fdd\u7559\u5217\u8868\u9879\uff09\n\n    \u8fd4\u56de (is_correct, reason)\u3002'
    final = _extract_final_answer(pred)
    ranges = _parse_ranges(golden)

    if ranges:
        # Preserved upstream logic; see the corresponding protocol documentation.
        pred_nums = _extract_numbers(final)
        if not pred_nums:
            pred_nums = _extract_numbers(pred)  # Preserved upstream logic; see the corresponding protocol documentation.
        if not pred_nums:
            return False, "no-num"
        hits = [any(lo - 1e-6 <= n <= hi + 1e-6 for n in pred_nums)
                for _, lo, hi in ranges]
        return all(hits), f"range:{sum(hits)}/{len(hits)}"

    # Preserved upstream logic; see the corresponding protocol documentation.
    if _normalize_answer(final) == _normalize_answer(golden):
        return True, "norm-exact"

    # Preserved upstream logic; see the corresponding protocol documentation.
    g_tokens = _tokenize_set(golden)
    p_tokens = _tokenize_set(final)
    if g_tokens and p_tokens:
        # Preserved upstream logic; see the corresponding protocol documentation.
        if g_tokens.issubset(p_tokens):
            return True, "set-cover"

    # Preserved upstream logic; see the corresponding protocol documentation.
    g_nums = _extract_numbers(golden)
    p_nums = _extract_numbers(final)
    if g_nums and p_nums and all(any(abs(a - b) < 1e-6 for b in p_nums) for a in g_nums):
        return True, "num-cover"

    return False, f"no-match:{len(final)}"
