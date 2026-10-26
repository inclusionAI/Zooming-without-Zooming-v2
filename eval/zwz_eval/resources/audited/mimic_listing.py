# Selected definitions from the project-adapted reference source.
# Not a relicense of the original. See MANIFEST.json and THIRD_PARTY_NOTICES.md.
from __future__ import annotations
import re
import json
from decimal import Decimal, ROUND_HALF_UP
NOISE_PATTERNS = [
    # Preserved upstream logic; see the corresponding protocol documentation.
    r"^based on .*?categor(y|ies)[^\n]*[\.\n]\s*",
    r"^based on .*?images?[^:]*[:：]\s*",
    # Preserved upstream logic; see the corresponding protocol documentation.
    r"^here is[^:]*[:：]\s*",
    r"^here is[^\.]*\.\s*",
    # "In the first/second/N image ... : "
    r"^in the (first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+\w*)[^:]*image[^:]*[:：]\s*",
    # "The items are/is: "
    r"^the items? (are|is)[:：]\s*",
    # "The sub-categories... : "
    r"^the (sub-?)?categor[^:]*[:：]\s*",
    # "List of ...: "
    r"^list of[^:]*[:：]\s*",
    # Preserved upstream logic; see the corresponding protocol documentation.
    r"^the categor(y|ies) (in|under|of)[^:]*[:：]\s*",
]

BULLET_RE = re.compile(r"^\s*[-•*‣▪◦]\s*")

NUM_PREFIX_RE = re.compile(r"^\s*\(?\d+[\)\.]\s*")

TRAILING_PUNCT_RE = re.compile(r"[\.\,\;\:\!\?]+$")

def parse_model_answer_to_objects(ma):
    "\u4ece\u81ea\u7531 model_answer \u6587\u672c\u63d0\u53d6 objects \u5217\u8868 (\u8ddf\u5b98\u65b9 pred.split(', ') \u5bf9\u9f50)"
    if not ma:
        return []
    ma = ma.strip()
    # Preserved upstream logic; see the corresponding protocol documentation.
    # Preserved upstream logic; see the corresponding protocol documentation.
    if "<answer>" in ma and "</answer>" in ma:
        ans_blocks = re.findall(r"<answer>(.*?)</answer>", ma, re.DOTALL)
        objects = []
        seen = set()
        for ans in ans_blocks:
            ans = ans.strip()
            if not ans:
                continue
            # Preserved upstream logic; see the corresponding protocol documentation.
            for item in ans.split(","):
                item = item.strip()
                item = TRAILING_PUNCT_RE.sub("", item).strip()
                if not item:
                    continue
                # Preserved upstream logic; see the corresponding protocol documentation.
                if len(item.split()) > 8:
                    continue
                if re.match(r"^(so|then|because|but|if|no|wait|final|the|in|this|that)", item, re.IGNORECASE):
                    continue
                key = item.lower()
                if key not in seen:
                    seen.add(key)
                    objects.append(key)
        return objects

    # Preserved upstream logic; see the corresponding protocol documentation.
    # Preserved upstream logic; see the corresponding protocol documentation.
    if "\n" not in ma and ", " in ma and len(ma) < 200:
        return [TRAILING_PUNCT_RE.sub("", s).strip().lower() for s in ma.split(", ") if s.strip()]

    # Preserved upstream logic; see the corresponding protocol documentation.
    objects = []
    # Preserved upstream logic; see the corresponding protocol documentation.
    if "\n\n" in ma:
        # Preserved upstream logic; see the corresponding protocol documentation.
        body = ma.split("\n\n", 1)[1].strip()
    else:
        # Preserved upstream logic; see the corresponding protocol documentation.
        body = ma
        for pat in NOISE_PATTERNS:
            body = re.sub(pat, "", body, flags=re.IGNORECASE | re.MULTILINE)
        body = body.strip()

    # Preserved upstream logic; see the corresponding protocol documentation.
    body = re.sub(r"^(?:based on|here is|in the \w+ image|the items?(?: are| is)?|the (?:sub-?)?categor|list of)[^\n]*\n", "", body, flags=re.IGNORECASE | re.MULTILINE)
    body = body.strip()

    # Preserved upstream logic; see the corresponding protocol documentation.
    for line in body.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Preserved upstream logic; see the corresponding protocol documentation.
        line = BULLET_RE.sub("", line).strip()
        line = NUM_PREFIX_RE.sub("", line).strip()
        if not line:
            continue
        # Preserved upstream logic; see the corresponding protocol documentation.
        for item in line.split(","):
            item = item.strip()
            item = TRAILING_PUNCT_RE.sub("", item).strip()
            if not item:
                continue
            # Preserved upstream logic; see the corresponding protocol documentation.
            if len(item.split()) > 8:  # Preserved upstream logic; see the corresponding protocol documentation.
                continue
            if re.match(r"^(in the|the items? are|here is|the sub-?categor|list of|based on|first image|second image|third image|fourth image|fifth image|sixth image|seventh image|eighth image|ninth image|tenth image|\w+ image)", item, re.IGNORECASE):
                continue
            objects.append(item.lower())
    return objects

def gt_objects_from_response(response):
    "\u5b98\u65b9: gt_obj = answer.split(', '), \u7136\u540e\u5c0f\u5199"
    if not response:
        return []
    return [o.strip().lower() for o in response.split(", ") if o.strip()]
