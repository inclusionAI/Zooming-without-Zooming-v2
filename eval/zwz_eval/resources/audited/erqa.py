# Selected definitions from the project-adapted reference source.
# Not a relicense of the original. See MANIFEST.json and THIRD_PARTY_NOTICES.md.
from __future__ import annotations
import re
import json
from decimal import Decimal, ROUND_HALF_UP
def extract_choice(answer):
    text = (answer or "").strip()
    tagged = re.findall(r"<answer>\s*([ABCD])\s*</answer>", text, re.I)
    if tagged:
        return tagged[-1].upper()
    if "</think>" in text:
        tail = text.rsplit("</think>", 1)[-1].strip()
        match = re.search(r"\b([ABCD])\b", tail, re.I)
        if match:
            return match.group(1).upper()
    final_matches = re.findall('(?:therefore|answer|option|\u7b54\u6848)[^ABCD]{0,40}\\b([ABCD])\\b', text[-600:], re.I)
    if final_matches:
        return final_matches[-1].upper()
    head = text[:160]
    match = re.search(r"(?:^|[<\s])([ABCD])(?:[.)：:]|\b)", head, re.I)
    return match.group(1).upper() if match else ""
