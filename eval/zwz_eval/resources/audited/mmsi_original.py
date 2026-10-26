# Selected definitions from the project-adapted reference source.
# Not a relicense of the original. See MANIFEST.json and THIRD_PARTY_NOTICES.md.
from __future__ import annotations
import re
import json
from decimal import Decimal, ROUND_HALF_UP
def extract_choice(pred: str | None) -> tuple[str | None, str]:
    """Read a model-native final tag first; otherwise retain official extraction."""
    if pred is None:
        return None, "missing"
    text = str(pred)
    tagged = re.findall(r"<answer>\s*\**([A-D])\**\s*</answer>", text, flags=re.IGNORECASE)
    if tagged:
        return tagged[-1].upper(), "answer_tag"
    match = re.search(r"``([^`]*)``", text)
    if match:
        text = match.group(1)
        source = "double_backtick"
    else:
        match = re.search(r"`([^`]*)`", text)
        if match:
            text = match.group(1)
            source = "backtick"
        else:
            source = "official_first_regex"
    match = re.search(r"\b[A-D]\b(?!\s[a-zA-Z])", text)
    return (match.group().upper(), source) if match else (None, "missing")
