# Selected definitions from the project-adapted reference source.
# Not a relicense of the original. See MANIFEST.json and THIRD_PARTY_NOTICES.md.
from __future__ import annotations
import re
import json
from decimal import Decimal, ROUND_HALF_UP
def extract_answer(text: str | None) -> str | None:
    """Prefer a model's final explicit answer tag, else use official extraction.

    The official evaluator reads a standalone A--D on the final output line.
    Some trained checkpoints explicitly define ``<answer>A</answer>`` as their
    response contract; honouring that complete tag avoids interpreting reasoning
    text as an answer.  The fallback keeps untagged models on official logic.
    """
    raw = str(text or "")
    tagged = re.findall(r"<answer>\s*([A-D])\s*</answer>", raw, re.IGNORECASE)
    if tagged:
        return tagged[-1].upper()
    final_line = raw.rstrip().split("\n")[-1]
    match = re.search(r"\b([A-D])\b", final_line, re.IGNORECASE)
    return match.group(1).upper() if match else None
