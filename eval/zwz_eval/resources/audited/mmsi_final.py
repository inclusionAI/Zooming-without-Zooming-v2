# Selected definitions from the project-adapted reference source.
# Not a relicense of the original. See MANIFEST.json and THIRD_PARTY_NOTICES.md.
from __future__ import annotations
import re
import json
from decimal import Decimal, ROUND_HALF_UP
def extract_choice(pred):
    text = str(pred or '')
    tags = re.findall(r'<answer>\s*\**([A-D])\**\s*</answer>', text, re.I)
    if tags:
        return tags[-1].upper(), 'answer_tag'
    quoted = re.findall(r'`{1,2}\s*([A-D])\s*`{1,2}', text, re.I)
    if quoted:
        return quoted[-1].upper(), 'explicit_quoted_option'
    lines = text.strip().splitlines()
    if lines:
        match = re.fullmatch(r'\s*\**([A-D])\**[.)]?\s*', lines[-1], re.I)
        if match:
            return match.group(1).upper(), 'final_line'
    matches = re.findall(r'(?:final answer|correct answer|correct option|answer is|answer:)\s*(?:is\s*)?\**([A-D])\b', text, re.I)
    if matches:
        return matches[-1].upper(), 'explicit_final_statement'
    # The official fallback remains appropriate for direct short answers.
    # Do not guess from option lists inside a long unfinished rationale.
    return original(text) if len(text) <= 160 else (None, 'missing')
