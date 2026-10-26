# Selected definitions from the project-adapted reference source.
# Not a relicense of the original. See MANIFEST.json and THIRD_PARTY_NOTICES.md.
from __future__ import annotations
import re
import json
from decimal import Decimal, ROUND_HALF_UP
def valid_verdict(entries, golden):
    for e in entries:
        if e['score'] not in (0, 1):
            return False
        if e['score'] == 1 and re.search(
            r'\b(?:does not match|fails to provide|fails to match|outside (?:the|this) (?:acceptable )?range|marked as incorrect|no (?:clear,? )?final answer)\b',
            e['rationale'], re.I):
            return False
    # A gold interval with separate tolerances for both endpoints.
    number = r'[-+]?\d+(?:\.\d+)?'
    bounds = re.findall(r'acceptable range\s+('+number+r')\s+to\s+('+number+r')', golden, re.I)
    if len(bounds) == 2 and len(entries) == 1 and entries[0]['score'] == 1:
        values = re.findall(number, entries[0]['extracted_answer'])
        if len(values) == 2:
            return all(float(lo) <= float(value) <= float(hi)
                       for value, (lo, hi) in zip(values, bounds))
    return True
