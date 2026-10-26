# Selected definitions from the project-adapted reference source.
# Not a relicense of the original. See MANIFEST.json and THIRD_PARTY_NOTICES.md.
from __future__ import annotations
import re
import json
from decimal import Decimal, ROUND_HALF_UP
def parse_json(text):
    match = re.search(r'\{\s*"verdict".*?\}', text or '', re.S)
    if not match: return None
    try: data = json.loads(match.group(0))
    except Exception: return None
    if data.get('verdict') not in ('correct', 'incorrect', 'ambiguous'): return None
    if data.get('selected') not in ('A', 'B', 'C', 'D', None, 'null'): return None
    return data
