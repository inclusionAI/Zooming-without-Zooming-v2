# Selected definitions from the project-adapted reference source.
# Not a relicense of the original. See MANIFEST.json and THIRD_PARTY_NOTICES.md.
from __future__ import annotations
import re
import json
from decimal import Decimal, ROUND_HALF_UP
def parse_json(s):
 m=re.search(r'\{\s*"verdict".*?\}',s,re.S)
 if not m: return None
 try:
  d=json.loads(m.group(0))
  if d.get('verdict') in ('correct','incorrect','ambiguous') and d.get('selected') in ('A','B','C','D',None,'null'): return d
 except: pass
 return None
