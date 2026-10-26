# Selected definitions from the project-adapted reference source.
# Not a relicense of the original. See MANIFEST.json and THIRD_PARTY_NOTICES.md.
from __future__ import annotations
import re
import json
from decimal import Decimal, ROUND_HALF_UP
def parse_answer_tag(text: str) -> str:
    '\u63d0\u53d6 <answer>...</answer> \u6216 Answer: \u540e\u5185\u5bb9'
    if not text: return ""
    if '<answer>' in text and '</answer>' in text:
        start = text.find('<answer>') + len('<answer>')
        end = text.find('</answer>')
        return text[start:end].strip()
    elif 'Answer:' in text:
        return text[text.find('Answer:') + len('Answer:'):].strip()
    return text

def parse_first_letter(text: str) -> str:
    '\u63d0\u53d6\u9996\u4e2a\u5927\u5199\u5b57\u6bcd\u9009\u9879 (A-Z)'
    if not text: return ""
    match = re.search(r'\(([A-Z])\)', text)
    if match: return match.group(1)
    match = re.search(r'([A-Z])[.、)\s]', text)
    if match: return match.group(1)
    for char in text:
        if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            return char
    return ""
