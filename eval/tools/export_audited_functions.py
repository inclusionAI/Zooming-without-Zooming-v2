#!/usr/bin/env python3
"""Extract named definitions, not deployment code, from a private source audit.

This maintainer tool does not connect to servers or publish anything. It retains
the selected source text verbatim and records source + excerpt hashes. Attribution
and redistribution approval remain release gates, even for small excerpts.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = {
    'platform_messages': ('perception/core_inference.py', ['build_messages']),
    'erqa_wrong_review': ('erqa/judge_wrong.py', ['parse_json']),
    'erqa_initial_review': ('erqa/judge_initial.py', ['parse_json']),
    'spar_review': ('spar/llm_audit.py', ['JUDGE_SYSTEM', 'parse_json']),
    'mmsi_original': ('mmsi/score.py', ['extract_choice']),
    'mmsi_final': ('mmsi/score_rerun.py', ['extract_choice']),
    'viewspatial': ('viewspatial/score.py', ['extract_answer']),
    'erqa': ('erqa/adapter.py', ['extract_choice']),
    'mindcube': ('mindcube/extractors.py', ['extract_answer', 'get_setting_from_id']),
    'platform': ('perception/parsers.py', ['parse_answer_tag', 'parse_first_letter']),
    'mimic_listing': ('mimic/listing.py', [
        'NOISE_PATTERNS', 'BULLET_RE', 'NUM_PREFIX_RE', 'TRAILING_PUNCT_RE',
        'parse_model_answer_to_objects', 'gt_objects_from_response']),
    'mimic_judge': ('mimic/judge.py', [
        'PROMPT_TEMPLATE', 'SUBSTITUTIONS', 'REMOVED_EXPRESSIONS',
        'normalize_final_answer', 'string_match', 'parse_llm_judge_response', 'extract_answer']),
    'spar': ('spar/adapter.py', [
        'MCA', 'NA', 'MULTIVIEW_NA', 'ANSWER_FORMAT_SYSTEM', 'LOW', 'MIDDLE', 'HIGH',
        'official_prompt', 'exact_match', 'first_number', 'mra',
        'parse_instruction', 'view_change_score', 'extract_final_answer', 'official_score']),
    'chartography': ('chartography/judge.py', [
        'JUDGE_PROMPT_TEMPLATE', 'JUDGE_GUIDED_JSON', '_judge_prompt', '_parse_verdict', '_verdict_passes']),
    'chartography_guard': ('chartography/guard.py', ['valid_verdict']),
    'chartography_rule': ('chartography/infer.py', [
        '_NUM', '_RANGE_RE', '_to_float', '_parse_ranges', '_extract_numbers',
        '_norm', '_extract_final_answer', '_tokenize_set', '_normalize_answer', 'score_one']),
}
HEADER = '''# Selected definitions from the project-adapted reference source.
# Not a relicense of the original. See MANIFEST.json and THIRD_PARTY_NOTICES.md.
from __future__ import annotations
import re
import json
from decimal import Decimal, ROUND_HALF_UP
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-audit', type=Path, required=True)
    args = p.parse_args()
    dest = ROOT / 'zwz_eval/resources/audited'
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (relative, selected) in EXPORTS.items():
        source = (args.source_audit / relative).read_text()
        tree = ast.parse(source)
        found = {}
        for node in tree.body:
            names = [node.name] if isinstance(node, ast.FunctionDef) else (
                [n.id for n in node.targets if isinstance(n, ast.Name)]
                if isinstance(node, ast.Assign) else [])
            for key in names:
                if key in selected:
                    found[key] = ast.get_source_segment(source, node)
        if set(found) != set(selected):
            raise ValueError(f'Missing selected definitions in {relative}: {set(selected)-set(found)}')
        text = HEADER + '\n\n'.join(found[key] for key in selected) + '\n'
        ast.parse(text)
        path = dest / (name + '.py')
        path.write_text(text)
        records.append({'module': name, 'source': relative,
                        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                        'excerpt_sha256': hashlib.sha256(text.encode()).hexdigest(),
                        'definitions': selected, 'redistribution_review': 'pending'})
    (dest / 'MANIFEST.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
