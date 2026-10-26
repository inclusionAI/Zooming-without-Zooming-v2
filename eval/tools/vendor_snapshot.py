#!/usr/bin/env python3
"""Build the reviewed scorer snapshot from an existing protocol pack (maintainer only).

This is a bulk mechanical source export. It never downloads datasets or weights.
Path/default-only patches are recorded; metric/rubric functions are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'zwz_eval/resources/vendor'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--memlens-converter', type=Path, required=True)
    parser.add_argument('--redact-endpoint', action='append', default=[],
                        help='Exact private endpoint to replace with a local placeholder; maintainer only')
    parser.add_argument('--redact-path-prefix', action='append', default=[],
                        help='Private deployment path prefix to remove; maintainer only')
    args = parser.parse_args()
    src = args.source.resolve()
    pairs = []
    mml = src / 'benchmarks/official/MMLongBench'
    for name in ['arguments.py', 'data.py', 'utils.py', 'utils_prompt.py', 'eval_api_batch.py']:
        pairs.append((mml / name, Path('mmlongbench') / name))
    for name in ['__init__.py', 'model_utils.py', 'vllm_models.py', 'openai_model.py']:
        pairs.append((mml / 'vlm_model' / name, Path('mmlongbench/vlm_model') / name))
    pairs += [(src / 'benchmarks/submit_bench/judge_memlens_official.py', Path('memlens/judge.py')),
              (src / 'benchmarks/submit_bench/cal_milebench_with_llm.py', Path('milebench/score.py')),
              (src / 'benchmarks/mimic_eval/infer.py', Path('common/infer.py')),
              (args.memlens_converter, Path('memlens/prepare.py'))]
    agent = src / 'agentvista'
    for path in agent.rglob('*'):
        if path.is_file() and path.suffix in {'.py', '.yaml'} and 'scripts' not in path.relative_to(agent).parts:
            pairs.append((path, Path('agentvista') / path.relative_to(agent)))
    gui = src / 'scoring_sources/gui'
    for rel in ['scripts/eval_grounding_topology.py', 'official/screenspot-pro/models/qwen3_5.py']:
        pairs.append((gui / rel, Path('gui') / rel))
    records = []
    for original, relative in pairs:
        raw = original.read_bytes()
        text = raw.decode('utf-8')
        patches = []
        if str(relative) == 'common/infer.py':
            text = re.sub(r'DEFAULT_IMAGE_ROOTS = \[.*?\]', 'DEFAULT_IMAGE_ROOTS = []', text, count=1, flags=re.S)
            patches.append('Remove private default image search root; callers pass explicit image roots.')
        if str(relative) == 'milebench/score.py':
            text = re.sub(r'"/[^"\n]+/eval_logs/milebench"', '"outputs/milebench"', text)
            patches.append('Replace legacy output default; no metric changes.')
        if str(relative) == 'mmlongbench/vlm_model/__init__.py':
            # All configured services are explicitly OpenAI-compatible; model IDs
            # containing gpt/claude must not override --use_vllm endpoint routing.
            marker = '    if "gpt" in args.model_name_or_path or "claude" in args.model_name_or_path:'
            text = text.replace(marker, '    if args.use_vllm:\n        from .vllm_models import VLLMModel\n        model_cls = VLLMModel\n    elif "gpt" in args.model_name_or_path or "claude" in args.model_name_or_path:', 1)
            patches.append('Prioritize explicitly selected compatible API backend over model-name heuristics.')
        if str(relative) == 'gui/scripts/eval_grounding_topology.py':
            text = text.replace('official" / "screenspot-pro"', 'official" / "screenspot-pro"')
        # Documentation/examples can carry private defaults even when never used.
        # Prefixes are supplied by the maintainer; no private mount namespace is
        # embedded in the portable public tool. Omitted redaction fails release scan.
        before = text
        for prefix in args.redact_path_prefix:
            text = re.sub(re.escape(prefix.rstrip('/') + '/') + r'[^\s\"\'`<>)]*',
                          '<LEGACY_DEPLOYMENT_PATH>', text)
        for endpoint in args.redact_endpoint:
            text = text.replace(endpoint, 'http://127.0.0.1:8000/v1')
        if before != text:
            patches.append('Redact legacy deployment path/default endpoint strings; execution uses explicit configuration.')
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
        records.append({'path': relative.as_posix(), 'origin_pack_path': str(original.relative_to(src)) if original.is_relative_to(src) else 'project-role-preserving-memlens-converter',
                        'original_sha256': hashlib.sha256(raw).hexdigest(),
                        'packaged_sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'patches': patches})
    # Only this model branch is included and supported, not dozens of unused frontends.
    factory = DEST / 'gui/official/screenspot-pro/model_factory.py'
    factory.write_text('''def build_model(args):
    if args.model_type != "qwen3_5_vllm":
        raise ValueError("The preserved native GUI branch is qwen3_5_vllm only")
    from models.qwen3_5 import Qwen3_5_VLLM_Model
    model = Qwen3_5_VLLM_Model()
    model.load_model(model_name_or_path=args.model_name_or_path)
    return model
''')
    manifest = {'note': 'Project-adapted snapshots; see THIRD_PARTY_NOTICES.md. Rebuild only during a versioned release.', 'files': records}
    (DEST / 'SOURCES.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'Exported {len(records)} source files to {DEST}')


if __name__ == '__main__':
    main()
