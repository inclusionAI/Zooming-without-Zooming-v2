#!/usr/bin/env python3
"""Maintainer-only provenance update for reviewed deployment/auth/error-handling edits."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'zwz_eval/resources/vendor'
CHANGES = {
    'mmlongbench/data.py': 'Bind the preserved VH fallback cycle [0,1,1,0] to selected-row position, eliminating concurrency/resume-dependent fallback scores. Prompts, ID selection and valid Yes/No parsing are unchanged; this explicit protocol fix requires a new run ID.',
    'mmlongbench/eval_api_batch.py': 'Propagate native task errors in explicitly selected strict wrapper mode; preserve default upstream prompts, sampling, parsing and metrics.',
    'memlens/judge.py': 'Read judge API key from environment by default; no rubric changes.',
    'milebench/score.py': 'Read judge API key from environment; optional strict transport errors; no metric/prompt changes.',
    'mmlongbench/vlm_model/vllm_models.py': 'Read main API key from environment; propagate transport/malformed-response errors without caching synthetic empty answers; preserve valid empty output with token accounting and finish metadata. No generation/prompt/parser/metric changes.',
}


def main():
    path = VENDOR / 'SOURCES.json'
    manifest = json.loads(path.read_text())
    for record in manifest['files']:
        if record['path'] in CHANGES:
            patch = CHANGES[record['path']]
            if patch not in record['patches']:
                record['patches'].append(patch)
            record['packaged_sha256'] = hashlib.sha256((VENDOR / record['path']).read_bytes()).hexdigest()
    factory = 'gui/official/screenspot-pro/model_factory.py'
    if not any(r['path'] == factory for r in manifest['files']):
        manifest['files'].append({'path': factory, 'origin_pack_path': 'new-wrapper-supported-native-branch',
            'original_sha256': None, 'packaged_sha256': hashlib.sha256((VENDOR / factory).read_bytes()).hexdigest(),
            'patches': ['New minimal factory; preserves only the qwen3_5_vllm native backend.']})
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
