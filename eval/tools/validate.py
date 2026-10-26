#!/usr/bin/env python3
"""Record the tests actually run; never imply real GPU/data acceptance."""
import ast
import contextlib
import importlib.metadata
import io
import platform
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zwz_eval.io import atomic_json
from zwz_eval.release import check


def main():
    report = {'python': platform.python_version(), 'real_gpu_tested': False,
              'full_real_datasets_tested': False, 'dependencies': {}}
    for name in ('openai', 'Pillow', 'PyYAML', 'tqdm', 'requests'):
        try:
            report['dependencies'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['dependencies'][name] = None
    files = sorted((ROOT / 'zwz_eval').rglob('*.py'))
    for path in files:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    report['python_sources_parsed'] = len(files)
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    log = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    release_report = check()
    successful = result.wasSuccessful() and not release_report['source_integrity_errors']
    report.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=[test.id() for test, _ in result.skipped],
                  unit_tests_successful=result.wasSuccessful(), successful=successful,
                  release_check=release_report)
    atomic_json(ROOT / 'validation_report.json', report)
    print(log.getvalue())
    print('Wrote validation_report.json (no real GPU or full dataset test claimed)')
    if release_report['source_integrity_errors']:
        print('Source integrity/export check failed; see validation_report.json')
    return 0 if successful else 1


if __name__ == '__main__':
    raise SystemExit(main())
