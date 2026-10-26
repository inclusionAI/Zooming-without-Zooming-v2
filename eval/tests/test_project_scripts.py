"""Project-directory shortcuts preserve the explicit configuration entrypoints."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class ProjectScripts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / 'project with spaces'
        self.project.mkdir()
        for name in ('model.json', 'datasets.json', 'ready.json'):
            (self.project / name).write_text('{}')
        self.python = self.root / 'python-stub'
        self.python.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                               'print(json.dumps(sys.argv[1:]))\n'
                               'sys.exit(int(os.environ.get("STUB_EXIT", "0")))\n')
        self.python.chmod(0o700)
        self.scripts = Path(__file__).resolve().parents[1] / 'scripts'

    def run_script(self, script, *args, code=0):
        env = {**os.environ, 'PYTHON': str(self.python), 'ZWZ_PLATFORM': 'generic', 'STUB_EXIT': str(code)}
        result = subprocess.run(['bash', str(self.scripts / script), *map(str, args)],
                                cwd=self.root, env=env, text=True, capture_output=True)
        return result, json.loads(result.stdout) if result.stdout else []

    def test_prepare_directory_defaults(self):
        result, args = self.run_script('prepare_benchmarks.sh', self.project)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(args, ['-m', 'zwz_eval', 'prepare-suite', '--manifest', str(self.project / 'datasets.json'),
                               '--bench', 'all', '--config', str(self.project / 'model.json'),
                               '--output-config', str(self.project / 'ready.json')])

    def test_prepare_directory_subset_and_option(self):
        result, args = self.run_script('prepare_benchmarks.sh', self.project, 'hr4k,hr8k', '--materialize-images')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(args[args.index('--bench') + 1], 'hr4k,hr8k')
        self.assertEqual(args[-1], '--materialize-images')

    def test_prepare_manifest_remains_compatible(self):
        result, args = self.run_script('prepare_benchmarks.sh', self.project / 'datasets.json', 'hr4k',
                                       '--config', 'custom.json', '--output-config', 'result.json')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(args[-4:], ['--config', 'custom.json', '--output-config', 'result.json'])

    def test_prepare_directory_missing_config_fails_before_python(self):
        (self.project / 'model.json').unlink()
        result, args = self.run_script('prepare_benchmarks.sh', self.project)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(args, [])

    def test_directory_does_not_silently_override_custom_configs(self):
        result, args = self.run_script('prepare_benchmarks.sh', self.project, '--config', 'custom.json')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(args, [])

    def test_evaluate_directory_defaults(self):
        result, args = self.run_script('evaluate.sh', self.project)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(args, ['-m', 'zwz_eval', 'evaluate', '--config', str(self.project / 'ready.json'),
                               '--bench', 'all', '--phase', 'all'])

    def test_evaluate_subset_phase_external(self):
        result, args = self.run_script('evaluate.sh', self.project, 'hr4k,hr8k', 'infer', '--services', 'external')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(args[-6:], ['--bench', 'hr4k,hr8k', '--phase', 'infer', '--services', 'external'])

    def test_explicit_config_remains_compatible(self):
        result, args = self.run_script('evaluate.sh', self.project / 'ready.json', 'hr4k', 'run')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(args[-4:], ['--bench', 'hr4k', '--phase', 'all'])

    def test_external_directory_without_explicit_subset(self):
        result, args = self.run_script('evaluate.sh', self.project, '--services', 'external')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(args[-6:], ['--bench', 'all', '--phase', 'all', '--services', 'external'])

    def test_child_failure_is_not_reported_as_success(self):
        result, _ = self.run_script('evaluate.sh', self.project, code=7)
        self.assertEqual(result.returncode, 7)


if __name__ == '__main__':
    unittest.main()
