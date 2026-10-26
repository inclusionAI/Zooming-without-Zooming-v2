"""Synthetic fixtures: exercise one-command plumbing, not benchmark accuracy."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zwz_eval.cli import main
from zwz_eval.config import load_config
from zwz_eval.data import audit
from zwz_eval.io import atomic_json, read_json, run_lock
from zwz_eval.protocols import MILE_TASKS, PROTOCOLS
from zwz_eval.workflow import convert_data, prepare_suite, run_suite, selection, one_file


class Workflow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'downloaded-milebench'
        for task in MILE_TASKS:
            atomic_json(self.source / task / (task + '.json'), {
                'meta_data': {'task_instruction': ['Keep original instruction.'], 'question_type': 'open-ended'},
                'data': [{'sample_id': task + '-0', 'task_instruction_id': 0, 'response': 'gold',
                          'task_instance': {'context': 'Original question.', 'images_path': []}}]})
        self.endpoint = {'model': 'fixture', 'base_urls': ['http://127.0.0.1:1/v1']}
        self.config_path = self.root / 'config.json'
        atomic_json(self.config_path, {'model_id': 'model', 'run_id': 'run', 'output_root': './outputs',
            'main': self.endpoint, 'judge': self.endpoint, 'datasets': {
                'milebench': {'root': './old-mile', 'image_root': './stale-images'},
                'memlens': {'root': './old-mem'}}, 'workers': 1})

    def test_ordered_selection_and_reject_typos_duplicates_empty(self):
        self.assertEqual(selection('memlens, milebench'), ['memlens', 'milebench'])
        self.assertEqual(selection('all', ['milebench', 'memlens']), ['milebench', 'memlens'])
        for text in ('', 'milebench,', 'unknown', 'milebench,milebench', 'all,milebench'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                selection(text)
        with self.assertRaises(ValueError):
            selection('mmsi', ['milebench'])

    def test_raw_milebench_conversion_and_hash_validated_resume(self):
        target = self.root / 'prepared/milebench'
        receipt = convert_data('milebench', self.source, target)
        self.assertFalse(receipt['reused'])
        self.assertEqual(len(receipt['audit']['sets']), 29)
        self.assertEqual(receipt['dataset']['root'], str(target / 'prepared'))
        record = read_json(target / 'prepared/milebench_ALFRED_test.json')[0]
        self.assertEqual(record['query'], 'Original question.')
        self.assertEqual(record['system_prompt'], 'Keep original instruction.')
        self.assertEqual(receipt['audit']['manifest_sha256'], audit('milebench', receipt['dataset'])['manifest_sha256'])
        self.assertTrue(convert_data('milebench', self.source, target)['reused'])
        row_file = self.source / 'ALFRED/ALFRED.json'
        data = read_json(row_file)
        data['data'][0]['response'] = 'changed'
        atomic_json(row_file, data)
        with self.assertRaisesRegex(ValueError, 'Source/options changed'):
            convert_data('milebench', self.source, target)
        self.assertEqual(read_json(target / 'prepared/milebench_ALFRED_test.json')[0]['response'], 'gold')

    def test_changed_prepared_outputs_refuse_resume(self):
        target = self.root / 'prepared/milebench'
        convert_data('milebench', self.source, target)
        path = target / 'prepared/milebench_ALFRED_test.json'
        data = read_json(path)
        data[0]['response'] = 'tampered'
        atomic_json(path, data)
        with self.assertRaisesRegex(ValueError, 'Prepared annotations/images changed'):
            convert_data('milebench', self.source, target)

    def test_failed_conversion_never_publishes_partial_bundle(self):
        (self.source / 'ALFRED/ALFRED.json').unlink()
        target = self.root / 'prepared/milebench'
        with self.assertRaises(FileNotFoundError):
            convert_data('milebench', self.source, target)
        self.assertFalse(target.exists())
        self.assertFalse(list(target.parent.glob('.zwz-convert-*')))

    def test_native_is_registered_not_copied(self):
        target = self.root / 'registered/milebench'
        raw = convert_data('milebench', self.source, self.root / 'first')
        bundle = Path(raw['dataset']['root'])
        receipt = convert_data('milebench', bundle, target, format='prepared')
        self.assertEqual(receipt['dataset']['root'], str(bundle))
        self.assertEqual({p.name for p in target.iterdir()}, {'dataset.json'})

    def test_nested_source_target_existing_unreceipted_and_concurrent_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'separate preparation target'):
            convert_data('milebench', self.source, self.source / 'generated')
        target = self.root / 'prepared/milebench'
        target.mkdir(parents=True)
        atomic_json(target / 'keep.json', {'keep': True})
        with self.assertRaisesRegex(ValueError, 'without a complete dataset receipt'):
            convert_data('milebench', self.source, target)
        self.assertTrue((target / 'keep.json').is_file())
        other = self.root / 'prepared/other'
        with run_lock(other.parent / '.prepare-other'):
            with self.assertRaisesRegex(RuntimeError, 'Another process'):
                convert_data('milebench', self.source, other)

    def test_manifest_paths_resolve_against_manifest_and_project_new_config(self):
        manifest = self.root / 'manifest.json'
        atomic_json(manifest, {'output_root': './prepared', 'datasets': {
            'milebench': {'source': './downloaded-milebench', 'format': 'raw'}}})
        destination = self.root / 'configs/model.json'
        result = prepare_suite(manifest, 'milebench', config_path=self.config_path, output_config=destination)
        config = load_config(destination)
        self.assertEqual(list(config['datasets']), ['milebench'])
        self.assertEqual(config['datasets']['milebench']['root'], str(self.root / 'prepared/milebench/prepared'))
        self.assertEqual(config['datasets']['milebench']['image_root'], str(self.source))
        self.assertEqual(config['output_root'], str(self.root / 'outputs'))
        self.assertEqual(result['config'], str(destination))
        prepare_suite(manifest, 'milebench', config_path=self.config_path, output_config=destination)
        config['model_id'] = 'other'
        atomic_json(destination, config)
        with self.assertRaisesRegex(ValueError, 'Output config exists'):
            prepare_suite(manifest, 'milebench', config_path=self.config_path, output_config=destination)

    def test_manifest_does_not_overwrite_template_or_guess_extra_options(self):
        manifest = self.root / 'manifest.json'
        atomic_json(manifest, {'output_root': './prepared', 'datasets': {
            'milebench': {'source': str(self.source), 'unexpected_prompt': 'do not add'}}})
        with self.assertRaisesRegex(ValueError, 'Invalid preparation entry'):
            prepare_suite(manifest)
        with self.assertRaisesRegex(ValueError, 'new local config'):
            prepare_suite(manifest, config_path=self.config_path, output_config=self.config_path)
        with self.assertRaisesRegex(ValueError, 'both --config'):
            prepare_suite(manifest, config_path=self.config_path)

    def test_file_discovery_refuses_ambiguity(self):
        for name in ('a.json', 'b.json'):
            atomic_json(self.root / name, [])
        with self.assertRaisesRegex(ValueError, 'Expected one annotation'):
            one_file(self.root, ('a.json', 'b.json'))
        self.assertEqual(one_file(self.root / 'a.json'), self.root / 'a.json')

    def test_invalid_generated_config_is_not_published_and_template_options_not_inherited(self):
        manifest = self.root / 'manifest.json'
        atomic_json(manifest, {'output_root': './prepared', 'datasets': {
            'milebench': {'source': str(self.source), 'options': {'judge_profile': 'missing'}}}})
        destination = self.root / 'invalid-config.json'
        with self.assertRaisesRegex(ValueError, 'judge_profile'):
            prepare_suite(manifest, config_path=self.config_path, output_config=destination)
        self.assertFalse(destination.exists())
        # Receipt/data remain useful; no wrong model config masquerades as valid.
        self.assertTrue((self.root / 'prepared/milebench/dataset.json').is_file())

    def test_unsupported_image_map_and_nonhistorical_normalized_input_rejected(self):
        mapping = self.root / 'mapping.json'
        atomic_json(mapping, {})
        with self.assertRaisesRegex(ValueError, 'supported only for normalized'):
            convert_data('milebench', self.source, self.root / 'new', image_map=mapping)
        source = self.root / 'unrelated.json'
        atomic_json(source, [{'query': 'invented', 'response': 'A', 'images': ['x.png']}])
        with self.assertRaisesRegex(ValueError, 'Historical input mismatch'):
            convert_data('mmstar', source, self.root / 'normalized-mmstar', format='normalized')
        self.assertFalse((self.root / 'normalized-mmstar').exists())

    def test_published_receipt_audit_paths_are_final_never_temporary(self):
        target = self.root / 'prepared/milebench'
        receipt = convert_data('milebench', self.source, target)
        self.assertNotIn('.zwz-convert-', json.dumps(receipt))
        self.assertEqual(receipt['audit'], audit('milebench', receipt['dataset']))

    def test_normalized_absolute_prefix_is_passed_to_historical_verification(self):
        source = self.root / 'source.json'
        atomic_json(source, [])
        with patch('zwz_eval.normalization.verify_source', side_effect=RuntimeError('gate reached')) as verify:
            with self.assertRaisesRegex(RuntimeError, 'gate reached'):
                convert_data('mmstar', source, self.root / 'target', format='normalized',
                             image_prefix='/old/prefix')
        verify.assert_called_once_with('mmstar', source, image_prefix='/old/prefix')

    def test_run_suite_preserves_order_phase_and_failure_status(self):
        calls = []
        def run(config, bench, phase):
            calls.append((bench, phase))
            if bench == 'milebench':
                raise RuntimeError('synthetic transport failure')
        with patch('zwz_eval.runner.run', side_effect=run), contextlib.redirect_stdout(io.StringIO()):
            report = run_suite(self.config_path, 'milebench,memlens', 'all')
        self.assertEqual(calls, [('milebench', 'all')])
        self.assertFalse(report['complete'])
        self.assertEqual([r['status'] for r in report['results']], ['failed', 'not_started'])
        calls.clear()
        with patch('zwz_eval.runner.run', side_effect=run), contextlib.redirect_stdout(io.StringIO()):
            report = run_suite(self.config_path, 'milebench,memlens', 'infer', True)
        self.assertEqual(calls, [('milebench', 'infer'), ('memlens', 'infer')])
        self.assertFalse(report['complete'])
        self.assertTrue((self.root / 'outputs/model/run/suite_infer.json').is_file())

    def test_plan_no_data_or_network_writes_and_cli_failure_nonzero(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report = run_suite(self.config_path, 'milebench,memlens', 'plan')
        self.assertTrue(report['complete'])
        self.assertFalse((self.root / 'outputs').exists())
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['run-suite', '--config', str(self.config_path), '--bench', 'typo']), 2)
        with patch('zwz_eval.runner.run', side_effect=RuntimeError('fixture')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['run-suite', '--config', str(self.config_path), '--bench', 'milebench']), 2)

    def test_public_bash_entrypoints_run_outside_repository(self):
        repository = Path(__file__).resolve().parents[1]
        env = {**os.environ, 'PYTHON': sys.executable}
        result = subprocess.run(['bash', str(repository / 'scripts/run_benchmarks.sh'), str(self.config_path),
                                 'milebench,memlens', 'plan'], cwd=self.root, env=env,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(['bash', str(repository / 'scripts/prepare_benchmarks.sh')],
                                cwd=self.root, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Usage:', result.stderr)

    def test_all_25_benchmarks_have_explicit_preparation_entries(self):
        manifest = read_json(Path(__file__).resolve().parents[1] / 'examples/datasets.json')
        self.assertEqual(set(manifest['datasets']), set(PROTOCOLS))


if __name__ == '__main__':
    unittest.main()
