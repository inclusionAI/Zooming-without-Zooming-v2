"""Preparation plumbing: image work belongs to requests unless explicitly selected."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from zwz_eval.cli import main
from zwz_eval.data import audit
from zwz_eval.io import atomic_json, file_hash, read_json
from zwz_eval.protocols import PLATFORM_COUNTS
from zwz_eval.workflow import convert_data, prepare_suite


class OnDemandWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source.json'
        self.image = self.root / 'original.png'
        Image.new('RGB', (8, 8), 'red').save(self.image)
        atomic_json(self.source, [{'query': '<image>Keep question.', 'response': 'A',
                                  'images': ['original.png']}])
        self.counts = patch.dict(PLATFORM_COUNTS, {'mmstar': 1})
        self.counts.start()
        self.addCleanup(self.counts.stop)
        self.verify = patch('zwz_eval.normalization.verify_source')
        self.verify.start()
        self.addCleanup(self.verify.stop)

    def convert(self, target, **kwargs):
        return convert_data('mmstar', self.source, self.root / target,
                            format='normalized', image_root=self.root, **kwargs)

    def test_default_prepare_and_reuse_do_not_open_preprocess_or_hash_images(self):
        self.image.write_bytes(b'not decodable: preparation must not decode it')
        def annotation_hash(path):
            if Path(path) == self.image:
                raise AssertionError('image bytes hashed during ordinary preparation')
            return file_hash(path)
        with patch('zwz_eval.general_visual.preprocess', side_effect=AssertionError('decoded')), \
             patch('zwz_eval.general_visual.file_hash', side_effect=annotation_hash), \
             patch('zwz_eval.data.file_hash', side_effect=annotation_hash), \
             patch('zwz_eval.workflow.audit', wraps=audit) as verify:
            receipt = self.convert('prepared')
            resumed = self.convert('prepared')
        self.assertTrue(resumed['reused'])
        self.assertTrue(all(call.kwargs['verify_images'] is False for call in verify.call_args_list))
        self.assertEqual(receipt['audit']['images_checked'], 0)
        self.assertEqual(receipt['audit']['image_sha256'], {})
        manifest = read_json(Path(receipt['dataset']['root']) / 'preparation.json')
        self.assertEqual(manifest['image_mode'], 'on-demand')
        self.assertFalse(list((self.root / 'prepared').rglob('*.png')))
        self.assertNotIn('.zwz-convert-', str(receipt))

    def test_explicit_audit_reads_images_without_changing_prepared_files(self):
        receipt = self.convert('prepared')
        records = Path(receipt['dataset']['root']) / 'records.json'
        before = records.read_bytes()
        result = audit('mmstar', receipt['dataset'])
        self.assertEqual(result['images_checked'], 1)
        self.assertTrue(result['image_sha256'])
        self.assertEqual(records.read_bytes(), before)

    def test_explicit_offline_materialization_and_old_prepared_registration(self):
        with patch('zwz_eval.general_visual.preprocess',
                   side_effect=lambda paths: [Image.open(p).convert('RGB') for p in paths]) as pixels:
            receipt = self.convert('offline', materialize_images=True)
        self.assertEqual(pixels.call_count, 1)
        self.assertTrue(receipt['input']['materialize_images'])
        prepared = Path(receipt['dataset']['root'])
        manifest = read_json(prepared / 'preparation.json')
        self.assertEqual(manifest.pop('image_mode'), 'materialized')
        atomic_json(prepared / 'preparation.json', manifest)  # legacy fixture: no mode field
        before = {str(p.relative_to(prepared)): p.read_bytes() for p in prepared.rglob('*') if p.is_file()}
        with patch('zwz_eval.general_visual.preprocess', side_effect=AssertionError('redecoded')):
            registered = convert_data('mmstar', prepared, self.root / 'registered', format='prepared')
            self.assertTrue(convert_data('mmstar', prepared, self.root / 'registered', format='prepared')['reused'])
        self.assertEqual(registered['audit']['images_checked'], 1)
        self.assertEqual(before, {str(p.relative_to(prepared)): p.read_bytes() for p in prepared.rglob('*') if p.is_file()})
        with self.assertRaisesRegex(ValueError, 'never reprocess prepared'):
            convert_data('mmstar', prepared, self.root / 'different', format='prepared', materialize_images=True)

    def test_on_demand_reuse_still_rejects_changed_annotations(self):
        receipt = self.convert('prepared')
        path = Path(receipt['dataset']['root']) / 'records.json'
        rows = read_json(path)
        rows[0]['query'] = 'changed'
        atomic_json(path, rows)
        with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
            self.convert('prepared')

    def test_suite_manifest_explicit_per_dataset_choice_is_forwarded(self):
        manifest = self.root / 'datasets.json'
        atomic_json(manifest, {'output_root': './prepared', 'datasets': {
            'mmstar': {'source': './source.json', 'format': 'normalized', 'materialize_images': True}}})
        with patch('zwz_eval.workflow.convert_data', return_value={'dataset': {'root': 'fixture'}}) as convert:
            prepare_suite(manifest)
        self.assertIs(convert.call_args.kwargs['materialize_images'], True)
        with patch('zwz_eval.workflow.convert_data', return_value={'dataset': {'root': 'fixture'}}) as convert:
            manifest_data = read_json(manifest)
            del manifest_data['datasets']['mmstar']['materialize_images']
            atomic_json(manifest, manifest_data)
            prepare_suite(manifest)
            self.assertIs(convert.call_args.kwargs['materialize_images'], False)
            prepare_suite(manifest, materialize_images=True)
            self.assertIs(convert.call_args.kwargs['materialize_images'], True)

    def test_cli_defaults_and_explicit_offline_flag_are_forwarded(self):
        cases = [
            ('prepare', ['--bench', 'mmstar', '--source', 'input', '--target', 'output'],
             'zwz_eval.general_visual.prepare'),
            ('convert', ['--bench', 'mmstar', '--source', 'input', '--target', 'output'],
             'zwz_eval.workflow.convert_data'),
            ('prepare-suite', ['--manifest', 'input'], 'zwz_eval.workflow.prepare_suite')]
        for command, args, target in cases:
            for flag in ([], ['--materialize-images']):
                with self.subTest(command=command, flag=flag), patch(target, return_value={}) as call, \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main([command, *args, *flag]), 0)
                    self.assertIs(call.call_args.kwargs['materialize_images'], bool(flag))

    def test_materialization_switch_does_not_silently_apply_to_other_pipelines(self):
        with self.assertRaisesRegex(ValueError, 'only to new raw/normalized platform'):
            convert_data('milebench', self.source, self.root / 'target', materialize_images=True)
        with self.assertRaisesRegex(ValueError, 'must be a boolean'):
            self.convert('target', materialize_images='false')
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['prepare', '--bench', 'milebench', '--source', str(self.source),
                                   '--target', str(self.root / 'target'), '--materialize-images']), 2)
        self.assertFalse((self.root / 'target').exists())


if __name__ == '__main__':
    unittest.main()
