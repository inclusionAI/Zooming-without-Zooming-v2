import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from zwz_eval.io import atomic_json, read_json
from zwz_eval.normalization import identities
from zwz_eval.raw_mimic import convert, normalize, normalize_row, source_files


class RawMIMIC(unittest.TestCase):
    """Synthetic local fixtures, not redistributed benchmark questions/images."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'snapshot'
        self.target = self.root / 'normalized'
        self.images = self.root / 'raw-images'
        self.images.mkdir()
        Image.new('RGB', (3, 4), 'red').save(self.images / 'a.png')
        Image.new('RGB', (3, 4), 'blue').save(self.images / 'b.png')
        self.counts = {'common': 1, 'counting': 2, 'listing': 1, 'oddone': 1}
        self.data, self.recipe, self.expected = {}, {'tasks': {}}, {}
        for task, count in self.counts.items():
            self.data[task] = []
            self.recipe['tasks'][task] = []
            for index in range(count):
                row = {'question': f'Synthetic {task} {index} <image>\n',
                       'images': ['a.png', 'b.png'], 'gt': 'A', 'task': 'two-images'}
                if task == 'counting':
                    row.update(idx=0, class_id='synthetic', gt=index + 1)
                else:
                    row['id'] = 'raw-' + task
                self.data[task].append(row)
                self.recipe['tasks'][task].append(['a.png', 'b.png'])
            self.expected[task] = identities([normalize_row(row, task, [task + '/a.png', task + '/b.png'])
                                               for row in self.data[task]])
        self.write_source()

    def write_source(self):
        for task, rows in self.data.items():
            atomic_json(self.source / task / 'test.json', rows)

    def gate(self, bench, data, task=None):
        self.assertEqual(bench, 'mimic')
        if identities(data) != self.expected[task]:
            raise ValueError('Historical input mismatch in synthetic contract')
        return {'benchmark': f'mimic-{task}', **identities(data),
                'matches_historical_annotations': True, 'image_bytes_verified': False}

    def run_conversion(self):
        with patch('zwz_eval.raw_mimic.COUNTS', self.counts), \
             patch('zwz_eval.raw_mimic.read_json', return_value=self.recipe), \
             patch('zwz_eval.raw_mimic.verify_rows', side_effect=self.gate):
            return convert(self.source, self.target, self.images)

    def test_raw_conversion_preserves_questions_images_and_duplicate_counting_ids(self):
        result = self.run_conversion()
        counting = read_json(self.target / 'mimic_counting.json')
        self.assertEqual(len(counting), 2)
        self.assertEqual([row['id'] for row in counting], ['counting_0', 'counting_0'])
        self.assertEqual([row['response'] for row in counting], ['1', '2'])
        self.assertEqual(counting[0]['query'], self.data['counting'][0]['question'])
        self.assertEqual(counting[0]['images'], ['counting/a.png', 'counting/b.png'])
        self.assertEqual(counting[0]['meta'], {'class_id': 'synthetic'})
        self.assertEqual((self.target / 'images/counting/a.png').read_bytes(), (self.images / 'a.png').read_bytes())
        self.assertTrue(result['matches_historical_annotations'])
        self.assertFalse(result['historical_image_bytes_verified'])
        self.assertEqual(result['image_paths_restored_from_recipe'], 0)
        self.assertNotIn(str(self.root), (self.target / 'normalization.json').read_text())

    def test_embedded_images_without_paths_use_only_content_free_identity_recipe(self):
        data = self.data
        for task, rows in data.items():
            for row in rows:
                row['images'] = [{'bytes': (self.images / name).read_bytes(), 'path': None} for name in ['a.png', 'b.png']]
        # Parquet provides bytes; JSON is metadata-only, so mock just its reader.
        with patch('zwz_eval.raw_mimic.raw_rows', side_effect=lambda paths: iter(data[paths[0].parent.name])):
            report = self.run_conversion()
        self.assertEqual(report['image_paths_restored_from_recipe'], 10)
        self.assertEqual((self.target / 'images/common/a.png').read_bytes(), (self.images / 'a.png').read_bytes())

    def test_mismatched_prompt_rejected_and_partial_outputs_removed(self):
        self.data['listing'][0]['question'] += ' unauthorized addition'
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'Historical input mismatch'):
            self.run_conversion()
        self.assertFalse(self.target.exists())
        self.assertFalse(list(self.root.glob('.normalized-mimic-*')))

    def test_wrong_image_order_is_rejected(self):
        self.data['counting'][0]['images'].reverse()
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'identity/order'):
            self.run_conversion()
        self.assertFalse(self.target.exists())

    def test_short_counting_rejected_never_fills_or_deduplicates(self):
        self.data['counting'].pop()
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'expected 2 rows, got 1'):
            self.run_conversion()
        self.assertFalse(self.target.exists())

    def test_missing_image_and_path_traversal_fail_closed(self):
        (self.images / 'a.png').unlink()
        with self.assertRaisesRegex(ValueError, 'no bytes and no readable local file'):
            self.run_conversion()
        self.assertFalse(self.target.exists())
        self.data['common'][0]['images'] = ['../a.png', 'b.png']
        self.write_source()
        with self.assertRaisesRegex(ValueError, 'Unsafe/remote'):
            self.run_conversion()

    def test_existing_target_untouched_and_wrong_benchmark_rejected(self):
        self.target.mkdir()
        marker = self.target / 'keep'
        marker.write_text('existing')
        with self.assertRaisesRegex(ValueError, 'must not already exist'):
            convert(self.source, self.target, self.images)
        self.assertEqual(marker.read_text(), 'existing')
        with self.assertRaisesRegex(ValueError, 'only converts'):
            normalize('milebench', self.source, self.root / 'other')

    def test_parquet_shards_must_be_complete_and_naturally_ordered(self):
        folder = self.root / 'parquet/common'
        folder.mkdir(parents=True)
        (folder / 'test-00001-of-00002.parquet').touch()
        with self.assertRaisesRegex(ValueError, 'missing or duplicate'):
            source_files(folder.parent, 'common')
        (folder / 'test-00000-of-00002.parquet').touch()
        self.assertEqual([p.name for p in source_files(folder.parent, 'common')],
                         ['test-00000-of-00002.parquet', 'test-00001-of-00002.parquet'])
        atomic_json(folder / 'test.json', [])
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            source_files(folder.parent, 'common')

    def test_original_hf_columns_required_and_counting_types_not_guessed(self):
        with self.assertRaisesRegex(ValueError, 'original HF columns'):
            normalize_row({'query': 'not a raw row'}, 'common', ['common/a.png'])
        raw = dict(self.data['counting'][0], idx='0')
        with self.assertRaisesRegex(ValueError, 'idx must be an integer'):
            normalize_row(raw, 'counting', ['counting/a.png'])
        raw = dict(self.data['counting'][0], gt=True)
        with self.assertRaisesRegex(ValueError, 'gt must be an integer'):
            normalize_row(raw, 'counting', ['counting/a.png'])

    def test_real_contract_rejects_tiny_fixture_even_if_shape_is_valid(self):
        # The production gate cannot be disabled by a JSON export or CLI flag.
        with patch('zwz_eval.raw_mimic.COUNTS', self.counts), \
             patch('zwz_eval.raw_mimic.read_json', return_value=self.recipe):
            with self.assertRaisesRegex(ValueError, 'Historical input mismatch'):
                convert(self.source, self.target, self.images)
        self.assertFalse(self.target.exists())


if __name__ == '__main__':
    unittest.main()
