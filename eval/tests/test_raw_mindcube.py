import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zwz_eval.raw_mindcube import PREFIX, NAME, convert, generate_row, locate, verify_historical


class RawMindCube(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.source = self.data / 'raw/MindCube_tinybench.jsonl'
        self.source.parent.mkdir(parents=True)
        self.row = {'id': 'around-synthetic', 'question': 'Synthetic? A. Left B. Right',
                    'images': ['views/second.png', 'views/first.png'], 'gt_answer': 'B',
                    'category': ['retained', 'metadata']}
        self.source.write_text(json.dumps(self.row) + '\n')
        for name in self.row['images']:
            path = self.data / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'synthetic-fixture')
        self.target = self.root / 'normalized'

    def test_official_raw_qa_prefix_and_original_fields_retained(self):
        row = generate_row(self.row)
        self.assertEqual(row['input_prompt'], PREFIX + self.row['question'])
        self.assertEqual(row['grounded_output'], '<answer>B. Right</answer>')
        self.assertEqual(row['images'], self.row['images'])
        self.assertEqual(row['category'], ['retained', 'metadata'])
        self.assertNotIn('input_prompt', self.row)
        self.assertEqual(generate_row(row), row)

    def test_convert_raw_layout_checks_images_and_writes_original_row_order(self):
        with patch('zwz_eval.raw_mindcube.verify_historical', return_value={'rows': 1}) as gate:
            report = convert(self.root, self.target)
        output = [json.loads(line) for line in (self.target / NAME).read_text().splitlines()]
        self.assertEqual(output[0]['input_prompt'], PREFIX + self.row['question'])
        self.assertEqual(output[0]['images'], ['views/second.png', 'views/first.png'])
        self.assertEqual(report['image_root'], str(self.data.resolve()))
        self.assertEqual(len(report['image_sha256']), 2)
        gate.assert_called_once()
        self.assertFalse((self.target / 'images').exists())

    def test_ready_prompt_location_infers_data_root(self):
        ready = self.data / 'prompts/general' / NAME
        ready.parent.mkdir(parents=True)
        ready.write_text(json.dumps(generate_row(self.row)) + '\n')
        self.assertEqual(locate(ready), (ready, self.data))

    def test_wrong_prompt_or_unsafe_images_never_silently_rewritten(self):
        with self.assertRaisesRegex(ValueError, 'not the historical'):
            generate_row(dict(self.row, input_prompt='Different instruction'))
        with self.assertRaisesRegex(ValueError, 'safe paths relative'):
            generate_row(dict(self.row, images=['../outside.png']))
        with self.assertRaisesRegex(ValueError, 'grounded_output differs'):
            generate_row(dict(self.row, grounded_output='<answer>A. Left</answer>'))

    def test_missing_images_fail_before_output_is_published(self):
        (self.data / self.row['images'][0]).unlink()
        with patch('zwz_eval.raw_mindcube.verify_historical', return_value={'rows': 1}):
            with self.assertRaisesRegex(ValueError, 'Missing/escaped'):
                convert(self.root, self.target)
        self.assertFalse(self.target.exists())

    def test_existing_output_retained_and_real_gate_requires_all_1050(self):
        with self.assertRaisesRegex(ValueError, '1050 unique'):
            convert(self.root, self.target)
        self.assertFalse(self.target.exists())
        self.target.mkdir()
        with self.assertRaisesRegex(ValueError, 'must not exist'):
            convert(self.root, self.target)


if __name__ == '__main__':
    unittest.main()
