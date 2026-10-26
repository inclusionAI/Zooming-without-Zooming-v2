import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('annotation_export', Path(__file__).resolve().parents[1] / 'tools/export_normalized_annotations.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExportTests(unittest.TestCase):
    def test_preserves_fields_order_and_repeated_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            rows = [{'query': '<image> then <image>', 'images': ['/old/root/z.png', '/old/root/a.png'],
                     'response': 'A', 'LLM_JUDGE_PROMPT': 'original'},
                    {'query': 'second', 'images': ['/old/root/z.png'], 'response': 'B'}]
            (p / 'source.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            manifest = module.export(p / 'source.jsonl', p / 'new.jsonl')
            result = [json.loads(l) for l in (p / 'new.jsonl').read_text().splitlines()]
            self.assertEqual(result[0]['images'], ['z.png', 'a.png'])
            self.assertEqual(result[1]['images'], ['z.png'])
            self.assertEqual(module.semantic(rows), module.semantic(result))
            self.assertEqual(manifest['image_references'], 3)
            self.assertFalse(manifest['image_bytes_verified'])
            with self.assertRaises(ValueError):
                module.export(p / 'source.jsonl', p / 'new.jsonl')

    def test_explicit_prefix_keeps_task_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / 'source.json').write_text(json.dumps([{'query': 'q', 'images': ['/old/root/counting/a.jpg'], 'response': '1'}]))
            module.export(p / 'source.json', p / 'out.json', '/old/root')
            self.assertEqual(json.loads((p / 'out.json').read_text())[0]['images'], ['counting/a.jpg'])
