import base64
import io
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image

from zwz_eval.data import audit
from zwz_eval.io import atomic_json, digest, read_journal
from zwz_eval.spatial import infer, score, messages, image_url, prepare
from zwz_eval.references import load
from zwz_eval.runner import run
from zwz_eval.io import read_json
from test_adapters import configuration


class Client:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.requests = []

    def complete(self, request, **kwargs):
        self.requests.append((request, kwargs))
        value = next(self.answers)
        if isinstance(value, Exception):
            raise value
        message = value if isinstance(value, dict) else {'content': value}
        return {'choices': [{'message': message, 'finish_reason': 'stop'}], 'usage': {}}


class SpatialPipeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new('RGB', (30, 20), 'red').save(self.root / 'image.png')
        self.config = configuration(self.root)
        self.row = {'id': 'sample', 'images': ['image.png'], 'question': 'Pick one.',
                    'question_type': 'test', 'choices': '\nA. red\nB. blue', 'answer': 'A'}

    def dataset(self, records):
        atomic_json(self.root / 'records.json', records)

    def test_image_transform_and_exact_prompt(self):
        rendered = image_url(self.root / 'image.png', 'viewspatial')
        with Image.open(io.BytesIO(base64.b64decode(rendered.split(',')[1]))) as image:
            self.assertEqual(image.size, (420, 280))
            self.assertEqual(image.format, 'JPEG')
        request = messages(self.row, 'viewspatial', self.root)
        self.assertEqual([p['type'] for p in request[0]['content']], ['image_url', 'text'])
        self.assertEqual(request[0]['content'][-1]['text'], 'Pick one.\nA. red\nB. blue\nReply only to the corresponding option.\nAnswer:')
        request = messages(self.row, 'mmsi', self.root)
        self.assertTrue(request[0]['content'][-1]['text'].endswith("Enclose the option's letter within ``."))
        # MMSI preserves original bytes, unlike ViewSpatial's resize.
        self.assertEqual(base64.b64decode(request[0]['content'][0]['image_url']['url'].split(',')[1]),
                         (self.root / 'image.png').read_bytes())

    def test_view_review_failure_then_resume(self):
        records = [self.row, {**self.row, 'id': 'other', 'answer': 'B'}]
        self.dataset(records)
        self.config['workers'] = 1
        infer(self.config, 'viewspatial', self.root, Client(['<answer>A</answer>', 'blue']))
        with self.assertRaises(RuntimeError):
            score(self.config, 'viewspatial', self.root, Client([RuntimeError('offline')]))
        self.assertFalse((self.root / 'scores/metrics.json').exists())
        judge = Client(['CORRECT'])
        result = score(self.config, 'viewspatial', self.root, judge)
        self.assertEqual(result['initial_correct'], 1)
        self.assertEqual(result['reviewed_correct'], 2)
        self.assertEqual(judge.requests[0][1]['max_tokens'], 16)
        self.assertFalse(judge.requests[0][1]['thinking'])
        self.assertTrue(all(isinstance(p['content'], str) for p in judge.requests[0][0]))
        # Successful review is never called again.
        score(self.config, 'viewspatial', self.root, Client([]))
        self.assertEqual(len(read_journal(self.root / 'scores/reviews.jsonl')), 2)

    def test_empty_eos_is_retained_and_not_resampled(self):
        self.dataset([self.row])
        result = infer(self.config, 'mmsi', self.root, Client(['']))
        self.assertEqual(result['empty_model_outputs'], 1)
        infer(self.config, 'mmsi', self.root, Client([]))
        report = score(self.config, 'mmsi', self.root)
        self.assertEqual(report['primary_accuracy'], 0)
        self.dataset([{**self.row, 'question': 'changed'}])
        with self.assertRaises(ValueError):
            infer(self.config, 'mmsi', self.root, Client([]))

    def test_mindcube_setting_filter_and_channel(self):
        record = {'id': 'around_1', 'input_prompt': 'Original prompt.',
                  'images': ['image.png'], 'gt_answer': 'B'}
        self.dataset([record, {**record, 'id': 'translation_1'}])
        self.config['workers'] = 1
        main = Client([{'content': '', 'reasoning': 'B'}, 'A'])
        infer(self.config, 'mindcube', self.root, main)
        report = score(self.config, 'mindcube', self.root)
        self.assertEqual(report['primary_total'], 1)
        self.assertEqual(report['primary_accuracy'], 1)
        self.assertEqual(report['unfiltered_accuracy'], .5)
        self.assertEqual(report['reasoning_fallbacks'], 1)
        self.assertEqual(main.requests[0][1]['reasoning_effort'], 'low')
        self.assertEqual(main.requests[0][1]['max_tokens'], 1536)

    def test_audit_rejects_partial_and_invalid_gold(self):
        self.dataset([self.row])
        with self.assertRaises(ValueError):
            audit('mmsi', self.config['datasets']['mmsi'])
        records = [{**self.row, 'id': i} for i in range(1000)]
        records[0]['answer'] = ''
        self.dataset(records)
        with self.assertRaises(ValueError):
            audit('mmsi', self.config['datasets']['mmsi'])
        records[0]['answer'] = 'A'
        self.dataset(records)
        result = audit('mmsi', self.config['datasets']['mmsi'])
        self.assertEqual(result['sets']['all']['rows'], 1000)
        self.assertEqual(len(result['image_sha256']), 1)

    def test_no_score_for_incomplete_journal(self):
        self.dataset([self.row])
        with self.assertRaises(ValueError):
            score(self.config, 'mmsi', self.root)

    def test_mindcube_audit_checks_reference_setting_distribution(self):
        records = [{'id': f'{setting}_{i}', 'input_prompt': 'Original question',
                    'gt_answer': 'A', 'images': ['image.png']}
                   for setting,n in [('around',250),('rotation',200),('among',600)] for i in range(n)]
        self.dataset(records)
        result = audit('mindcube', self.config['datasets']['mindcube'])
        self.assertEqual(result['sets']['all']['setting_counts'], {'around':250,'rotation':200,'among':600})
        records[0]['id'] = 'translation_0'
        self.dataset(records)
        with self.assertRaisesRegex(ValueError, 'settings differ'):
            audit('mindcube', self.config['datasets']['mindcube'])

    def test_complete_mmsi_framework_and_resume(self):
        # Full fixed denominator, synthetic repeated image, no actual model/API.
        self.dataset([{**self.row, 'id': i} for i in range(1000)])
        calls = []
        class Stub:
            def __init__(self, *args):
                pass
            def health(self):
                return True
            def complete(self, request, **kwargs):
                calls.append(kwargs)
                return {'choices': [{'message': {'content': '<answer>A</answer>'},
                                     'finish_reason': 'stop'}]}
        with patch('zwz_eval.runner.CompatibleAPI', Stub), patch('zwz_eval.spatial.CompatibleAPI', Stub):
            directory = run(self.config, 'mmsi', 'all')
            result = read_json(directory / 'score.json')
            self.assertEqual(result['macro'], 100)
            self.assertEqual(result['diagnostics']['total'], 1000)
            self.assertEqual(len(calls), 1000)
            run(self.config, 'mmsi', 'all')
            self.assertEqual(len(calls), 1000)
        self.assertTrue(all(c['max_tokens'] == 4096 and c['thinking'] is False for c in calls))
        self.assertIn('predictions.jsonl', result['source_outputs_sha256'])


if __name__ == '__main__':
    unittest.main()
