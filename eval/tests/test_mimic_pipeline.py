import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from zwz_eval.data import annotation_sets, rows, sample_uid
from zwz_eval.inference import infer_records
from zwz_eval.io import atomic_json, read_json, canonical, digest
from zwz_eval.mimic import prepare, formatter, score, math_grader, COUNTS
from zwz_eval.runner import run
from test_adapters import configuration
from test_spatial_pipeline import Client


class MIMICPipeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new('RGB', (4, 4), 'white').save(self.root / 'image.png')
        self.config = configuration(self.root)
        self.config['workers'] = 1
        self.config['datasets']['mimic']['mathruler_mode'] = 'disabled'

    def row(self, task, index, reference='1', rawid=None):
        row = {'id': index if rawid is None else rawid, 'query': f'{task} question {index} <image>',
               'response': reference, 'images': [str(self.root / 'image.png')]}
        row['sample_uid'] = formatter().make_sample_uid(row, task)
        return row

    def test_counting_identity_rules_and_micro_listing(self):
        dataset = {
            'common': [self.row('mimic-common', 0)],
            'counting': [self.row('mimic-counting', 0, '2', rawid=0), self.row('mimic-counting', 1, '3', rawid=0)],
            'listing': [self.row('mimic-listing', 0, 'cat, dog'), self.row('mimic-listing', 1, 'bird')],
            'oddone': [self.row('mimic-oddone', 0, 'A')],
        }
        self.assertNotEqual(*[r['sample_uid'] for r in dataset['counting']])
        for task, data in dataset.items():
            atomic_json(self.root / f'mimic_{task}.json', data)
        with contextlib.redirect_stdout(io.StringIO()):
            infer_records(self.config, 'mimic', self.root, Client(['001', '2', 'not right', 'cat', 'bird', 'Answer: B']))
        judge = Client(['Yes', 'No'])
        result = score(self.config, self.root, judge)
        self.assertEqual(result['source_counts']['string'], 2)
        self.assertEqual(len(judge.requests), 2)
        self.assertAlmostEqual(result['per_task']['mimic-listing'], .8)
        self.assertEqual(result['listing']['total_gt'], 3)
        self.assertEqual(result['listing']['total_pred'], 2)
        self.assertEqual(result['per_task']['mimic-counting'], 1)
        self.assertEqual(result['per_task']['mimic-oddone'], 0)
        self.assertTrue(all(r[1]['max_tokens'] == 8192 and r[1]['thinking'] is False for r in judge.requests))
        self.assertTrue(all('<image>' not in r[0][0]['content'] for r in judge.requests))
        score(self.config, self.root, Client([]))

    def test_mathruler_policy_is_not_environment_autodetection(self):
        with patch('zwz_eval.mimic.importlib.metadata.version', side_effect=AssertionError('should not inspect')):
            self.assertIsNone(math_grader(self.config))
        del self.config['datasets']['mimic']['mathruler_mode']
        with self.assertRaises(ValueError):
            math_grader(self.config)
        self.config['datasets']['mimic']['mathruler_mode'] = 'required'
        with self.assertRaises(ValueError):
            math_grader(self.config)

    def test_legacy_2500_counting_is_rejected_before_preparation(self):
        raw = self.root / 'legacy'
        atomic_json(raw / 'mimic_common.json',
                    [self.row('mimic-common', i) for i in range(1000)])
        atomic_json(raw / 'mimic_counting.json',
                    [self.row('mimic-counting', i) for i in range(2500)])
        target = self.root / 'prepared'
        with self.assertRaisesRegex(ValueError, 'counting: expected 5000'):
            prepare(raw, target)
        self.assertFalse(target.exists())

    def test_full_8000_prepared_dataset_and_completed_journal_resume(self):
        raw = self.root / 'raw'
        prepared = self.root / 'prepared'
        for task, count in COUNTS.items():
            data = [self.row(f'mimic-{task}', i, 'cat' if task == 'listing' else '1', rawid=0 if task == 'counting' else None)
                    for i in range(count)]
            # Inputs can contain old IDs; Counting reference ignores an old supplied UID.
            atomic_json(raw / f'mimic_{task}.json', data)
        prepare(raw, prepared)
        counting = rows(prepared / 'mimic_counting.json')
        self.assertEqual(len({r['id'] for r in counting}), 1)
        self.assertEqual(len({r['sample_uid'] for r in counting}), 5000)
        self.config['datasets']['mimic']['root'] = str(prepared)
        directory = self.root / 'model/reference/mimic'
        directory.mkdir(parents=True)
        with (directory / 'predictions.jsonl').open('w') as out:
            for task, path in annotation_sets('mimic', prepared).items():
                for index, row in enumerate(rows(path)):
                    out.write(canonical({'uid': f'{task}/{sample_uid(row,index)}', 'task': task,
                        'input_sha256': digest(row), 'status': 'ok', 'raw_response': row['response'],
                        'model_answer': row['response']})+'\n')
        class Stub:
            def __init__(self, *args): pass
            def health(self): return True
            def complete(self, *args, **kwargs): raise AssertionError('Completed records must not be called again')
        with patch('zwz_eval.runner.CompatibleAPI', Stub), patch('zwz_eval.inference.CompatibleAPI', Stub), patch('zwz_eval.mimic.CompatibleAPI', Stub):
            directory = run(self.config, 'mimic', 'all')
        result = read_json(directory / 'score.json')
        self.assertEqual(result['macro'], 100)
        self.assertEqual(result['diagnostics']['total'], 8000)
        self.assertEqual(len(result['tasks']), 4)
        self.assertEqual(result['diagnostics']['per_task_total']['mimic-counting'], 5000)
        # A 2500-row legacy resume cannot masquerade as completed 5000-row inference.
        journal = directory / 'predictions.jsonl'
        records = journal.read_text().splitlines()
        kept, counting = [], 0
        import json
        for line in records:
            if json.loads(line)['task'] == 'mimic-counting':
                counting += 1
                if counting > 2500:
                    continue
            kept.append(line)
        journal.write_text('\n'.join(kept) + '\n')
        with self.assertRaisesRegex(ValueError, 'incomplete/foreign'):
            score(self.config, directory, Stub())


if __name__ == '__main__':
    unittest.main()
