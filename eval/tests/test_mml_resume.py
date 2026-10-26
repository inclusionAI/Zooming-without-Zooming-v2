"""Execute the native VH loader/writer with synthetic data, without GPU imports."""
import ast
import contextlib
import io
import json
import logging
import os
import random
import statistics
import tempfile
import threading
import time
import types
import unittest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from zwz_eval.protocols import RESOURCES
from test_parsers import functions


class Rows:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.records]
        return self.records[key]

    def filter(self, predicate, **kwargs):
        return Rows([row for row in self.records if predicate(row)])

    def map(self, function, *, with_indices=False, **kwargs):
        return Rows([{**row, **(function(row, index) if with_indices else function(row))}
                     for index, row in enumerate(self.records)])


class Inputs:
    def __init__(self, data, model, processor):
        self.data = data

    def __getitem__(self, index):
        return {'index': index}, 'synthetic prompt'


class Model:
    api_model = True
    processor = None

    def __init__(self, fail_at=None):
        self.fail_at = fail_at
        self.second_finished = threading.Event()
        self.completion_order = []

    def generate(self, inputs):
        index = inputs['index']
        # Force a different completion order than selected dataset order.
        if index == 0 and not self.second_finished.wait(5):
            raise RuntimeError('Synthetic concurrency fixture did not start')
        if index == self.fail_at:
            raise RuntimeError('Synthetic interrupted API')
        self.completion_order.append(index)
        if index == 1:
            self.second_finished.set()
        return {'output': 'uncertain', 'input_len': 10, 'output_len': 1}


class MMLResume(unittest.TestCase):
    def setUp(self):
        source = RESOURCES / 'vendor/mmlongbench/data.py'
        tree = ast.parse(source.read_text())
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name in ('DefaultAnswerGenerator', 'load_visual_haystack')]
        metric = functions('mmlongbench/utils.py', ['remove_articles', 'white_space_fix', 'lower',
            'normalize_answer_with_punc', 'extract_binary_label', 'calculate_metrics'])
        self.records = [{'id': index, 'ctxs': [f'{index}.jpg'], 'answer': 'No'} for index in range(8)]
        namespace = {'os': os, 'random': random, 'calculate_metrics': metric['calculate_metrics'],
                     'load_dataset': lambda *args, **kwargs: {'train': Rows(self.records)}}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
        self.load = namespace['load_visual_haystack']
        self.args = types.SimpleNamespace(test_file_root='/synthetic', image_file_root='/images',
                                         preprocessing_num_workers=1)

    def test_fallback_is_indexed_after_selection_and_repeated_calls_do_not_advance(self):
        random.seed(42)
        data = self.load(self.args, 'vh.jsonl', max_test_samples=5)
        self.assertEqual(len(data['data']), 5)
        expected = [0, 1, 1, 0, 0]
        self.assertEqual([row['default_answer'] for row in data['data'].records], expected)
        for index in (4, 0, 3, 3, 1, 2, 0):
            metrics, details = data['post_process']({'output': 'Answer: uncertain'}, data['data'][index])
            self.assertEqual(details['default_answer'], expected[index])
            self.assertEqual(metrics['acc'], int(expected[index] == 0))

    def test_valid_yes_no_does_not_use_fallback(self):
        data = self.load(self.args, 'vh.jsonl')
        for row in data['data'].records:
            self.assertEqual(data['post_process']({'output': 'Answer: No'}, row)[0], {'acc': 1})
            self.assertEqual(data['post_process']({'output': 'Answer: Yes'}, row)[0], {'acc': 0})

    def native_runner(self):
        source = RESOURCES / 'vendor/mmlongbench/eval_api_batch.py'
        node = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'run_test')
        namespace = {'os': os, 'json': json, 'time': time, 'logger': logging.getLogger(__name__),
            'defaultdict': defaultdict, 'set_seed': random.seed, 'ThreadPoolExecutor': ThreadPoolExecutor,
            'tqdm': lambda values, *args, **kwargs: values,
            'np': types.SimpleNamespace(mean=lambda values: float(statistics.mean(values))),
            'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 0)),
            'TestItemDataset': Inputs,
            'DataLoader': lambda dataset, **kwargs: types.SimpleNamespace(dataset=dataset),
            'load_data': lambda args, *unused: self.load(args, 'vh.jsonl')}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        return namespace['run_test']

    def test_native_concurrent_full_run_equals_interrupted_resume(self):
        runner = self.native_runner()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            outputs = []
            for workers in (2, 4):
                for interrupted in (False, True):
                    folder = Path(tmp) / f'{workers}-{interrupted}'
                    folder.mkdir()
                    args = types.SimpleNamespace(**vars(self.args), output_dir=str(folder),
                        overwrite=False, debug=False, dry_run=False, max_test_samples=None,
                        input_max_length=8192, do_sample=False, generation_max_length=2048,
                        generation_min_length=0, temperature=0.0, top_p=1.0,
                        use_chat_template=True, seed=42, num_workers=0, model_name_or_path='synthetic',
                        batch_size=workers, count_tokens=False)
                    if interrupted:
                        with self.assertRaisesRegex(RuntimeError, 'Synthetic interrupted API'):
                            runner(args, Model(fail_at=3), 'vh_multi', 'vh.jsonl')
                        cache = next(folder.glob('*.cache'))
                        self.assertEqual(len(cache.read_text().splitlines()), 3)
                    model = Model()
                    output = Path(runner(args, model, 'vh_multi', 'vh.jsonl'))
                    if not interrupted:
                        self.assertLess(model.completion_order.index(1), model.completion_order.index(0))
                    result = json.loads(output.read_text())
                    outputs.append((result['data'], result['metrics'], result['averaged_metrics']))
            for result in outputs[1:]:
                self.assertEqual(result, outputs[0])
            self.assertEqual(outputs[0][2]['acc'], 50.0)
