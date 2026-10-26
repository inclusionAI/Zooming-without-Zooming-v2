"""Synthetic end-to-end exercise: real formatter, journal, scorer subprocess and macro.

This is deliberately not a real benchmark or GPU accuracy test.
"""
import contextlib
import importlib.util
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from zwz_eval.config import load_config
from zwz_eval.io import atomic_json, read_json, read_journal
from zwz_eval.protocols import MILE_TASKS
from zwz_eval.runner import run
from zwz_eval.data import load_module
from zwz_eval.protocols import RESOURCES

HAS_API = all(importlib.util.find_spec(x) for x in ('openai', 'PIL', 'tqdm'))


@unittest.skipUnless(HAS_API, 'Install the api extra to test the real legacy formatter/scorer')
class Pipeline(unittest.TestCase):
    def test_memlens_roles_dates_and_images_are_not_flattened(self):
        converter = load_module(RESOURCES / 'vendor/memlens/prepare.py', '_test_mem_converter')
        formatter = load_module(RESOURCES / 'vendor/common/infer.py', '_test_mem_formatter')
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image
            root = Path(tmp)
            Image.new('RGB', (16, 16), 'white').save(root / 'fixture.png')
            row = converter.convert_record({'question_id': 'fixture', 'question': 'What did I show?',
                'answer': 'An image', 'question_date': '2026-01-03', 'question_type': 'information_extraction',
                'haystack_sessions': [{'date': '2026-01-01', 'session': [
                    {'role': 'user', 'content': 'Here <image>', 'images': [{'file': 'fixture.png'}]},
                    {'role': 'assistant', 'content': 'I see it.'}]}]}, '32k', root)
            messages = formatter.build_record_messages(row, 'memlens-32k', [str(root)])
            self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'user'])
            self.assertIn('Insufficient information', messages[0]['content'][0]['text'])
            self.assertIn('2026-01-01', messages[1]['content'][0]['text'])
            self.assertEqual(messages[1]['content'][1]['type'], 'image_url')
            self.assertIn('2026-01-03', messages[-1]['content'][0]['text'])

    def test_milebench_synthetic_29_and_resume(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, value):
                raw = json.dumps(value).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                self.send({'data': [{'id': 'fixture-model'}]})

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(request)
                text = 'Paris'
                if request['max_tokens'] == 10:
                    text = 'A' if 'A. Paris' in request['messages'][0]['content'] else 'B'
                self.send({'id': 'fixture', 'object': 'chat.completion', 'created': 0, 'model': 'fixture-model',
                    'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': text}}]})
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = root / 'data'
                for index, task in enumerate(MILE_TASKS):
                    atomic_json(data / f'milebench_{task}_test.json', [{
                        'sample_id': 'same-id-across-tasks', 'task': task, 'query': 'A fixture question',
                        'images': [], 'response': 'Paris', 'system_prompt': 'Fixture instruction',
                        'meta': {'question_type': 'multi-choice' if index < 2 else 'open-ended',
                                 'choice_list': ['Rome', 'Paris'] if index == 0 else ['Paris', 'Rome'] if index == 1 else []}}])
                endpoint = {'model': 'fixture-model', 'base_urls': [f'http://127.0.0.1:{server.server_port}/v1'], 'api_key_env': 'FIXTURE_API_KEY'}
                config_path = root / 'config.json'
                atomic_json(config_path, {'model_id': 'fixture-model', 'run_id': 'synthetic',
                    'output_root': str(root / 'outputs'), 'main': endpoint, 'judge': endpoint,
                    'datasets': {'milebench': {'root': str(data)}}, 'workers': 4, 'timeout': 5, 'retries': 1})
                config = load_config(config_path)
                with contextlib.redirect_stdout(io.StringIO()):
                    directory = run(config, 'milebench', 'all')
                score = read_json(directory / 'score.json')
                self.assertEqual(len(score['tasks']), 29)
                self.assertEqual(score['macro'], 100)
                self.assertEqual(len(requests), 31)
                for request in requests:
                    self.assertFalse(request['chat_template_kwargs']['enable_thinking'])
                    if request['max_tokens'] == 8192:
                        self.assertIn('Fixture instruction', request['messages'][0]['content'][0]['text'])
                ids = [read_journal(p)[0]['sample_id'] for p in (directory / 'legacy').rglob('*_answer.jsonl')]
                self.assertEqual(len(set(ids)), 29)
                with contextlib.redirect_stdout(io.StringIO()):
                    run(config, 'milebench', 'all')
                self.assertEqual(len(requests), 31, 'Resume must not re-query valid samples or cached extraction')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
