"""First-user CLI journey: synthetic raw data/local API, never GPU accuracy."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from zwz_eval.io import atomic_json, read_json
from zwz_eval.protocols import MILE_TASKS


class PortableCLI(unittest.TestCase):
    def test_init_raw_conversion_doctor_two_stage_scoring_summary_and_resume(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.reply({'data': [{'id': 'synthetic-main'}, {'id': 'synthetic-judge'}]})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(payload)
                answer = 'Paris'
                if payload['max_tokens'] == 10:
                    answer = 'A' if 'A. Paris' in payload['messages'][0]['content'] else 'B'
                self.reply({'id': 'synthetic', 'object': 'chat.completion', 'created': 0,
                            'model': payload['model'], 'choices': [{'index': 0, 'finish_reason': 'stop',
                            'message': {'role': 'assistant', 'content': answer}}]})

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix='portable cli ') as temporary:
                root = Path(temporary).resolve()
                project = root / 'local project'
                storage = root / 'data storage'
                output = root / 'results'
                checkpoint = root / 'weights'
                checkpoint.mkdir()
                atomic_json(checkpoint / 'config.json', {'model_type': 'synthetic-only'})
                (checkpoint / 'chat_template.jinja').write_text('synthetic native template')
                env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                           MAIN_API_KEY='EMPTY', JUDGE_API_KEY='EMPTY',
                           NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
                def cli(*arguments):
                    result = subprocess.run([sys.executable, '-m', 'zwz_eval', *map(str, arguments)],
                                            cwd=root, env=env, capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr + '\n' + result.stdout[-4000:])
                    return result.stdout
                cli('init', '--directory', project, '--bench', 'milebench', '--model-id', 'synthetic-main',
                    '--model-path', checkpoint, '--data-root', storage, '--output-root', output)
                manifest = read_json(project / 'datasets.json')
                raw = Path(manifest['datasets']['milebench']['source'])
                for index, task in enumerate(MILE_TASKS):
                    images = raw / task / 'images'
                    images.mkdir(parents=True)
                    Image.new('RGB', (8, 8), 'blue').save(images / 'fixture.png')
                    choices = ['Rome', 'Paris'] if index == 0 else ['Paris', 'Rome'] if index == 1 else []
                    atomic_json(raw / task / f'{task}.json', {
                        'meta_data': {'task_instruction': ['Synthetic instruction'],
                                      'question_type': 'multi-choice' if choices else 'open-ended'},
                        'data': [{'sample_id': 'same-id-across-tasks', 'response': 'Paris',
                                  'task_instruction_id': 0, 'task_instance': {
                                      'context': 'Synthetic image {image#1}: question?',
                                      'images_path': ['fixture.png'], 'choice_list': choices}}]})
                config_path = project / 'model.json'
                config = read_json(config_path)
                url = f'http://127.0.0.1:{server.server_port}/v1'
                config['main']['base_urls'] = config['judge']['base_urls'] = [url]
                config['judge']['model'] = 'synthetic-judge'
                config.update(workers=4, timeout=5, retries=1)
                config['identity'].update(model_revision='synthetic-main-1', judge_revision='synthetic-judge-1',
                    chat_template_sha256=hashlib.sha256(b'synthetic native template').hexdigest())
                atomic_json(config_path, config)
                ready = project / 'ready.json'
                cli('prepare-suite', '--manifest', project / 'datasets.json', '--config', config_path,
                    '--output-config', ready)
                report = json.loads(cli('doctor', '--config', ready, '--phase', 'infer',
                                        '--audit-data', '--check-endpoints'))
                self.assertTrue(report['checks_passed'])
                self.assertEqual(requests, [], 'Doctor must never request completions')
                cli('run-suite', '--config', ready, '--phase', 'infer')
                self.assertEqual(len(requests), 29)
                cli('doctor', '--config', ready, '--phase', 'score', '--check-endpoints')
                cli('run-suite', '--config', ready, '--phase', 'score')
                self.assertEqual(len(requests), 31)
                for payload in requests:
                    self.assertFalse(payload['chat_template_kwargs']['enable_thinking'])
                    self.assertEqual(payload['max_tokens'], 8192 if payload['model'] == 'synthetic-main' else 10)
                scores = json.loads(cli('summarize', '--root', output))
                self.assertEqual(len(scores), 1)
                self.assertEqual((scores[0]['macro'], len(scores[0]['tasks'])), (100, 29))
                cli('run-suite', '--config', ready, '--phase', 'infer')
                cli('run-suite', '--config', ready, '--phase', 'score')
                self.assertEqual(len(requests), 31, 'Resume must not repeat model or judge requests')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
