"""Four full synthetic denominators through the retained judge subprocess.

The HTTP stub checks transport/rubric routing, not real LLM judgment quality.
Inference is a precompleted synthetic journal; no GPU or actual dataset is used.
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

from PIL import Image

from zwz_eval.data import annotation_sets, prepare_memlens, rows, sample_uid
from zwz_eval.io import atomic_json, canonical, digest, read_json, read_journal
from zwz_eval.protocols import MEM_SPLITS
from zwz_eval.runner import run
from test_adapters import configuration


@unittest.skipUnless(importlib.util.find_spec('openai'), 'Install the api extra')
class MemLensNativePipeline(unittest.TestCase):
    def test_four_789_splits_native_judge_routing_and_cache_binding(self):
        requests, failures = [], []
        lock = threading.Lock()
        class Server(ThreadingHTTPServer):
            request_queue_size = 128
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def reply(self, value):
                raw = json.dumps(value).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def do_GET(self):
                self.reply({'data': [{'id': 'fixture-model'}, {'id': 'fixture-judge'}]})
            def do_POST(self):
                value = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with lock:
                    requests.append((value.get('model'), value.get('max_tokens'),
                                     value.get('temperature'), value.get('chat_template_kwargs')))
                    if '[Current Case]' not in value['messages'][0]['content']:
                        failures.append('Missing preserved rubric/current-case prompt')
                self.reply({'choices': [{'message': {'role': 'assistant', 'content': '{"answer_score": 1}'},
                                         'finish_reason': 'stop'}]})
        server = Server(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                Image.new('RGB', (4, 4), 'white').save(root / 'frame.png')
                raw, normalized = root / 'raw', root / 'normalized'
                types = [('information_extraction', 'previnfo', 'gold'),
                         ('information_extraction', '', 'gold'),
                         ('knowledge_update', '', 'gold'),
                         ('answer_refusal', '', 'Insufficient information'),
                         ('temporal_reasoning', 'duration_comparison', 'gold'),
                         ('temporal_reasoning', 'order_ranking', 'gold'),
                         ('temporal_reasoning', 'temporal_info_extraction', 'gold'),
                         ('multi_session_reasoning', 'arithmetic', 'gold'),
                         ('multi_session_reasoning', 'counting', 'gold'),
                         ('multi_session_reasoning', 'entity_resolution', 'Yes')]
                for split in MEM_SPLITS:
                    original = []
                    for i in range(789):
                        kind, subtype, answer = types[i % len(types)]
                        original.append({'question_id': str(i), 'question': f'Fixture question {i}',
                            'answer': answer, 'question_type': kind, 'question_subtype': subtype,
                            'question_date': '2026-01-02', 'haystack_sessions': [
                                {'date': '2026-01-01', 'session': [
                                    {'role': 'user', 'content': 'Image <image>', 'images': [{'file': 'frame.png'}]},
                                    {'role': 'assistant', 'content': 'Recorded.'}]}]})
                    atomic_json(raw / f'dataset_{split}.json', original)
                prepare_memlens(raw, root, normalized)
                config = configuration(root / 'outputs')
                config['datasets']['memlens'] = {'root': str(normalized), 'image_root': str(root)}
                endpoint = f'http://127.0.0.1:{server.server_port}/v1'
                config['main'].update(model='fixture-model', base_urls=[endpoint])
                config['judge'].update(model='fixture-judge', base_urls=[endpoint])
                directory = root / 'outputs/model/reference/memlens'
                directory.mkdir(parents=True)
                with (directory / 'predictions.jsonl').open('w') as stream:
                    for task, source in annotation_sets('memlens', normalized).items():
                        for i, row in enumerate(rows(source)):
                            stream.write(canonical({'uid': f'{task}/{sample_uid(row,i)}', 'task': task,
                                'input_sha256': digest(row), 'status': 'ok',
                                'raw_response': row['response'], 'model_answer': row['response']})+'\n')
                with contextlib.redirect_stdout(io.StringIO()):
                    run(config, 'memlens', 'all')
                self.assertEqual(read_json(directory / 'score.json')['macro'], 100)
                self.assertEqual(len(requests), 4*789)
                self.assertFalse(failures)
                self.assertTrue(all(r == ('fixture-judge', 2048, 0.0, {'enable_thinking': False}) for r in requests))
                expected_keys = {'IE_PreviousInfo', 'IE_Entity', 'KU_KnowledgeUpdate', 'AR_AnswerRefusal',
                                 'TR_DurationComparison', 'TR_OrderRanking', 'TR_DateExtraction',
                                 'MSR_Arithmetic', 'MSR_Counting', 'MSR_YesNo'}
                for split in MEM_SPLITS:
                    metrics = read_json(directory / 'scores' / f'memlens-{split}' / 'judge_metrics.json')
                    self.assertEqual(metrics['overall']['total'], 789)
                    self.assertEqual(set(metrics['by_task_key']), expected_keys)
                with contextlib.redirect_stdout(io.StringIO()):
                    run(config, 'memlens', 'all')
                self.assertEqual(len(requests), 4*789, 'Resume must reuse bound judge records')
                # Same IDs, but a corrected model response: old native idx/qid cache
                # alone would accept it. The wrapper must reject reuse before HTTP.
                projection = next((directory / 'legacy').rglob('*_seed42_answer.jsonl'))
                changed = read_journal(projection)
                changed[0]['model_answer'] = 'different model response'
                projection.write_text(''.join(canonical(row)+'\n' for row in changed))
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'refusing stale verdict'):
                    run(config, 'memlens', 'score')
                self.assertEqual(len(requests), 4*789)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
