import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from zwz_eval.api import APIError, CompatibleAPI


class APIContract(unittest.TestCase):
    def setUp(self):
        self.requests = []
        outer = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, data, code=200):
                raw = json.dumps(data).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                self.reply({'data': [{'id': 'served-model'}]})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                outer.requests.append(payload)
                if payload['model'] == 'missing-model':
                    self.reply({'error': 'not found'}, 404)
                elif self.path == '/v1/completions':
                    self.reply({'choices': [{'text': 'Yes', 'finish_reason': 'stop'}]})
                else:
                    self.reply({'choices': [{'message': {'content': '<answer>B</answer>'}, 'finish_reason': 'stop'}]})
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = {'model': 'served-model', 'base_urls': [f'http://127.0.0.1:{self.server.server_port}/v1'], 'api_key_env': 'ZWZ_TEST_NONEXISTENT_KEY'}

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_endpoint_identity_and_nothinking_payload(self):
        client = CompatibleAPI(self.endpoint)
        self.assertTrue(client.health())
        answer = client.complete([{'role': 'user', 'content': 'Question'}], thinking=False, max_tokens=8192)
        self.assertEqual(answer['choices'][0]['message']['content'], '<answer>B</answer>')
        self.assertEqual(self.requests[0]['chat_template_kwargs'], {'enable_thinking': False})
        self.assertEqual(self.requests[0]['max_tokens'], 8192)

    def test_native_default_thinking_not_forced(self):
        CompatibleAPI(self.endpoint).complete([], thinking=None, max_tokens=10)
        self.assertNotIn('chat_template_kwargs', self.requests[0])

    def test_native_raw_completion_transport(self):
        client = CompatibleAPI(self.endpoint)
        prompt = 'native template<think>\n\n</think>\n\n'
        response = client.complete_prompt(prompt, max_tokens=96)
        self.assertEqual(response['choices'][0]['text'], 'Yes')
        self.assertEqual(self.requests[0]['prompt'], prompt)
        self.assertEqual(self.requests[0]['max_tokens'], 96)
        self.assertNotIn('messages', self.requests[0])
        self.assertNotIn('chat_template_kwargs', self.requests[0])

    def test_wrong_model_rejected_and_404_not_retried(self):
        endpoint = {**self.endpoint, 'model': 'missing-model'}
        client = CompatibleAPI(endpoint, retries=3)
        with self.assertRaises(APIError):
            client.health()
        with self.assertRaises(APIError):
            client.complete([], thinking=False, max_tokens=10)
        self.assertEqual(len(self.requests), 1)


if __name__ == '__main__':
    unittest.main()
