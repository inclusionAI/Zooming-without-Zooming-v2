import unittest
from unittest.mock import patch

from zwz_eval.api import APIError, CompatibleAPI


class MalformedResponses(unittest.TestCase):
    def setUp(self):
        self.client = CompatibleAPI({'model': 'model', 'base_urls': ['http://127.0.0.1:8000/v1']})

    def test_malformed_payload_is_not_a_successful_empty_model_answer(self):
        bodies = [None, [], {'choices': {}}, {'choices': [None]},
                  {'choices': [{'message': None}]}, {'choices': [{'message': {}}]}]
        for body in bodies:
            with self.subTest(body=body), patch.object(self.client, '_request', return_value=body) as request:
                with self.assertRaises(APIError):
                    self.client.complete([], thinking=False, max_tokens=10)
                self.assertEqual(request.call_count, 1)

    def test_explicit_empty_eos_answers_remain_valid_model_outputs(self):
        for content in ('', None):
            body = {'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}
            with self.subTest(content=content), patch.object(self.client, '_request', return_value=body):
                self.assertEqual(self.client.complete([], thinking=False, max_tokens=10), body)

    def test_raw_completion_and_model_list_validate_structure(self):
        for body in ({'choices': [None]}, {'choices': [{'text': None}]}):
            with patch.object(self.client, '_request', return_value=body):
                with self.assertRaises(APIError):
                    self.client.complete_prompt('rendered', max_tokens=96)
        with patch.object(self.client, '_request', return_value={'data': [None]}):
            with self.assertRaises(APIError):
                self.client.health()
