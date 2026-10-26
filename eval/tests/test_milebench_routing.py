"""CPU-only judge routing tests; no API, model, GPU or benchmark data required."""
import ast
import hashlib
import importlib.util
import json
import os
import tempfile
import threading
import types
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from zwz_eval.execution import command_plan


SOURCE = (Path(__file__).resolve().parents[1] /
          'zwz_eval/resources/vendor/milebench/score.py')


class MileBenchRouting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.calls = []
        self.created = []
        self.lock = threading.Lock()
        self.reply = lambda url, kwargs: 'A'
        fake_openai = types.ModuleType('openai')

        def factory(*, base_url, api_key):
            self.created.append((base_url, api_key))

            def create(**kwargs):
                with self.lock:
                    self.calls.append((base_url, kwargs))
                content = self.reply(base_url, kwargs)
                return types.SimpleNamespace(choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content=content))])
            return types.SimpleNamespace(chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)))

        fake_openai.OpenAI = factory
        spec = importlib.util.spec_from_file_location('_milebench_routing_fixture', SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {'openai': fake_openai}), patch.dict(os.environ, {
                'MILEBENCH_MODELS': 'fixture-model', 'NUM_SHARDS': '1', 'SHARD_ID': '0',
                'EVAL_ROOT_MILEBENCH': self.tmp.name,
                'CACHE_DIR_MILEBENCH': self.tmp.name,
                'JUDGE_API_KEY': 'fixture-key'}):
            spec.loader.exec_module(self.module)
        self.env = patch.dict(os.environ, {'JUDGE_API_KEY': 'fixture-key',
                                          'ZWZ_STRICT_ERRORS': '1'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.sleep = patch.object(self.module.time, 'sleep')
        self.sleep_mock = self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def test_execution_passes_all_endpoints_without_changing_job_count(self):
        urls = ['http://127.0.0.1:%d/v1' % (9000 + n) for n in range(8)]
        config = {'model_id': 'fixture-model', 'workers': 16,
                  'datasets': {'milebench': {'root': '/example/data'}},
                  'judge': {'base_urls': urls, 'model': 'fixture-judge',
                            'api_key_env': 'TEST_KEY'}}
        jobs = command_plan(config, 'milebench', 'score', '/example/out')
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['env']['JUDGE_URL'], ','.join(urls))
        self.assertEqual(jobs[0]['env']['JUDGE_MODEL'], 'fixture-judge')
        self.assertEqual(jobs[0]['env']['NUM_SHARDS'], '1')
        self.assertEqual(jobs[0]['env']['JUDGE_API_KEY']['from_env'], 'TEST_KEY')

    def test_concurrent_requests_use_every_replica_once_per_turn(self):
        urls = ['http://127.0.0.1:%d/v1' % (9000 + n) for n in range(8)]
        self.module.init_client(','.join(urls), 'fixture-judge')
        with ThreadPoolExecutor(max_workers=64) as pool:
            letters = list(pool.map(lambda n: self.module.llm_extract_letter(
                'question-%d' % n, {'B': 'beta', 'A': 'alpha'}, 'A'), range(64)))
        self.assertEqual(letters, ['A'] * 64)
        self.assertEqual(Counter(url for url, _ in self.calls), dict.fromkeys(urls, 8))
        self.assertEqual(self.created, [(url, 'fixture-key') for url in urls])
        for _, kwargs in self.calls:
            self.assertEqual(kwargs['model'], 'fixture-judge')
            self.assertEqual(kwargs['temperature'], 0.0)
            self.assertEqual(kwargs['max_tokens'], 10)
            self.assertEqual(kwargs['extra_body'], {'chat_template_kwargs': {'enable_thinking': False}})
            self.assertIn('A. alpha\nB. beta', kwargs['messages'][0]['content'])

    def test_transport_retries_stay_on_original_replica(self):
        self.module.init_client('http://first/v1,http://second/v1')
        failures = [0]

        def reply(url, kwargs):
            if url == 'http://first/v1' and failures[0] < 2:
                failures[0] += 1
                raise OSError('synthetic transport interruption')
            return 'A'
        self.reply = reply
        self.assertEqual(self.module.llm_extract_letter('q', {'A': 'a'}, 'A'), 'A')
        self.assertEqual(self.module.llm_extract_letter('next', {'A': 'a'}, 'A'), 'A')
        self.assertEqual([url for url, _ in self.calls],
                         ['http://first/v1'] * 3 + ['http://second/v1'])
        self.assertEqual([call.args for call in self.sleep_mock.call_args_list], [(1,), (3,)])
        self.assertEqual(self.calls[0][1], self.calls[1][1])
        self.assertEqual(self.calls[1][1], self.calls[2][1])

    def test_final_transport_failure_is_not_failover_or_new_sampling(self):
        self.module.init_client('http://first/v1,http://second/v1')

        def reply(url, kwargs):
            raise OSError('synthetic permanent transport failure')
        self.reply = reply
        with self.assertRaisesRegex(RuntimeError, 'extractor transport failed'):
            self.module.llm_extract_letter('q', {'A': 'a'}, 'A')
        self.assertEqual([url for url, _ in self.calls], ['http://first/v1'] * 3)

    def test_successful_invalid_letter_does_not_retry(self):
        self.module.init_client('http://first/v1,http://second/v1')
        self.reply = lambda url, kwargs: '??'
        self.assertEqual(self.module.llm_extract_letter('q', {'A': 'a'}, 'A'), '?')
        self.assertEqual(len(self.calls), 1)
        self.sleep_mock.assert_not_called()

    def test_resume_reuses_success_including_wrong_and_invalid_answers(self):
        self.module.init_client('http://first/v1,http://second/v1')
        rows = [{'sample_id': str(n), 'query': 'q', 'task': 'ALFRED',
                 'response': 'alpha', 'model_answer': 'A',
                 'meta': {'question_type': 'multi-choice', 'choice_list': ['alpha', 'beta']}}
                for n in range(6)]
        self.module._llm_cache[('fixture-model', '0')] = 'B'
        self.module._llm_cache[('fixture-model', '1')] = '?'
        # Persist prior valid responses, even if wrong/invalid, before resuming.
        for sample, letter in [('0', 'B'), ('1', '?')]:
            self.module.save_llm_record({'model': 'fixture-model', 'sample_id': sample,
                                         'extracted_letter': letter})
        self.assertEqual(self.module.evaluate_subtask_with_llm(rows, 'fixture-model'),
                         [0, 0, 1, 1, 1, 1])
        self.assertEqual(len(self.calls), 4)
        self.module._llm_cache.clear()
        self.module.load_llm_cache()
        self.assertEqual(self.module.evaluate_subtask_with_llm(rows, 'fixture-model'),
                         [0, 0, 1, 1, 1, 1])
        self.assertEqual(len(self.calls), 4, 'Do not re-query completed extractions')
        cache = [json.loads(line) for line in Path(self.module.LLM_CACHE).read_text().splitlines()]
        self.assertEqual(len(cache), 6)
        self.assertEqual({row['sample_id'] for row in cache}, {str(n) for n in range(6)})
        self.assertEqual({row['model'] for row in cache}, {'fixture-model'})

    def test_single_url_compatibility_and_empty_endpoint_rejection(self):
        self.module.init_client(' http://single/v1 ')
        self.assertEqual(self.module.llm_extract_letter('q', {'A': 'a'}, 'A'), 'A')
        self.assertEqual(self.created, [('http://single/v1', 'fixture-key')])
        for value in ('', ' ', 'http://one/v1,', ',http://two/v1'):
            with self.assertRaisesRegex(ValueError, 'nonempty base URLs'):
                self.module.init_client(value)


class MileBenchScoringAST(unittest.TestCase):
    def test_scoring_cache_concurrency_and_prompts_match_pre_routing_ast(self):
        # Baseline AST digests were recorded before this endpoint-only fix.
        # For extraction only, normalize the new request-local client selection.
        expected = {
            'get_category': '91e345b61273afd4c4f905559acc69b936b61675fc50c48dbb09386f42c542f2',
            'process': 'add76dde695e409b9a61a08145d75dc00081e9b9a6a341e95016bacca526d6ce',
            '_preprocess_option': '1c6bd3b615d5c90001e33f6a5b5a5f7967a50fb2bdfd412f2cfaa1a746670a56',
            'char_for': 'a9e0545f9b8187069d27226e38408e8a6c4771314b9ca1bde14e76e602fda336',
            '_lcs_len': '284c3d7df421f3ccd60b6ec9db4f468c82fe197e0341d703e859bb0c9a64dd1c',
            'rouge_l_f1': '7ffec53482b8ae81d2662f0038a8c4a659e12929b69ab6732fdb0e8c17fb3b5e',
            'judge_needle': 'f9bc36b2e9427e26316f0061091807e2a37b3eb08185d56e817421119af1d8a4',
            # Same extraction logic, pinned after the strict-errors environment namespace rename.
            'llm_extract_letter': '29e0012afa394b7c239c72a6edab13e1f24f5741506208e24ed3049495b9af89',
            'load_llm_cache': '1da2ef998b467d8868ad975ca1a4c003376faa404407366074f3b3738f19b650',
            'save_llm_record': 'a2d55b8c83d617b7c3ae6217f78143fde974b99aed41feeab05d82a093549429',
            'evaluate_subtask_with_llm': '2c7021cafe37aac1e5fc4c6febe37eac2563fb1c9078a94dfa97e3c999d40045',
            'worker': '9c6c66a445ed79c7f87ba215cc79b50bcce0ff3bd80787274dc98d7f43a2b488',
            'print_summary': '6df9f91a9484457b09315c04c261e0058a1338b31cd7af2f10651e971381ba29',
        }

        def stable_ast(node):
            # Empty generic type parameters were added to FunctionDef in 3.12.
            if isinstance(node, ast.AST):
                return [type(node).__name__, [[key, stable_ast(value)]
                        for key, value in ast.iter_fields(node)
                        if not (key == 'type_params' and not value)]]
            if isinstance(node, list):
                return [stable_ast(value) for value in node]
            return node

        class NormalizeRouting(ast.NodeTransformer):
            def visit_Assign(self, node):
                assignment = ast.parse('client = _next_client()').body[0]
                if ast.dump(node) == ast.dump(assignment):
                    return None
                return self.generic_visit(node)

            def visit_Name(self, node):
                if node.id == 'client':
                    node.id = '_client'
                return node

        functions = {node.name: node for node in ast.parse(SOURCE.read_text()).body
                     if isinstance(node, ast.FunctionDef)}
        for name, expected_hash in expected.items():
            node = functions[name]
            if name == 'llm_extract_letter':
                node = NormalizeRouting().visit(node)
            serialized = json.dumps(stable_ast(node), ensure_ascii=True, separators=(',', ':'))
            actual = hashlib.sha256(serialized.encode()).hexdigest()
            self.assertEqual(actual, expected_hash, name)


if __name__ == '__main__':
    unittest.main()
