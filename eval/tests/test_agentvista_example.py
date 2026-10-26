"""The portable AgentVista example uses the standard benchmark entrypoints."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from zwz_eval.config import load_config
from zwz_eval.execution import command_plan
from zwz_eval.packaging import source_files

ROOT = Path(__file__).resolve().parents[1]


class AgentVistaExample(unittest.TestCase):
    def test_user_supplied_configuration_keeps_retained_plan(self):
        settings = {
            'AGENTVISTA_DATA_ROOT': '/path/to/prepared/agentvista',
            'AGENTVISTA_IMAGE_ROOT': '/path/to/prepared/agentvista/images',
            'EVAL_OUTPUT_ROOT': '/path/to/results',
            'MAIN_MODEL_ID': 'test-main', 'MAIN_BASE_URL': 'https://model-api.example/v1',
            'MAIN_MODEL_REVISION': 'test-revision', 'MAIN_CHAT_TEMPLATE_SHA256': 'a' * 64,
            'JUDGE_MODEL_ID': 'test-judge', 'JUDGE_BASE_URL': 'https://judge-api.example/v1',
            'JUDGE_MODEL_REVISION': 'test-judge-revision',
        }
        with patch.dict(os.environ, settings):
            config = load_config(ROOT / 'examples/agentvista.json')
        self.assertEqual(set(config['datasets']), {'agentvista'})
        infer, = command_plan(config, 'agentvista', 'infer', Path('/path/to/results/agentvista'))
        argv = infer['argv']
        for flag, value in [('--max-completion-tokens', '1024'), ('--max-turns', '30'),
                            ('--max-total-tokens', '65536'), ('--system-prompt-key', 'multi_tool_agent_search_only')]:
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertEqual(infer['env']['REASONING_END_POINT'], settings['MAIN_BASE_URL'])
        self.assertEqual(infer['env']['AGENTVISTA_TOTAL_TOOL_BUDGET'], '12')
        self.assertEqual(infer['env']['AGENTVISTA_FREE_VISIT_CHARS'], '150000')
        score, = command_plan(config, 'agentvista', 'score', Path('/path/to/results/agentvista'))
        self.assertEqual(score['env']['VERIFIER_END_POINT'], settings['JUDGE_BASE_URL'])

    def test_standard_export_contains_complete_implementation_and_example(self):
        exported = {str(p.relative_to(ROOT)) for p in source_files(ROOT)}
        for relative in ('examples/agentvista.json',
                         'scripts/evaluate.sh', 'scripts/run_benchmarks.sh',
                         'zwz_eval/resources/vendor/agentvista/infer.py',
                         'zwz_eval/resources/vendor/agentvista/score_final_answers.py',
                         'zwz_eval/resources/vendor/agentvista/tools/free_backends.py'):
            self.assertIn(relative, exported)
        self.assertFalse((ROOT / 'agentvista').exists())
