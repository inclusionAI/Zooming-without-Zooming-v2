"""Exercise the actual native registry in isolation; never call live search."""
import os
import ast
import hashlib
import importlib.util
import types
import subprocess
import sys
import unittest
from unittest.mock import patch

from zwz_eval.execution import VENDOR
from zwz_eval.execution import command_plan
from test_adapters import configuration
from pathlib import Path


class AgentFreeTools(unittest.TestCase):
    def test_live_probe_does_not_accept_empty_error_or_challenge_pages(self):
        from zwz_eval.agent_tools import response_works, check_live_tools
        for value in ('', 'Error: ConnectionError', 'No results found', '<html>Access denied</html>'):
            for tool in ('web_search', 'image_search', 'visit'):
                self.assertFalse(response_works(tool, value))
        self.assertTrue(response_works('web_search', '1. title\nURL: https://example.org'))
        self.assertTrue(response_works('image_search', '1. image\nImage URL: https://example.org/image.png'))
        self.assertTrue(response_works('visit', 'Example Domain This domain is for use in examples.'))
        with patch('zwz_eval.agent_tools.subprocess.run', side_effect=subprocess.TimeoutExpired('probe', 90)):
            self.assertFalse(check_live_tools()['checks_passed'])

    def test_project_saved_system_prompt_hash_matches_selected_launcher(self):
        job, = command_plan(configuration('/data'), 'agentvista', 'infer', Path('/out'))
        argv = job['argv']
        self.assertEqual(argv[argv.index('--system-prompt-key')+1], 'multi_tool_agent_search_only')
        self.assertEqual(job['env']['AGENTVISTA_FREE_VISIT_CHARS'], '150000')
        root = VENDOR / 'agentvista'
        spec = importlib.util.spec_from_file_location('fixture_agent_prompt_loader', root / 'configs/prompt_loader.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.dict(os.environ, {'ENABLED_TOOLS': job['env']['ENABLED_TOOLS']}):
            prompt = module.load_inference_prompts(types.SimpleNamespace(
                inference_prompts_path=root / 'prompts/inference_prompts.yaml',
                system_prompt_key=argv[argv.index('--system-prompt-key')+1]))['SYSTEM_PROMPT']
        self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(),
                         'cb34ba159c2ebc59fb9495dec74016db1bfcc7e9ce88f3c04767986cf5845a3c')

    def test_native_cli_search_limits_are_not_handler_fallback_defaults(self):
        tree = ast.parse((VENDOR / 'agentvista/infer.py').read_text())
        defaults = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                if node.args and isinstance(node.args[0], ast.Constant):
                    for keyword in node.keywords:
                        if keyword.arg == 'default' and isinstance(keyword.value, ast.Constant):
                            defaults[node.args[0].value] = keyword.value.value
        self.assertEqual(defaults['--image-search-max-calls'], 100)
        self.assertEqual(defaults['--web-search-max-calls'], 100)

    def test_registry_and_free_tool_http_routes_without_paid_keys(self):
        script = r'''
from unittest.mock import patch
from types import SimpleNamespace
import requests
from tools import get_tool
assert [get_tool(n).__name__ for n in ('web_search','image_search','visit')] == [
    'FreeWebSearch','FreeImageSearch','FreeVisit']
def page(text='', data=None):
    return SimpleNamespace(text=text, raise_for_status=lambda:None, json=lambda:data)
html = '<a class="result__a" href="https://example.org">Title</a><a class="result__snippet">Snippet</a>'
with patch('requests.get', return_value=page(html)) as get:
    assert 'https://example.org' in get_tool('web_search')().call({'query':'fixture'})
    assert get.call_args.args[0] == 'https://html.duckduckgo.com/html/'
with patch('requests.get', side_effect=requests.Timeout()) as get:
    assert get_tool('web_search')().call({'query':'fixture'}).startswith('Error:')
with patch('requests.get') as get:
    result = get_tool('image_search')().call({'search_type':'reverse','image_url':'private.png'})
    assert 'does not upload' in result
    get.assert_not_called()
with patch('requests.get', side_effect=[page("vqd='123-456'"), page(data={'results':[
        {'image':'https://example.org/image.png','title':'Picture','url':'https://example.org'}]})]) as get:
    assert 'Image URL: https://example.org/image.png' in get_tool('image_search')().call({'query':'fixture'})
    assert get.call_args.args[0] == 'https://duckduckgo.com/i.js'
with patch('requests.get', side_effect=[requests.Timeout(),page('Reader text')]) as get:
    assert get_tool('visit')({'max_content_length':6}).call({'url':'https://example.org','goal':'fixture'}) == 'Reader'
    assert get.call_args.args[0] == 'https://r.jina.ai/http://example.org'
with patch('requests.post') as post, patch('requests.get', return_value=page('<p>Readable</p>')):
    assert get_tool('visit')().call({'url':'https://example.org','goal':'fixture'}) == 'Readable'
    post.assert_not_called()  # Free visit does not invoke the YAML LLM summarizer.
'''
        env = os.environ.copy()
        for key in ('SERPAPI_KEY', 'JINA_API_KEY', 'IMGBB_API_KEY', 'REASONING_API_KEY',
                    'VERIFIER_API_KEY', 'EVALUATOR_API_KEY', 'AGENTVISTA_FREE_VISIT_CHARS'):
            env.pop(key, None)
        env['AGENTVISTA_FREE_TOOLS'] = '1'
        env['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
        result = subprocess.run([sys.executable, '-c', script], cwd=VENDOR / 'agentvista',
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
