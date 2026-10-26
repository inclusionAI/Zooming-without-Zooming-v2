"""Exercise the retained native generate method without importing GPU runtimes."""
import ast
import os
from functools import partial
import types
import unittest
from unittest.mock import Mock, patch

from zwz_eval.protocols import RESOURCES


class MMLTransport(unittest.TestCase):
    def test_native_task_errors_propagate_when_wrapper_selects_strict_mode(self):
        source = RESOURCES / 'vendor/mmlongbench/eval_api_batch.py'
        text = source.read_text()
        handlers = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.ExceptHandler)]
        handler = next(n for n in handlers if 'ZWZ_STRICT_ERRORS' in ast.get_source_segment(text,n))
        tree = ast.Module(body=[ast.Try(body=[ast.Raise(exc=ast.Call(func=ast.Name(id='RuntimeError',ctx=ast.Load()),
                                  args=[ast.Constant(value='offline')],keywords=[]),cause=None)],
                                  handlers=[handler],orelse=[],finalbody=[])],type_ignores=[])
        ast.fix_missing_locations(tree)
        namespace = {'os':os,'args':types.SimpleNamespace(debug=False),'logger':Mock(),'dataset':'fixture'}
        with patch.dict(os.environ, {'ZWZ_STRICT_ERRORS':'1'}):
            with self.assertRaisesRegex(RuntimeError,'offline'):
                exec(compile(tree,str(source),'exec'),namespace)

    def model(self, reply=None, error=None):
        source = RESOURCES / 'vendor/mmlongbench/vlm_model/vllm_models.py'
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VLLMModel')
        selected = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(selected)
        create = Mock(return_value=reply, side_effect=error)
        namespace = {'LLM': object, 'partial': partial,
                     'call_api': lambda func, **kwargs: func(),
                     'time': types.SimpleNamespace(time=lambda: 1.0, sleep=lambda _: None)}
        exec(compile(selected, str(source), 'exec'), namespace)
        model = object.__new__(namespace['VLLMModel'])
        model._next_client = lambda: types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)))
        model.model_name = 'fixture'
        model.generation_max_length = 2048
        model.temperature, model.top_p, model.do_sample, model.api_sleep = 1.0, 1.0, False, 0
        return model, create

    def request(self, model):
        return model.generate(inputs={'messages': [{'role':'user', 'content':[{'type':'text','text':'Original prompt'}]}]})

    def reply(self, text):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=text), finish_reason='stop')],
            usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=0 if text is None else 4))

    def test_transport_failure_is_not_cached_as_empty_text(self):
        model,_ = self.model(error=RuntimeError('offline'))
        with self.assertRaisesRegex(RuntimeError, 'API request failed; no result cached'):
            self.request(model)

    def test_valid_none_eos_keeps_real_usage(self):
        model,_ = self.model(reply=self.reply(None))
        record = self.request(model)
        self.assertEqual((record['output'],record['input_len'],record['output_len']), ('',100,0))
        self.assertEqual(record['api_status'], 'ok')

    def test_request_generation_settings_and_answer_are_unchanged(self):
        model,create = self.model(reply=self.reply('<answer>B</answer>'))
        record = self.request(model)
        self.assertEqual(record['output'], '<answer>B</answer>')
        kwargs = create.call_args.kwargs
        self.assertEqual((kwargs['temperature'],kwargs['top_p'],kwargs['max_tokens']), (0.0,1.0,2048))
        self.assertEqual(kwargs['extra_body'], {'chat_template_kwargs':{'enable_thinking':False}})

    def test_malformed_response_is_not_a_model_failure(self):
        model,_ = self.model(reply=types.SimpleNamespace(choices=[]))
        with self.assertRaisesRegex(RuntimeError, 'Malformed MMLongBench'):
            self.request(model)
