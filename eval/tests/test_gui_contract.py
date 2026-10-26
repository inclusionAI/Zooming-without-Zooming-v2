import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
import io

from zwz_eval.gui_grounding import (extract_coordinate, legacy_point, normalize_coordinate,
                                         parse_point, point_in_box, summarize)
from zwz_eval.gui_api import collect_stream, image_payload, system_prompt, api_payload
from zwz_eval.protocols import RESOURCES


def call(x=500, y=500):
    return json.dumps({'name': 'computer_use', 'arguments': {'action': 'left_click', 'coordinate': [x, y]}})


class GUIContract(unittest.TestCase):
    def test_native_tool_request_has_one_unambiguous_schema(self):
        from types import SimpleNamespace
        args = SimpleNamespace(coordinate_space='normalized_1000', prompt_contract='explicit',
             native_tools=True, model='fixture', max_tokens=8192, reasoning_effort='low')
        p = api_payload(args, {'url': 'fixture', 'size': [3840, 2144]}, 'click target')
        self.assertEqual(p['messages'][1]['content'][1]['text'], 'click target')
        self.assertNotIn('<tools>', p['messages'][0]['content'])
        self.assertIn('x=1000*px/3840', p['messages'][0]['content'])
        self.assertEqual(p['tool_choice']['function']['name'], 'computer_use')
        schema = p['tools'][0]['function']['parameters']
        self.assertEqual(schema['required'], ['action', 'coordinate'])
        self.assertEqual(schema['properties']['coordinate']['items']['maximum'], 1000)
        self.assertFalse(p['parallel_tool_calls'])

    def test_explicit_grid_prompt_and_diagnostics(self):
        prompt = system_prompt('normalized_1000', (3840, 1088), 'explicit')
        self.assertIn('x=1000*px/3840', prompt)
        self.assertIn('y=1000*py/1088', prompt)
        self.assertIn('NOT screenshot pixels', prompt)
        self.assertNotEqual(prompt, system_prompt('normalized_1000', (3840, 1088)))
        items = [{'id': str(i), 'subset': 'all'} for i in range(3)]
        rows = [{'id': str(i), 'correct': False, 'point': None, 'error': None,
                 'coordinate_space': 'normalized_1000', 'image_input': {'size': [3840, 1088]},
                 'raw_response': raw} for i, raw in enumerate([call(), call(1120, 1099), 'no answer'])]
        report = summarize(items, rows, 'screenspot_pro')
        self.assertEqual(report['diagnostics'], {'unparseable_tool_call': 1,
                         'coordinate_out_of_range': 1, 'format_only_legacy_failure': 1})
        self.assertEqual(report['accuracy'], 0)

    def test_exact_legacy_parser_unchanged(self):
        path = RESOURCES / 'vendor/gui/official/screenspot-pro/models/qwen3_5.py'
        tree = ast.parse(path.read_text())
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3_5Model')
        method = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'ground_only_positive')
        parse = next(n for n in method.body if isinstance(n, ast.Try))
        module = ast.Module(body=[parse], type_ignores=[])
        for text in (call(), '<tool_call>' + call() + '</tool_call>',
                     '<tool_call>\n' + call() + '\n</tool_call>', '',
                     '<tool_call>\n' + call(-1, 0) + '\n</tool_call>'):
            namespace = {'json': json, 'response': text, 'result_dict': {'point': None}}
            exec(compile(module, '<retained-parser>', 'exec'), namespace)
            self.assertEqual(legacy_point(text), namespace['result_dict']['point'])

    def test_compact_tool_call_diagnostic_does_not_replace_legacy(self):
        text = '<tool_call>' + call() + '</tool_call>'
        self.assertIsNone(legacy_point(text))
        self.assertEqual(extract_coordinate(text), [500, 500])

    def test_units_are_explicit_not_inferred_from_magnitude(self):
        p = [30, 130]
        self.assertEqual(normalize_coordinate(p, 'normalized_1000', (3840, 1080)), [.03, .13])
        self.assertEqual(normalize_coordinate(p, 'image_pixels', (3840, 1080)), [30/3840, 130/1080])
        self.assertNotEqual(normalize_coordinate([1, 1], 'normalized_1000', (100, 100)), [1, 1])

    def test_bad_coordinates_rejected(self):
        for value in ([True, 0], [float('nan'), 1], [float('inf'), 2], [-1, 1], [1001, 2]):
            self.assertIsNone(normalize_coordinate(value, 'normalized_1000', (3840, 2160)))

    def test_multiple_calls_wrong_action_and_prose_rejected(self):
        good = '<tool_call>' + call() + '</tool_call>'
        self.assertIsNone(extract_coordinate(good + good))
        self.assertIsNone(extract_coordinate(call().replace('left_click', 'right_click')))
        self.assertIsNone(extract_coordinate('Maybe coordinate [200,300]'))

    def test_inclusive_edges(self):
        self.assertTrue(point_in_box([.1, .2], [10, 20, 30, 40], [100, 100]))
        self.assertTrue(point_in_box([.3, .4], [10, 20, 30, 40], [100, 100]))
        self.assertFalse(point_in_box([.31, .4], [10, 20, 30, 40], [100, 100]))

    def test_ui_reversed_bbox_upstream_rule(self):
        self.assertTrue(point_in_box([.2, .3], [30, 40, 10, 20], [100, 100], reorder=True))

    def test_no_partial_accuracy_and_invalid_stays_denominator(self):
        items = [{'id': '1', 'subset': 'all'}, {'id': '2', 'subset': 'all'}]
        rows = [{'id': '1', 'correct': True, 'point': [.5,.5], 'error': None}]
        self.assertIsNone(summarize(items, rows, 'screenspot_pro')['accuracy'])
        rows.append({'id': '2', 'correct': False, 'point': None, 'error': None})
        self.assertEqual(summarize(items, rows, 'screenspot_pro')['accuracy'], .5)
        rows[1]['error'] = 'HTTP failed'
        self.assertIsNone(summarize(items, rows, 'screenspot_pro')['accuracy'])

    def test_ui_requires_three_splits_and_macro_not_micro(self):
        items = [{'id': str(i), 'subset': 'basic' if i < 2 else 'functional' if i < 3 else 'spatial'} for i in range(4)]
        rows = [{'id': str(i), 'correct': i < 2, 'point': [0,0], 'error': None} for i in range(4)]
        self.assertAlmostEqual(summarize(items, rows, 'ui_vision')['accuracy'], 1/3)
        self.assertIsNone(summarize(items[:2], rows[:2], 'ui_vision')['accuracy'])

    def test_different_units_cannot_be_used_with_legacy_parser(self):
        with self.assertRaises(ValueError):
            parse_point(call(), parser='native_xml', coordinate_space='image_pixels', sent_size=(2000,1000))

    def test_stream_has_no_reasoning_fallback(self):
        lines = ['data: ' + json.dumps({'choices': [{'delta': {'reasoning_content': call()}, 'finish_reason': 'stop'}]})]
        result = collect_stream(lines)
        self.assertEqual(result['raw_response'], '')
        self.assertIsNone(extract_coordinate(result['raw_response']))
        with self.assertRaisesRegex(RuntimeError, 'finish_reason'):
            collect_stream(['data: ' + json.dumps({'choices': [{'delta': {'content': call()}}]})])

    def test_stream_native_function_call_supported(self):
        lines = ['data: ' + json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {
            'name': 'computer_use', 'arguments': json.dumps({'action': 'left_click', 'coordinate': [500, 500]})}}]},
            'finish_reason': 'tool_calls'}]})]
        self.assertEqual(legacy_point(collect_stream(lines)['raw_response']), [.5, .5])

    def test_byte_stream_without_charset_and_invalid_calls_are_scoreable(self):
        event = {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {
            'name': 'computer_use', 'arguments': 'bad json'}}]}, 'finish_reason': 'tool_calls'}]}
        result = collect_stream([('data: ' + json.dumps(event)).encode()])
        self.assertIsNone(legacy_point(result['raw_response']))
        self.assertIsNone(extract_coordinate(result['raw_response']))
        event['choices'][0]['delta']['tool_calls'].append({'index': 1, 'function': {
            'name': 'computer_use', 'arguments': '{}'}})
        self.assertIsNone(legacy_point(collect_stream(['data: ' + json.dumps(event)])['raw_response']))

    def test_prompt_source_and_coordinate_declaration(self):
        native = system_prompt('normalized_1000', (2000, 1000))
        self.assertIn('1000x1000', native)
        pixels = system_prompt('image_pixels', (2000, 1500))
        self.assertIn('2000x1500', pixels)
        self.assertNotIn('1000x1000', pixels)

    def test_image_bytes_lossless_and_input_dimensions_checked(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'image.png'
            Image.new('RGB', (2000, 1000)).save(p)
            encoded = image_payload(p, [2000, 1000], 8294400)
            self.assertTrue(encoded['url'].startswith('data:image/png;base64,'))
            self.assertEqual(encoded['size'], [1984, 992])
            with self.assertRaises(ValueError):
                image_payload(p, [1000, 2000], 8294400)

    def test_api_contract_partial_resume_and_changed_image_fail_closed(self):
        from zwz_eval.gui_api import main
        from zwz_eval.io import file_hash
        # Synthetic two-row loader isolates orchestration; real loader enforces
        # 1581/5479 and the official three split denominators independently.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'fixture.png'
            image.write_bytes(b'synthetic-image-bytes-only')
            items = [{'id': str(i), 'subset': 'all', 'instruction': 'click target',
                      'image': str(image), 'bbox': [0, 0, 100, 100], 'size': [100, 100]} for i in range(2)]
            args = ['--bench', 'screenspot_pro', '--data-root', tmp, '--out-root', str(root/'run'),
                    '--api', 'http://127.0.0.1:1/v1', '--model', 'synthetic-fixture', '--limit', '1']
            def fake(item, args):
                return {**item, 'point': [.5,.5], 'correct': True, 'error': None,
                        'raw_response': '<tool_call>\n' + call() + '\n</tool_call>',
                        'image_input': {'size': [100,100], 'original_sha256': file_hash(image)}}
            with patch('zwz_eval.gui_api.load_items', return_value=(items, {'fixture':'test'})), \
                 patch('zwz_eval.gui_api.infer_one', side_effect=fake) as run, redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 2)
                self.assertIsNone(json.loads((root/'run/metrics.json').read_text())['accuracy'])
                self.assertEqual(main(args), 0)
                self.assertEqual(main(args), 0)
                self.assertEqual(run.call_count, 2)
                self.assertEqual(json.loads((root/'run/metrics.json').read_text())['accuracy'], 1)
                image.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Screenshot changed'):
                    main(args)
                image.write_bytes(b'synthetic-image-bytes-only')
                with self.assertRaisesRegex(ValueError, 'contract changed'):
                    main(args + ['--max-tokens', '2048'])

    def test_api_rejects_historical_uncontracted_journal(self):
        from zwz_eval.gui_api import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'predictions.jsonl').write_text('{}\n')
            with patch('zwz_eval.gui_api.load_items', return_value=([], {})):
                with self.assertRaisesRegex(ValueError, 'Historical predictions'):
                    main(['--bench', 'screenspot_pro', '--data-root', tmp, '--out-root', tmp,
                          '--api', 'http://127.0.0.1:1/v1', '--model', 'synthetic-fixture'])
