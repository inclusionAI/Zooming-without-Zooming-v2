import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from zwz_eval.config import for_benchmark, load_config
from zwz_eval.data import audit
from zwz_eval.io import atomic_json, read_json
from zwz_eval.runner import run
from zwz_eval.protocols import PLATFORM_COUNTS, PLATFORM_SOURCE_FILES, PROTOCOLS
from zwz_eval.general_visual import (
    prepare, preprocess, messages, infer, initial_score, score, NativeJudgeRenderer, EMPTY_THINK)
from test_adapters import configuration
from test_spatial_pipeline import Client


class RawClient:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.requests = []
    def complete_prompt(self, prompt, **kwargs):
        self.requests.append((prompt, kwargs))
        value = next(self.answers)
        if isinstance(value, Exception): raise value
        return {'choices': [{'text': value}]}


class GeneralVisual(unittest.TestCase):
    def test_additional_platform_counts_and_identity(self):
        expected = {'vstar': 191, 'hr4k': 800, 'hr8k': 800, 'mme_en': 23609,
                    'mme_cn': 5917, 'zoombench': 845, 'perception': 3000,
                    'treebench': 405, 'o3': 345, 'countqa': 1528}
        for bench, count in expected.items():
            self.assertEqual(PLATFORM_COUNTS[bench], count)
            self.assertEqual(PROTOCOLS[bench]['max_tokens'], 16384)
            self.assertEqual(PROTOCOLS[bench]['context'], 32768)
            config = configuration(self.root)
            with self.assertRaisesRegex(ValueError, 'qwen3.5-35b-a3b'):
                for_benchmark(config, bench)
            source = self.root / f'{bench}.jsonl'
            source.write_text(json.dumps(self.row) + '\n')
            with self.assertRaisesRegex(ValueError, f'expected {count}'):
                prepare(bench, source, self.root / bench)
        self.assertIn('realworld', PLATFORM_SOURCE_FILES['mme_en'])
        self.assertIn('realworld-cn', PLATFORM_SOURCE_FILES['mme_cn'])

    def test_all_ten_additional_platforms_prepare_infer_judge_score_and_resume(self):
        # Small synthetic fixtures exercise every dispatch path, not real GPU/data acceptance.
        added = [b for b in PLATFORM_COUNTS if b not in ('mmstar', 'babyvision')]
        def preprocess_fixture(paths):
            return [Image.open(p).convert('RGB') for p in paths]
        for bench in added:
            with self.subTest(bench=bench), patch.dict(PLATFORM_COUNTS, {bench: 2}):
                source = self.root / f'{bench}.jsonl'
                source.write_text('\n'.join(json.dumps({**self.row, 'id': 7}) for _ in range(2))+'\n')
                prepared = self.root / bench
                prepare(bench, source, prepared, image_root=self.root, preprocessor=preprocess_fixture,
                        materialize_images=True)
                self.config['datasets'][bench]['root'] = str(prepared)
                calls, judgments = [], []
                answers = iter(['<answer>A</answer><answer>B</answer>', 'incorrect response'])
                class Stub:
                    def __init__(self, *args): pass
                    def health(self): return True
                    def complete(self, request, **kwargs):
                        calls.append(kwargs)
                        return {'choices': [{'message': {'content': next(answers)}, 'finish_reason': 'stop'}]}
                    def complete_prompt(self, request, **kwargs):
                        judgments.append((request, kwargs))
                        return {'choices': [{'text': 'Yes'}]}
                with patch('zwz_eval.runner.CompatibleAPI', Stub), patch('zwz_eval.spatial.CompatibleAPI', Stub), patch('zwz_eval.general_visual.CompatibleAPI', Stub), patch('zwz_eval.general_visual.tokenizer_manifest', return_value={'fixture': 'tokenizer-sha'}), patch('zwz_eval.general_visual.NativeJudgeRenderer', return_value=lambda s: s+EMPTY_THINK):
                    directory = run(self.config, bench, 'all')
                    result = read_json(directory / 'score.json')
                    self.assertEqual(result['macro'], 100)
                    self.assertEqual(result['diagnostics']['initial_correct'], 1)
                    self.assertEqual(result['diagnostics']['total'], 2)
                    run(self.config, bench, 'all')
                self.assertEqual(len(calls), 2)
                self.assertEqual(len(judgments), 1)
                self.assertEqual(judgments[0][1], {'temperature': 0.0, 'max_tokens': 96})
                self.assertTrue(judgments[0][0].endswith(EMPTY_THINK))
                self.assertTrue(all(c['max_tokens'] == 16384 and c['thinking'] is False for c in calls))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new('RGB', (8, 8), 'white').save(self.root / 'frame.png')
        self.config = configuration(self.root)
        self.config['workers'] = 1
        self.config['identity']['judge_family'] = 'qwen3.5-35b-a3b'
        self.row = {'id': 0, 'query': ['Before<image>after'],
                    'images': ['frame.png'], 'system': '', 'response': 'A'}

    def test_native_judge_template_and_explicit_prefix(self):
        calls = []
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                calls.append((messages, kwargs))
                return 'native-rendered-prompt'
        renderer = NativeJudgeRenderer(self.config, tokenizer=Tokenizer())
        self.assertEqual(renderer('question'), 'native-rendered-prompt'+EMPTY_THINK)
        self.assertEqual(calls, [([{'role': 'user', 'content': 'question'}],
                                 {'tokenize': False, 'add_generation_prompt': True})])

    def test_preprocess_source_pixel_budget_wiring(self):
        calls = []
        def vision(request, **kwargs):
            calls.append((request, kwargs))
            return [Image.new('RGB', (8, 8))], None
        module = types.SimpleNamespace(process_vision_info=vision)
        with patch.dict(sys.modules, {'qwen_vl_utils': module}):
            preprocess([self.root / 'frame.png'])
        block = calls[0][0][0]['content'][0]
        self.assertEqual((block['min_pixels'], block['max_pixels']), (65536, 16777216))
        self.assertEqual(calls[0][1], {'image_patch_size': 14})

    def test_messages_keep_empty_system_and_user_turns(self):
        row = {**self.row, 'query': ['Before<image>after', 'Another turn']}
        request = messages(row, 'mmstar', self.root)
        self.assertEqual([m['role'] for m in request], ['system', 'user', 'user'])
        self.assertEqual(request[0]['content'], [{'type': 'text', 'text': ''}])
        self.assertEqual([b['type'] for b in request[1]['content']], ['text', 'image_url', 'text'])
        self.assertEqual(request[-1]['content'], [{'type': 'text', 'text': 'Another turn'}])

    def test_mutating_source_image_during_prepare_is_rejected(self):
        original = self.root / 'original.json'
        atomic_json(original, [dict(self.row) for _ in range(388)])
        target = self.root / 'prepared'
        def mutate(paths):
            result = Image.open(paths[0]).convert('RGB')
            paths[0].write_bytes(b'replaced during preprocessing')
            return [result]
        with self.assertRaisesRegex(ValueError, 'changed during preprocessing'):
            prepare('babyvision', original, target, image_root=self.root, preprocessor=mutate,
                    materialize_images=True)
        self.assertFalse(target.exists())

    def test_previously_cached_image_is_rechecked_before_commit(self):
        Image.new('RGB', (8, 8), 'black').save(self.root / 'second.png')
        original = self.root / 'original.json'
        atomic_json(original, [{**self.row, 'images': ['frame.png', 'second.png']} for _ in range(388)])
        target = self.root / 'prepared'
        def mutate_earlier(paths):
            result = Image.open(paths[0]).convert('RGB')
            if paths[0].name == 'second.png':
                (self.root / 'frame.png').write_bytes(b'earlier cached image changed')
            return [result]
        with self.assertRaisesRegex(ValueError, 'changed during preparation'):
            prepare('babyvision', original, target, image_root=self.root, preprocessor=mutate_earlier,
                    materialize_images=True)
        self.assertFalse(target.exists())

    def test_prepared_images_are_staged_on_disk_before_next_image(self):
        Image.new('RGB', (8, 8), 'black').save(self.root / 'second.png')
        original = self.root / 'original.json'
        atomic_json(original, [{**self.row, 'images': ['frame.png', 'second.png']} for _ in range(388)])
        calls = []
        def inspect_staging(paths):
            if calls:
                self.assertEqual(len(list(self.root.glob('.zwz-prepare-*/prepared/images/*.png'))), 1)
            calls.append(paths)
            return [Image.open(paths[0]).convert('RGB')]
        target = self.root / 'prepared'
        prepare('babyvision', original, target, image_root=self.root, preprocessor=inspect_staging,
                materialize_images=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(list((target / 'images').glob('*.png'))), 2)
        self.assertEqual(list(self.root.glob('.zwz-prepare-*')), [])

    def test_first_tag_and_short_mcq_gate(self):
        self.assertTrue(initial_score(self.row, '<answer>A</answer><answer>B</answer>')[2])
        self.assertTrue(initial_score(self.row, '(A)')[2])
        self.assertFalse(initial_score(self.row, 'A ' + 'reasoning '*30)[2])

    def test_wrong_only_native_judge_and_custom_prompt(self):
        records = [self.row, {**self.row, 'id': 1, 'response': 'B'},
                   {**self.row, 'id': 2, 'response': 'C', 'LLM_JUDGE_PROMPT': 'Q={question};G={gt};P={parsed_pred}'},
                   {**self.row, 'id': 3, 'response': 'B'}]
        with patch.dict(PLATFORM_COUNTS, {'mmstar': len(records)}):
            source = self.root / 'original.json'
            atomic_json(source, records)
            prepare('mmstar', source, self.root / 'prepared', image_root=self.root,
                    preprocessor=lambda paths: [Image.open(p).convert('RGB') for p in paths],
                    materialize_images=True)
            self.config['datasets']['mmstar']['root'] = str(self.root / 'prepared')
            infer(self.config, 'mmstar', self.root,
                  Client(['<answer>A</answer><answer>B</answer>', '(B)', 'C ' + 'reasoning '*30, 'A']))
        client = RawClient(['Yes', 'No'])
        result = score(self.config, 'mmstar', self.root, client, render=lambda s: 'native:'+s+EMPTY_THINK)
        self.assertEqual(result['initial_correct'], 2)
        self.assertEqual(result['correct'], 3)
        self.assertEqual(result['llm_judge_count'], 2)
        self.assertEqual(result['platform_reported_final_score'], .75)
        self.assertIn("Q=['Before<image>after'];G=C;P=", client.requests[0][0])
        self.assertTrue(all(r[0].endswith(EMPTY_THINK) and r[1]['max_tokens'] == 96 for r in client.requests))
        score(self.config, 'mmstar', self.root, RawClient([]), render=lambda s: 'native:'+s+EMPTY_THINK)

    def test_judge_transport_failure_is_not_a_wrong_answer(self):
        with patch.dict(PLATFORM_COUNTS, {'babyvision': 1}):
            source = self.root / 'original.json'
            atomic_json(source, [self.row])
            prepare('babyvision', source, self.root / 'prepared', image_root=self.root,
                    preprocessor=lambda paths: [Image.open(p).convert('RGB') for p in paths],
                    materialize_images=True)
            self.config['datasets']['babyvision']['root'] = str(self.root / 'prepared')
            infer(self.config, 'babyvision', self.root, Client(['B']))
        with self.assertRaises(RuntimeError):
            score(self.config, 'babyvision', self.root, RawClient([RuntimeError('offline')]), render=lambda s: s)
        self.assertFalse((self.root / 'scores/metrics.json').exists())
        self.assertEqual(score(self.config, 'babyvision', self.root, RawClient(['Yes']), render=lambda s: s)['correct'], 1)

    def test_full_prepared_388_framework_and_resume(self):
        original = self.root / 'original.json'
        atomic_json(original, [{**self.row, 'id': i, 'response': 'B'} for i in range(388)])
        prepared = self.root / 'prepared'
        def fake_preprocess(paths):
            return [Image.open(path).convert('RGB') for path in paths]
        prepare('babyvision', original, prepared, image_root=self.root, preprocessor=fake_preprocess,
                materialize_images=True)
        self.config['datasets']['babyvision']['root'] = str(prepared)
        calls = []
        class Stub:
            def __init__(self, *args): pass
            def health(self): return True
            def complete(self, request, **kwargs):
                calls.append(kwargs)
                return {'choices': [{'message': {'content': '<answer>B</answer><answer>A</answer>'}, 'finish_reason': 'stop'}]}
        with patch('zwz_eval.runner.CompatibleAPI', Stub), patch('zwz_eval.spatial.CompatibleAPI', Stub), patch('zwz_eval.general_visual.tokenizer_manifest', return_value={'fixture': 'tokenizer-sha'}), patch('zwz_eval.general_visual.NativeJudgeRenderer', return_value=lambda s: s):
            directory = run(self.config, 'babyvision', 'all')
            self.assertEqual(read_json(directory / 'score.json')['macro'], 100)
            run(self.config, 'babyvision', 'all')
            self.assertEqual(len(calls), 388)
        self.assertTrue(all(c['max_tokens'] == 16384 and c['thinking'] is False for c in calls))
        with (prepared / 'records.json').open() as f:
            records = json.load(f)
        (prepared / records[0]['images'][0]).write_bytes(b'changed')
        with self.assertRaises(ValueError):
            audit('babyvision', self.config['datasets']['babyvision'])

    def test_judge_profile_does_not_inherit_other_family_tokenizer(self):
        self.config['identity']['judge_tokenizer_path'] = '/old/wrong-tokenizer'
        self.config['judge_profiles'] = {'native96': {
            'endpoint': {'model': 'native-judge', 'base_urls': ['http://127.0.0.1:9001/v1'], 'api_key_env': 'KEY'},
            'model_path': '/models/qwen35-a3b', 'family': 'qwen3.5-35b-a3b'}}
        self.config['datasets']['mmstar']['judge_profile'] = 'native96'
        selected = for_benchmark(self.config, 'mmstar')
        self.assertEqual(selected['judge']['model'], 'native-judge')
        self.assertEqual(selected['identity']['judge_model_path'], '/models/qwen35-a3b')
        self.assertNotIn('judge_tokenizer_path', selected['identity'])
        self.assertIn('judge_tokenizer_path', self.config['identity'])


if __name__ == '__main__':
    unittest.main()
