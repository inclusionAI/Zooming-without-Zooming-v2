"""Lazy platform image loading must preserve the existing pixel/message contract."""
import ast
import base64
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from zwz_eval import general_visual as visual
from zwz_eval.data import audit, rows
from zwz_eval.io import atomic_json, digest, file_hash, read_journal, read_json
from zwz_eval.protocols import PLATFORM_COUNTS


class Client:
    def __init__(self):
        self.requests = []

    def complete(self, request, **kwargs):
        self.requests.append((request, kwargs))
        return {'choices': [{'message': {'content': 'A'}, 'finish_reason': 'stop'}]}


class PlatformOnDemand(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'run').mkdir()
        self.images = self.root / 'original-images'
        self.images.mkdir()
        for name, color in [('a.png', 'red'), ('b.png', 'blue')]:
            Image.new('RGB', (17, 13), color).save(self.images / name)
        self.original = [
            {'id': 1, 'query': ['before<image>middle<image>after', 'second turn'],
             'images': ['a.png', 'b.png'], 'response': 'A', 'system': ''},
            {'id': 2, 'query': 'without a marker', 'images': ['b.png'], 'response': 'A',
             'system': 'original system', 'LLM_JUDGE_PROMPT': 'original {gt}'},
            {'id': 3, 'query': 'before<image>between<image>after',
             'images': ['a.png'], 'response': 'A', 'extra': ['preserve', 'this']},
        ]
        self.source = self.root / 'source.json'
        atomic_json(self.source, self.original)
        counts = patch.dict(PLATFORM_COUNTS, {'mmstar': len(self.original)})
        counts.start()
        self.addCleanup(counts.stop)
        self.preprocess_calls = []

    def preprocess_fixture(self, paths):
        # Deterministic CPU stand-in; both paths must use this exact operation.
        self.preprocess_calls.append(list(paths))
        result = []
        for path in paths:
            with Image.open(path) as image:
                result.append(image.convert('RGB').resize((28, 28), Image.Resampling.BICUBIC))
        return result

    def prepare(self, name='prepared', materialize=False):
        target = self.root / name
        visual.prepare('mmstar', self.source, target, image_root=self.images,
                       preprocessor=self.preprocess_fixture, materialize_images=materialize)
        return target

    def config(self, target):
        return {'datasets': {'mmstar': {'root': str(target)}},
                'workers': 1, 'timeout': 10, 'retries': 1, 'main': {}}

    def test_default_prepare_and_annotation_audit_do_not_touch_images(self):
        original_open, original_stat = Path.open, Path.stat

        def guarded_open(path, *args, **kwargs):
            if path.parent == self.images:
                raise AssertionError('prepare/audit must not read original images')
            return original_open(path, *args, **kwargs)

        def guarded_stat(path, *args, **kwargs):
            if path.parent == self.images:
                raise AssertionError('prepare/audit must not stat original images')
            return original_stat(path, *args, **kwargs)

        with patch.object(Path, 'open', guarded_open), patch.object(Path, 'stat', guarded_stat), \
                patch.object(visual, 'resolve_image', side_effect=AssertionError('no image resolution')):
            target = self.prepare()
            report = audit('mmstar', {'root': str(target)}, verify_images=False)
        self.assertEqual(self.preprocess_calls, [])
        manifest = read_json(target / 'preparation.json')
        self.assertEqual(manifest['image_mode'], 'on-demand')
        self.assertNotIn('assets', manifest)
        self.assertFalse((target / 'images').exists())
        self.assertEqual(report['images_checked'], 0)
        self.assertEqual(report['image_sha256'], {})
        for before, after in zip(self.original, rows(target / 'records.json')):
            self.assertEqual({k: v for k, v in after.items() if k not in ('images', 'sample_uid')},
                             {k: v for k, v in before.items() if k != 'images'})

    def test_on_demand_messages_match_materialized_pixels_and_layout(self):
        materialized = self.prepare('offline', True)
        on_demand = self.prepare('lazy')
        self.preprocess_calls.clear()
        with patch.object(visual, 'preprocess', self.preprocess_fixture):
            for old_row, new_row in zip(rows(materialized / 'records.json'), rows(on_demand / 'records.json')):
                old = visual.messages(old_row, 'mmstar', materialized)
                new = visual.on_demand_messages(new_row, 'mmstar', on_demand)
                self.assertEqual(old, new)
                for message in new:
                    for block in message['content']:
                        if block['type'] == 'image_url':
                            raw = base64.b64decode(block['image_url']['url'].split(',', 1)[1])
                            with Image.open(io.BytesIO(raw)) as image:
                                self.assertEqual((image.format, image.size), ('PNG', (28, 28)))
        self.assertEqual(len(self.preprocess_calls), 4)

    def test_inference_selects_verified_mode_and_resume_does_not_decode_successes(self):
        target = self.prepare()
        client = Client()
        with patch.object(visual, 'preprocess', self.preprocess_fixture):
            summary = visual.infer(self.config(target), 'mmstar', self.root / 'run', client)
            first_calls = len(self.preprocess_calls)
            visual.infer(self.config(target), 'mmstar', self.root / 'run', client)
        self.assertEqual(summary['completed'], 3)
        self.assertEqual(first_calls, 4)
        self.assertEqual(len(self.preprocess_calls), first_calls)
        self.assertEqual(len(client.requests), 3)
        journal = read_journal(self.root / 'run/predictions.jsonl')
        self.assertEqual([row['request_sha256'] for row in journal],
                         [digest(request) for request, _ in client.requests])
        self.assertTrue(all(kwargs == {'thinking': False, 'max_tokens': 16384, 'top_p': 1.0}
                            for _, kwargs in client.requests))

    def test_legacy_materialized_has_no_second_preprocessing(self):
        target = self.prepare(materialize=True)
        manifest = read_json(target / 'preparation.json')
        del manifest['image_mode']  # Existing releases did not have a mode key.
        atomic_json(target / 'preparation.json', manifest)
        with patch.object(visual, 'preprocess', side_effect=AssertionError('must not resize again')):
            summary = visual.infer(self.config(target), 'mmstar', self.root / 'run', Client())
            audit('mmstar', {'root': str(target)})
        self.assertEqual(summary['completed'], 3)

    def test_first_request_runs_before_remaining_images_are_preprocessed(self):
        target = self.prepare()
        processed_at_request = []
        client = Client()
        complete = client.complete

        def recording_complete(request, **kwargs):
            processed_at_request.append(len(self.preprocess_calls))
            return complete(request, **kwargs)
        client.complete = recording_complete
        with patch.object(visual, 'preprocess', self.preprocess_fixture):
            visual.infer(self.config(target), 'mmstar', self.root / 'run', client)
        self.assertEqual(processed_at_request, [2, 3, 4])

    def test_bad_or_missing_images_fail_at_request_and_never_reach_api(self):
        for content in (None, b'not an image'):
            with self.subTest(content=content):
                if (self.images / 'a.png').exists():
                    (self.images / 'a.png').unlink()
                if content is not None:
                    (self.images / 'a.png').write_bytes(content)
                target = self.prepare('missing' if content is None else 'broken')
                (target / 'run').mkdir()
                client = Client()
                with patch.object(visual, 'preprocess', self.preprocess_fixture):
                    with self.assertRaisesRegex(RuntimeError, '2 unresolved inference errors'):
                        visual.infer(self.config(target), 'mmstar', target / 'run', client)
                self.assertEqual(len(client.requests), 1)
                self.assertEqual(read_json(target / 'run/inference_summary.json')['failed'], 2)

    def test_original_image_mutation_during_request_is_rejected(self):
        target = self.prepare()
        row = rows(target / 'records.json')[0]

        def mutation(paths):
            result = self.preprocess_fixture(paths)
            paths[0].write_bytes(b'changed while preprocessing')
            return result
        with patch.object(visual, 'preprocess', mutation):
            with self.assertRaisesRegex(ValueError, 'changed during request'):
                visual.on_demand_messages(row, 'mmstar', target)

    def test_unknown_mode_and_forged_materialized_claim_fail_annotation_audit(self):
        target = self.prepare()
        original = read_json(target / 'preparation.json')
        for update in ({'image_mode': 'raw-bypass'}, {'assets': {}}, {'image_mode': 'materialized'}):
            atomic_json(target / 'preparation.json', {**original, **update})
            with self.assertRaises(ValueError):
                audit('mmstar', {'root': str(target)}, verify_images=False)
        atomic_json(target / 'preparation.json', original)
        records = rows(target / 'records.json')
        records[0]['image_mode'] = 'raw-bypass'
        atomic_json(target / 'records.json', records)
        # A per-row field cannot override dispatch; changing annotations is detected.
        with self.assertRaisesRegex(ValueError, 'annotation provenance'):
            visual.infer(self.config(target), 'mmstar', self.root / 'run', Client())
        # Even when an otherwise valid annotation carries a mode-like field,
        # it cannot switch off the manifest-selected preprocessing operation.
        original['records_sha256'] = file_hash(target / 'records.json')
        atomic_json(target / 'preparation.json', original)
        with patch.object(visual, 'preprocess', self.preprocess_fixture):
            visual.infer(self.config(target), 'mmstar', self.root / 'run', Client())
        self.assertEqual(len(self.preprocess_calls), 4)

    def test_materialized_hash_and_asset_validation_remain_strict(self):
        target = self.prepare(materialize=True)
        manifest = read_json(target / 'preparation.json')
        record = rows(target / 'records.json')[0]
        image = record['images'][0]
        (target / image).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'differs from its preprocessing manifest'):
            audit('mmstar', {'root': str(target)})
        del manifest['assets'][image]
        atomic_json(target / 'preparation.json', manifest)
        with self.assertRaisesRegex(ValueError, 'absent from its preprocessing manifest'):
            audit('mmstar', {'root': str(target)}, verify_images=False)

    def test_explicit_full_audit_hashes_originals_without_claiming_old_hashes(self):
        target = self.prepare()
        report = audit('mmstar', {'root': str(target)}, verify_images=True)
        self.assertEqual(report['images_checked'], 2)
        self.assertEqual(report['image_sha256'], {str((self.images / name).resolve()):
                                                file_hash(self.images / name) for name in ('a.png', 'b.png')})
        self.assertNotIn('assets', report['preparation'])

    def test_raw_bundle_relative_paths_survive_atomic_parent_rename(self):
        stage = self.root / 'staging/bundle'
        normalized = stage / 'normalized'
        normalized.mkdir(parents=True)
        for name in ('a.png', 'b.png'):
            (normalized / name).write_bytes((self.images / name).read_bytes())
        source = normalized / 'normalized.json'
        atomic_json(source, self.original)
        visual.prepare('mmstar', source, stage / 'prepared')
        prepared_rows = rows(stage / 'prepared/records.json')
        self.assertEqual(prepared_rows[0]['images'], ['../normalized/a.png', '../normalized/b.png'])
        final = self.root / 'final-bundle'
        stage.rename(final)
        with patch.object(visual, 'preprocess', self.preprocess_fixture):
            summary = visual.infer(self.config(final / 'prepared'), 'mmstar', self.root / 'run', Client())
        self.assertEqual(summary['completed'], 3)

    def test_source_symlink_parent_segments_are_not_lexically_collapsed(self):
        actual = self.root / 'actual'
        (actual / 'nested').mkdir(parents=True)
        (self.root / 'link').symlink_to(actual / 'nested', target_is_directory=True)
        Image.new('RGB', (17, 13), 'red').save(actual / 'a.png')
        Image.new('RGB', (17, 13), 'blue').save(self.root / 'a.png')
        reference = str(self.root / 'link/../a.png')
        self.original[0]['images'][0] = reference
        atomic_json(self.source, self.original)
        target = self.prepare()
        row = rows(target / 'records.json')[0]
        self.assertEqual(row['images'][0], reference)
        with patch.object(visual, 'preprocess', self.preprocess_fixture):
            request = visual.on_demand_messages(row, 'mmstar', target)
        block = next(block for message in request for block in message['content']
                     if block['type'] == 'image_url')
        with Image.open(io.BytesIO(base64.b64decode(block['image_url']['url'].split(',', 1)[1]))) as image:
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

    def test_annotation_mode_policy_and_rows_are_all_audited(self):
        target = self.prepare()
        original = read_json(target / 'preparation.json')
        for field, value in [('rows', 4), ('benchmark', 'mme_en')]:
            atomic_json(target / 'preparation.json', {**original, field: value})
            with self.assertRaises(ValueError):
                audit('mmstar', {'root': str(target)}, verify_images=False)
        original['preprocessing']['max_pixels'] = 1
        atomic_json(target / 'preparation.json', original)
        with self.assertRaisesRegex(ValueError, 'policy differs'):
            audit('mmstar', {'root': str(target)}, verify_images=False)


class PlatformSemanticsAST(unittest.TestCase):
    def test_original_preprocessor_layout_and_all_scoring_definitions_are_unchanged(self):
        baseline = {
            'preprocess': 'c5ee16aa614552fb6a879fa7cfe05ad9711f2f7c45fe1b70b005cadefe3f4d4b',
            'messages': '5b3a1d7e7cf8674cd4c61e2d1eb7900d5926612190f9a48b60744012440894a6',
            'tokenizer_manifest': '0325f142e58356399d3ddf21f5daab14936eca5a8b231440181a26243e7bc324',
            'NativeJudgeRenderer': 'ef02cde554d7b761a1016b8f9cfb3c10a59ba6259066360b1d70fd88645a5cfc',
            'initial_score': '391c49370af75f4db09a79b96db6bbca684f04da36f92d720ea0cd2239e1967f',
            'judge_prompt': '86bdf4e04326c960c449e9be424c245cc723a180cbe50166b14493dad211d9f4',
            'parse_judge': 'bd3f69e41b40223f46a54491daf463c74bb7fb741b1befe90f20b0b21b406ebb',
            'score': '7b80a023767c7f8dc8bbc02d827412da8e93b0a82eaed4606f3ec8c08d2cb148',
        }

        def stable(node):
            if isinstance(node, ast.AST):
                return [type(node).__name__, [[key, stable(value)] for key, value in ast.iter_fields(node)
                                             if not (key == 'type_params' and not value)]]
            if isinstance(node, list):
                return [stable(value) for value in node]
            return node

        source = Path(visual.__file__)
        definitions = {node.name: node for node in ast.parse(source.read_text()).body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for name, expected in baseline.items():
            payload = json.dumps(stable(definitions[name]), ensure_ascii=True, separators=(',', ':'))
            self.assertEqual(hashlib.sha256(payload.encode()).hexdigest(), expected, name)


if __name__ == '__main__':
    unittest.main()
