import base64
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zwz_eval.data import rows
from zwz_eval.io import canonical
from zwz_eval.normalization import fingerprint, identities
from zwz_eval.protocols import PLATFORM_BENCHMARKS
from zwz_eval.raw_platform import (
    MC_SUFFIX, SYSTEM, _candidate_rows, _order_candidates, convert_platform, load_recipe, _source_files, _vstar_sidecar,
)


def image_uri(color):
    from PIL import Image
    output = io.BytesIO()
    Image.new('RGB', (3, 2), color).save(output, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode()


def normalized(query, answer, images):
    return {'query': query, 'response': answer, 'images': images, 'system': SYSTEM}


def recipe(benchmark, expected):
    return {'schema': 1, 'benchmark': benchmark,
            'order': [[fingerprint({k: v for k, v in row.items() if k != 'images'}), row['images']]
                      for row in expected]}


class RawPlatform(unittest.TestCase):
    def test_vstar_literal_numeric_and_period_quirks_are_source_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'direct_attributes'
            folder.mkdir()
            path = folder / 'fixture.json'
            raw = {'question': 'Synthetic?', 'options': ['no person', 'right']}
            path.write_text(json.dumps(raw))
            projection = {'raw_projections': {'direct_attributes/fixture.jpg': {
                'raw_metadata_sha256': fingerprint(raw),
                'option_slices': [[1, 0, 5, 'append_period'], [0, 0, 9, 'no_person_to_zero_people']]}}}
            result = _vstar_sidecar(path, projection)
            self.assertEqual(result['text'], 'Synthetic?\n(A) right.\n(B) 0 people\n' + MC_SUFFIX)
            self.assertEqual(result['label'], 'B')
    def test_official_zoom_single_parquet_and_vstar_sidecar_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'data').mkdir()
            (root / 'data/test.parquet').touch()
            self.assertEqual(_source_files('zoombench', root), [root / 'data/test.parquet'])
            for folder in ('direct_attributes', 'relative_position', 'OCR', 'GPT4V-hard'):
                (root / folder).mkdir()
                (root / folder / 'synthetic.json').write_text('{}')
            self.assertEqual(_source_files('vstar', root), [root / 'direct_attributes/synthetic.json',
                                                           root / 'relative_position/synthetic.json'])

    def test_official_vstar_sidecars_reconstruct_mc_options_without_shipping_question_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, expected, raw_projections = Path(temporary), [], {}
            for folder, color in [('direct_attributes', 'red'), ('relative_position', 'blue')]:
                directory = root / folder
                directory.mkdir()
                raw = {'question': 'Synthetic spatial question?',
                       'options': ['The fixture is left.', 'The fixture is right.'],
                       'target_object': ['fixture'], 'bbox': [[0, 0, 1, 1]]}
                (directory / 'fixture.json').write_text(json.dumps(raw))
                (directory / 'fixture.jpg').write_bytes(base64.b64decode(image_uri(color).split(',', 1)[1]))
                relative = folder + '/fixture.jpg'
                raw_projections[relative] = {'raw_metadata_sha256': fingerprint(raw),
                                            'option_slices': [[1, 15, 20], [0, 15, 19]]}
                expected.append(normalized('<image>Synthetic spatial question?\n(A) right\n(B) left\n' + MC_SUFFIX,
                                           'B', [relative]))
            projection = recipe('vstar', expected)
            projection['raw_projections'] = raw_projections
            with patch('zwz_eval.raw_platform.load_recipe', return_value=projection), \
                 patch('zwz_eval.normalization.read_json', return_value={'inputs': {'vstar': identities(expected)}}):
                convert_platform('vstar', root, root / 'output')
            self.assertEqual(rows(root / 'output/normalized.jsonl'), expected)
            # Original metadata edits cannot silently reuse the shortened-option recipe.
            path = root / 'direct_attributes/fixture.json'
            raw = json.loads(path.read_text())
            raw['options'].reverse()
            path.write_text(json.dumps(raw))
            with patch('zwz_eval.raw_platform.load_recipe', return_value=projection):
                with self.assertRaisesRegex(ValueError, 'sidecar metadata differs'):
                    convert_platform('vstar', root, root / 'changed')
            self.assertFalse((root / 'changed').exists())

    def convert(self, benchmark, raw, expected, root):
        source, output = root / 'raw.json', root / 'converted'
        source.write_text(json.dumps(raw))
        contract = {'inputs': {benchmark: identities(expected)}}
        with patch('zwz_eval.raw_platform.load_recipe', return_value=recipe(benchmark, expected)), \
                patch('zwz_eval.normalization.read_json', return_value=contract):
            report = convert_platform(benchmark, source, output)
        self.assertEqual(rows(output / 'normalized.jsonl'), expected)
        self.assertTrue(report['matches_historical_annotations'])
        self.assertFalse(report['image_bytes_verified'])
        for row in expected:
            for image in row['images']:
                self.assertTrue((output / image).is_file())
        self.assertNotIn(str(root), (output / 'conversion.json').read_text())
        return output

    def test_all_twelve_supported_raw_schemas_and_fixed_historical_order(self):
        red, blue = image_uri('red'), image_uri('blue')
        cases = {}
        raw = [{'index': 11, 'question': 'Q? Options: A: x, B: y, C: nan, D: nan', 'answer': 'A', 'image': red},
               {'index': 12, 'question': 'R? Options: A: x, B: y', 'answer': 'B', 'image': blue}]
        expected = [normalized('<image>R? Select from the following choices.\n(A) x\n(B) y', '(B)', ['12.jpg']),
                    normalized('<image>Q? Select from the following choices.\n(A) x\n(B) y', '(A)', ['11.jpg'])]
        cases['mmstar'] = raw, expected
        cases['babyvision'] = ([{'question': 'Q?', 'ansType': 'choice', 'options': ['x', 'y'], 'choiceAns': 1,
                                  'image': {'path': 'test-uuid.png', 'bytes': None}},
                                 {'question': 'R?', 'ansType': 'blank', 'options': [], 'blankAns': '2', 'image': blue}],
                                [normalized('<image>R?', '2', ['other.png']),
                                 normalized('<image>Q?\nSelect from the following choices.\nA:x\nB:y\n', 'B', ['test-uuid.png'])])
        cases['vstar'] = ([{'text': 'Q?\n(A) x\n(B) y\n' + MC_SUFFIX.replace('choices.', 'choices directly.'),
                            'label': 'B', 'image': red}],
                          [normalized('<image>Q?\n(A) x\n(B) y\n' + MC_SUFFIX, 'B', ['direct_attributes/x.jpg'])])
        for bench in ('hr4k', 'hr8k'):
            raw = [{'question': q, 'answer': a, 'A': 'x', 'B': 'y', 'C': 'z', 'D': 'w', 'image': im}
                   for q, a, im in [('Q?', 'A', red), ('R?', 'B', red), ('S?', 'C', blue)]]
            expected = [normalized('<image>' + q + '\nA. x\nB. y\nC. z\nD. w\n' + MC_SUFFIX, a, [p])
                        for q, a, p in [('S?', 'C', 'image_19.png'), ('Q?', 'A', 'image_7.png'), ('R?', 'B', 'image_7.png')]]
            cases[bench] = raw, expected
        for bench, lead in [('mme_en', 'Select from the following choices.'), ('mme_cn', '\u8bf7\u4ece\u4ee5\u4e0b\u9009\u9879\u4e2d\u9009\u62e9.')]:
            raw = [{'Question_id': 'fixture', 'Question Type': 'Perception', 'Image': red,
                    'Text': 'Q?', 'Answer choices': ['(A) x.', '(B) y.'], 'Ground truth': 'B',
                    'Task': 'fixture', 'Subtask': 'fixture', 'Category': 'fixture', 'Dataset': 'fixture'}]
            cases[bench] = raw, [normalized('<image>Q?\n' + lead + '\nA:(A) x..\nB:(B) y..', 'B', ['source.png'])]
        cases['zoombench'] = ([{'id': 'abc123', 'query': 'Q?\nA. x\nB. y', 'response': 'B',
                               'image': red, 'crop_image': blue}],
                             [normalized('<image>Q?\nA. x\nB. y\n' + MC_SUFFIX, 'B', ['abc123.jpg'])])
        cases['perception'] = ([{'index': 7, 'problem': '<|image_1|><|image_2|>Q?', 'answer': '2',
                                 'image': [red, blue], 'hint': ''}],
                               [normalized('<image><image>Q?', '2', ['7_01.jpg', '7_02.jpg'])])
        cases['treebench'] = ([{'index': 8, 'question': ' Q? ', 'answer': 'B', 'A': 'x', 'B': 'y', 'C': None,
                                'image': red}],
                              [normalized('<image> Q?  Select from the following choices.\n(A) x\n(B) y', '(B)', ['00008.png'])])
        raw = {'question': 'Q?', 'options': 'A. x\nB. y', 'answer': 'A', 'image': red,
               'subset': 'map', 'explanation': 'fixture explanation', 'target_layouts': [1],
               'layout_infos': {'1': {'bbox': [0.0, 1.0, 2.0, 3.0]}}, 'img_width': 3, 'img_height': 2}
        expected = {k: v for k, v in raw.items() if k not in ('question', 'image')}
        expected.update(query='<image>\nQ?\nA. x\nB. y', response='A', images=['map/fixture.png'])
        cases['o3'] = [raw], [expected]
        cases['countqa'] = ([{'questions': ['Q?', 'R?'], 'answers': ['2', '3'], 'image': red},
                             {'questions': ['Q?'], 'answers': ['2'], 'image': blue}],
                           [normalized('<image>Q?', '2', ['image_1.png']),
                            normalized('<image>R?', '3', ['image_0.png']),
                            normalized('<image>Q?', '2', ['image_0.png'])])
        self.assertEqual(set(cases), set(PLATFORM_BENCHMARKS))
        for benchmark, (raw, expected) in cases.items():
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                # Exercise path-based HF image objects as well as embedded data URIs.
                (root / 'test-uuid.png').write_bytes(base64.b64decode(red.split(',', 1)[1]))
                output = self.convert(benchmark, raw, expected, root)
                if benchmark == 'zoombench':
                    self.assertEqual((output / 'abc123.jpg').read_bytes(), base64.b64decode(red.split(',', 1)[1]))
                if benchmark == 'perception':
                    self.assertNotEqual((output / '7_01.jpg').read_bytes(), (output / '7_02.jpg').read_bytes())

    def test_mismatch_missing_image_and_ambiguous_question_leave_no_output(self):
        raw = [{'index': 0, 'question': 'Q? Options: A: x, B: y', 'answer': 'B', 'image': image_uri('red')}]
        expected = [normalized('<image>Q? Select from the following choices.\n(A) x\n(B) y', '(A)', ['0.jpg'])]
        for mode in ('gold', 'missing_image', 'ambiguous', 'traversal'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture, wanted = copy.deepcopy(raw), copy.deepcopy(expected)
                if mode != 'gold':
                    fixture[0]['answer'] = 'A'
                if mode == 'missing_image':
                    fixture[0]['image'] = 'missing.jpg'
                elif mode == 'ambiguous':
                    wanted.append({**wanted[0], 'images': ['1.jpg']})
                elif mode == 'traversal':
                    wanted[0]['images'] = ['../escape.jpg']
                source = root / 'raw.json'
                source.write_text(json.dumps(fixture))
                with patch('zwz_eval.raw_platform.load_recipe', return_value=recipe('mmstar', wanted)), \
                        patch('zwz_eval.normalization.read_json', return_value={'inputs': {'mmstar': identities(expected)}}):
                    with self.assertRaises(ValueError):
                        convert_platform('mmstar', source, root / 'out')
                self.assertFalse((root / 'out').exists())

    def test_normalized_row_preserves_extra_fields_and_existing_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'fixture.png').write_bytes(base64.b64decode(image_uri('red').split(',', 1)[1]))
            expected = [{'query': '<image> Q ', 'response': 'A', 'images': ['fixture.png'],
                         'LLM_JUDGE_PROMPT': 'custom {gt}', 'custom': ['must', 'survive']}]
            output = self.convert('o3', expected, expected, root)
            with self.assertRaisesRegex(ValueError, 'already exist'):
                convert_platform('o3', root / 'raw.json', output)
            self.assertEqual(rows(output / 'normalized.jsonl'), expected)

    def test_content_free_recipes_cover_every_platform_and_actual_contract_counts(self):
        from zwz_eval.protocols import PLATFORM_COUNTS
        for benchmark in PLATFORM_BENCHMARKS:
            value = load_recipe(benchmark)
            self.assertEqual(len(value['order']), PLATFORM_COUNTS[benchmark])
            for digest, images in value['order']:
                self.assertRegex(digest, r'^[0-9a-f]{64}$')
                self.assertIsInstance(images, list)
                for image in images:
                    self.assertFalse(Path(image).is_absolute())
                    self.assertNotIn('..', Path(image).parts)
            text = canonical(value)
            self.assertNotIn('"query"', text)
            self.assertNotIn('"response"', text)

    def test_tsv_image_payload_and_numbered_marker_whitespace_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'MMStar.tsv'
            fields = {'index': '7', 'question': '<image 1> Q with <image 2> internally?',
                      'A': 'first', 'B': 'second', 'C': 'nan', 'answer': 'B',
                      'image': image_uri('blue').split(',', 1)[1]}
            with source.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fields, delimiter='\t')
                writer.writeheader()
                writer.writerow(fields)
            expected = [normalized('<image> Q with <image 2> internally? Select from the following choices.\n(A) first\n(B) second', '(B)', ['7.jpg'])]
            with patch('zwz_eval.raw_platform.load_recipe', return_value=recipe('mmstar', expected)), \
                    patch('zwz_eval.normalization.read_json', return_value={'inputs': {'mmstar': identities(expected)}}):
                result = convert_platform('mmstar', source, root / 'out')
            self.assertEqual(rows(root / 'out/normalized.jsonl'), expected)
            self.assertTrue(result['matches_historical_annotations'])

    def test_known_wrong_image_identity_cannot_be_relabelled_to_inflate_parity(self):
        expected = [normalized('<image>Q', 'A', ['a.png']), normalized('<image>R', 'B', ['b.png'])]
        value = recipe('mmstar', expected)
        candidates = [{'row': {k: v for k, v in expected[0].items() if k != 'images'},
                       'hash': value['order'][0][0], 'assets': ['fixture-sha'], 'hints': ['b.png']}]
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            _order_candidates('mmstar', candidates, value)


if __name__ == '__main__':
    unittest.main()
