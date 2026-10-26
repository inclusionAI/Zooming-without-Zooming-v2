"""Image relocation must choose the same files before and during inference."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image

from zwz_eval.data import audit, image_root_for
from zwz_eval.execution import command_plan
from zwz_eval.inference import infer_records
from zwz_eval.io import atomic_json, read_json
from zwz_eval.protocols import MILE_TASKS, PLATFORM_BENCHMARKS, PROTOCOLS, mml_specs
from zwz_eval.runner import runtime_inputs
from zwz_eval.serving import plan
from test_adapters import configuration


class ImageRoots(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                    if k != 'CUDA_VISIBLE_DEVICES'}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.data = self.root / 'annotations'
        self.external = self.root / 'separate images'
        self.external.mkdir()
        Image.new('RGB', (8, 8), 'red').save(self.external / 'fixture.png')

    def test_defaults_for_every_profile_and_explicit_override(self):
        flat = set(PLATFORM_BENCHMARKS) | {'mmsi', 'viewspatial', 'mindcube', 'chartography', 'spar', 'erqa', 'mimic'}
        for bench in PROTOCOLS:
            with self.subTest(benchmark=bench):
                expected = self.data if bench in flat else self.data / 'images'
                if bench == 'mmlongbench':
                    expected = self.data / 'images/mmlb_image'
                self.assertEqual(image_root_for(bench, {'root': str(self.data)}), expected)
                self.assertEqual(image_root_for(bench, {'root': str(self.data), 'image_root': str(self.external)}), self.external)

    def test_mml_separate_images_used_by_audit_inference_and_server_allowlist(self):
        for item in mml_specs():
            annotation = self.data / 'mmlb_data' / item['file']
            annotation.parent.mkdir(parents=True, exist_ok=True)
            row = {'id': 'synthetic'}
            if item['family'] in ('infoseek', 'viquae'):
                row['image'] = 'fixture.png'
            elif item['family'] == 'vh_multi':
                row['ctxs'] = ['fixture.png']
            else:
                row.update(image_list=['fixture.png'], choices_image=['fixture.png'])
            annotation.write_text(json.dumps(row) + '\n')
        config = configuration(self.root / 'output')
        config['datasets']['mmlongbench'] = {'root': str(self.data), 'image_root': str(self.external)}
        checked = audit('mmlongbench', config['datasets']['mmlongbench'])
        self.assertEqual(set(checked['image_sha256']), {str(self.external / 'fixture.png')})
        for job in command_plan(config, 'mmlongbench', 'infer', self.root / 'run'):
            self.assertEqual(job['argv'][job['argv'].index('--image_file_root')+1], str(self.external))
        server = plan(config, 'mmlongbench')['commands'][0]['argv']
        self.assertEqual(server[server.index('--allowed-local-media-path')+1], str(self.external))
        with self.assertRaises(FileNotFoundError):
            audit('mmlongbench', {'root': str(self.data)})

    def test_gui_separate_images_used_by_both_command_plans_and_screen_audit(self):
        atomic_json(self.data / 'annotations/fixture.json', [
            {'id': index, 'img_filename': 'fixture.png'} for index in range(1581)])
        config = configuration(self.root / 'output')
        for bench in ('screenspot_pro', 'ui_vision'):
            config['datasets'][bench] = {'root': str(self.data), 'image_root': str(self.external)}
            for job in command_plan(config, bench, 'infer', self.root / bench):
                self.assertEqual(job['argv'][job['argv'].index('--screenspot_imgs')+1], str(self.external))
        checked = audit('screenspot_pro', config['datasets']['screenspot_pro'])
        self.assertEqual(set(checked['image_sha256']), {str(self.external / 'fixture.png')})

    def test_default_mile_image_root_matches_shared_formatter_after_audit(self):
        images = self.data / 'images'
        images.mkdir(parents=True)
        Image.new('RGB', (8, 8), 'blue').save(images / 'fixture.png')
        for task in MILE_TASKS:
            atomic_json(self.data / f'milebench_{task}_test.json', [
                {'sample_id': task, 'query': '<image>Question?', 'images': ['fixture.png'],
                 'response': 'Paris', 'system_prompt': 'Synthetic task',
                 'meta': {'question_type': 'open-ended'}}])
        config = configuration(self.root / 'outputs')
        config['datasets']['milebench'] = {'root': str(self.data)}
        checked = audit('milebench', config['datasets']['milebench'])
        self.assertEqual(set(checked['image_sha256']), {str(images / 'fixture.png')})
        calls = []
        class Client:
            def complete(self, messages, **kwargs):
                calls.append(messages)
                return {'choices': [{'message': {'content': 'Paris'}, 'finish_reason': 'stop'}]}
        directory = self.root / 'run'
        directory.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            infer_records(config, 'milebench', directory, client=Client())
        self.assertEqual(read_json(directory / 'inference_summary.json')['failed'], 0)
        self.assertEqual(len(calls), 29)
        self.assertTrue(all(any(block['type'] == 'image_url' for block in messages[-1]['content']) for messages in calls))
