"""CPU-only checks: device allocation changes, never prompts/model parameters."""
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zwz_eval.cli import main
from zwz_eval.config import gpu_groups
from zwz_eval.protocols import RESOURCES
from zwz_eval.serving import plan


def configuration(gpus=2):
    return {
        'model_id': 'fixture', 'run_id': 'cpu-test', 'output_root': './outputs',
        'identity': {'model_path': '/fixture/model', 'judge_model_path': '/fixture/judge'},
        'hardware': {'gpus': gpus, 'tp': 1, 'judge_tp': 2},
        'main': {'model': 'fixture', 'base_urls': ['http://127.0.0.1:23100/v1',
                                                  'http://127.0.0.1:23101/v1']},
        'judge': {'model': 'judge', 'base_urls': ['http://127.0.0.1:24100/v1']},
        'datasets': {'spar': {'root': '/fixture/data'}},
    }


class GPUAllocation(unittest.TestCase):
    def test_unset_preserves_historical_sixteen_card_mapping(self):
        self.assertEqual(gpu_groups(16, 1, environment={}), [[str(i)] for i in range(16)])
        self.assertEqual(gpu_groups(16, 2, environment={}),
                         [[str(i), str(i + 1)] for i in range(0, 16, 2)])

    def test_physical_identifiers_and_order_are_preserved(self):
        env = {'CUDA_VISIBLE_DEVICES': '8,9'}
        self.assertEqual(gpu_groups(2, 1, environment=env), [['8'], ['9']])
        self.assertEqual(gpu_groups(2, 2, environment=env), [['8', '9']])
        self.assertEqual(gpu_groups(2, 1, environment={'CUDA_VISIBLE_DEVICES': '9, 8, 7'}),
                         [['9'], ['8']])
        self.assertEqual(env, {'CUDA_VISIBLE_DEVICES': '8,9'})

    def test_uuid_and_mig_identifiers_pass_through(self):
        for visible in ('GPU-deadbeef,GPU-cafefeed',
                        'MIG-GPU-deadbeef/1/0,MIG-GPU-deadbeef/2/0'):
            self.assertEqual(gpu_groups(2, 2, environment={'CUDA_VISIBLE_DEVICES': visible}),
                             [visible.split(',')])

    def test_empty_disabled_malformed_duplicates_and_short_allocations_rejected(self):
        for visible in ('', ' ', '-1', '0,-1', '8', '8,', ',8', '8,,9', '8,8',
                        '08,8', 'GPU-dead,GPU-dead', 'all', '8;9', '8,unknown'):
            with self.subTest(visible=visible), self.assertRaises(ValueError):
                gpu_groups(2, 1, environment={'CUDA_VISIBLE_DEVICES': visible})
        for gpus, tp in ((True, 1), (2, False), (0, 1), (2, 0), (3, 2), (1, 2)):
            with self.subTest(gpus=gpus, tp=tp), self.assertRaises(ValueError):
                gpu_groups(gpus, tp, environment={})

    def test_service_plan_changes_only_device_assignment(self):
        config = configuration()
        before = copy.deepcopy(config)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1'}):
            original = {role: plan(config, 'spar', role)['commands'] for role in ('main', 'judge')}
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '8,9'}):
            mapped = {role: plan(config, 'spar', role)['commands'] for role in ('main', 'judge')}
        self.assertEqual([j['env']['CUDA_VISIBLE_DEVICES'] for j in mapped['main']], ['8', '9'])
        self.assertEqual(mapped['judge'][0]['env']['CUDA_VISIBLE_DEVICES'], '8,9')
        for role in original:
            for old, new in zip(original[role], mapped[role]):
                old['env'].pop('CUDA_VISIBLE_DEVICES')
                new['env'].pop('CUDA_VISIBLE_DEVICES')
                self.assertEqual(old, new)
        self.assertEqual(config, before)

    def test_invalid_dry_plan_fails_before_service_factory(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            cfg = Path(tmp) / 'config.json'
            cfg.write_text(json.dumps(configuration()))
            for visible in ('', '8', '8,8'):
                with self.subTest(visible=visible), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': visible}), \
                        patch('zwz_eval.service_lifecycle.ServicePool') as pool, \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(['serve-plan', '--config', str(cfg), '--bench', 'spar']), 2)
                    pool.assert_not_called()
            self.assertFalse((Path(tmp) / 'outputs').exists())

    def test_gui_actual_group_expression_and_initializer_keep_selected_physical_ids(self):
        path = RESOURCES / 'vendor/gui/scripts/eval_grounding_topology.py'
        tree = ast.parse(path.read_text())
        main_function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        assignment = next(n for n in ast.walk(main_function) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'groups' for t in n.targets))
        expression = ast.Expression(body=assignment.value)
        args = SimpleNamespace(num_gpu=2, tp=1)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '8,9'}):
            groups = eval(compile(expression, str(path), 'eval'), {'gpu_groups': gpu_groups, 'args': args})
        self.assertEqual(groups, [['8'], ['9']])
        initializer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'initializer')
        devices = []
        fake_os = SimpleNamespace(environ={})
        namespace = {'os': fake_os, 'torch': SimpleNamespace(cuda=SimpleNamespace(set_device=devices.append)),
                     'build_model': lambda _: object()}
        exec(compile(ast.Module(body=[initializer], type_ignores=[]), str(path), 'exec'), namespace)
        for group in groups:
            with contextlib.redirect_stdout(io.StringIO()):
                namespace['initializer'](group, args)
            self.assertEqual(fake_os.environ['CUDA_VISIBLE_DEVICES'], group[0])
        self.assertEqual(devices, [0, 0], 'each child selects logical 0 inside its own physical allocation')
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '8'}), self.assertRaises(ValueError):
            eval(compile(expression, str(path), 'eval'), {'gpu_groups': gpu_groups, 'args': args})


if __name__ == '__main__':
    unittest.main()
