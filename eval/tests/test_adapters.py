import copy
import io
import json
import os
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from pathlib import Path

from zwz_eval.downloads import safe_extract
from zwz_eval.execution import command_plan
from zwz_eval.io import atomic_json, append_jsonl, lock_contract
from zwz_eval.protocols import PROTOCOLS, MEM_SPLITS, mml_specs
from zwz_eval.scoring import aggregate, number
from zwz_eval.serving import plan


def configuration(root):
    return {'model_id': 'model', 'run_id': 'reference', 'output_root': str(root),
            'workers': 16, 'timeout': 600, 'retries': 3,
            'identity': {'model_path': '/models/checkpoint', 'judge_model_path': '/models/judge'},
            'hardware': {'gpus': 16, 'tp': 1},
            'main': {'model': 'model', 'base_urls': ['http://127.0.0.1:8000/v1'], 'api_key_env': 'MAIN_TEST_KEY'},
            'judge': {'model': 'judge', 'base_urls': ['http://127.0.0.1:9000/v1'], 'api_key_env': 'JUDGE_TEST_KEY'},
            'datasets': {b: {'root': str(root)} for b in PROTOCOLS}}


class Adapters(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                    if k != 'CUDA_VISIBLE_DEVICES'}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def test_all_plans_no_shell_or_embedded_key(self):
        config = configuration('/datasets')
        for bench in PROTOCOLS:
            for phase in ('infer', 'score'):
                if phase == 'infer' and bench in ('milebench', 'memlens'):
                    continue
                jobs = command_plan(config, bench, phase, Path('/outputs'))
                for job in jobs:
                    self.assertIsInstance(job['argv'], list)
                    self.assertTrue(Path(job['argv'][1]).is_file())
                    self.assertNotIn('--api-key', job['argv'])
                    self.assertNotIn('--judge-api-key', job['argv'])

    def test_mml_only_requested_tasks(self):
        jobs = command_plan(configuration('/data'), 'mmlongbench', 'infer', Path('/out'))
        self.assertEqual(len(jobs), 25)
        self.assertNotIn('counting', json.dumps(jobs))
        for job in jobs:
            argv = job['argv']
            self.assertEqual(argv[argv.index('--generation_max_length')+1], '2048')
            self.assertEqual(argv[argv.index('--seed')+1], '42')
            self.assertEqual(job['env']['ZWZ_STRICT_ERRORS'], '1')

    def test_mml_filtered_parsed_output_cannot_be_a_final_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'tasks/infoseek_K8'
            atomic_json(folder / 'fixture.json', {'data':[{'parsed_output':'valid'}]})
            atomic_json(folder / 'fixture.json.score', {'sub_em':100})
            append_jsonl(folder / 'fixture.json.cache', {'id':0,'parsed_output':'valid'})
            append_jsonl(folder / 'fixture.json.cache', {'id':1,'parsed_output':'','output':'nonempty truncated model response'})
            with self.assertRaisesRegex(ValueError, 'excluded empty outputs'):
                aggregate(configuration(root), 'mmlongbench', root)
            self.assertFalse((root / 'score.json').exists())

    def test_gui_thinking_environment_matches_backend(self):
        jobs = command_plan(configuration('/data'), 'ui_vision', 'infer', Path('/out'))
        self.assertEqual(len(jobs), 3)
        self.assertEqual(jobs[0]['env']['QWEN3_5_ENABLE_THINKING'], 'false')

    def test_memlens_macro_and_missing_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, split in enumerate(MEM_SPLITS):
                atomic_json(root / 'scores' / f'memlens-{split}' / 'judge_metrics.json',
                            {'overall': {'accuracy': i/4, 'total': 789}})
            self.assertEqual(aggregate(configuration(root), 'memlens', root)['macro'], 37.5)
            atomic_json(root / 'scores/memlens-256k/judge_metrics.json', {'overall': {'accuracy': 1, 'total': 788}})
            with self.assertRaises(ValueError):
                aggregate(configuration(root), 'memlens', root)

    def test_invalid_score(self):
        for value in (float('nan'), float('inf'), -1, 101, True):
            with self.assertRaises(ValueError):
                number(value)

    def test_summarize_rejects_changed_root_judge_journal(self):
        from contextlib import redirect_stdout, redirect_stderr
        from zwz_eval.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_contract(root, {'fixture': 'immutable'})
            for split in MEM_SPLITS:
                atomic_json(root / 'scores' / f'memlens-{split}' / 'judge_metrics.json',
                            {'overall': {'accuracy': 1, 'total': 789}})
            journal = root / 'verdicts.jsonl'
            append_jsonl(journal, {'uid': 'fixture', 'correct': True})
            result = aggregate(configuration(root), 'memlens', root)
            self.assertIn('verdicts.jsonl', result['source_outputs_sha256'])
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(['summarize', '--root', tmp]), 0)
                append_jsonl(journal, {'uid': 'changed', 'correct': False})
                self.assertEqual(main(['summarize', '--root', tmp]), 2)

    def mml_score_fixture(self, root):
        lock_contract(root, {'fixture': 'immutable'})
        for item in mml_specs():
            folder = root / 'tasks' / f"{item['family']}_K{item['k']}"
            atomic_json(folder / 'fixture.json', {'data': [{'id': 'fixture', 'parsed_output': 'Yes'}]})
            atomic_json(folder / 'fixture.json.score', {item['metric']: 100})
            append_jsonl(folder / 'fixture.json.cache', {'id': 'fixture', 'parsed_output': 'Yes'})
        return aggregate(configuration(root), 'mmlongbench', root)

    def test_summarize_rejects_modified_or_missing_native_mml_cache(self):
        from contextlib import redirect_stdout, redirect_stderr
        from zwz_eval.cli import main
        for mutation in ('append', 'edit', 'delete'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result = self.mml_score_fixture(root)
                relative = 'tasks/infoseek_K8/fixture.json.cache'
                self.assertIn(relative, result['source_outputs_sha256'])
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(main(['summarize', '--root', tmp]), 0)
                    cache = root / relative
                    if mutation == 'append':
                        append_jsonl(cache, {'id': 'unexpected', 'parsed_output': ''})
                    elif mutation == 'edit':
                        cache.write_text('{"id":"fixture","parsed_output":"No"}\n')
                    else:
                        cache.unlink()
                    self.assertEqual(main(['summarize', '--root', tmp]), 2)

    def test_summarize_rejects_old_mml_score_without_bound_native_caches(self):
        from contextlib import redirect_stdout, redirect_stderr
        from zwz_eval.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.mml_score_fixture(root)
            result['source_outputs_sha256'] = {key: value for key, value in result['source_outputs_sha256'].items()
                                               if not key.endswith('.cache')}
            atomic_json(root / 'score.json', result)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(['summarize', '--root', tmp]), 2)

    def test_serve_resource_constraints(self):
        config = configuration('/data')
        self.assertEqual(len(plan(config, 'memlens')['commands']), 1)
        config['hardware']['tp'] = 3
        with self.assertRaises(ValueError):
            plan(config, 'memlens')

    def test_text_only_judge_capacity_is_not_main_vision_capacity(self):
        config = configuration('/data')
        main = plan(config, 'spar', 'main')['commands'][0]['argv']
        judge = plan(config, 'spar', 'judge')['commands'][0]['argv']
        self.assertEqual(main[main.index('--max-model-len')+1], '131072')
        self.assertEqual(judge[judge.index('--max-model-len')+1], '32768')
        self.assertEqual(judge[judge.index('--limit-mm-per-prompt')+1], '{"image":0,"video":0}')

    def test_archive_traversal_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as out:
                out.writestr('../escape', 'bad')
            with self.assertRaises(ValueError):
                safe_extract(archive, root / 'target')
            self.assertFalse((root / 'target').exists())

    def test_archive_links_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'bad.tar'
            with tarfile.open(archive, 'w') as out:
                member = tarfile.TarInfo('alias')
                member.type = tarfile.SYMTYPE
                member.linkname = '/etc/passwd'
                out.addfile(member)
            with self.assertRaises(ValueError):
                safe_extract(archive, root / 'target')


if __name__ == '__main__':
    unittest.main()
