import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from zwz_eval.downloads import SOURCES, download
from zwz_eval.protocols import RESOURCES
from zwz_eval.scope import REQUESTED


class DownloadCatalogue(unittest.TestCase):
    def test_all_25_have_documented_hf_sources_without_faking_access_or_parity(self):
        self.assertEqual(set(SOURCES), set(REQUESTED))
        self.assertEqual(len(SOURCES), 25)
        for name, source in SOURCES.items():
            with self.subTest(bench=name):
                self.assertRegex(source['repo'], r'^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')
                self.assertEqual(source['hf_url'], 'https://huggingface.co/datasets/' + source['repo'])
                self.assertTrue(source['note'])
                if source['observed_http_status'] == 200:
                    self.assertRegex(source['observed_revision'], r'^[0-9a-f]{40}$')
        self.assertEqual(SOURCES['chartography']['observed_http_status'], 401)
        self.assertIsNone(SOURCES['chartography']['observed_revision'])
        self.assertEqual(SOURCES['erqa']['provenance'], 'converted_mirror')
        self.assertEqual(SOURCES['o3']['repo'], 'm-Just/O3-Bench')
        text = (RESOURCES.parent.parent / 'docs/DOWNLOADS.md').read_text()
        documented = {line.split('`')[1] for line in text.splitlines() if line.startswith('| `')}
        self.assertEqual(documented, set(REQUESTED))

    def test_all_plans_are_offline_and_do_not_create_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(sys.modules, {'huggingface_hub': None}):
                for bench in SOURCES:
                    target = root / bench
                    info = download(bench, target)
                    self.assertFalse(info['executed'])
                    self.assertFalse(info['automatic_extraction'])
                    self.assertFalse(target.exists())
                    with self.assertRaisesRegex(ValueError, 'revision'):
                        download(bench, target, execute=True)

    def test_download_pins_commit_and_rejects_foreign_resume(self):
        commit = 'a'*40
        snapshots = []
        class API:
            def dataset_info(self, repo, revision):
                self.request = (repo, revision)
                return types.SimpleNamespace(sha=commit)
        def snapshot(**kwargs):
            snapshots.append(kwargs)
            (kwargs['local_dir'] / 'fixture.txt').write_text('fixture data')
        module = types.SimpleNamespace(HfApi=API, snapshot_download=snapshot)
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {'huggingface_hub': module}):
            root = Path(temporary) / 'download'
            info = download('vstar', root, revision='main', execute=True)
            self.assertEqual(info['revision'], commit)
            self.assertEqual(snapshots[0]['revision'], commit)
            self.assertEqual(json.loads((root / 'download_state.json').read_text()),
                             {'repo': 'craigwu/vstar_bench', 'revision': commit})
            self.assertIn('fixture.txt', info['files'])
            download('vstar', root, revision='main', execute=True)
            with self.assertRaisesRegex(ValueError, 'other data/revision'):
                download('hr4k', root, revision='main', execute=True)

    def test_access_failure_is_explicit_and_does_not_leave_a_completed_target(self):
        class API:
            def dataset_info(self, *args, **kwargs):
                raise RuntimeError('private service detail should not be echoed')
        module = types.SimpleNamespace(HfApi=API, snapshot_download=lambda **k: self.fail('must not download'))
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {'huggingface_hub': module}):
            root = Path(temporary) / 'download'
            with self.assertRaises(RuntimeError) as caught:
                download('chartography', root, revision='main', execute=True)
            self.assertIn('surgeai/chartography', str(caught.exception))
            self.assertNotIn('private service detail', str(caught.exception))
            self.assertFalse(root.exists())


if __name__ == '__main__':
    unittest.main()
