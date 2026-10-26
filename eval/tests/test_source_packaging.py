import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zwz_eval.packaging import security_issue, source_files


class SourcePackaging(unittest.TestCase):
    def test_actual_source_tree_has_no_integrity_or_export_errors(self):
        from zwz_eval.release import check
        self.assertEqual(check()['source_integrity_errors'], [])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'project'
        self.root.mkdir()
        (self.root / 'README.md').write_text('Public source\n')

    def test_weights_secrets_and_previous_manifests_are_not_selected(self):
        (self.root / '.env').write_text('PRIVATE_VALUE=not-exportable')
        (self.root / 'SOURCE_MANIFEST.json').write_text('{}')
        (self.root / 'RELEASE_CHECK.json').write_text('{}')
        (self.root / 'models').mkdir()
        (self.root / 'models/weights.safetensors').write_bytes(b'not source')
        self.assertEqual([p.name for p in source_files(self.root)], ['README.md'])

    def test_unreviewed_binary_in_source_folder_is_rejected(self):
        (self.root / 'docs').mkdir()
        (self.root / 'docs/leftover.parquet').write_bytes(b'not source')
        with self.assertRaisesRegex(ValueError, 'Unreviewed file'):
            source_files(self.root)

    def test_local_user_configuration_is_not_exported(self):
        local = self.root / 'local/my-eval/model.json'
        local.parent.mkdir(parents=True)
        local.write_text(json.dumps({'model_path': '/private/user/checkpoint', 'api_key': 'sk-' + 'a'*40}))
        self.assertNotIn(local, source_files(self.root))
        self.assertTrue(local.exists())

    @unittest.skipUnless(shutil.which('git'), 'git is needed to check actual ignore behavior')
    def test_git_ignores_root_storage_but_not_vendored_model_code(self):
        ignore = Path(__file__).resolve().parents[1] / '.gitignore'
        (self.root / '.gitignore').write_text(ignore.read_text())
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True, capture_output=True)
        def ignored(relative):
            result = subprocess.run(['git', '-c', 'core.excludesFile=/dev/null', 'check-ignore',
                                     '--no-index', relative], cwd=self.root, capture_output=True, text=True)
            self.assertIn(result.returncode, (0, 1), result.stderr)
            return result.returncode == 0
        self.assertTrue(ignored('local/my-eval/model.json'))
        self.assertTrue(ignored('models/private/config.json'))
        self.assertTrue(ignored('data/images/example.png'))
        self.assertFalse(ignored('zwz_eval/resources/vendor/gui/official/screenspot-pro/models/qwen3_5.py'))

    def test_private_distribution_data_is_not_implicitly_reexported_as_public_source(self):
        path = self.root / 'bundled_data/authorized-private-release/example.parquet'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'private authorized data, not public code')
        self.assertNotIn(path, source_files(self.root))

    def test_only_shared_public_entrypoint_is_exported(self):
        scripts = self.root / 'scripts'
        scripts.mkdir()
        public = scripts / 'evaluate.sh'
        public.write_text('set -eu\n')
        private = scripts / 'machine_specific_retry.py'
        private.write_text('PRIVATE_LOCAL_CONFIGURATION = True\n')
        selected = source_files(self.root)
        self.assertIn(public, selected)
        self.assertNotIn(private, selected)
        self.assertTrue(private.exists())

    def test_nested_models_python_code_is_exported_not_treated_as_weights(self):
        path = self.root / 'zwz_eval/resources/vendor/gui/models/qwen.py'
        path.parent.mkdir(parents=True)
        path.write_text('MODEL_FAMILY = "qwen"\n')
        self.assertIn(path, source_files(self.root))

    def test_missing_tracked_vendor_file_is_detected(self):
        from zwz_eval import release
        resources = self.root / 'resources'
        (resources / 'vendor').mkdir(parents=True)
        (resources / 'vendor/SOURCES.json').write_text(json.dumps({'files': [
            {'path': 'models/missing.py', 'packaged_sha256': 'missing'}]}))
        with patch.object(release, 'RESOURCES', resources):
            self.assertTrue(any('Missing tracked vendor file' in error
                                for error in release.check()['source_integrity_errors']))

    def test_private_material_in_documentation_is_rejected(self):
        examples = ['sk-' + 'a'*40, '-----BEGIN ' + 'RSA ' + 'PRIVATE KEY-----',
                    '/' + 'mnt/' + 'nas/' + 'someone/private.py']
        for value in examples:
            with self.subTest(value_kind=value[:8]):
                (self.root / 'README.md').write_text(value)
                with self.assertRaises(ValueError):
                    source_files(self.root)

    def test_personal_paths_run_ids_and_machine_names_are_rejected(self):
        values = ['/' + 'Users/' + 'example/private.py', '/' + 'home/' + 'example/model',
                  '/' + 'mnt/' + 'storage/model', 'global_' + 'step_12_merged',
                  'run_' + '20261025' + '_123456_eval',
                  'workflow_' + '123456', 'experiment-' + '123456',
                  'job_' + '123456', 'aistudio-' + '123456']
        for value in values:
            with self.subTest(kind=type(value).__name__):
                self.assertIsNotNone(security_issue(value))

    def test_private_names_in_archive_members_are_rejected(self):
        path = self.root / 'docs' / 'private-fixture-run_notes.md'
        path.parent.mkdir()
        path.write_text('No private content here.')
        with patch.dict('os.environ', {'ZWZ_PRIVATE_IDENTIFIERS': 'private-fixture-run'}), \
             self.assertRaisesRegex(ValueError, 'source member name'):
            source_files(self.root)

    def test_external_private_identifier_list_is_case_insensitive_and_not_echoed(self):
        values = ['Private-Fixture-Person', 'prefix-private-fixture-run-suffix',
                  'https://private-fixture-host.example/v1']
        configured = ' private-fixture-person , private-fixture-run,private-fixture-host.example,, '
        with patch.dict('os.environ', {'ZWZ_PRIVATE_IDENTIFIERS': configured}):
            for value in values:
                self.assertEqual(security_issue(value), 'Configured private identifier')
                (self.root / 'README.md').write_text(value)
                with self.assertRaises(ValueError) as raised:
                    source_files(self.root)
                self.assertNotIn(value, str(raised.exception))
        with patch.dict('os.environ', {'ZWZ_PRIVATE_IDENTIFIERS': ''}):
            self.assertIsNone(security_issue('Private-Fixture-Person'))

    def test_bare_credentials_in_named_assignments_are_rejected(self):
        token = 'aB3dE6gH9jK2mN5pQ8sT1vW4yZ7cF0iL'
        for value in ['API_KEY=' + token, '"api_key": "' + token + '"',
                      'password = "' + token + '"', 'access_token="' + token + '"']:
            self.assertEqual(security_issue(value), 'Possible credential assignment')

    def test_public_placeholders_urls_and_image_names_are_allowed(self):
        values = ['../models/my-checkpoint', '/path/to/checkpoint', '/old/root/example.png',
                  'http://127.0.0.1:8000/v1', 'https://huggingface.co/org/dataset',
                  'https://github.com/org/repository', 'MAIN_API_KEY=EMPTY',
                  '{"api_key_env": "MAIN_API_KEY"}', 'diagram@2x.webp',
                  'api_key=build_environment_configuration',
                  'financial-ri' + 'sk-' + 'A' * 40 + '.png']
        for value in values:
            self.assertIsNone(security_issue(value), value)

    def test_credential_boundary_does_not_exempt_data_files(self):
        ordinary_filename = 'financial-ri' + 'sk-' + 'A' * 40 + '.png'
        self.assertIsNone(security_issue(json.dumps({'image': ordinary_filename})))
        self.assertEqual(security_issue(json.dumps({'image': 'sk-' + 'A' * 40 + '.png'})),
                         'Possible credential')

    def test_source_symlink_is_not_followed(self):
        (self.root / 'docs').symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            source_files(self.root)

    def test_rebuilding_an_unpacked_archive_has_no_self_hash_or_duplicate_paths(self):
        path = Path(__file__).resolve().parents[1] / 'tools/build_source_archive.py'
        spec = importlib.util.spec_from_file_location('source_archive_builder_fixture', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = self.root.parent / 'first.tar.gz'
        report = {'source_integrity_errors': [], 'release_ready': False}
        with patch.object(module, 'ROOT', self.root), patch.object(module, 'check', return_value=report), \
             patch.object(sys, 'argv', ['build', '--draft', '--output', str(output)]), contextlib.redirect_stdout(io.StringIO()):
            module.main()
        with tarfile.open(output) as source:
            source.extractall(self.root.parent / 'unpacked', filter='data') if sys.version_info >= (3, 12) else source.extractall(self.root.parent / 'unpacked')
        unpacked = self.root.parent / 'unpacked/eval'
        second = self.root.parent / 'second.tar.gz'
        with patch.object(module, 'ROOT', unpacked), patch.object(module, 'check', return_value=report), \
             patch.object(sys, 'argv', ['build', '--draft', '--output', str(second)]), contextlib.redirect_stdout(io.StringIO()):
            module.main()
        with tarfile.open(second) as source:
            names = source.getnames()
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue(all(name.startswith('eval/') for name in names))
            manifest = json.load(source.extractfile('eval/SOURCE_MANIFEST.json'))
            self.assertNotIn('SOURCE_MANIFEST.json', manifest)
            self.assertNotIn('RELEASE_CHECK.json', manifest)
            for relative, expected in manifest.items():
                self.assertEqual(hashlib.sha256(source.extractfile('eval/'+relative).read()).hexdigest(), expected)
