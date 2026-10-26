"""Installed-wheel-safe local project scaffolding; never download or run a model."""
from __future__ import annotations

import tempfile
from pathlib import Path

from .config import identifier, load_config, for_benchmark
from .io import atomic_json, read_json, run_lock
from .protocols import RESOURCES, PLATFORM_BENCHMARKS
from .workflow import selection


def initialize(directory, selected='all', *, model_id='my-qwen-model', model_path=None,
               data_root=None, output_root=None):
    supplied = Path(directory).expanduser()
    if supplied.is_symlink():
        raise ValueError('Initialization target must not be a symlink')
    target = supplied.resolve()
    identifier(model_id, 'model_id')
    template = read_json(RESOURCES / 'templates/config.json')
    manifest = read_json(RESOURCES / 'templates/datasets.json')
    names = selection(selected, manifest['datasets'])
    if target.exists() or target.is_symlink():
        raise ValueError('Initialization target already exists; existing projects are never overwritten')
    external_data = Path(data_root).expanduser().resolve() if data_root else None

    def relocate(value):
        if isinstance(value, str) and value.startswith('../data/'):
            return str(external_data / value[len('../data/'):]) if external_data else value[3:]
        if isinstance(value, str) and value.startswith('../'):
            return value[3:]
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        return value

    config, manifest = relocate(template), relocate(manifest)
    config['model_id'] = config['main']['model'] = model_id
    if model_path:
        config['identity']['model_path'] = str(Path(model_path).expanduser().resolve())
    if output_root:
        config['output_root'] = str(Path(output_root).expanduser().resolve())
    config['datasets'] = {name: config['datasets'][name] for name in names}
    manifest['datasets'] = {name: manifest['datasets'][name] for name in names}
    if not any(name in PLATFORM_BENCHMARKS for name in names):
        config.pop('judge_profiles', None)
    target.parent.mkdir(parents=True, exist_ok=True)
    with run_lock(target.parent / ('.init-' + target.name)):
        if target.exists() or target.is_symlink():
            raise ValueError('Initialization target appeared concurrently')
        with tempfile.TemporaryDirectory(prefix='.zwz-init-', dir=target.parent) as temporary:
            stage = Path(temporary) / 'project'
            stage.mkdir()
            atomic_json(stage / 'model.json', config)
            atomic_json(stage / 'datasets.json', manifest)
            checked = load_config(stage / 'model.json')
            for name in names:
                for_benchmark(checked, name)
            (stage / 'README.md').write_text(
                '# Local evaluation project\n\n'
                'Edit model.json (model/judge/service/output paths) and datasets.json '
                '(downloaded annotations/images). Paths are relative to each JSON file, '
                'not the terminal working directory. No data/model/service was created.\n\n'
                'Use the installed package from this directory:\n\n'
                '```bash\n'
                'python -m zwz_eval prepare-suite --manifest datasets.json '
                '--config model.json --output-config ready.json\n'
                'python -m zwz_eval doctor --config ready.json --phase infer --audit-data\n'
                '# One command: deploy, infer, switch to judge, score and clean up.\n'
                'python -m zwz_eval evaluate --config ready.json\n'
                '# For already deployed model APIs, add --services external.\n'
                'python -m zwz_eval summarize --root outputs\n'
                '```\n\n'
                'Use --bench BENCH or --bench BENCH_A,BENCH_B to select a serial queue. '
                'The command automatically uses each benchmark-specific judge profile. '
                'For a custom output_root, use that path with summarize. '
                'Never replace the reference dataset, prompts or parser just to make data pass. '
                'The default revision/hash placeholders must be replaced with actual provenance; '
                '/v1/models does not attest actual weights or chat templates.\n', encoding='utf-8')
            if target.exists():
                raise ValueError('Initialization target appeared concurrently')
            stage.rename(target)
    return {'directory': str(target), 'benchmarks': names,
            'config': str(target / 'model.json'), 'manifest': str(target / 'datasets.json'),
            'created_files': ['model.json', 'datasets.json', 'README.md'],
            'downloaded_data': False, 'started_services': False}
