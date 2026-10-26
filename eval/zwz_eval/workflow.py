"""Downloaded data -> preserved adapters -> selected-benchmark execution.

This is orchestration, not a new prompt/scoring implementation. Large native
datasets are registered in place rather than copied or flattened into three fields.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config, for_benchmark, public_config, run_directory
from .data import audit
from .io import atomic_json, digest, file_hash, read_json, run_lock
from .protocols import PLATFORM_BENCHMARKS, PROTOCOLS

NATIVE = ('mmlongbench', 'screenspot_pro', 'ui_vision')
FORMATS = ('raw', 'normalized', 'prepared', 'native')


def selection(value, available=None):
    """Explicit, ordered selection; never silently ignore misspelled benchmarks."""
    available = list(PROTOCOLS if available is None else available)
    if value == 'all':
        if not available:
            raise ValueError('No benchmarks configured')
        return available
    names = [name.strip() for name in value.split(',')]
    if not names or any(not name for name in names):
        raise ValueError('Use a nonempty comma-separated benchmark list, or all')
    if len(names) != len(set(names)):
        raise ValueError('Duplicate selected benchmark')
    unknown = [name for name in names if name not in available or name not in PROTOCOLS]
    if unknown:
        raise ValueError('Unknown/unconfigured benchmarks: ' + ', '.join(unknown))
    return names


def local_path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('A nonempty local path is required')
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def source_identity(source):
    """Hash annotations, not every multi-GB archive/weight in a downloaded folder.

    Prepared image pixels are separately hashed by audit. No source modification,
    new downloads, recursive archive unpacking, or filename-only resume decisions.
    """
    source = Path(source)
    if source.is_file():
        return {source.name: file_hash(source)}
    if not source.is_dir():
        raise FileNotFoundError(f'Downloaded source is missing: {source}')
    suffixes = {'.json', '.jsonl', '.parquet', '.tsv', '.tfrecord'}
    result = {}
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source)
        if '.cache' in relative.parts or path.suffix.lower() not in suffixes or not path.is_file():
            continue
        result[str(relative)] = file_hash(path)
    if not result:
        raise ValueError(f'No extracted annotations found in {source}; extract the downloaded release first')
    return result


def one_file(source, names=(), pattern=None):
    """Recognize known release filenames; ambiguous snapshots fail closed."""
    source = Path(source)
    if source.is_file():
        return source
    found = [source / name for name in names if (source / name).is_file()]
    if not found and pattern:
        found = sorted(source.glob(pattern))
    if len(found) != 1:
        raise ValueError(f'Expected one annotation file under {source}; pass its exact path')
    return found[0]


def _preparation_audit(bench, data):
    """New on-demand platform inputs validate annotations, not all image bytes.

    Explicit audit remains independent. Old materialized receipts retain their
    original full-hash validation and are never converted or decoded on reuse.
    """
    manifest = Path(data['root']) / 'preparation.json'
    on_demand = (bench in PLATFORM_BENCHMARKS and manifest.is_file()
                 and read_json(manifest).get('image_mode') == 'on-demand')
    return audit(bench, data, verify_images=not on_demand)


def convert_data(bench, source, target, *, format='raw', image_root=None,
                 image_map=None, image_prefix=None, options=None, materialize_images=False):
    """Convert/audit transactionally; reuse only a matching complete receipt.

    raw = downloaded original schema; normalized = exact project annotations;
    prepared = already prepared bundle; native = benchmark-native release tree.
    Only the first two invoke converters. Native/prepared data stays in place.
    """
    if bench not in PROTOCOLS or format not in FORMATS:
        raise ValueError('Unknown benchmark or input format')
    if not isinstance(materialize_images, bool):
        raise ValueError('materialize_images must be a boolean')
    if materialize_images and (bench not in PLATFORM_BENCHMARKS or format not in ('raw', 'normalized')):
        raise ValueError('materialize_images applies only to new raw/normalized platform imports; never reprocess prepared data')
    source, target = Path(source).resolve(), Path(target).resolve()
    image_root = Path(image_root).resolve() if image_root else None
    image_map = Path(image_map).resolve() if image_map else None
    if options is not None and not isinstance(options, dict):
        raise ValueError('Dataset options must be an object')
    options = dict(options or {})
    if image_prefix is not None and (not isinstance(image_prefix, str) or not image_prefix.startswith('/')):
        raise ValueError('image-prefix must be the exact original absolute POSIX prefix')
    if image_map and not (format == 'normalized' and (bench in PLATFORM_BENCHMARKS or bench == 'mimic')):
        raise ValueError('--image-map is supported only for normalized platform/MIMIC imports')
    if image_prefix and not (format == 'normalized' and (bench in PLATFORM_BENCHMARKS or bench == 'mimic')):
        raise ValueError('--image-prefix is supported only for normalized platform/MIMIC imports')
    allowed = {'judge_profile'} | ({'mathruler_mode', 'mathruler_version'} if bench == 'mimic' else set())
    if set(options) - allowed:
        raise ValueError('Unknown dataset options: ' + ', '.join(sorted(set(options)-allowed)))
    if 'judge_profile' in options and (not isinstance(options['judge_profile'], str) or not options['judge_profile']):
        raise ValueError('judge_profile must be a nonempty profile name')
    if bench == 'mimic':
        options.setdefault('mathruler_mode', 'disabled')
        if options['mathruler_mode'] not in ('disabled', 'required'):
            raise ValueError('Explicit valid MIMIC mathruler_mode required')
        if options['mathruler_mode'] == 'required' and not options.get('mathruler_version'):
            raise ValueError('MIMIC required mathruler needs an explicit version')
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Use a separate preparation target, not the source or its ancestor/child')
    identity = {'benchmark': bench, 'format': format, 'source': str(source),
                'source_files': source_identity(source), 'image_root': str(image_root) if image_root else None,
                'image_map_sha256': file_hash(image_map) if image_map else None,
                'image_prefix': image_prefix, 'options': options}
    if materialize_images:
        identity['materialize_images'] = True
    target.parent.mkdir(parents=True, exist_ok=True)
    with run_lock(target.parent / ('.prepare-' + target.name)):
        receipt_path = target / 'dataset.json'
        if target.exists():
            if not receipt_path.is_file():
                raise ValueError('Target exists without a complete dataset receipt; choose a new target')
            receipt = read_json(receipt_path)
            if receipt.get('input') != identity:
                raise ValueError('Source/options changed; existing prepared data is not overwritten')
            actual = _preparation_audit(bench, receipt['dataset'])
            if actual['manifest_sha256'] != receipt['audit']['manifest_sha256']:
                raise ValueError('Prepared annotations/images changed; refusing stale data reuse')
            return {**receipt, 'reused': True}
        with tempfile.TemporaryDirectory(prefix='.zwz-convert-', dir=target.parent) as temporary:
            stage = Path(temporary) / 'bundle'
            stage.mkdir()
            data = _convert(bench, source, stage, format, image_root, image_map, image_prefix,
                            materialize_images=materialize_images)
            data.update(options)
            # Validate every annotation before publication. New platform images
            # are decoded/preprocessed only by the inference request that uses them.
            staged_audit = _preparation_audit(bench, data)
            if source_identity(source) != identity['source_files']:
                raise ValueError('Source annotations changed during conversion')
            if image_map and file_hash(image_map) != identity['image_map_sha256']:
                raise ValueError('Image map changed during conversion')
            # Only descriptors contain these temporary prefixes; converter rows
            # themselves use relative paths or stable original source paths.
            final_data = {}
            for key, value in data.items():
                if key in ('root', 'image_root'):
                    path = Path(value)
                    value = str(target / path.relative_to(stage)) if path.is_relative_to(stage) else str(path)
                final_data[key] = value
            # The receipt is published in the same atomic directory rename,
            # with stable final paths (including the audit's image keys).
            def relocate(value):
                if isinstance(value, str) and (value == str(stage) or value.startswith(str(stage) + '/')):
                    return str(target) + value[len(str(stage)):]
                if isinstance(value, list):
                    return [relocate(item) for item in value]
                if isinstance(value, dict):
                    return {relocate(key): relocate(item) for key, item in value.items()}
                return value
            final_audit = relocate(staged_audit)
            final_audit['manifest_sha256'] = digest({'sets': final_audit['sets'],
                'images': final_audit['image_sha256'], 'preparation': final_audit.get('preparation')})
            receipt = {'benchmark': bench, 'input': identity, 'dataset': final_data,
                       'audit': final_audit, 'reused': False}
            atomic_json(stage / 'dataset.json', receipt)
            if target.exists():
                raise ValueError('Preparation target appeared concurrently')
            stage.rename(target)
        return receipt


def _convert(bench, source, stage, format, image_root, image_map, image_prefix, *, materialize_images=False):
    prepared = stage / 'prepared'
    if format in ('native', 'prepared') or bench in NATIVE:
        if bench in NATIVE and format == 'normalized':
            raise ValueError(f'{bench} requires its native release layout, not flattened JSON')
        result = {'root': str(source)}
        if image_root:
            result['image_root'] = str(image_root)
        return result
    if bench in PLATFORM_BENCHMARKS:
        from .general_visual import prepare
        if format == 'raw':
            from .raw_platform import convert_platform
            normalized = stage / 'normalized'
            convert_platform(bench, source, normalized, image_root=image_root)
            source, image_root, image_map = normalized / 'normalized.jsonl', normalized, None
        else:
            from .normalization import verify_source
            verify_source(bench, source, image_prefix=image_prefix)
        prepare(bench, source, prepared, image_root, image_map, materialize_images=materialize_images)
        return {'root': str(prepared)}
    if bench == 'mimic':
        from .mimic import prepare
        if format == 'raw':
            from .raw_mimic import convert
            normalized = stage / 'normalized'
            convert(source, normalized, image_root=image_root)
            # Relative task/basename paths keep Counting's sample_uid stable
            # across transactional publication and other machines.
            prepare(normalized, prepared)
            return {'root': str(prepared), 'image_root': str(normalized / 'images')}
        from .normalization import verify_source
        verify_source(bench, source, image_prefix=image_prefix)
        prepare(source, prepared, image_root, image_map)
        return {'root': str(prepared), 'image_root': str(image_root or source)}
    if format == 'normalized':
        raise ValueError(f'{bench}: use raw for the official importer, or prepared for an existing audited bundle')
    if bench in ('mmsi', 'viewspatial', 'mindcube'):
        from .spatial import prepare
        if bench == 'mindcube':
            from .raw_mindcube import convert, NAME
            normalized = stage / 'normalized'
            report = convert(source, normalized, image_root=image_root)
            prepare(bench, normalized / NAME, prepared)
            return {'root': str(prepared), 'image_root': report['image_root']}
        names = {'mmsi': ('MMSI_Bench.parquet', 'data/MMSI_Bench.parquet'),
                 'viewspatial': ('ViewSpatial-Bench.json',),
                 'mindcube': ('MindCube_tinybench_raw_qa.jsonl',
                              'prompts/general/MindCube_tinybench_raw_qa.jsonl')}[bench]
        prepare(bench, one_file(source, names), prepared)
        original_images = image_root or (source.parent if source.is_file() else source)
        return {'root': str(prepared), **({'image_root': str(original_images)}
                                         if bench != 'mmsi' else {})}
    if bench in ('chartography', 'spar'):
        from .structured import prepare
        annotation = one_file(source, ('data.parquet',)) if bench == 'chartography' else source
        prepare(bench, annotation, prepared)
        return {'root': str(prepared), **({'image_root': str(image_root or annotation.parent)}
                                         if bench == 'chartography' else {})}
    if bench == 'erqa':
        from .erqa import prepare
        prepare(one_file(source, ('erqa.tfrecord', 'data/erqa.tfrecord')), prepared)
        return {'root': str(prepared)}
    if bench == 'milebench':
        from .data import prepare_milebench
        prepare_milebench(source, prepared)
        return {'root': str(prepared), 'image_root': str(image_root or source)}
    if bench == 'memlens':
        from .data import prepare_memlens
        if not image_root:
            raise ValueError('MemLens raw conversion requires the release --image-root')
        prepare_memlens(source, image_root, prepared)
        return {'root': str(prepared), 'image_root': str(image_root)}
    if bench == 'agentvista':
        from .data import prepare_agentvista
        prepare_agentvista(source, prepared)
        return {'root': str(prepared), 'image_root': str(prepared / 'images')}
    raise ValueError(f'No preserved converter for {bench}; no generic prompt will be invented')


def prepare_suite(manifest_path, selected='all', *, config_path=None, output_config=None, materialize_images=False):
    if not isinstance(materialize_images, bool):
        raise ValueError('materialize_images must be a boolean')
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    if not isinstance(manifest, dict) or set(manifest)-{'output_root', 'datasets'}:
        raise ValueError('Preparation manifest accepts only output_root and datasets')
    datasets = manifest.get('datasets')
    if not isinstance(datasets, dict) or not datasets or set(datasets)-set(PROTOCOLS):
        raise ValueError('Manifest datasets must be a nonempty map of known benchmarks')
    output_root = local_path(manifest['output_root'], path.parent)
    names = selection(selected, datasets)
    if bool(config_path) != bool(output_config):
        raise ValueError('Supply both --config and --output-config, or neither')
    template = load_config(config_path) if config_path else None
    if output_config and Path(output_config).resolve() == Path(config_path).resolve():
        raise ValueError('Write a new local config instead of replacing its template')
    result = {'benchmarks': names, 'datasets': {}, 'receipts': {}}
    for bench in names:
        entry = datasets[bench]
        allowed = {'source', 'format', 'image_root', 'image_map', 'image_prefix', 'options', 'materialize_images'}
        if not isinstance(entry, dict) or set(entry)-allowed or 'source' not in entry:
            raise ValueError(f'Invalid preparation entry: {bench}')
        receipt = convert_data(bench, local_path(entry['source'], path.parent), output_root / bench,
                               format=entry.get('format', 'raw'),
                               image_root=local_path(entry['image_root'], path.parent) if entry.get('image_root') else None,
                               image_map=local_path(entry['image_map'], path.parent) if entry.get('image_map') else None,
                               image_prefix=entry.get('image_prefix'),
                               options=entry.get('options'),
                               materialize_images=entry.get('materialize_images',
                                   materialize_images if bench in PLATFORM_BENCHMARKS else False))
        result['datasets'][bench] = receipt['dataset']
        result['receipts'][bench] = str(output_root / bench / 'dataset.json')
    if template is not None:
        config = public_config(template)
        config['datasets'] = {}
        for bench, data in result['datasets'].items():
            inherited = template.get('datasets', {}).get(bench, {})
            # Only named judge routing can be inherited; data paths and scoring
            # options come from this exact conversion, never stale template data.
            config['datasets'][bench] = {**({'judge_profile': inherited['judge_profile']}
                                          if 'judge_profile' in inherited else {}), **data}
        destination = Path(output_config).resolve()
        if destination.exists() and read_json(destination) != config:
            raise ValueError('Output config exists with different contents; choose a new filename')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.zwz-config-', dir=destination.parent) as temporary:
            validation = Path(temporary) / 'config.json'
            atomic_json(validation, config)
            checked = load_config(validation)
            for bench in checked['datasets']:
                for_benchmark(checked, bench)
        atomic_json(destination, config)
        result['config'] = str(destination)
    return result


def run_suite(config_path, selected='all', phase='all', continue_on_error=False):
    if phase not in ('infer', 'score', 'all', 'audit', 'plan'):
        raise ValueError('Invalid suite phase')
    config = load_config(config_path)
    names = selection(selected, config['datasets'])
    # Validate the entire selection before opening any expensive service calls.
    for bench in names:
        for_benchmark(config, bench)
    from .runner import run
    results = []
    for bench in names:
        item = {'benchmark': bench, 'phase': phase, 'output': str(run_directory(config, bench))}
        print(json.dumps({'starting': item}), flush=True)
        try:
            if phase == 'audit':
                item['audit'] = audit(bench, config['datasets'][bench])
            elif phase == 'plan':
                from .execution import command_plan
                effective = for_benchmark(config, bench)
                item['protocol'] = PROTOCOLS[bench]
                item['services_started'] = False
                item['backend'] = 'native' if bench in NATIVE or bench == 'agentvista' else 'in-process'
                if item['backend'] == 'native':
                    item['infer'] = command_plan(effective, bench, 'infer', run_directory(config, bench))
            else:
                run(config, bench, phase)
            item['status'] = 'complete'
        except Exception as exc:
            item.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            results.append(item)
            if not continue_on_error:
                break
            continue
        results.append(item)
    done = {item['benchmark'] for item in results}
    results.extend({'benchmark': name, 'phase': phase, 'status': 'not_started'}
                   for name in names if name not in done)
    report = {'model_id': config['model_id'], 'run_id': config['run_id'], 'phase': phase,
              'checked_at': datetime.now(timezone.utc).isoformat(), 'results': results,
              'complete': all(item['status'] == 'complete' for item in results)}
    if phase not in ('plan', 'audit'):
        report_path = Path(config['output_root']) / config['model_id'] / config['run_id'] / ('suite_' + phase + '.json')
        atomic_json(report_path, report)
    return report
