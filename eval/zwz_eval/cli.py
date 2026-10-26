"""Portable entry point. Unsupported integrations fail rather than silently improvise."""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

from .config import load_config, public_config, run_directory, for_benchmark
from .data import audit, prepare_memlens, prepare_milebench, prepare_agentvista
from .io import atomic_json, file_hash, lock_contract, run_lock
from .protocols import PROTOCOLS, RESOURCES, PLATFORM_BENCHMARKS


def main(argv=None):
    parser = argparse.ArgumentParser(prog='zwz-eval')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list')
    sub.add_parser('coverage')
    sub.add_parser('check-source')
    p = sub.add_parser('init', help='Create portable local configuration; never overwrite an existing project')
    p.add_argument('--directory', required=True)
    p.add_argument('--bench', default='all')
    p.add_argument('--model-id', default='my-qwen-model')
    p.add_argument('--model-path')
    p.add_argument('--data-root')
    p.add_argument('--output-root')
    p = sub.add_parser('doctor', help='Read-only path/dependency/optional data and endpoint checks')
    p.add_argument('--config', required=True)
    p.add_argument('--bench', default='all')
    p.add_argument('--phase', choices=('infer', 'score', 'all'), default='infer')
    p.add_argument('--audit-data', action='store_true')
    p.add_argument('--check-endpoints', action='store_true', help='Query /v1/models only; never request completions')
    p.add_argument('--check-tools', action='store_true', help='Probe AgentVista free search/visit with public example queries; no model calls or dataset downloads')
    for name in ('plan', 'audit', 'infer', 'score', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--bench', required=True, choices=PROTOCOLS)
    p = sub.add_parser('review', help='Separate diagnostic audit; does not replace primary scoring')
    p.add_argument('--config', required=True)
    p.add_argument('--bench', required=True, choices=('spar',))
    p = sub.add_parser('summarize')
    p.add_argument('--root', required=True)
    p = sub.add_parser('download')
    p.add_argument('--bench', required=True, choices=PROTOCOLS)
    p.add_argument('--target', required=True)
    p.add_argument('--revision', help='Explicit upstream dataset revision; resolved to immutable commit')
    p.add_argument('--execute', action='store_true', help='Without this, print source information only')
    p = sub.add_parser('extract')
    p.add_argument('--archive', required=True)
    p.add_argument('--target', required=True)
    for name in ('serve-plan', 'serve'):
        p = sub.add_parser(name, help=('Print local service commands without executing' if name == 'serve-plan'
                                     else 'Run owned local services in the foreground; Ctrl-C cleans up this pool'))
        p.add_argument('--config', required=True)
        p.add_argument('--bench', required=True, choices=PROTOCOLS)
        p.add_argument('--role', choices=('main', 'judge'), default='main')
        p.add_argument('--main-internal-port-base', type=int, default=25000)
        p.add_argument('--judge-internal-port-base', type=int, default=29000)
        p.add_argument('--internal-port-block-size', type=int, default=32)
        p.add_argument('--service-start-attempts', type=int, default=3)
        if name == 'serve':
            p.add_argument('--log-dir', required=True)
            p.add_argument('--startup-timeout', type=float, default=1800)
    sub.add_parser('release-check')
    p = sub.add_parser('prepare')
    p.add_argument('--bench', required=True, choices=('memlens', 'milebench', 'agentvista', 'mmsi', 'viewspatial', 'mindcube', 'chartography', 'spar', 'erqa', 'mimic') + PLATFORM_BENCHMARKS)
    p.add_argument('--source', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--image-root')
    p.add_argument('--image-map', help='Exact original-image-path to local-file JSON map for normalized MMStar/BabyVision imports')
    p.add_argument('--materialize-images', action='store_true', help='Platform tasks only: explicitly preprocess all images offline; default is on-demand inference')
    p = sub.add_parser('convert', help='Downloaded raw data -> preserved project input format')
    p.add_argument('--bench', required=True, choices=PROTOCOLS)
    p.add_argument('--source', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--format', choices=('raw', 'normalized', 'prepared', 'native'), default='raw')
    p.add_argument('--image-root')
    p.add_argument('--image-map')
    p.add_argument('--image-prefix', help='Exact old absolute image prefix for historical normalized annotation verification')
    p.add_argument('--materialize-images', action='store_true', help='Platform raw/normalized imports only: explicitly preprocess all images offline')
    p = sub.add_parser('prepare-suite', help='Prepare selected downloaded datasets from one manifest')
    p.add_argument('--manifest', required=True)
    p.add_argument('--bench', default='all', help='Comma-separated benchmark IDs, or all manifest entries')
    p.add_argument('--config', help='Model/service configuration template (optional)')
    p.add_argument('--output-config', help='Write a new runnable local config using prepared paths')
    p.add_argument('--materialize-images', action='store_true', help='Explicit offline image preprocessing for selected platform imports; other tasks retain their native preparation')
    p = sub.add_parser('run-suite', help='Run selected benchmarks with their existing prompts/scorers')
    p.add_argument('--config', required=True)
    p.add_argument('--bench', default='all', help='Comma-separated benchmark IDs, or all configured entries')
    p.add_argument('--phase', choices=('infer', 'score', 'all', 'audit', 'plan'), default='all')
    p.add_argument('--continue-on-error', action='store_true', help='Record failures and continue; final exit remains nonzero')
    p = sub.add_parser('evaluate', help='One-terminal deployment, inference, judge and scoring')
    p.add_argument('--config', required=True)
    p.add_argument('--bench', default='all')
    p.add_argument('--phase', choices=('infer', 'score', 'all'), default='all')
    p.add_argument('--services', choices=('local', 'external'), default='local')
    p.add_argument('--startup-timeout', type=float, default=1800)
    args = parser.parse_args(argv)
    try:
        if args.command == 'evaluate':
            if args.services == 'local':
                from .managed import evaluate
                report = evaluate(args.config, args.bench, args.phase, startup_timeout=args.startup_timeout)
            else:
                from .workflow import run_suite
                report = run_suite(args.config, args.bench, args.phase)
            print(json.dumps(report, indent=2))
            return 0 if report['complete'] else 2
        if args.command == 'init':
            from .onboarding import initialize
            report = initialize(args.directory, args.bench, model_id=args.model_id,
                                model_path=args.model_path, data_root=args.data_root, output_root=args.output_root)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.command == 'doctor':
            from .preflight import doctor
            report = doctor(args.config, args.bench, args.phase,
                            audit_data=args.audit_data, check_endpoints=args.check_endpoints,
                            check_tools=args.check_tools)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report['checks_passed'] else 2
        if args.command == 'convert':
            from .workflow import convert_data
            report = convert_data(args.bench, args.source, args.target, format=args.format,
                                  image_root=args.image_root, image_map=args.image_map,
                                  image_prefix=args.image_prefix, materialize_images=args.materialize_images)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.command == 'prepare-suite':
            from .workflow import prepare_suite
            report = prepare_suite(args.manifest, args.bench, config_path=args.config,
                                   output_config=args.output_config, materialize_images=args.materialize_images)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.command == 'run-suite':
            from .workflow import run_suite
            report = run_suite(args.config, args.bench, args.phase, args.continue_on_error)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report['complete'] else 2
        if args.command == 'coverage':
            from .scope import coverage, EXCLUDED
            print(json.dumps({'requested': coverage(), 'excluded': EXCLUDED}, indent=2))
            return 0
        if args.command == 'list':
            print(json.dumps(PROTOCOLS, indent=2))
            return 0
        if args.command == 'check-source':
            files = sorted(Path(__file__).parent.rglob('*.py'))
            for path in files:
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            print(f'PASS: {len(files)} Python sources parse; this is not a GPU execution test')
            return 0
        if args.command == 'release-check':
            from .release import check
            report = check()
            print(json.dumps(report, indent=2))
            return 0 if report['release_ready'] else 2
        if args.command == 'download':
            from .downloads import download
            print(json.dumps(download(args.bench, args.target, args.revision, args.execute), indent=2))
            return 0
        if args.command == 'extract':
            from .downloads import safe_extract
            safe_extract(args.archive, args.target)
            print(f'Extracted to {args.target}; original archive retained')
            return 0
        if args.command == 'summarize':
            records = []
            from .io import read_json
            for path in sorted(Path(args.root).rglob('score.json')):
                record = read_json(path)
                if record.get('status') == 'complete':
                    contract = path.parent / 'contract.json'
                    if not contract.exists() or record.get('contract_sha256') != read_json(contract)['sha256']:
                        raise ValueError(f'Score lacks matching run contract: {path}')
                    if record.get('benchmark') == 'mmlongbench':
                        from .protocols import mml_specs
                        tracked = record.get('source_outputs_sha256', {})
                        for item in mml_specs():
                            folder = path.parent / 'tasks' / f"{item['family']}_K{item['k']}"
                            caches = list(folder.glob('*.json.cache'))
                            if len(caches) != 1 or str(caches[0].relative_to(path.parent)) not in tracked:
                                raise ValueError(f'MMLongBench score lacks native cache integrity: {path}; recompute before summarizing')
                    for relative, expected in record.get('source_outputs_sha256', {}).items():
                        source = path.parent / relative
                        if not source.is_file() or file_hash(source) != expected:
                            raise ValueError(f'Score source output changed: {source}; recompute before summarizing')
                    records.append({'path': str(path), **record})
            print(json.dumps(records, indent=2))
            return 0
        if args.command == 'prepare':
            if args.materialize_images and args.bench not in PLATFORM_BENCHMARKS:
                raise ValueError('--materialize-images applies only to platform benchmark imports')
            if args.bench in PLATFORM_BENCHMARKS:
                from .general_visual import prepare
                prepare(args.bench, args.source, args.target, args.image_root, args.image_map,
                        materialize_images=args.materialize_images)
            elif args.bench == 'mimic':
                from .mimic import prepare
                prepare(args.source, args.target, args.image_root, args.image_map)
            elif args.bench == 'erqa':
                from .erqa import prepare
                prepare(args.source, args.target)
            elif args.bench in ('chartography', 'spar'):
                from .structured import prepare
                prepare(args.bench, args.source, args.target)
            elif args.bench in ('mmsi', 'viewspatial', 'mindcube'):
                from .spatial import prepare
                prepare(args.bench, args.source, args.target)
            elif args.bench == 'memlens':
                if not args.image_root:
                    raise ValueError('MemLens requires --image-root')
                prepare_memlens(args.source, args.image_root, args.target)
            elif args.bench == 'milebench':
                prepare_milebench(args.source, args.target)
            else:
                prepare_agentvista(args.source, args.target)
            return 0
        config = load_config(args.config)
        if args.bench not in config.get('datasets', {}):
            raise ValueError('Configure this benchmark dataset explicitly')
        config = for_benchmark(config, args.bench)
        directory = run_directory(config, args.bench)
        if args.command in ('serve-plan', 'serve'):
            from .serving import plan
            service_plan = plan(config, args.bench, args.role,
                main_internal_port_base=args.main_internal_port_base,
                judge_internal_port_base=args.judge_internal_port_base,
                internal_port_block_size=args.internal_port_block_size,
                service_start_attempts=args.service_start_attempts)
            if args.command == 'serve-plan':
                print(json.dumps(service_plan, indent=2))
                return 0
            from .service_lifecycle import ServicePool
            import signal
            pool = ServicePool(service_plan['commands'], args.log_dir,
                               readiness_timeout=args.startup_timeout,
                               max_start_retries=args.service_start_attempts - 1,
                               log_prefix=args.role)
            def terminate(signum, frame):
                raise KeyboardInterrupt
            old_handler = signal.signal(signal.SIGTERM, terminate)
            try:
                records = pool.start()
                print(json.dumps({'status': 'ready', 'role': args.role, 'services': records}), flush=True)
                pool.wait()
                return 0
            except KeyboardInterrupt:
                return 130
            finally:
                try:
                    pool.stop()
                finally:
                    signal.signal(signal.SIGTERM, old_handler)
        if args.command == 'plan':
            from .execution import command_plan
            print(json.dumps({'output': str(directory), 'protocol': PROTOCOLS[args.bench],
                              'config': public_config(config),
                              'infer': 'benchmark-specific in-process journal adapter' if args.bench in PLATFORM_BENCHMARKS or args.bench in ('milebench', 'memlens', 'mmsi', 'viewspatial', 'mindcube', 'chartography', 'spar', 'erqa', 'mimic') else command_plan(config, args.bench, 'infer', directory),
                              'score': 'benchmark-specific in-process scorer' if args.bench in PLATFORM_BENCHMARKS or args.bench in ('mmsi', 'viewspatial', 'mindcube', 'chartography', 'spar', 'erqa', 'mimic') else command_plan(config, args.bench, 'score', directory)}, indent=2))
            return 0
        if args.command == 'audit':
            manifest = audit(args.bench, config['datasets'][args.bench])
            print(json.dumps(manifest, indent=2))
            return 0
        from .runner import run
        print(run(config, args.bench, 'all' if args.command == 'run' else args.command))
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
