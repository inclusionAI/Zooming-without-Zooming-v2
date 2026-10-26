"""Auditable OpenAI-compatible static GUI adapter (not an agent roll-out).

Run with ``python -m zwz_eval.gui_api --help``. Native XML inference is
unchanged. This API transport records its own immutable contract and never
resumes a historical GLM directory without that contract.
"""
from __future__ import annotations

import argparse
import ast
import base64
import concurrent.futures
import hashlib
import importlib.metadata
import io
import json
import math
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

from .gui_grounding import COORDINATE_SPACES, PARSERS, parse_point, point_in_box, summarize
from .io import append_jsonl, atomic_json, digest, file_hash, lock_contract, read_journal, run_lock, repair_torn_tail
from .protocols import GUI_EXPECTED, RESOURCES


def system_prompt(coordinate_space, sent_size, prompt_contract='historical', native_tools=False):
    # Read the retained prompt literal without importing torch/transformers.
    source = RESOURCES / 'vendor/gui/official/screenspot-pro/models/qwen3_5.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_qwen3_5_prompt_msg')
    texts = [n.value for n in ast.walk(function) if isinstance(n, ast.Constant)
             and isinstance(n.value, str) and '# Tools' in n.value]
    if len(texts) != 1:
        raise ValueError('Retained GUI prompt literal is ambiguous')
    prompt = texts[0]
    if native_tools:
        if prompt_contract != 'explicit' or coordinate_space != 'normalized_1000':
            raise ValueError('Native API tools require explicit normalized_1000 coordinates')
        prompt = prompt.split('\n\n# Tools', 1)[0]
    if coordinate_space == 'image_pixels':
        w, h = sent_size
        prompt = prompt.replace('1000x1000', f'{w}x{h}')
        prompt += f'\nReturn coordinates in the provided image pixels: x in [0,{w}], y in [0,{h}].'
    elif coordinate_space == 'normalized_1':
        prompt = prompt.replace('1000x1000', '1x1')
        prompt += '\nReturn normalized fractional coordinates x,y in [0,1].'
    if prompt_contract not in ('historical', 'explicit'):
        raise ValueError('Unknown prompt contract')
    if prompt_contract == 'explicit':
        w, h = sent_size
        prompt += f'\nThe supplied image has dimensions {w}x{h} pixels.'
        if coordinate_space == 'normalized_1000':
            prompt += ('\nOutput coordinates on a normalized 0–1000 grid, NOT screenshot pixels. '
                       f'For a location (px, py) in the supplied image, use x=1000*px/{w} '
                       f'and y=1000*py/{h}. The top-left is (0,0), centre is (500,500), '
                       'and bottom-right is (1000,1000).')
        prompt += ('\nReturn exactly one computer_use left_click tool call with a two-number '
                   'coordinate [x,y]. Do not include reasoning, alternate coordinates, or multiple calls.')
    return prompt


def api_payload(args, image, instruction):
    messages = [{'role': 'system', 'content': system_prompt(args.coordinate_space, image['size'],
                args.prompt_contract, args.native_tools)}, {'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': image['url']}},
                {'type': 'text', 'text': instruction}]}]
    payload = {'model': args.model, 'temperature': 0, 'top_p': 1, 'max_tokens': args.max_tokens,
               'messages': messages, 'stream': True}
    if args.reasoning_effort:
        payload['reasoning_effort'] = args.reasoning_effort
    if args.native_tools:
        payload['tools'] = [{'type': 'function', 'function': {'name': 'computer_use',
            'description': 'Click the requested UI element using a normalized 0–1000 coordinate grid.',
            'parameters': {'type': 'object', 'properties': {
                'action': {'type': 'string', 'enum': ['left_click']},
                'coordinate': {'type': 'array', 'minItems': 2, 'maxItems': 2,
                    'items': {'type': 'number', 'minimum': 0, 'maximum': 1000}}},
                'required': ['action', 'coordinate'], 'additionalProperties': False}}}]
        payload['tool_choice'] = {'type': 'function', 'function': {'name': 'computer_use'}}
        payload['parallel_tool_calls'] = False
    return payload


def load_items(benchmark, root, image_root=None):
    root = Path(root).resolve()
    image_root = Path(image_root).resolve() if image_root else root / 'images'
    if benchmark == 'screenspot_pro':
        files = sorted((root / 'annotations').glob('*.json'))
    else:
        files = [root / 'annotations/element_grounding' / f'element_grounding_{s}.json'
                 for s in GUI_EXPECTED['ui_vision']]
    if not files:
        raise FileNotFoundError('No official GUI annotation files found')
    items, hashes = [], {}
    for source in files:
        data = json.loads(source.read_text())
        hashes[str(source.relative_to(root))] = file_hash(source)
        split = 'all' if benchmark == 'screenspot_pro' else source.stem.removeprefix('element_grounding_')
        if benchmark == 'ui_vision' and len(data) != GUI_EXPECTED[benchmark][split]:
            raise ValueError(f'Wrong denominator in {source.name}')
        for i, r in enumerate(data):
            ui = benchmark == 'ui_vision'
            path = (image_root / r['image_path' if ui else 'img_filename']).resolve()
            if not path.is_relative_to(image_root) or not path.is_file():
                raise ValueError(f'Missing/escaping image path: {path}')
            size = r['image_size' if ui else 'img_size']
            box = list(r['bbox'])
            if len(size) != 2 or min(size) <= 0 or len(box) != 4 or not all(math.isfinite(v) for v in box):
                raise ValueError('Invalid annotation dimensions/bbox')
            # Upstream UI-Vision sorts reversed endpoints; ScreenSpot does not.
            if ui:
                box = [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]
            items.append({'id': f'{split}-{i}' if ui else str(r['id']), 'subset': split,
                          'annotation': str(source.relative_to(root)), 'row_index': i,
                          'instruction': r['prompt_to_evaluate' if ui else 'instruction'],
                          'size': size, 'bbox': box, 'image': str(path)})
    if len(items) != sum(GUI_EXPECTED[benchmark].values()):
        raise ValueError('The official GUI item count differs from the project protocol')
    if len({item['id'] for item in items}) != len(items):
        raise ValueError('Duplicate GUI annotation IDs')
    return items, hashes


def image_payload(path, size, max_pixels):
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert('RGB')
        if list(image.size) != list(size):
            raise ValueError('Actual screenshot dimensions differ from annotation')
        # Reproduce native factor-32 smart_resize, with explicit resampling.
        h, w = image.height, image.width
        if max(h, w) / min(h, w) > 200:
            raise ValueError('Native smart_resize rejects aspect ratios above 200')
        rh, rw = max(32, round(h / 32) * 32), max(32, round(w / 32) * 32)
        if rh * rw > max_pixels:
            scale = math.sqrt(h * w / max_pixels)
            rh, rw = max(32, math.floor(h / scale / 32) * 32), max(32, math.floor(w / scale / 32) * 32)
        image = image.resize((rw, rh), Image.Resampling.BICUBIC)
        stream = io.BytesIO()
        image.save(stream, format='PNG')
    data = stream.getvalue()
    return {'url': 'data:image/png;base64,' + base64.b64encode(data).decode(),
            'sha256': hashlib.sha256(data).hexdigest(), 'original_sha256': file_hash(path),
            'size': [rw, rh], 'media_type': 'image/png', 'resample': 'BICUBIC'}


def collect_stream(lines):
    content, reasoning, calls, finish, usage, model = [], [], {}, None, None, None
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        if not line or not line.startswith('data:'):
            continue
        value = line[5:].strip()
        if value == '[DONE]':
            break
        event = json.loads(value)
        if event.get('error'):
            raise RuntimeError('API stream error: ' + str(event['error']))
        model = event.get('model', model)
        usage = event.get('usage') or usage
        for choice in event.get('choices', []):
            if choice.get('index', 0) != 0:
                raise ValueError('Only one completion is supported')
            delta = choice.get('delta') or {}
            content.append(delta.get('content') or '')
            reasoning.append(delta.get('reasoning_content') or delta.get('reasoning') or '')
            for call in delta.get('tool_calls', []):
                current = calls.setdefault(call.get('index', 0), {'name': '', 'arguments': ''})
                for key in ('name', 'arguments'):
                    current[key] += call.get('function', {}).get(key) or ''
            finish = choice.get('finish_reason') or finish
    if finish is None:
        raise RuntimeError('Stream closed without a finish_reason; not a completed prediction')
    raw = ''.join(content)
    if calls:
        if len(calls) != 1 or raw.strip():
            # A malformed/ambiguous model answer is incorrect, not an API error
            # that disappears from the denominator or gets sampled repeatedly.
            raw = json.dumps({'ambiguous_content': raw, 'tool_calls': list(calls.values())})
        else:
            call = dict(next(iter(calls.values())))
            try:
                call['arguments'] = json.loads(call['arguments'])
            except ValueError:
                pass  # Retain invalid arguments; primary parser will mark wrong.
            # Canonical serialization allows the exact historical newline parser.
            raw = '<tool_call>\n' + json.dumps(call, ensure_ascii=False) + '\n</tool_call>'
    return {'raw_response': raw, 'content': ''.join(content), 'reasoning': ''.join(reasoning),
            'native_tool_calls': calls, 'finish_reason': finish, 'usage': usage,
            'response_model': model, 'response_channel': 'tool_calls' if calls else 'content'}


def infer_one(item, args):
    import requests
    started = time.monotonic()
    image = image_payload(item['image'], item['size'], args.max_pixels)
    payload = api_payload(args, image, item['instruction'])
    # Store exact JSON request sans base64 bytes and authorization headers.
    request_record = json.loads(json.dumps(payload))
    request_record['messages'][1]['content'][0]['image_url']['url'] = 'sha256:' + image['sha256']
    record = {**item, 'gold_bbox': item['bbox'], 'request': request_record,
              'image_input': {k: v for k, v in image.items() if k != 'url'},
              'coordinate_space': args.coordinate_space, 'parser': args.parser}
    error = None
    for attempt in range(args.retries + 1):
        try:
            headers = {}
            if args.api_key_env and os.environ.get(args.api_key_env):
                headers['Authorization'] = 'Bearer ' + os.environ[args.api_key_env]
            with requests.post(args.api.rstrip('/') + '/chat/completions', json=payload,
                               headers=headers, stream=True, timeout=args.timeout) as response:
                response.raise_for_status()
                result = collect_stream(response.iter_lines(decode_unicode=False))
            point = parse_point(result['raw_response'], parser=args.parser,
                                coordinate_space=args.coordinate_space, sent_size=image['size'])
            return {**record, **result, 'point': point, 'correct': point_in_box(point, item['bbox'], item['size']),
                    'error': None, 'elapsed_s': time.monotonic() - started, 'completed_at': time.time()}
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            # Do not persist URLs/authentication headers from exception messages.
            error = {'type': type(exc).__name__, 'attempts': attempt + 1,
                     'http_status': getattr(getattr(exc, 'response', None), 'status_code', None)}
            if attempt < args.retries:
                time.sleep(min(10, 2 ** attempt))
    return {**record, 'point': None, 'correct': False, 'error': error,
            'elapsed_s': time.monotonic() - started, 'completed_at': time.time()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', choices=GUI_EXPECTED, required=True)
    parser.add_argument('--data-root', required=True, type=Path, help='One benchmark root containing annotations/ and images/')
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--out-root', required=True, type=Path)
    parser.add_argument('--api', required=True, help='OpenAI-compatible base URL ending in /v1')
    parser.add_argument('--model', required=True)
    parser.add_argument('--api-key-env', default='MAIN_API_KEY')
    parser.add_argument('--coordinate-space', choices=COORDINATE_SPACES, default='normalized_1000')
    parser.add_argument('--parser', choices=PARSERS, default='native_xml')
    parser.add_argument('--prompt-contract', choices=('historical', 'explicit'), default='historical',
                        help='explicit clarifies sent dimensions and units; changes historical prompt text')
    parser.add_argument('--native-tools', action='store_true',
                        help='Use API tools/tool_choice instead of manually emulated XML tool signatures')
    parser.add_argument('--reasoning-effort', choices=('low',))
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--max-pixels', type=int, default=8294400)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--limit', type=int, help='Diagnostic smoke only; never changes full denominators')
    args = parser.parse_args(argv)
    if min(args.workers, args.max_tokens, args.timeout) <= 0 or args.max_pixels < 1024 or args.retries < 0:
        parser.error('Invalid resource/transport limits')
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if args.parser == 'native_xml' and args.coordinate_space != 'normalized_1000':
        parser.error('native_xml parser requires normalized_1000; explicitly select tool_call for another coordinate space')
    if args.native_tools and (args.prompt_contract != 'explicit' or args.coordinate_space != 'normalized_1000' or args.parser != 'tool_call'):
        parser.error('--native-tools requires --prompt-contract explicit --coordinate-space normalized_1000 --parser tool_call')
    if '?' in args.api or '#' in args.api or '@' in args.api:
        parser.error('Do not embed credentials in the API URL')
    parsed_api = urlsplit(args.api)
    if parsed_api.scheme not in ('http', 'https') or not parsed_api.hostname:
        parser.error('Supply an http(s) OpenAI-compatible API base URL')
    items, hashes = load_items(args.bench, args.data_root, args.image_root)
    contract = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ('workers', 'limit', 'timeout', 'retries', 'out_root')}
    contract.update(protocol='gui-api-explicit-coordinates-v1', annotation_sha256=hashes,
                    items_sha256=digest(items), Pillow=importlib.metadata.version('Pillow'),
                    implementation_sha256={name: file_hash(Path(__file__).with_name(name))
                                           for name in ('gui_api.py', 'gui_grounding.py')},
                    system_prompt_sha256=digest(system_prompt(args.coordinate_space, (1000, 1000), args.prompt_contract, args.native_tools)))
    with run_lock(args.out_root):
        journal = args.out_root / 'predictions.jsonl'
        if journal.exists() and not (args.out_root / 'contract.json').exists():
            raise ValueError('Historical predictions have no matching API contract; use a new output directory')
        signature = lock_contract(args.out_root, contract)
        repair_torn_tail(journal)
        previous = read_journal(journal)
        by_id = {str(item['id']): item for item in items}
        checked_images = {}
        for record in previous:
            item = by_id[str(record['id'])]
            if record.get('error'):
                continue
            image_path = item['image']
            if image_path not in checked_images:
                checked_images[image_path] = file_hash(image_path)
            if record['image_input']['original_sha256'] != checked_images[image_path]:
                raise ValueError('Screenshot changed since the stored prediction; use a new output directory')
            point = parse_point(record['raw_response'], parser=args.parser,
                                coordinate_space=args.coordinate_space, sent_size=record['image_input']['size'])
            record['point'] = point
            record['correct'] = point_in_box(point, item['bbox'], item['size'])
        done = {str(r['id']) for r in previous if not r.get('error')}
        pending = [item for item in items if str(item['id']) not in done]
        if args.limit:
            pending = pending[:args.limit]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(infer_one, item, args): item for item in pending}
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                append_jsonl(journal, row)
                previous.append(row)
                print(json.dumps({'id': row['id'], 'correct': row['correct'], 'error': row['error']}), flush=True)
        report = summarize(items, previous, args.bench)
        report['contract_sha256'] = signature
        report['predictions_sha256'] = file_hash(journal) if journal.exists() else None
        atomic_json(args.out_root / 'metrics.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
