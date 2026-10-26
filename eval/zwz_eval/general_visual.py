"""Project evaluate_v3 Qwen + ACC profiles, preserving normalized task prompts.

Not a substitute upstream boxed-answer evaluator. Native judge templates and
their explicit empty-think assistant prefix are preserved via raw completions.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import io
import os
import tempfile
from pathlib import Path

from .api import CompatibleAPI
from .data import resolve_image, rows
from .io import atomic_json, digest, file_hash, read_json
from .references import load
from .protocols import PLATFORM_COUNTS
from .spatial import infer as journal_infer
from .structured import checked_predictions, judge_jobs

COUNTS = PLATFORM_COUNTS
EMPTY_THINK = '<think>\n\n</think>\n\n'
PIXEL_MIN, PIXEL_MAX, PATCH_SIZE = 256*256, 4096*4096, 14


def validate_row(row):
    queries = row.get('query')
    queries = [queries] if isinstance(queries, str) else queries
    if not isinstance(queries, list) or not queries or not all(isinstance(q, str) for q in queries):
        raise ValueError('Original query string/list required')
    if row.get('videos'):
        raise ValueError('These project profiles are image-only; no invented video adaptation')
    images = row.get('images')
    if not isinstance(images, list) or not images or len(images) > 10 or not all(isinstance(p, str) and p for p in images):
        raise ValueError('Original/prepared ordered image paths required')
    reference = row.get('parsed_response', row.get('response'))
    if reference is None or not str(reference).strip():
        raise ValueError('Missing benchmark gold response')
    if row.get('system') is not None and not isinstance(row['system'], str):
        raise ValueError('System prompt must be text')
    if 'LLM_JUDGE_PROMPT' in row and not isinstance(row['LLM_JUDGE_PROMPT'], str):
        raise ValueError('Per-item custom judge prompt must be text')


def preprocess(paths):
    """Exact source Qwen preprocessing boundary; no black-image fallback."""
    from qwen_vl_utils import process_vision_info
    vision = [{'type': 'image', 'image': str(p), 'min_pixels': PIXEL_MIN,
               'max_pixels': PIXEL_MAX} for p in paths]
    vision.append({'type': 'text', 'text': 'placeholder'})
    images, videos = process_vision_info([{'role': 'user', 'content': vision}], image_patch_size=PATCH_SIZE)
    if videos or images is None or len(images) != len(paths):
        raise ValueError('Image preprocessing did not preserve ordered image count')
    return images


def prepare(bench, source, target, image_root=None, image_map=None, preprocessor=None,
            *, materialize_images=False):
    if not isinstance(materialize_images, bool):
        raise ValueError('materialize_images must be boolean')
    target = Path(target)
    if target.exists():
        raise ValueError('Preparation target must not exist')
    target.parent.mkdir(parents=True, exist_ok=True)
    # Default preparation is annotation-only. Optional offline image exports are
    # staged on disk; neither mode publishes a partially prepared target.
    with tempfile.TemporaryDirectory(prefix='.zwz-prepare-', dir=target.parent) as temporary:
        staged = Path(temporary) / 'prepared'
        _prepare(bench, source, staged, image_root, image_map, preprocessor,
                 materialize_images=materialize_images, reference_root=target)
        if target.exists():
            raise ValueError('Preparation target was created concurrently')
        staged.rename(target)


def _prepare(bench, source, target, image_root=None, image_map=None, preprocessor=None,
             *, materialize_images=False, reference_root=None):
    source, target = Path(source), Path(target)
    if target.exists():
        raise ValueError('Preparation target must not exist')
    source_sha = file_hash(source)
    map_sha = file_hash(image_map) if image_map else None
    original = rows(source)
    if len(original) != COUNTS[bench]:
        raise ValueError(f'{bench}: expected {COUNTS[bench]} normalized source rows')
    remap = read_json(image_map) if image_map else {}
    if not isinstance(remap, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in remap.items()):
        raise ValueError('image-map must map exact original image paths to local paths')
    preprocessor = preprocessor or preprocess
    target.mkdir(parents=True)
    if materialize_images:
        (target / 'images').mkdir()
    fingerprints, cache, records, source_images = {}, {}, [], {}
    for index, original_row in enumerate(original):
        row = dict(original_row)
        image_paths = row.get('images')
        image_paths = [image_paths] if isinstance(image_paths, str) else image_paths
        if not isinstance(image_paths, list):
            raise ValueError('Ordered source images required')
        prepared_paths = []
        for original_path in image_paths:
            if not isinstance(original_path, str) or not original_path:
                raise ValueError('Nonempty source image paths required')
            if not materialize_images:
                # Lexical paths only: do not stat/resolve/read all NAS images.
                # Bundle-local raw images remain valid after an atomic rename.
                path = Path(remap.get(original_path, original_path))
                if not path.is_absolute():
                    path = Path(image_root or source.parent) / path
                if not path.is_absolute():
                    path = Path.cwd() / path
                destination = Path(reference_root or target)
                if not destination.is_absolute():
                    destination = Path.cwd() / destination
                # Do not collapse symlink/..: only the request-time filesystem
                # resolution can determine its original meaning.
                if ('..' not in path.parts and '..' not in destination.parts and
                        path.is_relative_to(destination.parent)):
                    prepared_paths.append(os.path.relpath(path, destination))
                else:
                    prepared_paths.append(str(path))
                continue
            path = resolve_image(remap.get(original_path, original_path), image_root or source.parent)
            key = str(path)
            if key not in cache:
                original_sha = file_hash(path)
                source_images[path] = original_sha
                image = preprocessor([path])[0]
                if file_hash(path) != original_sha:
                    raise ValueError('Source image changed during preprocessing')
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')  # lossless transport of source preprocessed pixels
                raw = buffer.getvalue()
                relative = 'images/' + digest({'source_sha256': original_sha, 'pixels':
                                               hashlib.sha256(raw).hexdigest()}) + '.png'
                (target / relative).write_bytes(raw)
                cache[key] = relative
                fingerprints[relative] = {'source_sha256': original_sha,
                    'prepared_sha256': hashlib.sha256(raw).hexdigest(),
                    'size': list(image.size), 'mode': image.mode}
            prepared_paths.append(cache[key])
        row['images'] = prepared_paths
        row['sample_uid'] = f'{bench}:row-{index}'
        validate_row(row)
        records.append(row)
    if file_hash(source) != source_sha or (image_map and file_hash(image_map) != map_sha):
        raise ValueError('Source annotations/image map changed during preparation')
    if any(file_hash(path) != sha for path, sha in source_images.items()):
        raise ValueError('Source image changed during preparation')
    atomic_json(target / 'records.json', records)
    versions = {}
    for name in ('qwen-vl-utils', 'Pillow', 'torch'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifest = {
        'benchmark': bench, 'source_sha256': source_sha,
        'records_sha256': file_hash(target / 'records.json'),
        'image_map_sha256': map_sha,
        'image_mode': 'materialized' if materialize_images else 'on-demand',
        'preprocessing': {'min_pixels': PIXEL_MIN, 'max_pixels': PIXEL_MAX,
                          'image_patch_size': PATCH_SIZE, 'versions': versions},
        'rows': len(records),
        'transform': ('source Qwen process_vision_info -> lossless PNG; original query/system/gold/judge prompt retained'
                      if materialize_images else
                      'annotation/path projection; source Qwen process_vision_info -> in-memory lossless PNG at request time'),
    }
    if materialize_images:
        manifest['assets'] = fingerprints
    atomic_json(target / 'preparation.json', manifest)


def preparation_mode(preparation):
    """Missing mode means the original materialized format, never raw passthrough."""
    mode = preparation.get('image_mode', 'materialized')
    if mode not in ('on-demand', 'materialized'):
        raise ValueError('Unknown platform image preparation mode')
    if mode == 'materialized' and not isinstance(preparation.get('assets'), dict):
        raise ValueError('Materialized preprocessing manifest requires image assets')
    if mode == 'on-demand' and 'assets' in preparation:
        raise ValueError('On-demand manifests must not claim materialized image hashes')
    return mode


def messages(row, bench, image_root):
    # Preserve legacy empty system role, per-query user turns, and source repair
    # behavior for unequal image placeholders. No joining turns or new suffixes.
    original = dict(row)
    layout = load('platform_messages').build_messages(original, default_system='', message_policy='legacy')
    images = iter(row['images'])
    result = []
    for message in layout:
        blocks = []
        for block in message['content']:
            if block['type'] == 'image':
                path = resolve_image(next(images), image_root)
                url = 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode('ascii')
                blocks.append({'type': 'image_url', 'image_url': {'url': url}})
            elif block['type'] == 'text':
                blocks.append(dict(block))
            else:
                raise ValueError('Unsupported multimodal block')
        result.append({'role': message['role'], 'content': blocks})
    return result


def on_demand_messages(row, bench, image_root):
    """Keep the old layout; decode/resize only inside the inference worker."""
    original = dict(row)
    layout = load('platform_messages').build_messages(original, default_system='', message_policy='legacy')
    images = iter(row['images'])
    result = []
    for message in layout:
        blocks = []
        for block in message['content']:
            if block['type'] == 'image':
                path = resolve_image(next(images), image_root)
                before = path.stat()
                image = preprocess([path])[0]
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                after = path.stat()
                fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
                if any(getattr(before, key) != getattr(after, key) for key in fields):
                    raise ValueError('Source image changed during request preprocessing')
                url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
                blocks.append({'type': 'image_url', 'image_url': {'url': url}})
            elif block['type'] == 'text':
                blocks.append(dict(block))
            else:
                raise ValueError('Unsupported multimodal block')
        result.append({'role': message['role'], 'content': blocks})
    return result


def infer(config, bench, directory, client=None):
    from .data import audit
    report = audit(bench, config['datasets'][bench], verify_images=False)
    mode = preparation_mode(report['preparation'])
    builder = on_demand_messages if mode == 'on-demand' else messages
    return journal_infer(config, bench, directory, client, message_builder=builder,
                         generation_options={'top_p': 1.0})


def tokenizer_manifest(config):
    identity = config.get('identity', {})
    value = identity.get('judge_tokenizer_path') or identity.get('judge_model_path')
    if not value or not Path(value).is_dir():
        raise ValueError('Same-profile native judge requires a local Qwen3.5-35B-A3B tokenizer directory')
    root = Path(value)
    names = ('tokenizer_config.json', 'tokenizer.json', 'chat_template.jinja', 'special_tokens_map.json',
             'added_tokens.json', 'vocab.json', 'merges.txt', 'tokenizer.model')
    files = {name: file_hash(root / name) for name in names if (root / name).is_file()}
    if 'tokenizer_config.json' not in files or not any(k in files for k in ('tokenizer.json', 'vocab.json', 'tokenizer.model')):
        raise ValueError('Incomplete judge tokenizer assets')
    return files


class NativeJudgeRenderer:
    def __init__(self, config, tokenizer=None):
        if tokenizer is None:
            tokenizer_manifest(config)
            from transformers import AutoTokenizer
            identity = config['identity']
            path = identity.get('judge_tokenizer_path') or identity['judge_model_path']
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.tokenizer = tokenizer

    def __call__(self, text):
        prompt = self.tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                                   tokenize=False, add_generation_prompt=True)
        return prompt + EMPTY_THINK  # Do not pass a new enable_thinking template kwarg.


def initial_score(row, raw):
    parser = load('platform')
    prediction = parser.parse_answer_tag(raw)
    truth = str(row.get('parsed_response', row.get('response', '')))
    correct = prediction.strip().lower() == truth.strip().lower()
    source = 'exact' if correct else None
    letters = ''.join(c for c in truth if c not in ['(', ')', '（', '）']).strip()
    if not correct and len(letters) == 1 and letters.upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' and len(prediction) < 201:
        correct = parser.parse_first_letter(prediction).upper() == parser.parse_first_letter(truth).strip().upper()
        if correct:
            source = 'short_mcq_first_letter'
    return prediction, truth, correct, source


def judge_prompt(row, parsed, truth):
    question = str(row.get('query', ''))
    if 'LLM_JUDGE_PROMPT' in row:
        return row['LLM_JUDGE_PROMPT'].format(question=question, gt=truth, parsed_pred=parsed[:16384])
    return (f'Your task is to judge whether the response expresses the same meaning as the answer of a question.\n'
            f'The question is: {question}\nThe answer is: {truth}\nThe response is: {parsed[:16384]}\n'
            'Please check and compare them and then judge. If the response is correct, your output should be Yes. '
            'Otherwise, your output should be No. Directly give me your output.')


def parse_judge(raw):
    result = raw.strip().lower()
    if 'verdict' in result and 'yes' in result:
        result = 'yes'
    return result.startswith(('yes', '\u662f', '\u4e00\u81f4', '1'))


def score(config, bench, directory, client=None, render=None):
    directory = Path(directory)
    expected, predictions = checked_predictions(config, bench, directory)
    scored, requests = {}, {}
    render = render or NativeJudgeRenderer(config)
    for uid, row in expected.items():
        # The platform's parser sees raw model_output, not the shared last-box normalizer.
        parsed, truth, correct, source = initial_score(row, predictions[uid]['raw_response'])
        scored[uid] = {'uid': uid, 'parsed_prediction': parsed, 'gold': truth,
                       'initial_correct': correct, 'correct': correct, 'source': source}
        if not correct:
            requests[uid] = render(judge_prompt(row, parsed, truth))
    client = client or CompatibleAPI(config['judge'], config['timeout'], config['retries'])

    def call(prompt):
        response = client.complete_prompt(prompt, temperature=0.0, max_tokens=96)
        raw = response['choices'][0]['text'].strip()
        return {'raw_response': raw, 'correct': parse_judge(raw)}

    judged = judge_jobs(config, directory, 'platform_judgments.jsonl', requests, call)
    for uid in requests:
        scored[uid].update(correct=judged[uid]['correct'], source='llm')
    initial = sum(item['initial_correct'] for item in scored.values())
    correct = sum(item['correct'] for item in scored.values())
    result = {'benchmark': bench, 'total': len(scored), 'initial_correct': initial,
              'correct': correct, 'strict_accuracy': initial/len(scored),
              'raw_accuracy_with_llm': correct/len(scored),
              'accuracy_with_llm': round(correct/len(scored), 4),
              'platform_reported_final_score': round(correct/len(scored), 4),
              'llm_judge_count': len(requests), 'llm_judge_correct': correct-initial,
              'judge_generation': 'native template + explicit empty-think prefix; 96 tokens'}
    atomic_json(directory / 'scores/scored_records.json', list(scored.values()))
    atomic_json(directory / 'scores/metrics.json', result)
    return result
