"""Project ERQA inference and conservative two-direction answer audit.

TensorFlow is needed only for importing original TFRecords. Portable API
inference/scoring uses prepared PNGs and original question/visual indices.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from .api import CompatibleAPI
from .io import atomic_json, file_hash
from .data import resolve_image
from .references import load
from .spatial import infer as journal_infer
from .structured import checked_predictions, judge_jobs

EXPECTED = 400


def gold(row):
    value = row.get('gold')
    if not isinstance(value, str):
        raise ValueError('ERQA requires original textual gold')
    choice = load('erqa').extract_choice(value) or value.strip().upper()
    if choice not in ('A', 'B', 'C', 'D'):
        raise ValueError('Unparseable ERQA gold option')
    return choice


def validate_row(row):
    gold(row)
    if not isinstance(row.get('question'), str) or not row['question']:
        raise ValueError('Original ERQA question required')
    images, indices = row.get('images'), row.get('visual_indices')
    if not isinstance(images, list) or not images or not all(isinstance(p, str) and p for p in images):
        raise ValueError('Original ordered PNG paths required')
    if len(images) > 16:
        raise ValueError('ERQA image count exceeds reference service limit')
    if not isinstance(indices, list) or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError('ERQA visual_indices must be nonnegative integers')
    if indices and len(indices) != len(images):
        raise ValueError('Image/index count mismatch: refusing silent zip truncation')


def prepare(source, target):
    """Run in a separate TensorFlow preparation environment if necessary."""
    source, target = Path(source), Path(target)
    if target.exists():
        raise ValueError('Preparation target must not exist')
    import tensorflow as tf
    from PIL import Image
    features = {'answer': tf.io.FixedLenFeature([], tf.string),
                'image/encoded': tf.io.VarLenFeature(tf.string),
                'question_type': tf.io.VarLenFeature(tf.string),
                'visual_indices': tf.io.VarLenFeature(tf.int64),
                'question': tf.io.FixedLenFeature([], tf.string)}
    records, assets = [], []
    for index, raw in enumerate(tf.data.TFRecordDataset(str(source))):
        parsed = tf.io.parse_single_example(raw, features)
        parsed = {key: (tf.sparse.to_dense(value) if isinstance(value, tf.SparseTensor) else value)
                  for key, value in parsed.items()}
        paths = []
        for position, image in enumerate(parsed['image/encoded'].numpy()):
            # Exact source decode/re-encode boundary, not a JPEG/PIL substitute.
            decoded = Image.fromarray(tf.io.decode_image(image).numpy())
            buffer = io.BytesIO()
            decoded.save(buffer, format='PNG')
            relative = f'images/{index:04d}-{position}.png'
            paths.append(relative)
            assets.append((relative, buffer.getvalue()))
        qtype = parsed['question_type'].numpy()
        row = {'id': index, 'question': parsed['question'].numpy().decode(),
               'gold': parsed['answer'].numpy().decode(), 'images': paths,
               'visual_indices': parsed['visual_indices'].numpy().tolist(),
               'question_type': qtype[0].decode() if len(qtype) else 'Unknown'}
        validate_row(row)
        records.append(row)
    if len(records) != EXPECTED:
        raise ValueError('ERQA requires 400 examples')
    target.mkdir(parents=True)
    for relative, raw in assets:
        path = target / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
    atomic_json(target / 'records.json', records)
    atomic_json(target / 'preparation.json', {
        'benchmark': 'erqa', 'source_sha256': file_hash(source),
        'records_sha256': file_hash(target / 'records.json'), 'rows': len(records),
        'transform': 'source TensorFlow decode_image -> PIL PNG; original question/indices/gold retained'})


def messages(row, bench, image_root):
    images = []
    for value in row['images']:
        path = resolve_image(value, image_root)
        images.append({'type': 'image_url', 'image_url': {'url':
            'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode('ascii')}})
    indices, question = row['visual_indices'], row['question']
    pairs = sorted(zip(images, indices), key=lambda item: item[1])
    parts = []
    if not indices or all(index == 0 for index in indices):
        parts.extend([image for image, _ in pairs] if indices else images)
        parts.append({'type': 'text', 'text': question})
    else:
        last = 0
        for image, index in pairs:
            if index == 0:
                parts.append(image)
            elif index <= len(question):
                text = question[last:index]
                if text:
                    parts.append({'type': 'text', 'text': text})
                parts.append(image)
                last = index
            else:
                parts.append(image)
        if last < len(question):
            parts.append({'type': 'text', 'text': question[last:]})
        if not parts:
            parts = [{'type': 'text', 'text': question}, *images]
    return [{'role': 'user', 'content': parts}]


def infer(config, directory, client=None):
    return journal_infer(config, 'erqa', directory, client, message_builder=messages)


def review_request(row, prediction, initial):
    if initial:
        prompt = f"""Audit only whether the MODEL RESPONSE explicitly selects the GOLD multiple-choice option.
Do NOT solve the question. Read QUESTION only to interpret option references.
Return "correct" only for an explicit, unambiguous selection of GOLD; "incorrect" for an explicit different selection; otherwise "ambiguous".
Return exactly JSON, without markdown:
{{"verdict":"correct|incorrect|ambiguous","selected":"A|B|C|D|null","reason":"brief"}}

QUESTION:
{row['question']}

GOLD: {row['gold']}

MODEL RESPONSE:
{prediction}"""
    else:
        # Preserve the source's literal backslash-n delimiters, not new newlines.
        prompt = ('Audit only whether a model response explicitly selects the GOLD multiple-choice option. '
                  'Do NOT solve the question. Read QUESTION only to interpret option references. '
                  'Return correct only for an explicit unambiguous selection of GOLD; incorrect for an '
                  'explicit different selection; ambiguous otherwise. Return exactly JSON, no markdown: '
                  '{"verdict":"correct|incorrect|ambiguous","selected":"A|B|C|D|null","reason":"brief"}'
                  f"\\n\\nQUESTION:\\n{row['question']}\\n\\nGOLD: {row['gold']}\\n\\nRESPONSE:\\n{prediction}")
    return {'messages': [{'role': 'user', 'content': prompt}], 'initial_correct': initial,
            'max_tokens': 160 if initial else 1024}


def score(config, directory, client=None):
    directory = Path(directory)
    expected, predictions = checked_predictions(config, 'erqa', directory)
    initial = {uid: load('erqa').extract_choice(predictions[uid]['model_answer']) == gold(row)
               for uid, row in expected.items()}
    requests = {uid: review_request(row, predictions[uid]['model_answer'], initial[uid])
                for uid, row in expected.items()}
    client = client or CompatibleAPI(config['judge'], config['timeout'], config['retries'])

    def call(request):
        parser = load('erqa_initial_review' if request['initial_correct'] else 'erqa_wrong_review').parse_json
        attempts = []
        for _ in range(3):
            try:
                response = client.complete(request['messages'], thinking=False, max_tokens=request['max_tokens'])
            except Exception as error:
                error.judge_attempts = attempts
                raise
            raw = response['choices'][0]['message'].get('content') or ''
            attempts.append(raw)
            parsed = parser(raw)
            if parsed:
                return {'verdict': parsed, 'raw_response': raw, 'judge_attempts': attempts}
        error = ValueError('Unparseable ERQA audit verdict, not semantic ambiguity')
        error.judge_attempts = attempts
        raise error

    reviews = judge_jobs(config, directory, 'erqa_reviews.jsonl', requests, call)
    scored = []
    for uid, row in expected.items():
        label = reviews[uid]['verdict']['verdict']
        corrected = (label != 'incorrect') if initial[uid] else (label == 'correct')
        scored.append({'uid': uid, 'initial_correct': initial[uid], 'verdict': label,
                       'reviewed_correct': corrected,
                       'extracted': load('erqa').extract_choice(predictions[uid]['model_answer'])})
    result = {'benchmark': 'erqa', 'total': len(scored),
              'initial_correct': sum(initial.values()),
              'initial_accuracy': sum(initial.values())/len(scored),
              'added': sum(not r['initial_correct'] and r['reviewed_correct'] for r in scored),
              'subtracted': sum(r['initial_correct'] and not r['reviewed_correct'] for r in scored),
              'ambiguous': sum(r['verdict'] == 'ambiguous' for r in scored),
              'primary_accuracy': sum(r['reviewed_correct'] for r in scored)/len(scored)}
    atomic_json(directory / 'scores/scored_records.json', scored)
    atomic_json(directory / 'scores/metrics.json', result)
    return result
