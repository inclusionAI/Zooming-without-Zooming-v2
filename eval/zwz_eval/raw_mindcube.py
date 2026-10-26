"""MindCube Tiny raw_qa prompt generation, not cognitive-map scaffolding.

Equivalent to the official RawQATemplate in
https://github.com/mll-lab-nu/MindCube/blob/main/src/prompt_generation/templates.py
The preserved project's 1050 actual input prompts all use this exact prefix.
Only a content-free historical fingerprint is embedded, never questions/gold.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

from .io import atomic_json, digest, file_hash

UPSTREAM_TEMPLATE_URL = 'https://github.com/mll-lab-nu/MindCube/blob/main/src/prompt_generation/templates.py'

PREFIX = """[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints.
[Answer Instruction]
You only need to provide *ONE* correct answer selecting from the options listed below. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should **only** be '<answer>A. Above</answer>'.

[Question]
"""
HISTORICAL_SHA256 = 'af15ada5d8cb9da99705a539b8eecf51e8cedc90affb4f56a5e369e6e40883fc'
SETTING_COUNTS = {'among': 600, 'around': 250, 'rotation': 200}
NAME = 'MindCube_tinybench_raw_qa.jsonl'


def generate_row(row):
    """Apply the upstream raw_qa template, retaining every original raw field."""
    if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k]
                                        for k in ('id', 'question', 'gt_answer')):
        raise ValueError('MindCube requires original id, question and gt_answer strings')
    if row['gt_answer'] not in 'ABCDE' or len(row['gt_answer']) != 1:
        raise ValueError('MindCube gt_answer must be its original option letter')
    images = row.get('images')
    if not isinstance(images, list) or not images or len(images) > 4:
        raise ValueError('MindCube requires the original ordered list of at most four images')
    for image in images:
        if not isinstance(image, str) or not image or '\\' in image or '\x00' in image:
            raise ValueError('Invalid MindCube image path')
        path = PurePosixPath(image)
        if path.is_absolute() or '..' in path.parts or '://' in image:
            raise ValueError('MindCube images must be safe paths relative to the extracted data directory')
    prompt = PREFIX + row['question']
    if 'input_prompt' in row and row['input_prompt'] != prompt:
        raise ValueError('MindCube input_prompt is not the historical official raw_qa prompt')
    options = re.findall(r'([A-Z]\.\s+.*?)(?=[A-Z]\.|$)', row['question'])
    answers = [option.strip() for option in options if option.strip().startswith(row['gt_answer'] + '.')]
    if len(answers) != 1:
        raise ValueError('MindCube gold option text cannot be uniquely extracted by the official template')
    grounded = '<answer>' + answers[0] + '</answer>'
    if 'grounded_output' in row and row['grounded_output'] != grounded:
        raise ValueError('MindCube grounded_output differs from the official raw_qa template')
    return dict(row, input_prompt=prompt, grounded_output=grounded)


def verify_historical(rows):
    if len(rows) != 1050 or len({row['id'] for row in rows}) != 1050:
        raise ValueError('MindCube Tiny requires 1050 unique original IDs')
    if dict(Counter(row['id'].split('_')[0] for row in rows)) != SETTING_COUNTS:
        raise ValueError('MindCube Tiny setting counts differ from the historical 600/250/200 inputs')
    projection = sorted([{key: row[key] for key in ('id', 'question', 'images', 'gt_answer', 'input_prompt')}
                         for row in rows], key=lambda row: row['id'])
    if digest(projection) != HISTORICAL_SHA256:
        raise ValueError('MindCube raw_qa question/gold/prompt/ordered-image identity differs from historical inputs')
    return {'rows': 1050, 'setting_counts': SETTING_COUNTS,
            'historical_input_projection_sha256': HISTORICAL_SHA256,
            'matches_historical_input_projection': True,
            'historical_input_row_order_verified': False,
            'historical_image_bytes_verified': False}


def locate(source):
    source = Path(source)
    if source.is_file():
        # Both a raw and an already-generated official input can be verified.
        root = (source.parent.parent if source.parent.name == 'raw' else
                source.parents[2] if source.parent.name == 'general' and source.parents[1].name == 'prompts'
                else source.parent)
        return source, root
    for data_root in (source, source / 'data'):
        for relative in ('raw/MindCube_tinybench.jsonl', 'prompts/general/' + NAME):
            path = data_root / relative
            if path.is_file():
                return path, data_root
    raise ValueError('Missing MindCube data/raw/MindCube_tinybench.jsonl or official Tiny raw_qa prompts; extract data.zip first')


def convert(source, target, image_root=None):
    """Generate only Tiny raw_qa; fail closed if any historical input differs.

    Images remain relative to image_root (normally the extracted ``data``
    directory). The report supplies this root to the selected-benchmark runner.
    No image rewriting, model inference or cognitive-map generation occurs.
    """
    path, inferred_root = locate(source)
    root, target = Path(image_root or inferred_root).resolve(), Path(target)
    if target.exists() or target.is_symlink():
        raise ValueError('MindCube conversion target must not exist')
    generated = []
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                generated.append(generate_row(json.loads(line)))
    report = verify_historical(generated)
    # Verify every local image before publishing any result; do not silently
    # substitute another similarly named image or alter list order.
    image_hashes = {}
    for row in generated:
        for name in row['images']:
            candidate = (root / name).resolve()
            if not candidate.is_relative_to(root) or not candidate.is_file():
                raise ValueError('Missing/escaped MindCube image: ' + name)
            if name not in image_hashes:
                image_hashes[name] = file_hash(candidate)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + target.name + '-mindcube-', dir=target.parent))
    try:
        with (stage / NAME).open('w', encoding='utf-8') as stream:
            for row in generated:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        report.update(benchmark='mindcube', task='raw_qa', source_sha256=file_hash(path),
                      upstream_template=UPSTREAM_TEMPLATE_URL,
                      prompt_prefix_sha256=digest(PREFIX), upstream_license='MIT (per official repository README)',
                      image_root=str(root), image_sha256=image_hashes,
                      transform='Official raw_qa prefix plus original question; original image and row order retained')
        atomic_json(stage / 'normalization.json', report)
        if target.exists() or target.is_symlink():
            raise ValueError('MindCube target appeared during conversion; refusing overwrite')
        stage.rename(target)
        return report
    except BaseException:
        shutil.rmtree(stage)
        raise
