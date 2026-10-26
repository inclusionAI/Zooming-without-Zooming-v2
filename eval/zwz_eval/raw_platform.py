"""Local raw-data conversion for the twelve historical evaluate_v3 profiles.

No evaluation questions or gold answers are bundled here.  Content-free recipes
restore the historical order and reject any changed prompt, gold, custom fields,
or image identity.  Matching these recipes is not a claim that server image bytes
or numerical scores have been independently reproduced.
"""
from __future__ import annotations

import base64
import collections
import csv
import hashlib
import io
import json
import math
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from .io import atomic_json, canonical, file_hash, read_json
from .normalization import fingerprint, verify_rows
from .protocols import PLATFORM_BENCHMARKS, RESOURCES

SYSTEM = ('You are a helpful assistant.\n The assistant first thinks about the reasoning process in the mind '
          'and then provides the user with the answer. The answer are enclosed within <answer> </answer> '
          'tags, respectively, i.e., reasoning process here <answer> answer here </answer>')
MC_SUFFIX = "Answer with the option's letter from the given choices."
SOURCE_NAMES = {
    'mmstar': ('mmstar.parquet', 'MMStar.tsv'),
    'babyvision': ('data/train-*.parquet',),
    'vstar': ('test_questions.jsonl',),
    'hr4k': ('hr_bench_4k.parquet', 'hr_bench_4k.tsv'),
    'hr8k': ('hr_bench_8k.parquet', 'hr_bench_8k.tsv'),
    'mme_en': ('MME_RealWorld.json',),
    'mme_cn': ('MME_RealWorld_CN.json',),
    'zoombench': ('data/test.parquet', 'data/test-*.parquet'),
    'perception': ('PerceptionBench.jsonl',),
    'treebench': ('TreeBench.tsv',),
    'o3': ('test/metadata.jsonl', 'metadata.jsonl'),
    'countqa': ('data/test-*.parquet',),
}


def load_recipe(benchmark):
    if benchmark not in PLATFORM_BENCHMARKS:
        raise ValueError('Not an evaluate_v3 platform benchmark')
    recipe = read_json(RESOURCES / 'normalization' / ('platform_' + benchmark + '.json'))
    if recipe.get('benchmark') != benchmark or recipe.get('schema') != 1:
        raise ValueError('Invalid content-free normalization recipe')
    return recipe


def _safe_relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or '\\' in str(p) or not str(p) or str(p) == '.':
        raise ValueError('Unsafe relative image identity')
    return str(p)


def _annotation(row):
    return {k: v for k, v in row.items() if k != 'images'}


def _annotation_hash(row):
    return fingerprint(_annotation(row))


def _valid(value):
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _text(value, name='text'):
    if not _valid(value):
        raise ValueError('Missing raw ' + name)
    return str(value)


def _source_files(benchmark, source):
    source = Path(source)
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise ValueError('Raw source must be an existing local file or download directory')
    if benchmark == 'vstar':
        groups = [sorted((source / name).glob('*.json'))
                  for name in ('direct_attributes', 'relative_position')]
        if any(groups):
            if not all(groups):
                raise ValueError('VStar requires both direct_attributes and relative_position; not OCR/GPT4V-hard')
            return sorted(path for group in groups for path in group)
    # Choose one canonical serialization, never concatenate TSV and Parquet copies.
    for pattern in SOURCE_NAMES[benchmark]:
        found = sorted(source.glob(pattern))
        if found:
            return found
    raise ValueError('No known raw annotation file for ' + benchmark + '; pass its exact file path')


def _vstar_sidecar(path, recipe):
    """Official per-image JSON -> the project's MC serialization.

    The author's documented gold is the first original option. Content-free
    slices restore the project's shortened/reordered option TEXT; the gold is
    derived from that original option, never invented or read from model output.
    https://github.com/penghao-wu/vstar#benchmark
    """
    raw = read_json(path)
    identity = path.parent.name + '/' + path.with_suffix('.jpg').name
    projection = recipe.get('raw_projections', {}).get(identity)
    if not projection or fingerprint(raw) != projection['raw_metadata_sha256']:
        raise ValueError('VStar official sidecar metadata differs from its verified raw conversion recipe')
    question, options = raw.get('question'), raw.get('options')
    if not isinstance(question, str) or not isinstance(options, list) or not 2 <= len(options) <= 4:
        raise ValueError('VStar original question and ordered options are required')
    slices = projection['option_slices']
    if len(slices) != len(options) or {spec[0] for spec in slices} != set(range(len(options))):
        raise ValueError('Invalid VStar original-to-project option mapping')
    transforms = {'identity': lambda value: value, 'lower': str.lower,
                  'upper': str.upper, 'capitalize': str.capitalize,
                  # Two literal serialization quirks in the retained input,
                  # applied only by metadata-hash-bound slice specifications.
                  'no_person_to_zero_people': lambda value: '0 people' if value == 'no person' else value,
                  'append_period': lambda value: value + '.'}
    restored, correct = [], None
    for position, spec in enumerate(slices):
        if len(spec) not in (3, 4):
            raise ValueError('Invalid VStar option slice')
        index, start, end = spec[:3]
        transform = spec[3] if len(spec) == 4 else 'identity'
        original = options[index]
        if (not isinstance(original, str) or any(type(x) is not int for x in (index, start, end))
                or not 0 <= start < end <= len(original) or transform not in transforms):
            raise ValueError('Invalid VStar option extraction')
        restored.append('(' + chr(65 + position) + ') ' + transforms[transform](original[start:end]))
        if index == 0:
            correct = chr(65 + position)
    return {'text': question + '\n' + '\n'.join(restored) + '\n' + MC_SUFFIX,
            'label': correct, 'image': str(path.with_suffix('.jpg').resolve())}


def _read_file(path):
    suffix = path.suffix.lower()
    if suffix == '.parquet':
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError('Install zwz-eval[data] for local Parquet conversion') from exc
        # Bound memory even when HR-Bench stores multi-megabyte images per row.
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1):
            yield from batch.to_pylist()
    elif suffix == '.jsonl':
        with path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f'Malformed raw JSONL at line {line_number}') from exc
                if not isinstance(row, dict):
                    raise ValueError('Each raw JSONL row must be an object')
                yield row
    elif suffix == '.json':
        value = read_json(path)
        if isinstance(value, dict) and isinstance(value.get('data'), list):
            value = value['data']
        if not isinstance(value, list) or not all(isinstance(r, dict) for r in value):
            raise ValueError('Raw JSON must contain an array of row objects')
        yield from value
    elif suffix in ('.tsv', '.csv'):
        previous = csv.field_size_limit()
        try:
            csv.field_size_limit(256 * 1024 * 1024)
            with path.open(encoding='utf-8', newline='') as stream:
                yield from csv.DictReader(stream, delimiter='\t' if suffix == '.tsv' else ',')
        finally:
            csv.field_size_limit(previous)
    else:
        raise ValueError('Supported raw annotations: JSON, JSONL, Parquet, TSV, CSV')


class _Assets:
    """Stage original encoded bytes; no image recompression or remote URL fetch."""
    def __init__(self, directory, roots):
        self.directory = directory
        self.roots = [Path(p).resolve() for p in roots]
        self.cache = {}

    def put(self, value):
        hint = None
        if isinstance(value, dict):
            hint = value.get('path')
            raw = value.get('bytes')
            if raw is not None:
                value = raw
            elif hint:
                value = hint
            else:
                raise ValueError('Image object has neither bytes nor path')
        if isinstance(value, (bytes, bytearray, memoryview)):
            raw = bytes(value)
        elif isinstance(value, str):
            if value.startswith(('http://', 'https://')):
                raise ValueError('Download images explicitly; conversion does not fetch remote URLs')
            if value.startswith('data:image/'):
                if ';base64,' not in value:
                    raise ValueError('Only base64 image data URIs are supported')
                raw = base64.b64decode(value.split(';base64,', 1)[1], validate=True)
            elif (len(value) > 512 or value.startswith(('/9j/', 'iVBORw0KGgo', 'UklGR', 'R0lGOD'))) and re.fullmatch(r'[A-Za-z0-9+/=\s]+', value):
                raw = base64.b64decode(''.join(value.split()), validate=True)
            else:
                hint = hint or value
                p = Path(value)
                candidates = [p] if p.is_absolute() else [r / p for r in self.roots]
                existing = [p.resolve() for p in candidates if p.is_file()]
                # Do not follow traversal or symlinks outside the explicitly allowed data roots.
                existing = [p for p in existing if any(p.is_relative_to(r) for r in self.roots)]
                if not existing:
                    raise ValueError('Image file is missing or outside the supplied image/data roots')
                hashes = {file_hash(p) for p in existing}
                if len(hashes) != 1:
                    raise ValueError('Ambiguous local image path resolves to different files')
                raw = existing[0].read_bytes()
        else:
            try:
                from PIL import Image
                is_image = isinstance(value, Image.Image)
            except ImportError:
                is_image = False
            if not is_image:
                raise ValueError('Unsupported raw image representation')
            buffer = io.BytesIO()
            value.save(buffer, format='PNG')
            raw = buffer.getvalue()
        sha = hashlib.sha256(raw).hexdigest()
        if sha not in self.cache:
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError('Install zwz-eval[data] for image validation') from exc
            with Image.open(io.BytesIO(raw)) as picture:
                picture.verify()
            target = self.directory / sha
            target.write_bytes(raw)
            self.cache[sha] = target
        return sha, str(hint) if hint else None


def _column_options(raw):
    return [(k, str(raw[k])) for k in 'ABCDEFGHIJK'
            if k in raw and _valid(raw[k]) and str(raw[k]) not in ('', 'nan', 'None')]


def _mmstar_candidates(raw):
    question = _text(raw.get('question'), 'question')
    options = _column_options(raw)
    if not options:
        if ' Options: ' not in question:
            raise ValueError('MMStar raw question lacks the upstream Options block')
        question, block = question.rsplit(' Options: ', 1)
        parts = re.split(r'(?:^|,\s)([A-K]):\s', block)
        if parts[0] or len(parts) < 3 or len(parts) % 2 != 1:
            raise ValueError('Unrecognized MMStar options serialization')
        options = [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)
                   if parts[i + 1] not in ('nan', 'None', '')]
    answer = _text(raw.get('answer'), 'answer').strip('()')
    # A source's leading image marker may be replaced by the transport marker.
    # Internal numbered markers are part of some actual historical questions.
    questions = (question, re.sub(r'^<image\s+\d+>', '', question))
    for q in dict.fromkeys(q for value in questions for q in (value, value.strip())):
        yield '<image>' + q + ' Select from the following choices.\n' + '\n'.join(
            '(' + letter + ') ' + option for letter, option in options), '(' + answer + ')'


def _basic_candidates(benchmark, raw):
    """Enumerate bounded known serialization variants, accepted only by exact hash."""
    if benchmark == 'mmstar':
        yield from _mmstar_candidates(raw)
        return
    if benchmark == 'babyvision':
        question = _text(raw.get('question'), 'question')
        options = raw.get('options') or []
        if isinstance(options, str):
            options = json.loads(options)
        if not isinstance(options, list):
            raise ValueError('BabyVision options must be a list')
        if str(raw.get('ansType')).lower() in ('choice', 'multiple-choice', 'mcq') or options:
            answer = raw.get('choiceAns')
            if not _valid(answer) or int(answer) != float(answer) or not 0 <= int(answer) < (len(options) or 26):
                raise ValueError('Invalid zero-based BabyVision choiceAns')
            # Some choice tasks embed all labels in their question/image rather
            # than providing separate textual options.
            yield '<image>' + question, chr(65 + int(answer))
            for sep in (':', '. ', '.'):  # historical sources contain all three option forms
                block = '\n'.join(chr(65 + i) + sep + str(v) for i, v in enumerate(options))
                for trailing in ('', '\n'):
                    yield '<image>' + question + '\nSelect from the following choices.\n' + block + trailing, chr(65 + int(answer))
            for trailing in ('', '\n'):
                yield '<image>' + question + '\nSelect from the following choices.\n' + '\n'.join(map(str, options)) + trailing, chr(65 + int(answer))
        else:
            yield '<image>' + question, _text(raw.get('blankAns'), 'blankAns')
        return
    if benchmark in ('hr4k', 'hr8k'):
        question = _text(raw.get('question'), 'question')
        options = _column_options(raw)
        if len(options) != 4:
            raise ValueError('HR-Bench requires its original four A-D choices')
        yield '<image>' + question + '\n' + '\n'.join(k + '. ' + v for k, v in options) + '\n' + MC_SUFFIX, _text(raw.get('answer'))
        return
    if benchmark == 'vstar':
        text = _text(raw.get('text'), 'text')
        yield '<image>' + text.replace("Answer with the option's letter from the given choices directly.", MC_SUFFIX), _text(raw.get('label'))
        return
    if benchmark == 'treebench':
        question = _text(raw.get('question'), 'question')
        options = _column_options(raw)
        block = '\n'.join('(' + k + ') ' + v for k, v in options)
        for q in (question, question.strip()):
            yield '<image>' + q + ' Select from the following choices.\n' + block, '(' + _text(raw.get('answer')).strip('()') + ')'
        return
    if benchmark in ('mme_en', 'mme_cn'):
        question = _text(raw.get('question'), 'question')
        options = raw.get('options', raw.get('choices'))
        if isinstance(options, str):
            options = json.loads(options)
        if isinstance(options, dict):
            options = list(options.values())
        if not isinstance(options, list) or not options:
            raise ValueError('MME-RealWorld requires the original ordered options list')
        lead = 'Select from the following choices.' if benchmark == 'mme_en' else '\u8bf7\u4ece\u4ee5\u4e0b\u9009\u9879\u4e2d\u9009\u62e9.'
        for final_dot in ('', '.'):
            block = '\n'.join(chr(65 + i) + ':' + str(v) + final_dot for i, v in enumerate(options))
            yield '<image>' + question + '\n' + lead + '\n' + block, _text(raw.get('answer')).strip('()')
        return
    if benchmark == 'zoombench':
        query = _text(raw.get('query', raw.get('prompt')), 'query/prompt')
        answer = _text(raw.get('response', raw.get('answer')))
        query = query if query.startswith('<image>') else '<image>' + query
        yield query, answer
        yield query + '\n' + MC_SUFFIX, answer
        yield query + '\n Please answer using Arabic numerals. ', answer
        return
    if benchmark == 'perception':
        problem = _text(raw.get('problem'), 'problem')
        problem = re.sub(r'<\|image_\d+\|>', '<image>', problem)
        yield problem, _text(raw.get('answer'))
        hint = raw.get('hint')
        if hint:
            yield problem + '\n' + str(hint), _text(raw.get('answer'))
        return
    raise ValueError('Unhandled raw schema')


def _image_values(benchmark, raw, position):
    if benchmark == 'perception':
        values = raw.get('image')
        if not isinstance(values, list) or not values:
            raise ValueError('PerceptionBench requires the original ordered image array')
        return values, [f'{int(raw.get("index", position))}_{i:02d}.jpg' for i in range(1, len(values) + 1)]
    value = raw.get('image', raw.get('image_path', raw.get('file_name')))
    if value is None:
        raise ValueError('Missing raw image')
    if benchmark == 'mmstar':
        return [value], [str(raw.get('index', position)) + '.jpg']
    if benchmark == 'treebench':
        return [value], [f'{int(raw.get("index", position)):05d}.png']
    if benchmark == 'countqa':
        return [value], [f'image_{position}.png']
    if benchmark == 'zoombench' and raw.get('id'):
        return [value], [str(raw['id']) + '.jpg']
    return [value], [None]


def _candidate_rows(benchmark, raw, position):
    if 'query' in raw and 'images' in raw and 'response' in raw:
        # Exact already-normalized data: never drop extra fields or add a system role.
        yield dict(raw), list(raw['images']), list(raw['images'])
        return
    if benchmark in ('mme_en', 'mme_cn'):
        # Official MME_RealWorld*.json uses these literal, capitalized keys.
        # The lowercase schema also occurs in pre-extracted local serializations.
        raw = dict(raw)
        for name, official in (('question', 'Text'), ('image', 'Image'),
                               ('options', 'Answer choices'), ('answer', 'Ground truth')):
            if name not in raw and official in raw:
                raw[name] = raw[official]
    values, hints = _image_values(benchmark, raw, position)
    if benchmark == 'countqa':
        questions, answers = raw.get('questions'), raw.get('answers')
        if not isinstance(questions, list) or not isinstance(answers, list) or not questions or len(questions) != len(answers):
            raise ValueError('CountQA questions and answers must be nonempty aligned lists')
        for question, answer in zip(questions, answers):
            yield {'query': '<image>' + _text(question), 'response': _text(answer), 'system': SYSTEM}, values, hints
    elif benchmark == 'o3':
        question = _text(raw.get('question'), 'question')
        options = _text(raw.get('options'), 'options')
        row = {k: v for k, v in raw.items() if k not in ('image', 'file_name', 'question')}
        row.update(query='<image>\n' + question + '\n' + options, response=_text(raw.get('answer')))
        yield row, values, hints
    else:
        seen = set()
        for query, response in _basic_candidates(benchmark, raw):
            key = (query, response)
            if key not in seen:
                seen.add(key)
                yield {'query': query, 'response': response, 'system': SYSTEM}, values, hints


def _hint_matches(hint, expected):
    if hint is None:
        return False
    value = PurePosixPath(str(hint))
    wanted = PurePosixPath(expected)
    return str(value) == str(wanted) or str(value).endswith('/' + str(wanted))


def _known_hint(hint, expected_paths):
    if hint is None:
        return False
    parts = PurePosixPath(str(hint)).parts
    return any(str(PurePosixPath(*parts[i:])) in expected_paths for i in range(len(parts)))


def _order_candidates(benchmark, candidates, recipe):
    expected = recipe['order']
    by_hash = collections.defaultdict(list)
    for slot, entry in enumerate(expected):
        by_hash[entry[0]].append((slot, entry[1]))
    if benchmark in ('hr4k', 'hr8k'):
        # Raw image byte identity, not input row index, groups cycle-expanded HR rows.
        expected_groups, raw_groups = collections.defaultdict(list), collections.defaultdict(list)
        for digest, paths in expected:
            if len(paths) != 1:
                raise ValueError('Invalid HR recipe image cardinality')
            expected_groups[paths[0]].append(digest)
        group_names = {}
        for path, values in expected_groups.items():
            key = fingerprint(sorted(values))
            if key in group_names:
                raise ValueError('Ambiguous historical HR image group')
            group_names[key] = path
        for candidate in candidates:
            if len(candidate['assets']) != 1:
                raise ValueError('HR raw row must contain one image')
            raw_groups[candidate['assets'][0]].append(candidate['hash'])
        assigned = {}
        for asset, values in raw_groups.items():
            key = fingerprint(sorted(values))
            if key not in group_names:
                raise ValueError('HR image/question group differs from the historical input')
            path = group_names[key]
            if path in assigned.values():
                raise ValueError('Two different raw HR images claim the same historical identity')
            assigned[asset] = path
        for candidate in candidates:
            candidate['hints'] = [assigned[candidate['assets'][0]]]
    result = [None] * len(expected)
    bindings = {}
    expected_paths = {path for _, paths in expected for path in paths}
    for candidate in candidates:
        available = [(slot, paths) for slot, paths in by_hash[candidate['hash']] if result[slot] is None]
        hinted = [(slot, paths) for slot, paths in available if len(paths) == len(candidate['hints']) and
                  all(_hint_matches(h, p) for h, p in zip(candidate['hints'], paths))]
        if not hinted and any(_known_hint(h, expected_paths) for h in candidate['hints']):
            raise ValueError('Raw image identity conflicts with its historical question')
        options = hinted or available
        # Duplicate semantic questions on different images must be disambiguated, never guessed.
        identities = {tuple(paths) for _, paths in options}
        if not options or len(identities) != 1:
            raise ValueError('Raw question/image mapping is missing, duplicated, or ambiguous')
        slot, paths = options[0]
        if len(paths) != len(candidate['assets']):
            raise ValueError('Raw image order/cardinality differs from the historical input')
        for path, asset in zip(paths, candidate['assets']):
            path = _safe_relative(path)
            if path in bindings and bindings[path] != asset:
                raise ValueError('Conflicting raw bytes for one historical image identity')
            bindings[path] = asset
        result[slot] = {**candidate['row'], 'images': list(paths)}
    if any(row is None for row in result):
        raise ValueError('Raw input does not reconstruct every historical question')
    return result, bindings


def convert_platform(benchmark, source, output, image_root=None):
    """Convert local official raw files, restoring exact historical annotation order.

    Returns a report; writes ``output/normalized.jsonl`` and images with historical
    relative identities. Output is installed only after all hashes/counts match.
    Download/extraction, model inference and evaluation are deliberately separate.
    """
    recipe = load_recipe(benchmark)
    files = _source_files(benchmark, source)
    source_hashes = {str(p.resolve()): file_hash(p) for p in files}
    output = Path(output)
    if output.exists():
        raise ValueError('Raw conversion output must not already exist')
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(source) if Path(source).is_dir() else Path(source).parent
    roots = [root] + ([Path(image_root)] if image_root else [])
    # O3 imagefolder metadata is relative to test/, and may use images/ explicitly.
    roots.extend(p.parent for p in files)
    roots.extend([root / 'test', root / 'test/images', root / 'images'])
    with tempfile.TemporaryDirectory(prefix='.zwz-raw-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'converted'
        stage.mkdir()
        pool_root = Path(temporary) / 'assets'
        pool_root.mkdir()
        pool = _Assets(pool_root, roots)
        known = {entry[0] for entry in recipe['order']}
        candidates, position, image_aliases = [], 0, {}
        for file in files:
            source_rows = ([_vstar_sidecar(file, recipe)]
                           if benchmark == 'vstar' and file.parent.name in ('direct_attributes', 'relative_position')
                           and file.suffix.lower() == '.json' else _read_file(file))
            for raw in source_rows:
                matched = {}
                expanded = []
                for row, values, hints in _candidate_rows(benchmark, raw, position):
                    digest = _annotation_hash(row)
                    if digest in known:
                        matched[digest] = (row, values, hints)
                        expanded.append((digest, row, values, hints))
                normalized = 'query' in raw and 'images' in raw and 'response' in raw
                is_counting_expansion = benchmark == 'countqa' and not normalized
                if not is_counting_expansion and len(matched) != 1:
                    raise ValueError(f'{benchmark}: raw row {position} does not match exactly one historical annotation')
                if is_counting_expansion and len(expanded) != len(raw.get('questions', [])):
                    raise ValueError('CountQA expanded QA differs from the historical annotation set')
                selected = expanded if is_counting_expansion else [(k, *v) for k, v in matched.items()]
                for digest, row, values, hints in selected:
                    assets, actual_hints = [], []
                    for value, hint in zip(values, hints):
                        if isinstance(value, str) and value.isdecimal() and value in image_aliases:
                            asset, original_hint = image_aliases[value], None
                        else:
                            asset, original_hint = pool.put(value)
                        assets.append(asset)
                        actual_hints.append(hint or original_hint)
                    candidates.append({'row': _annotation(row), 'hash': digest, 'assets': assets, 'hints': actual_hints})
                    if len(assets) == 1:
                        alias = str(raw.get('index', position))
                        if alias in image_aliases and image_aliases[alias] != assets[0]:
                            raise ValueError('Conflicting source image index')
                        image_aliases[alias] = assets[0]
                position += 1
        if len(candidates) != len(recipe['order']):
            raise ValueError('Converted question count differs from the historical recipe')
        rows, bindings = _order_candidates(benchmark, candidates, recipe)
        parity = verify_rows(benchmark, rows)
        for relative, asset in bindings.items():
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pool.cache[asset], target)
        with (stage / 'normalized.jsonl').open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(canonical(row) + '\n')
        if any(file_hash(path) != sha for path, sha in source_hashes.items()):
            raise ValueError('Raw annotation source changed during conversion')
        report = {'benchmark': benchmark, 'raw_rows': position, 'rows': len(rows),
                  'annotations': 'normalized.jsonl', 'source_sha256': list(source_hashes.values()),
                  'normalized_sha256': file_hash(stage / 'normalized.jsonl'),
                  'recipe_sha256': fingerprint(recipe), 'images': len(bindings),
                  'assets_sha256': {path: asset for path, asset in sorted(bindings.items())},
                  **parity, 'image_bytes_verified': False,
                  'note': 'Historical annotation/image-identity hashes match; server image bytes and model/score parity are not independently verified.'}
        atomic_json(stage / 'conversion.json', report)
        if output.exists():
            raise ValueError('Raw conversion output was created concurrently')
        stage.rename(output)
    return report
