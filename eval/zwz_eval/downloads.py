"""Explicit, revision-pinned downloads. No implicit multi-GB network activity."""
from __future__ import annotations

import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from .io import atomic_json, file_hash, read_json
from .protocols import RESOURCES

SOURCES = read_json(RESOURCES / 'dataset_sources.json')['benchmarks']
for _source in SOURCES.values():
    _source['hf_url'] = 'https://huggingface.co/datasets/' + _source['repo']


def download(bench, target, revision=None, execute=False):
    source = SOURCES[bench]
    info = {'benchmark': bench, **source, 'target': str(Path(target).resolve()),
            'revision': revision, 'executed': False, 'automatic_extraction': False}
    if not execute:
        return info
    if not source['repo']:
        raise ValueError('No verified public download for this benchmark')
    if not revision:
        raise ValueError('Choose --revision explicitly; moving default branches are not reproducible')
    from huggingface_hub import HfApi, snapshot_download
    try:
        sha = HfApi().dataset_info(source['repo'], revision=revision).sha
    except Exception:
        raise RuntimeError('Dataset metadata/revision is unavailable: ' + source['hf_url'] +
                           '. Check the requested revision and your HF access permissions; '
                           'authentication may be supplied through the Hugging Face environment.') from None
    if not isinstance(sha, str) or len(sha) != 40 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('Dataset revision did not resolve to an immutable commit')
    root = Path(target).resolve()
    state = root / 'download_state.json'
    identity = {'repo': source['repo'], 'revision': sha}
    if root.exists() and any(root.iterdir()) and (not state.exists() or read_json(state) != identity):
        raise ValueError('Target belongs to other data/revision; existing data is never overwritten')
    atomic_json(state, identity)
    snapshot_download(repo_id=source['repo'], repo_type='dataset', revision=sha, local_dir=root)
    files = {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
             for p in root.rglob('*') if p.is_file() and '.cache' not in p.relative_to(root).parts
             and p.name not in ('download_manifest.json', 'download_state.json')}
    info.update(revision=sha, executed=True, files=files)
    atomic_json(root / 'download_manifest.json', info)
    return info


def safe_extract(archive, target):
    """Extract only regular files/directories to a new directory; reject links/devices/traversal."""
    archive, target = Path(archive), Path(target).resolve()
    if target.exists():
        raise ValueError('Extraction target must not exist')
    def validate(name):
        name = name.replace('\\', '/')
        parts = PurePosixPath(name)
        if parts.is_absolute() or '..' in parts.parts or ':' in name:
            raise ValueError(f'Unsafe archive member: {name}')
        dest = target.joinpath(*parts.parts)
        if not dest.is_relative_to(target):
            raise ValueError('Archive member escaped destination')
        return dest
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as src:
            members = src.infolist()
            for member in members:
                validate(member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                    raise ValueError('Archive links/devices are prohibited')
            target.mkdir(parents=True)
            for member in members:
                dest = validate(member.filename)
                if member.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with src.open(member) as fin, dest.open('xb') as fout:
                        import shutil
                        shutil.copyfileobj(fin, fout)
    else:
        with tarfile.open(archive) as src:
            members = src.getmembers()
            for member in members:
                validate(member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError('Archive links/devices are prohibited')
            target.mkdir(parents=True)
            for member in members:
                dest = validate(member.name)
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with src.extractfile(member) as fin, dest.open('xb') as fout:
                        import shutil
                        shutil.copyfileobj(fin, fout)
