"""Atomic outputs, immutable contracts, and process-local append journals."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_journal(path):
    """Ignore only a torn final line; malformed complete/intermediate lines fail closed."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if i == len(lines)-1 and not line.endswith(b'\n'):
                break
            raise ValueError(f'Corrupt journal {path}, line {i+1}')
    return rows


def repair_torn_tail(path):
    """Call only while holding run_lock. Preserve a recoverable copy before truncation."""
    path = Path(path)
    if not path.exists():
        return
    raw = path.read_bytes()
    if not raw or raw.endswith(b'\n'):
        return
    tail = raw.rsplit(b'\n', 1)[-1]
    try:
        json.loads(tail)
    except (ValueError, UnicodeDecodeError):
        backup = path.with_name(path.name + '.torn-' + hashlib.sha256(raw).hexdigest()[:12])
        if not backup.exists():
            backup.write_bytes(raw)
        boundary = raw.rfind(b'\n') + 1
        with path.open('r+b') as stream:
            stream.truncate(boundary)
    else:
        with path.open('ab') as stream:
            stream.write(b'\n')


def append_jsonl(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def lock_contract(directory, contract):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'contract.json'
    value = {'sha256': digest(contract), 'contract': contract}
    if target.exists():
        if read_json(target) != value:
            raise ValueError('Run contract changed. Use a new run_id; existing outputs are untouched.')
    else:
        atomic_json(target, value)
    return value['sha256']


@contextlib.contextmanager
def run_lock(directory):
    """Advisory flock prevents two writers, including inference vs score, in one run."""
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.run.lock').open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process owns this run: {directory}') from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
