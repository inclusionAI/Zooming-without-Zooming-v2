"""Integrity-checked, dependency-light project reference functions.

These are scoring building blocks, not an assertion that new inference adapters
are complete. No endpoint access or argument-parsing side effects on import.
"""
import hashlib
import importlib.util
import json
from functools import lru_cache

from .protocols import RESOURCES


@lru_cache(maxsize=None)
def load(name):
    root = RESOURCES / 'audited'
    records = {item['module']: item for item in json.loads((root / 'MANIFEST.json').read_text())}
    if name not in records:
        raise ValueError('Unknown audited reference module')
    path = root / (name + '.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != records[name]['excerpt_sha256']:
        raise ValueError(f'Reference integrity mismatch: {name}')
    spec = importlib.util.spec_from_file_location('zwz_reference_' + name, path)
    module = importlib.util.module_from_spec(spec)
    if name == 'mmsi_final':
        module.original = load('mmsi_original').extract_choice
    spec.loader.exec_module(module)
    return module
