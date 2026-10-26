"""Release gates are reports, not optimistic labels based on syntax checks."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from .io import file_hash, read_json
from .protocols import RESOURCES


def check():
    package = Path(__file__).parent
    errors, warnings = [], []
    manifest = read_json(RESOURCES / 'vendor/SOURCES.json')
    tracked = {entry['path']: entry for entry in manifest['files']}
    for relative in tracked:
        if not (RESOURCES / 'vendor' / relative).is_file():
            errors.append(f'Missing tracked vendor file: {relative}')
    for path in sorted((RESOURCES / 'vendor').rglob('*')):
        if not path.is_file() or path.suffix not in ('.py', '.yaml', '.md', '.json') or path.name == 'SOURCES.json':
            continue
        relative = str(path.relative_to(RESOURCES / 'vendor'))
        if relative not in tracked:
            errors.append(f'Untracked vendor file: {relative}')
        elif file_hash(path) != tracked[relative]['packaged_sha256']:
            errors.append(f'Vendor source hash changed without provenance entry: {relative}')
    for path in sorted(package.rglob('*')):
        if not path.is_file() or path.suffix not in ('.py', '.yaml', '.md', '.json'):
            continue
        content = path.read_text(encoding='utf-8')
        if path.suffix == '.py':
            try:
                ast.parse(content, filename=str(path))
            except SyntaxError:
                errors.append(f'Invalid Python syntax: {path.relative_to(package)}')
        # Detect known deployment namespaces without embedding them in this scanner.
        if re.search(r'/(?:mnt/(?:nas2?|r-contentsecurity-p)|r-contentsecurity/share)/', content):
            # Exempt this detector's source; the expression does not contain literal private paths.
            errors.append(f'Private deployment path: {path.relative_to(package)}')
        if re.search(r'(?:sk-[A-Za-z0-9]{24,}|Bearer\s+[A-Za-z0-9_-]{30,})', content):
            errors.append(f'Possible credential: {path.relative_to(package)}')
    permission_file = RESOURCES / 'release_permissions.json'
    permissions = read_json(permission_file) if permission_file.is_file() else {}
    owner_authorized = permissions.get('owner_adaptations', {}).get('public_distribution_authorized') is True
    blockers = ['AgentVista upstream redistribution rights require confirmation',
                'Historical MIMIC mathruler policy and all-benchmark numerical integration acceptance pending',
                'Upstream license/attribution review for audited reference excerpts remains pending',
                'Project GUI/shared inference upstream attribution review required',
                'Full real-data/model integration acceptance has not been recorded']
    if not owner_authorized:
        blockers.append('Owner adaptation/contributor public distribution authorization is not recorded')
    audit_root = RESOURCES / 'audited'
    if audit_root.exists():
        entries = read_json(audit_root / 'MANIFEST.json')
        expected = {entry['module'] + '.py': entry['excerpt_sha256'] for entry in entries}
        for name, sha in expected.items():
            path = audit_root / name
            if not path.is_file() or file_hash(path) != sha:
                errors.append(f'Audited reference integrity mismatch: {name}')
        for path in audit_root.glob('*.py'):
            if path.name not in expected:
                errors.append(f'Untracked audited reference: {path.name}')
    # A source checkout has docs/tools/examples too; scan the actual export set,
    # not only the installed Python package. Wheels have no repository root here.
    root = package.parent
    if (root / 'pyproject.toml').is_file() and (root / 'tools/build_source_archive.py').is_file():
        from .packaging import source_files
        try:
            selected = {str(p.relative_to(root)) for p in source_files(root)}
            for relative in tracked:
                if 'zwz_eval/resources/vendor/' + relative not in selected:
                    errors.append(f'Tracked vendor file excluded from source export: {relative}')
        except ValueError as exc:
            errors.append(str(exc))
    return {'release_ready': not errors and not blockers, 'source_integrity_errors': errors,
            'release_blockers': blockers, 'warnings': warnings,
            'owner_adaptations_authorized': owner_authorized,
            'note': 'A passing unit test does not prove GPU compatibility or numerical parity.'}
