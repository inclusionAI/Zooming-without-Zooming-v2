"""Public source selection: never recursively bundle arbitrary project leftovers."""
from __future__ import annotations

import os
import re
from pathlib import Path

SOURCE_DIRS = {'.github', 'docs', 'examples', 'zwz_eval', 'scripts', 'tests', 'third_party', 'tools'}
ROOT_FILES = {'.gitignore', '.env.example', 'README.md', 'CONTRIBUTING.md', 'IMPLEMENTATION_STATUS.md',
              'LICENSE', 'NOTICE', 'SECURITY.md', 'THIRD_PARTY_NOTICES.md', 'pyproject.toml',
              'constraints-api.txt', 'validation_report.json'}
EXCLUDE = {'.git', '.venv', '__pycache__', 'outputs', 'data', 'bundled_data', 'checkpoints',
           'build', 'dist', '.pytest_cache'}
GENERATED = {'SOURCE_MANIFEST.json', 'RELEASE_CHECK.json'}
EXTENSIONS = {'.py', '.sh', '.json', '.yaml', '.yml', '.md', '.txt', '.toml', '.jinja'}
PUBLIC_SCRIPTS = {'scripts/evaluate.sh', 'scripts/prepare_benchmarks.sh', 'scripts/run_benchmarks.sh'}


def security_issue(content):
    # Patterns only; never echo a matched credential or private path in an error.
    if re.search(r'(?<![A-Za-z0-9_./-])/(?:Users|home|root|mnt)/[^\s\x27\"`<>]+', content):
        return 'Private deployment path'
    # Builder-specific names, aliases and internal addresses stay outside source
    # and archives. Report only the category, never the configured identifier.
    private_identifiers = (value.strip().casefold() for value in
                           os.environ.get('ZWZ_PRIVATE_IDENTIFIERS', '').split(','))
    folded_content = content.casefold()
    if any(value and value in folded_content for value in private_identifiers):
        return 'Configured private identifier'
    if re.search(r'\b(?:run_)?20\d{6}_\d{6}(?:_[A-Za-z0-9_.-]+)?\b', content):
        return 'Private run identifier'
    if re.search(r'global_step_\d+_merged', content):
        return 'Private checkpoint identifier'
    if re.search(r'\b(?:workflow|experiment|job)[_-]\d{5,}\b|\baistudio-\d{5,}\b', content, re.I):
        return 'Private machine identifier'
    # Token boundaries distinguish actual credential-shaped strings from image
    # basenames containing an incidental "sk-" inside a word. No data file is
    # exempt from this check and benchmark image names are not rewritten.
    if re.search(r'(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{24,}|Bearer\s+[A-Za-z0-9_-]{30,}', content):
        return 'Possible credential'
    quoted_secret = r'(?i)\b(?:api_?key|access_?token|secret|password)\b[\x27\"]?\s*[:=]\s*([\x27\"])[A-Za-z0-9_,.-]{24,}\1'
    shell_secret = r'(?m)^\s*(?:export\s+)?(?:[A-Z0-9_]+_)?(?:API_?KEY|ACCESS_?TOKEN|SECRET|PASSWORD)\s*=\s*[A-Za-z0-9_-]{24,}\s*$'
    if re.search(quoted_secret, content) or re.search(shell_secret, content):
        return 'Possible credential assignment'
    if re.search(r'-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----', content):
        return 'Private key material'
    return None


def source_files(root):
    root = Path(root)
    files = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if relative.parts[0] == 'scripts' and path.is_file() and str(relative) not in PUBLIC_SCRIPTS:
            continue  # Machine-specific helpers are not public protocol entrypoints.
        if relative.parts[0] in {'models', 'local'}:
            continue  # Private weights/configs, not nested vendored Python packages.
        if any(part in EXCLUDE or part.endswith('.egg-info') for part in relative.parts):
            continue
        if len(relative.parts) == 1 and path.name in GENERATED:
            continue  # Rebuild these; a manifest must never hash its previous self.
        if path.name == '.DS_Store' or path.name.startswith('.env') and path.name != '.env.example':
            continue
        if path.suffix in ('.pyc', '.pyo', '.log', '.whl', '.tar', '.gz', '.tmp'):
            continue
        issue = security_issue(str(relative))
        if issue:
            raise ValueError(f'{issue} in source member name')
        if path.is_symlink():
            raise ValueError(f'Source symlink requires review: {relative}')
        if not path.is_file():
            continue
        if len(relative.parts) == 1:
            allowed = path.name in ROOT_FILES
        else:
            allowed = relative.parts[0] in SOURCE_DIRS and path.suffix in EXTENSIONS
        if not allowed:
            raise ValueError(f'Unreviewed file is not eligible for a public source archive: {relative}')
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError(f'Large source file requires explicit review: {relative}')
        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            raise ValueError(f'Non-text file is not eligible for this source archive: {relative}') from None
        issue = security_issue(content)
        if issue:
            raise ValueError(f'{issue}: {relative}')
        files.append(path)
    return files
