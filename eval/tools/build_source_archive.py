#!/usr/bin/env python3
"""Build an auditable source archive; public-release mode fails while gates remain open."""
import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zwz_eval.release import check
from zwz_eval.packaging import source_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--draft', action='store_true', help='Internal review snapshot; not a cleared public release')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError('Write archive outside the source tree')
    report = check()
    if report['source_integrity_errors']:
        raise ValueError('Source integrity/security checks failed')
    if not args.draft and not report['release_ready']:
        raise ValueError('Release gates remain open; use --draft only for private review')
    if output.exists() or output.with_suffix(output.suffix + '.sha256').exists():
        raise ValueError('Archive/checksum exists; choose another output name')
    files = source_files(ROOT)
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    extra = {'SOURCE_MANIFEST.json': json.dumps(manifest, indent=2).encode(),
             'RELEASE_CHECK.json': json.dumps(report, indent=2).encode()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            def add(name, content):
                info = tarfile.TarInfo('eval/' + name)
                info.size, info.mtime, info.uid, info.gid, info.mode = len(content), 0, 0, 0, 0o644
                archive.addfile(info, io.BytesIO(content))
            for path in files:
                add(str(path.relative_to(ROOT)), path.read_bytes())
            for name, content in extra.items():
                add(name, content)
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + '.sha256').write_text(f'{sha}  {output.name}\n')
    print(json.dumps({'archive': str(output), 'files': len(files)+len(extra), 'sha256': sha,
                      'draft': args.draft, 'release_ready': report['release_ready']}, indent=2))


if __name__ == '__main__':
    main()
