"""Verify preserved bytes or reconstruct an isolated historical workspace."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def selected(manifest, group):
    for row in manifest['files']:
        if not group or row['workspace_path'].split('/')[1].startswith(group):
            yield row


def source(row):
    if 'archive_path' in row:
        return HERE / row['archive_path']
    return REPO / row['repository_path']


def check(row):
    path = source(row)
    if not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
        raise ValueError(f"Missing or different bytes: {path}; materialize the indicated Git LFS file if needed")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'restore'])
    parser.add_argument('--group', help='Original output directory prefix, e.g. rubik or cup-state-il')
    parser.add_argument('--with-references', action='store_true', help='Also hash existing main files reused by the archive')
    parser.add_argument('--destination', type=Path, help='New empty workspace directory for restore')
    parser.add_argument('--relocate', action='store_true', help='Replace original absolute workspace prefixes in restored source only')
    args = parser.parse_args()
    manifest = json.loads((HERE / 'manifest.json').read_text())
    rows = list(selected(manifest, args.group))
    if not rows:
        parser.error('No files match this group')
    if args.action == 'restore' and (args.destination is None or args.destination.exists()):
        parser.error('restore requires --destination naming a directory that does not yet exist')
    verified = ast_count = archives = 0
    for row in rows:
        if args.action == 'verify' and 'repository_path' in row and not args.with_references:
            continue
        path = check(row)
        verified += 1
        if row['workspace_path'].endswith('.py'):
            ast.parse(path.read_text(), filename=str(path))
            ast_count += 1
        if path.suffix in ['.zip', '.npz']:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f'ZIP CRC failure: {path}')
            archives += 1
    if args.action == 'restore':
        destination = args.destination.resolve()
        destination.mkdir(parents=True)
        relocated = []
        for row in rows:
            path = check(row)
            target = destination / row['workspace_path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if args.relocate and target.suffix in ['.py', '.cpp', '.h', '.hpp', '.sh', '.slurm']:
                old_root = row['source_location'].split('/outputs/')[0]
                text = target.read_text()
                changed = text.replace(old_root + '/outputs.nosync/', str(destination / 'outputs') + '/')
                changed = changed.replace(old_root + '/', str(destination) + '/')
                if changed != text:
                    target.write_text(changed)
                    relocated.append(row['workspace_path'])
        (destination / 'RESTORE.json').write_text(json.dumps({'files': len(rows), 'relocated_source_paths': relocated, 'algorithm_changes': False}, indent=2) + '\n')
    print(json.dumps({'verified_files': verified, 'parsed_python_sources': ast_count, 'zip_crc_checks': archives, 'action': args.action, 'group': args.group}, indent=2))


if __name__ == '__main__':
    main()
