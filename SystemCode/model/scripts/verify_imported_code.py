"""Verify frozen imported Python files against the shared source manifest."""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / 'Miscellaneous/model_artifacts/CODE_IMPORT_MANIFEST.json').read_text())
    failures = []
    count = 0
    for item in manifest['files']:
        if not item['path'].endswith('.py'):
            continue
        count += 1
        target = root / item['path']
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != item['sha256']:
            failures.append(item['path'])
    if failures:
        raise SystemExit('Frozen source changed or missing:\n' + '\n'.join(failures))
    print(f'PASS: {count} imported Python sources match original bytes.')


if __name__ == '__main__':
    main()
