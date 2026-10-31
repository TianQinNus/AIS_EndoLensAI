"""Verify imported experiment source without ML dependencies or data access."""
from pathlib import Path
import ast
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'docs/IMPORTED_EXPERIMENT_SOURCES.json').read_text(encoding='utf-8'))
checked = 0
for record in manifest['files']:
    path = root / record['path']
    assert path.is_file(), str(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], str(path)
    if path.suffix == '.py':
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    checked += 1
for number in range(14):
    assert (root / 'experiments' / ('E' + str(number)) / 'README.md').is_file()
print('EXPERIMENT_ARCHIVE_PASS', checked, 'imported files;', '14 stage guides')
