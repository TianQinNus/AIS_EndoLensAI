"""Import the reviewed teammate delivery without executing its training scripts."""
import stat
import zipfile
from pathlib import Path, PurePosixPath
from common import SOURCE_ZIP, STARTER, PACKAGE, sha, read_json, write_json, verify_weights


def prepare():
    if not SOURCE_ZIP.is_file():
        raise FileNotFoundError('Missing supplied starter ZIP: ' + str(SOURCE_ZIP))
    with zipfile.ZipFile(SOURCE_ZIP) as z:
        if z.testzip() is not None:
            raise ValueError('Starter ZIP CRC failure')
        for item in z.infolist():
            parts = PurePosixPath(item.filename).parts
            if not parts or parts[0] == '__MACOSX' or item.is_dir():
                continue
            if parts[0] != STARTER.name or any(p in ('.', '..') or ':' in p for p in parts):
                raise ValueError('Unsafe starter member: ' + item.filename)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('Symlinks not accepted')
            target = PACKAGE / 'resources' / Path(*parts)
            if not target.resolve().is_relative_to(STARTER.resolve()):
                raise ValueError('Starter member escapes destination')
            data = z.read(item)
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError('Existing extracted resource differs: ' + str(target))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    manifest = read_json(STARTER / 'manifest.json')
    for name, expected in manifest.items():
        if sha(STARTER / name) != expected:
            raise ValueError('Teammate manifest mismatch: ' + name)
    index = {'source_zip': str(SOURCE_ZIP), 'source_zip_sha256': sha(SOURCE_ZIP),
             'delivered_manifest_verified': len(manifest), 'weights': verify_weights(),
             'training_reference_executed': False,
             'caveats': ['R50 was historically selected after C6 inspection.',
                         'R50 initialization lineage includes nine later-quarantined images.',
                         'Archived R50 C6 score used rotate4 at 320, not the new common protocol.']}
    write_json(PACKAGE / 'resources/resource_index.json', index)
    print('RESOURCES_PASS', index['weights'], flush=True)


if __name__ == '__main__':
    prepare()
