"""Restore tracked metadata only; never download data or start training."""
from pathlib import Path
import hashlib
import shutil


def main():
    root = Path(__file__).resolve().parents[1]
    source = root.parents[1] / 'Miscellaneous/model_artifacts'
    targets = ['e2_data_seed42', 'e0_20261004_221345_872444']
    copies = []
    for name in targets:
        for item in sorted((source / name).iterdir()):
            if not item.is_file():
                raise ValueError(f'Unexpected entry: {item}')
            target = root / 'runs' / name / item.name
            if target.exists():
                if not target.is_file() or hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(item.read_bytes()).digest():
                    raise FileExistsError(f'Different local artifact; not overwritten: {target}. Restore assets to a clean checkout or retain your original run.')
            else:
                copies.append((item, target))
    for item, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(item.read_bytes()).digest():
            raise IOError(f'Copy checksum mismatch: {target}')
    print(f'Restored {len(copies)} files; matching files preserved. No images, weights or probability maps copied.')


if __name__ == '__main__':
    main()
