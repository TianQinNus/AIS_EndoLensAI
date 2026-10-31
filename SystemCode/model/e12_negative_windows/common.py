"""E12 utilities. Existing experiments and source datasets are read-only inputs."""
import contextlib
import csv
import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
DATA = ROOT / 'runs/e12_negative_data_seed42'
OUTPUT = ROOT / 'runs/e12_negative_comparison_seed42'
STARTER = PACKAGE / 'resources/Best_UNet_Negative_Starter_20261027'
POLYPGEN = ROOT / 'dataset/_downloads/PolypGen-v3/extracted/PolypGen2021_MultiCenterData_v3'
SOURCE_ZIP = ROOT.parent / 'Best_UNet_Negative_Starter_20261027.zip'
WEIGHTS = {
    'pvt': ROOT / 'model_pth/PolypPVT.pth',
    'r34': ROOT / 'runs/e8_B_expanded_seed42/checkpoints/best.pth',
    'r50': STARTER / 'weights/best.pt',
}
EXPECTED_WEIGHTS = {
    'pvt': 'aef5ac34f7795b85af0ecbb730502db251d416a73a4c6b04a74e30aa45554048',
    'r34': '2ee7145fe5509d6cdbeefd90c8a2f0d87ca7f1d9daf78ce79e7edb4764ee7b5f',
    'r50': 'f521346ea188bbfe68d67c6147a7d8f8c17b4744c1a4da268a3374af575df9bc',
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                      ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def safe_path(relative):
    p = canonical_path(ROOT / relative)
    if not p.is_relative_to(ROOT) or p == ROOT:
        raise ValueError('Path outside project: ' + str(relative) + '; resolved=' + str(p) + '; root=' + str(ROOT))
    return p


def relative(path):
    return canonical_path(path).relative_to(ROOT).as_posix()


def canonical_path(path):
    value = str(Path(path).resolve())
    # Windows may retain the extended-length prefix when a new file's parent
    # is being created by another worker. Normalize its spelling, not scope.
    if value.startswith('\\\\?\\UNC\\'):
        value = '\\\\' + value[8:]
    elif value.startswith('\\\\?\\'):
        value = value[4:]
    return Path(value)


def code_hashes():
    files = sorted(PACKAGE.glob('*.py'))
    files += [ROOT / 'e4_windows/core/models.py', ROOT / 'e4_windows/core/original_unet.py',
              ROOT / 'e6_windows/pvt_timm_compat.py', ROOT / 'lib/pvt.py', ROOT / 'lib/pvtv2.py',
              STARTER / 'model/resnet50_model.py']
    return {relative(p): sha(p) for p in files}


def verify_weights():
    actual = {k: sha(v) for k, v in WEIGHTS.items()}
    if actual != EXPECTED_WEIGHTS:
        raise ValueError('Frozen model weight identity mismatch: ' + str(actual))
    return actual


def say(folder, message):
    print(message, flush=True)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'console.log').open('a', encoding='utf-8') as f:
        f.write(message + '\n')


def pid_alive(pid):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() != 87  # access denied is not proof of exit
        try:
            result = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(result)):
                return True
            return result.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


@contextlib.contextmanager
def exclusive_lock(path):
    """Reuse an orphan lock only after its same-host process is proven absent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = {'pid': os.getpid(), 'host': socket.gethostname(), 'created_utc': utc()}
    if path.exists():
        previous = read_json(path)
        if previous.get('host') != token['host'] or pid_alive(previous['pid']):
            raise RuntimeError('An E12 process may still be active: ' + str(previous))
        archive = path.with_name(path.name + '.orphan.' + str(previous['pid']))
        if archive.exists():
            archive = archive.with_name(archive.name + '.' + str(os.getpid()))
        path.replace(archive)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(token, f)
        yield
    finally:
        if path.exists() and read_json(path) == token:
            path.unlink()


def load_data(verify_bytes=False):
    info = read_json(DATA / 'bundle.json')
    plain = {k: v for k, v in info.items() if k != 'signature'}
    if signature(plain) != info['signature']:
        raise ValueError('E12 data descriptor changed')
    for name, expected in info['input_sources'].items():
        if sha(safe_path(name)) != expected:
            raise ValueError('Protected source manifest changed: ' + name)
    rows = {}
    for name, digest in info['manifests'].items():
        path = DATA / (name + '.json')
        if sha(path) != digest:
            raise ValueError('E12 split changed: ' + name)
        rows[name] = read_json(path)
        if verify_bytes:
            for i, row in enumerate(rows[name], 1):
                for kind in ['image', 'mask'] + (['bbox'] if 'bbox' in row else []):
                    if sha(safe_path(row[kind])) != row[kind + '_sha256']:
                        raise ValueError('Input bytes changed: ' + row['sample_id'])
                if i % 500 == 0:
                    print(f'[verify {name}] {i}/{len(rows[name])}', flush=True)
    return info, rows
