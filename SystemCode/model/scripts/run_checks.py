"""Run CPU synthetic-data regression checks, not GPU training or experiments."""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    for test in ['e0_mac/test_e0.py', 'e1_windows/test_e1.py', 'e2_windows/test_e2.py',
                 'v3_windows/test_v3.py', 'e4_windows/test_e4.py']:
        print(f'CHECK {test}', flush=True)
        subprocess.run([sys.executable, str(root / test)], cwd=root, check=True)
    print('PASS: all CPU functional checks. Windows CUDA probe and smoke remain separate.')


if __name__ == '__main__':
    main()
