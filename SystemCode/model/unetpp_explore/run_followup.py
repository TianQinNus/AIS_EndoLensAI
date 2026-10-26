"""Queue two training arms after the current GPU evaluation, then evaluate winner."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from evaluate_tta import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait-for', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='mps', choices=['cpu', 'mps', 'cuda'])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(status='WAITING_FOR_INFERENCE', target=.85, metric='mean per-image Dice')
    save(args.output/'status.json', status)
    scripts = Path(__file__).resolve().parent

    def run(command, log):
        with log.open('w') as f:
            subprocess.run([sys.executable, '-B', *map(str, command)], stdout=f,
                           stderr=subprocess.STDOUT, check=True)

    try:
        deadline = time.monotonic()+6*3600
        while True:
            previous = json.loads(args.wait_for.read_text())
            if previous['status'] == 'COMPLETED':
                break
            if previous['status'] == 'FAILED':
                raise RuntimeError('Preceding inference failed; inspect its protocol')
            if time.monotonic() > deadline:
                raise TimeoutError('Preceding inference did not finish within six hours')
            time.sleep(30)
        scores = {}
        for arm in ['control', 'photometric']:
            status.update(status='TRAINING', arm=arm)
            save(args.output/'status.json', status)
            command = [scripts/'finetune.py', '--checkpoint', args.checkpoint,
                       '--output', args.output/arm, '--device', args.device]
            if arm == 'photometric':
                command.append('--photometric')
            run(command, args.output/f'{arm}.log')
            scores[arm] = json.loads((args.output/arm/'status.json').read_text())['best_val_dice']
        selected = max(scores, key=scores.get)
        save(args.output/'locked_arm.json', dict(selected=selected, validation_dice=scores,
                                                rule='maximum validation Dice; ties prefer control'))
        status.update(status='EVALUATING', selected=selected)
        save(args.output/'status.json', status)
        run([scripts/'evaluate_tta.py', '--checkpoint', args.output/selected/'best.pt',
             '--output', args.output/'selected_evaluation', '--device', args.device],
            args.output/'evaluation.log')
        summary = json.loads((args.output/'selected_evaluation/summary.json').read_text())
        status.update(status='COMPLETED', known_evaluation=summary,
                      target_reached=summary['ALL']['dice'] >= .85)
    except BaseException as e:
        status.update(status='FAILED', error=repr(e))
        raise
    finally:
        save(args.output/'status.json', status)


if __name__ == '__main__':
    main()
