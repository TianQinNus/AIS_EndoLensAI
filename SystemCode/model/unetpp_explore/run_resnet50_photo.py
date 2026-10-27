"""Bounded multiscale experiment, matched to the completed control."""
import json
from pathlib import Path
import subprocess
import sys
from evaluate_tta import ROOT, save


def main():
    output = ROOT.parents[1]/'outputs/unet_resnet50_photo_20261026'
    output.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    checkpoint = ROOT.parents[1]/'outputs/unet_resnet50_20261025/train/best.pt'
    status = dict(status='TRAINING', target=.85, factor='ImageNet ResNet50 V2 encoder + GN U-Net decoder; saturation/gamma augmentation continuation30 epochs',
                  control='outputs/unet_resnet50_20261025/train',
                  initialization=str(checkpoint), epochs=30)
    save(output/'status.json', status)
    try:
        with (output/'train.log').open('w') as f:
            subprocess.run([sys.executable, '-B', str(scripts/'train_resnet50_photo.py'),
                            '--checkpoint', str(checkpoint), '--output', str(output/'train'),
                            '--photometric', '--device', 'mps', '--epochs', '30', '--lr', '0.00001'],
                           stdout=f, stderr=subprocess.STDOUT, check=True)
        # Compare against the matched control using validation only before test.
        control = ROOT.parents[1]/status['control']
        scores = {str(p): json.loads((p/'status.json').read_text())['best_val_dice']
                  for p in [control, output/'train']}
        selected = Path(max(scores, key=scores.get))
        save(output/'locked_selection.json', dict(selected=str(selected), validation_dice=scores,
                                                rule='highest validation mean Dice; ties prefer control'))
        status.update(status='EVALUATING', selected=str(selected))
        save(output/'status.json', status)
        if selected == control:
            # Identical selected weight and protocol already evaluated; do not rerun.
            summary = json.loads((control.parent/'evaluation/summary.json').read_text())
            status['evaluation_reused_from'] = str(control.parent/'evaluation')
        else:
            with (output/'evaluation.log').open('w') as f:
                subprocess.run([sys.executable, '-B', str(scripts/'evaluate_resnet50.py'),
                                '--checkpoint', str(selected/'best.pt'), '--output', str(output/'evaluation'),
                                '--device', 'mps'], stdout=f, stderr=subprocess.STDOUT, check=True)
            summary = json.loads((output/'evaluation/summary.json').read_text())
        status.update(status='COMPLETED', known_evaluation=summary, target_reached=summary['ALL']['dice'] >= .85)
    except BaseException as e:
        status.update(status='FAILED', error=repr(e))
        raise
    finally:
        save(output/'status.json', status)


if __name__ == '__main__':
    main()
