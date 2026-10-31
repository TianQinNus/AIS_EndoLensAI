# Research reproduction requirements

## Branch and directory roles

Use `main` for the deployed web application and `model` for experiment development. The chronological E0-E13 navigation lives under `experiments/`; executable packages retain their original flat names under `SystemCode/model`. Moving their Python files into E-number navigation folders would break sibling imports, relative artifact paths and source fingerprints.

```powershell
git clone --branch model https://github.com/TianQinNus/AIS_EndoLensAI.git
cd AIS_EndoLensAI/SystemCode/model
conda activate endolens-e2-win
```

The environment name denotes the original Windows environment, not a bundled environment. Inspect stage requirements before installing dependencies. Historical experiment execution used Python 3.11, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, NumPy 1.26.4 and Pillow 10.2.0 for later training. Earlier stages have their own requirements. E14 additionally uses OpenCV. GPU launchers generally require CUDA rather than falling back to CPU. Do not downgrade an active training environment to reproduce an older package.

## External assets that Git does not supply

| Asset | Expected location relative to SystemCode/model | Role |
|---|---|---|
| Original positive images/masks | dataset/TrainDataset and dataset/TestDataset | Historical E0-E4 data; later filtered identities come from frozen manifests |
| PolypGen v3 extraction | dataset/_downloads/PolypGen-v3/extracted/PolypGen2021_MultiCenterData_v3 | Positive expansion and authored negative frames |
| Frozen Polyp-PVT checkpoint | model_pth/PolypPVT.pth | E0 inference; SHA256 recorded in E12 common.py |
| ResNet34 initialisation | e2_weights/resnet34-b627a593.pth | ImageNet encoder for B-family training |
| ResNet50 initialisation | e13_joint_windows/resources/resnet50-11ad3fa6.pth | E13 ImageNet V2 encoder; expected SHA256 in experiment.py |
| Collaborator starter archive | ../Best_UNet_Negative_Starter_20261027.zip | E12 verifies and extracts the original supplied archive; place in SystemCode when using the unchanged loader |
| Collaborator positive-only R50 | e12_negative_windows/resources/Best_UNet_Negative_Starter_20261027/weights/best.pt | Historical comparator; separate from final E13 weights |
| Fixed split identities and run caches | runs/ under this directory | Stage-specific signed manifests, earlier checkpoints and native probabilities |

The starter's reviewed Python/configuration dependencies are included, but its ZIP and weights are not newly uploaded here. Final portable E13 weights are available through [the SDK installer](../endolens_e13/README.md); a portable inference weight is not a substitute for optimiser state, original training checkpoints or signed experiment manifests.

## Data roles

E13 uses 2,269 positive and 1,907 negative training images; validation contains 494 positive and 733 negative images. The reused C6 evaluation contains 83 positives and 1,760 negatives. Positive C1-C4 train, C5 validates, and C6 evaluates; negative sequences 1-10 / 11-13 / 14-23 follow train / validation / evaluation. Negative single frames require verified empty annotations. Missing labels do not imply a negative image.

E8/E9 50/50 means original versus added positive data. E13 50/50 means positive versus negative draws; conditional positive draws retain original-source mass 0.5 and each C1-C4 mass 0.125. Each epoch has 1,160 replacement draws, not one full pass over all images.

FAR-POLYP-SEG is an external positive-image development dataset. The historical split has 160 calibration images in 75 patient groups and 272 selection images in 112 groups. C6 is the sixth PolypGen centre. Both were repeatedly used during development; neither supports an untouched-test claim for the final selected rule.

## Integrity and recovery

1. Restore the exact historical artifacts and verify the expected hashes and split signatures.
2. Read the stage guide, then run its checks/smoke entrypoint before expensive training.
3. Run one GPU experiment at a time. Preserve source, checkpoint and manifest fingerprints.
4. If a run reports an active process or inconsistent fingerprints, inspect the process and records before recovery. Do not delete locks or edit status files to force continuation.
5. A NO_PASS result blocks downstream promotion; diagnostic best candidates are not automatically approved test/deployment configurations.

The source snapshot is published for review and reconstruction. A clean clone alone cannot replay artifact-dependent historical experiments. Machine-specific paths in inherited artifacts require deliberate relocation and verification, not fabricated signatures. No new training or evaluation was run for this documentation update.

## Published source audit

Run `python scripts/verify_experiment_archive.py` from `SystemCode/model`. It checks imported file hashes and Python syntax without loading a model, reading datasets, or running training. The manifest covers newly imported E5-E14 source/configuration dependencies and the shared E4 Python core. The E4 core text is unchanged; its original local byte format is preserved so later source fingerprints can be checked. Other existing E0-E4 and collaborator folders retain their repository history. Presentation README text is separate from frozen Python/CMD bytes.
