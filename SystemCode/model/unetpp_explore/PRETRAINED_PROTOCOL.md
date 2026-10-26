# Pretrained U-Net experiment — 2026-10-25

Previous continuation, photometric, foreground-weight and multiscale trials did not close the evaluation gap. Retained control: validation 0.9030809542, known 798-image per-image Dice 0.7576873977 with validation-selected flip4; 139 below 0.5 and 42 zero Dice.

New bounded experiment: 100 epochs, seed42, official ImageNet ResNet34 encoder and freshly initialized GN U-Net decoder, input320, batch8 accumulated from microbatch2. Encoder parameters are trainable; BatchNorm running statistics are frozen. Model embeds ImageNet mean/std normalization for the existing RGB /255 data interface. AdamW LR1e-4 decays to1e-5, original BCE+Dice+0.2 boundary loss and paired augmentation. Architecture is adapted from the repository E2 implementation; no historical source was changed. This is a combined architecture/initialization change, not an isolated causal test of pretraining.

Train only original1160; validation290 with original256 targets. Select checkpoint using validation Dice including epoch0, compare with retained control, then select identity/flip4/rotate4 on validation and evaluate the selected model on unchanged798 native-resolution images. No evaluation images enter training or selection. If control remains selected, reuse its existing evaluation rather than reporting a new test of the unsuccessful model.

Runtime: project-local outputs/runtime contains torchvision0.23.0 matching existing torch2.8.0. Official encoder URL: https://download.pytorch.org/models/resnet34-b627a593.pth ; hash prefix verified before initialization. Full encoder hash is retained in checkpoint config. Official references: https://pytorch.org/get-started/previous-versions/ and https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet34 .

CPU smoke passed official checkpoint loading, output shape, complete backward and frozen-BN checks. Training/evaluation remain pending. Entry point: run_pretrained.py; results: outputs/unet_pretrained_20261025. Keep all prior failed runs and data unchanged. Target remains0.85, not yet achieved.

## Follow-up evidence

The100-epoch model selected epoch87 and rotation TTA on validation; known mean Dice0.8076202189254323,101 below0.5,27 zero. The40-epoch384 continuation selected flip4 on validation; known mean0.8119093727081123,91 below0.5,20 zero. ColonDB declined while ETIS improved; no universal subset improvement is claimed.

Next bounded run:20 epochs from384 best, fixed training-only difficulty sampling. Each of1160 draws per epoch uses probability0.5/N +0.5*(1-initial_training_Dice+0.05)/sum(1-initial_training_Dice+0.05). Scores use unaugmented training inputs and masks at384; validation and known evaluation do not determine sampling weights. All existing augmentation/loss settings retained, LR1e-5 to1e-6, effective batch8. Source train_pretrained_hard.py; run_pretrained_hard.py handles validation selection and evaluation. This tests a sampling policy with replacement, not a claim that evaluation failures have been fixed.

## ResNet50 V2 experiment

Hard sampling completed20 epochs without surpassing the initial validation score; retain384 ResNet34 baseline. Next run unet_resnet50_20261025 trains100 epochs with ImageNet V2 ResNet50 encoder, newly initialized GN decoder,320 input, effective batch8 via microbatch2, LR1e-4 to1e-5. Encoder BN stats frozen, parameters trainable, embedded normalization and previous loss/augmentation retained. It changes architecture and ImageNet initialization recipe together; no isolated capacity claim. Official weight https://download.pytorch.org/models/resnet50-11ad3fa6.pth verified by SHA256 prefix11ad3fa6; full digest saved in checkpoint config. Source confirmation: https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html . CPU smoke passed strict loading,320 forward/backward and frozen BN. Validate against retained384 model; evaluate new model only if selected. Evaluation remains the same798 known images. Entry run_resnet50.py.

## ResNet50 result and384 continuation

ResNet50 selected via validation, with flip4 TTA, achieved known798 mean Dice0.8184157503089938,100 images below0.5,12 zero. This improves average and zero count but worsens below0.5 count relative to ResNet34-384. Next bounded run unet_resnet50384_20261025:40 epochs from selected ResNet50 best, input384, LR1e-5 to1e-6, microbatch2/effective8, unchanged augmentation/loss. Source train_resnet50_384.py; orchestration run_resnet50384.py. Compare against prior ResNet50 on validation, retain baseline if no improvement, then evaluate only selected model. CPU selected-checkpoint384 forward/backward smoke passed.
