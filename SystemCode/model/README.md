# Final E13 model release

This directory contains the final E13 ResNet50 U-Net SDK and release record. Production application integration is maintained in `SystemCode/backend`.

- [Install weights and use the SDK](endolens_e13/README.md)
- [Architecture, training data, evaluation and limitations](docs/FINAL_MODEL_RELEASE.md)
- [Historical experiments on the model branch](https://github.com/TianQinNus/AIS_EndoLensAI/tree/model/SystemCode/model)
- [Model development record](https://github.com/TianQinNus/AIS_EndoLensAI/blob/model/SystemCode/model/docs/EXPERIMENT_STATUS.md)
- [Final presentation](https://youtu.be/qXlUmzXPsoE)

The deployed configuration is E13, epoch 94, with four-rotation logit averaging, native pixel threshold 0.5 and region mean score >=0.8. This is an exploratory lower-false-alarm operating point; no E13 checkpoint met every prespecified validation protection criterion. It is not a clinically validated system.

Weights remain available through the existing Google Drive links and checksum-verifying installer.
