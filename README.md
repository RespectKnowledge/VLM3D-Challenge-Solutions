# MR-RATE NeuroVFM-LoRA

NeuroVFM-based MRI abnormality classification developed for the
Forithmus MR-RATE challenge.

## Model

- Backbone: NeuroVFM
- Fine-tuned checkpoint: MrRate.full.88540.MERGED.pt
- Training checkpoint step: 88,540
- Classification: 32 diagnostic labels
- Classifier: Linear head (512 -> 32)
- Postprocessing: Per-class calibrated thresholds

## Repository structure

- contrastive-pretraining/: Training and feature extraction
- docker/app/: Inference entrypoint and label definitions
- docker/src/: Model implementation and inference dependencies
- docker/Dockerfile: Submission container definition

## Model weights

Weights are hosted separately on Hugging Face.

Expected files:

- neurovfm/MrRate.full.88540.MERGED.pt
- neurovfm/neurovfm-encoder/
- classifier/linear_head.pt
- classifier/organizer_f1_calibration.json

## Reproducibility

The repository contains the training and inference source used to
prepare the challenge submission. Dataset access, preprocessing,
training configuration and model dependencies must be configured
before reproducing the results.

The original MR-RATE imaging data and patient-level split files
are not redistributed.

## Intended use

Research use in volumetric brain MRI abnormality classification.
Not validated for clinical diagnosis.
