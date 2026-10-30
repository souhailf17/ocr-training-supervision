> Document d’étape : pour le bilan consolidé du 30 octobre et les identités des modèles, lire [le dossier encadrant](encadrant/01-vue-ensemble.md).

# Training architecture

## Data and model flow

The source model and OCR dataset are pinned by revision in `model.md` and `dataset.md`. Raw TAR shards and model files are retained in private object storage; the training job reads image byte ranges, verifies image/annotation hashes, preprocesses each page and forms supervised image-to-text examples. No source dataset or model weights are included in this Git repository.

The training process runs the Qwen 3.5 multimodal model with a frozen BF16 base and custom trainable LoRA adapters in the language projections. The saved snapshot used four-rank PyTorch DDP on H200 GPUs, FLA/causal-convolution kernels, activation checkpointing, length-checked batches, AdamW and a sample-based cosine learning-rate schedule.

## Storage and checkpoints

OpenShift mounts the prepared project volume and injects S3-compatible credentials at runtime. Checkpoints, logs and metric records are persisted to the project PVC and verified in MinIO. The dataset, model weights, credentials and checkpoint binaries stay outside Git. See `checkpoints.md` for the saved checkpoint and resume checks.

## Scope and reproducibility

The repository includes application scripts, tests, documentation and numerical evidence. It does not include the Forge Arsenal runtime itself, the model repository, the raw/processed dataset, cluster credentials or deployable model weights. A matching authorized runtime, storage access, and OpenShift configuration are required to reproduce the saved run.
