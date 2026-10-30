> Document d’étape : pour le bilan consolidé du 30 octobre et les identités des modèles, lire [le dossier encadrant](encadrant/01-vue-ensemble.md).

# Checkpoints and resume

## Saved checkpoint snapshot

At the 2 October 2026 snapshot, the run's best/latest marker recorded checkpoint step 8,487 after 1,430,144 examples, with validation loss 0.293362 and SHA-256 `2483f053aade0dfddff6d2de99d61c6ba1b4aea0c92c97b871308e6dfc3e01c9`. The marker and SHA are also summarized in `../metrics/training-snapshot-2026-10-02.json`. The checkpoint file and optimizer state are stored outside Git in private PVC/MinIO storage.

These repository files are a dated record; they do not establish current checkpoint availability. Before resuming, inspect the live job and verify that the checkpoint exists in both storage locations and that its computed SHA-256 matches the retained marker.

## Resume contents and safeguards

A resumable state includes:

- LoRA adapter parameters;
- optimizer state and learning-rate schedule/clock;
- full-pass sample cursor;
- Python, NumPy, CPU Torch and per-rank CUDA RNG states;
- world size, configuration and source-checkpoint identity.

Do not overwrite retained checkpoint names or resume from an unverified partial upload. The continuation performs a controlled update-and-replay check before training, then checks all ranks and the verified artifact round trip. Never treat `latest.json` or an old status file as live job status.

The repository intentionally excludes model weights, adapter/checkpoint binaries, raw OCR images and credentials.
