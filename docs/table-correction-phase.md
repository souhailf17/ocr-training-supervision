> Document d’étape : pour le bilan consolidé du 30 octobre et les identités des modèles, lire [le dossier encadrant](encadrant/01-vue-ensemble.md).

# Isolated tables-only corrective phase

The original full pass remains complete and its files are retained. This new
phase initializes the identical custom LoRA tensors from checkpoint 14355
(`e32116d77a324a8b970503e23440dbc83434a21b171a2ec3bcb87d14e6c503e6`).
It uses a fresh optimizer and a new cursor/schedule for the corrected objective.
It is not a continuation of the old completed full-stream cursor.

## Data

The CPU preparer selects 512 original Arabic train pages containing verified
tables from each of the first eight original shards: 4096 unique training groups.
It expands each table reference into its complete original HTML and preserves
the surrounding text. Training contains no pages without tables and no augmented
views in this initial pass. Original/augmentation group split membership remains
unchanged in the original manifests.

Separate held-out sets: 64 validation table pages, 32 validation control pages
without tables, and 16 reserved test table pages. Every accepted image/annotation,
source manifest, corrected label and actual processor token count is checked.
All files are copied to a separate PVC directory and SHA-verified MinIO prefix
before DATA COMPLETE is published. No source object or historical file is deleted.

Prepared counts: 5284 training tables / 123124 cells. Maximum observed combined
sequence length: 3204 tokens; configured limit remains 8192. No candidate was
rejected in this preparation. This is a bounded first correction pass, not all
table pages in the complete 103-shard corpus.

## Initial configuration

| Parameter | Value |
|---|---|
| Source adapter | checkpoint 14355 |
| GPU request | one unused H200, chosen by scheduler |
| Microbatch / accumulation / global batch | 1 / 32 / 32 |
| Passes | one deterministic pass, 128 updates |
| LoRA rank / alpha / compute | 16 / 32 / FP32 |
| Base model | pinned Qwen3.5-9B, BF16, SDPA |
| Gradient checkpointing | enabled, non-reentrant |
| Image thumbnail | 1024 pixels maximum side |
| Maximum combined sequence | 8192 tokens |
| Optimizer | fresh AdamW, weight decay 0.01 |
| Learning rate | eight-update warmup to 2e-5, cosine to 1e-5 |
| Gradient clipping | norm 1.0 |
| Kernels | existing isolated FLA / causal-conv1d / Triton 3.7.1 |
| CPU prefetch | ordered, depth 4; consumed cursor saved |
| Validation | baseline, every 32 updates and final; token-weighted loss |
| Checkpoints | initial, first actual update, every 32 updates and stop/final |

This runner uses PyTorch/Transformers directly with the identical verified custom
LoRA forward path. It does not rewrite the historical Arsenal trainer. There is
no DDP for a single-GPU job.

Before the pass, real forward/backward checks require finite gradients, an actual
adapter update, nonzero executed kernels, exact checkpoint/replay below 1e-5 and a
SHA-verified PVC/S3 checkpoint. Those temporary updates are then rolled back.
Only `TABLE_CORRECTION_PREFLIGHT_PASSED`, followed by `TABLE_CORRECTION_METRIC`
and a verified actual checkpoint, demonstrates that training started.

The runner refuses a busy GPU and stops on nonfinite/unstable values, repeated
validation degradation, a phase-owned STOP file or low disk, without deleting
files. Loss on the corrected labels is not directly comparable to the old loss.
OCR cell completeness on held-out pages must still be tested after the pass;
the runner deliberately makes no claim that tables are fixed from loss alone.

## Paths and explicit resume

PVC directory: `/work/q35ocr-20260921-p560/training-tables-correction-v1`.
MinIO prefix: `s3://qwen35-9b-ocr-finetuning/training-tables-correction-v1/`.

```sh
python scripts/train_table_correction.py \
  --root /work/q35ocr-20260921-p560/training-tables-correction-v1 \
  --model-dir /work/q35ocr-20260921-p560/model \
  --parent /work/q35ocr-20260921-p560/training-fast-gpu4-m3-from12995-v1/checkpoint-14355.pt
```

After an explicit user request to resume, use `--resume` with the path recorded
in this phase's verified `latest.json`. The dataset, parent, phase configuration,
optimizer, consumed cursor, RNG and validation health must match. A fresh runtime
gate is repeated and rolled back before further updates. Completed phases and
historical failed trainers are never restarted automatically.

The submitted job has priority 0, no requested node/affinity/priority class,
backoffLimit 0 and a six-hour deadline. Existing admission constraints are kept.
Pending means it is waiting for resources; it does not mean learning has started.
