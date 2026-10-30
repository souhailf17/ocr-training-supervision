> Document d’étape : pour le bilan consolidé du 30 octobre et les identités des modèles, lire [le dossier encadrant](encadrant/01-vue-ensemble.md).

# Training configuration and saved run

## Configuration

The following configuration was recorded in the 2 October 2026 snapshot:

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Training objective | Image-to-text OCR; assistant answer tokens supervised |
| Base weights | Frozen, BF16 |
| Adapters | LoRA rank 16, alpha 32, scale 2, no dropout |
| Adapted layers | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` in language-model layers |
| Optimizer | AdamW, weight decay 0.01, gradient clipping 1.0 |
| Learning rate | Sample-based cosine schedule, peak `1e-4`, minimum `1e-5`; restored from checkpoint |
| Distributed | Four-rank DDP replication on four H200 GPUs |
| Per-GPU microbatch | 3 examples |
| Gradient accumulation | 32 microsteps |
| Global batch | 384 examples per optimizer update |
| Maximum combined input length | 8192 tokens; oversized samples fail instead of being silently truncated |
| Image preprocessing | Aspect-preserving thumbnail, maximum 1024 × 1024 |
| Activation checkpointing | Enabled, non-reentrant |

The saved run kept trainable parameters in FP32 and computed its LoRA branch in FP32, while the frozen base model ran in BF16. The optimization candidate adds `Q35OCR_LORA_COMPUTE_DTYPE=bf16` for adapter matrix operations while preserving FP32 parameter storage. The default remains FP32; enable the candidate only after the GPU preflight and OCR quality gates in [`optimization-branch-2026-10-03.md`](optimization-branch-2026-10-03.md). This is a custom LoRA implementation, not a dependency on the Hugging Face PEFT package.

## Snapshot as of 2 October 2026

The saved status file reported the job as running at global step 8,494. The deterministic full pass had consumed 1,432,832 of 3,683,080 training examples (38.90%). The continuation began from a verified checkpoint at step 7,878 and 1,196,288 examples. This repository does not establish the job's current state after that snapshot.

The best checkpoint marker recorded step 8,487, 1,430,144 examples and validation loss 0.293362. The recorded baseline validation loss was 0.326533, a 10.16% decrease. Latest training loss was 0.171045. The 616 updates in this run segment averaged 32.675 seconds per update and 11.664 examples per second end to end.

The validation loss is token-weighted over a fixed 256-example set; train loss is logged as a mean of microbatch losses. Their absolute gap should not be interpreted as an overfitting measurement. OCR character/word error rate and page completeness require separate held-out inference evaluation.

See [`../metrics/training-snapshot-2026-10-02.json`](../metrics/training-snapshot-2026-10-02.json), [`../metrics/training-metrics-step7879-8494.csv`](../metrics/training-metrics-step7879-8494.csv), and the [dashboard](../metrics/training-dashboard-step8494.png). GPU telemetry covers 60 samples per GPU across approximately five minutes; sample timing averaged 5.244 seconds.

## Checkpointing and recovery

Training checkpoints contain adapter weights, optimizer, scheduler/clock, dataloader cursor and per-rank RNG state. The saved continuation preflight verified the parent checkpoint's local and S3 checksums, finite gradients, parameter updates, execution of the optimized FLA/causal-convolution kernels, and four-rank replay error of zero. The corresponding saved best-checkpoint SHA-256 is in the numerical snapshot. Model weights and checkpoint files are intentionally excluded from GitHub.

Do not reuse a saved status JSON as live status. Before resuming, inspect the actual OpenShift Job and all rank logs, verify the retained checkpoint against its SHA-256 in both storage locations, and preserve the exact dataset cursor and scheduler history.

## Runtime dependencies and reproducibility

The scripts expect the prepared project volume and a Forge Arsenal runtime. The recorded optimized runtime used Arsenal/Forge 2026.7.11, PyTorch 2.11.0+cu130, `fla-core` 0.5.2, `causal-conv1d` 1.7.0 and Triton 3.7.1. These runtime packages and model/data artifacts are not vendored in this repository. The scheduler module `lr_scheduler_basic` is provided by the training runtime and is not present among the project sources captured here; obtain that dependency from the authorized runtime before attempting reproduction.

Provide S3 credentials through an OpenShift Secret at runtime; never put tokens or credential values in a manifest or Git. Launch scripts refer to the original project namespace and storage conventions and need review before reuse in another cluster.
