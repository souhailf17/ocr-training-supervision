> Document d’étape : pour le bilan consolidé du 30 octobre et les identités des modèles, lire [le dossier encadrant](encadrant/01-vue-ensemble.md).

# Training-efficiency branch — 3 October 2026

This branch preserves `main` as the saved baseline and adds opt-in candidates for the next isolated four-H200 preflight. It changes training code only; it does not alter a running OpenShift Job or deploy a new one. The 2 October metrics remain a historical baseline, not live cluster telemetry.

## Changes in this branch

- **DDP health checks:** finite-loss state and loss summaries remain on the GPU during gradient accumulation. Gradient clipping and one coordinated finite-gradient decision happen before each optimizer update, so a rank cannot continue updating while another rank detects invalid gradients. The former per-microstep all-reduce and Python scalar reads are removed from the pilot path.
- **Rank-skew protection:** the existing `torch.cuda.synchronize()` plus barriers remain every 16 microsteps by default (every microstep in smoke mode), because earlier runs experienced rank skew. The per-microstep finite-loss collective and Python scalar reads are removed; synchronization frequency is not reduced until matched H200 profiling and a long-sequence distributed smoke show it is safe. `Q35OCR_RANK_SYNC_INTERVAL` allows controlled experiments.
- **LoRA BF16 candidate:** `Q35OCR_LORA_COMPUTE_DTYPE=bf16` uses autocast for adapter matrix operations while keeping LoRA parameters, optimizer state and checkpoint keys in FP32-compatible form. The default remains `fp32` until H200 forward/backward, resume, validation and held-out OCR gates pass.
- **Pinned transfers:** the microbatch-3 continuation now pins the concatenated CPU batch after padding. This preserves the existing nonblocking device-copy path and avoids relying on pinning each example before collation.
- **Microbatch experiment:** the continuation reads `MICROBATCH` (default 3) and derives accumulation from the fixed global batch of 384. Setting `MICROBATCH=4` yields 4 GPUs × 4 examples × 24 accumulation steps = 384. This is only a candidate: run a long-page memory preflight before any full-data continuation.
- **Safe candidate preflight:** `Q35OCR_PREFLIGHT_ONLY=1` runs checkpoint, finite-gradient, update and exact-resume gates, then exits without starting the full-data loop. Do not interpret this as proof that the 8,192-token tail fits; include long examples in the memory gate.

## Not changed yet

Length bucketing is deferred because it changes which examples share a microbatch, and the current optimizer averages microbatch losses. A sampler change therefore needs a token-weighting decision, full-coverage/resume state tests, and matched OCR validation. Reusing the image processor's work between prompt and target encoding is also deferred until the tokenizer assistant-boundary mask and image tensors can be shown identical on actual Qwen 3.5 inputs. These are not claimed as optimizations in this branch.

## Verification status and rollout gate

The local workspace does not include PyTorch, Arsenal, a GPU runtime, or the `oc` executable. CPU-only configuration and repository tests can check source contracts but cannot validate H200 speed, memory, NCCL behavior, BF16 adapter gradients, or model quality. The branch must not be treated as deployed or as an ETA improvement until a scheduler-placed, isolated four-rank preflight reports per-rank finite gradients, a parameter update, exact checkpoint replay, no long-page OOM, and verified checkpoint read-back. Then compare at least 30 matched updates per candidate, plus warm-up exclusion, validation loss, supervised tokens/s, allocated/reserved memory and held-out CER/WER/page completeness. Keep the prior checkpoint and training job available for rollback.
