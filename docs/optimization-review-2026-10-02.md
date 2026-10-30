> Document d’étape : pour le bilan consolidé du 30 octobre et les identités des modèles, lire [le dossier encadrant](encadrant/01-vue-ensemble.md).

# Qwen OCR — training optimization review

Analysis of the saved 2 October 2026 report, not a new live cluster status. No training setting, job, checkpoint or GPU allocation was changed for this review.

## Assessment

The measured run is learning and already uses four-GPU DDP, BF16, optimized FLA/causal-conv1d kernels, CPU prefetch and gradient accumulation. Adding DDP is therefore not the next improvement. The most concrete candidates are synchronization overhead, unnecessary FP32 work in custom LoRA, and inefficient batching of variable-length pages. Their individual speedups have not yet been measured.

Keep the current maximum sequence length of 8,192 and the existing LoRA rank/alpha while investigating throughput. Do not obtain a misleading speedup by dropping long pages, shortening targets, or reducing image detail.

## Evidence and limitations

| Measurement | Saved result |
|---|---:|
| Global step | 8,494 |
| Full-pass coverage | 1,432,832 / 3,683,080 examples — 38.90% |
| Updates analyzed in this continuation | 616, steps 7,879–8,494 |
| Examples processed in this continuation | 236,544 |
| Training-loop elapsed time | 20,279.82 seconds — 5 h 38 min |
| End-to-end throughput | 11.664 examples/s |
| Mean optimizer-update duration | 32.675 seconds |
| 95th percentile update duration | 34.644 seconds |
| Latest train loss | 0.171045 |
| Mean train loss over 616 updates | 0.175739 |
| Train loss range | 0.105560–0.250914; all finite |
| Baseline validation loss | 0.326533 |
| Latest/best recorded validation loss | 0.293362, step 8,487 |
| Relative validation-loss decrease | 10.16% |
| Latest learning rate | 0.0000676835 |

Validation improved consistently across the eight recorded evaluations. Mean train loss was 0.176153 over the first 100 updates and 0.177362 over the last 100: approximately flat on different training examples, without evidence of divergence in this window. This does not establish final OCR accuracy or rule out every generalization problem.

Train loss is an average of microbatch losses; validation loss is weighted by valid target-token counts. Their absolute gap is not directly interpretable as an overfitting measure. A 10.16% decrease in cross-entropy is not a 10.16% decrease in OCR character errors.

The latest recorded retained checkpoint is `checkpoint-8487.pt` (step 8,487; 1,430,144 examples), also identified as best in the saved markers, with SHA-256 `2483f053aade0dfddff6d2de99d61c6ba1b4aea0c92c97b871308e6dfc3e01c9`. These are saved verification records, not a fresh checksum run.

## Configuration already in use

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen3.5-9B |
| Parallelism | Four H200 GPUs, DistributedDataParallel |
| Physical microbatch | 3 examples per GPU |
| Gradient accumulation | 32 microsteps |
| Global batch | 4 × 3 × 32 = 384 examples |
| Maximum input sequence | 8,192 tokens, explicit rejection rather than silent truncation |
| Image preprocessing | Aspect-preserving thumbnail within 1,024 × 1,024 |
| Base-model precision | BF16 |
| Custom LoRA | Rank 16, alpha 32, scale 2, no dropout |
| Trainable parameters | 29,097,984 across 128 adapted modules |
| LoRA weights and current branch arithmetic | FP32 |
| Optimizer | AdamW, weight decay 0.01; gradient clipping 1.0 |
| Learning-rate schedule | Sample-based cosine, configured maximum 1e-4 and minimum 1e-5; restored schedule history |
| Activation checkpointing | Enabled, non-reentrant |
| Attention/kernels | SDPA; fla-core 0.5.2; causal-conv1d 1.7.0; Triton 3.7.1 |
| Prefetch | One producer thread per rank; depth 4 |

The runtime preflight recorded actual optimized-kernel calls and zero checkpoint-replay error on all four ranks. Do not describe kernel installation or DDP activation as new work still required.

## GPU measurements

| GPU | Mean utilization | Samples at ≥90% | Reported memory used / total |
|---|---:|---:|---:|
| 0 | 83.77% | 66.67% | 113.35 / 140.40 GiB |
| 1 | 80.93% | 60.00% | 61.51 / 140.40 GiB |
| 2 | 75.32% | 38.33% | 134.29 / 140.40 GiB |
| 3 | 76.65% | 51.67% | 84.12 / 140.40 GiB |

All four GPUs were active; their combined mean was 79.17%. The capture contains 60 measurements per GPU, 240 total. Correcting the previous nominal description: first-to-last measurement spanned 309.39 seconds, and mean spacing was 5.244 seconds, because collection time was added to the five-second sleep.

These are sampled readings, not continuous five-second averages. NVIDIA defines GPU utilization as the fraction of a short sampling window with a kernel executing, not percentage of theoretical compute throughput. Therefore 100% does not establish optimal performance, and 79.17% does not quantify recoverable performance. [NVIDIA documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

Memory differences warrant investigation, but do not prove a leak or current rank imbalance. Cached allocations also appear as used in NVIDIA reporting; collect both allocated and reserved PyTorch peaks during real long-page updates. [PyTorch memory management](https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management)

## Recommended experiments, in order

### 1. Measure and reduce synchronization inside accumulation

In the inspected `PilotTrainer.compute_loss`, every microstep performs a scalar distributed finite-loss check, reads `finite.item()`, and converts GPU loss to Python floats. The inspected `AccumulationMixin` also explicitly synchronizes CUDA and runs a barrier every 16 microsteps. With accumulation 32, that means 32 finite-check collectives plus two explicit synchronization/barrier points per normal update, in addition to required gradient synchronization.

DDP `no_sync()` is already present for intermediate backward passes. It does not suppress these explicit finite-check collectives. Host scalar reads also restrict CPU/GPU overlap. [PyTorch tuning guidance](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization)

First capture a short all-rank trace. Then test GPU-resident loss accumulation and a coordinated finite-loss/gradient decision at the update boundary, strictly **before** any optimizer update. Retain fail-fast behavior and coordinated failure across ranks. Reduce extra barriers only if the trace supports it and long-page stress tests remain healthy; they were introduced after historical rank-skew/NCCL failures. Do not merely delete checks or raise the timeout.

Source: saved trainer imports `PilotTrainer`; inspected workspace dependencies `pilot_train.py:120` and `large_batch.py:68`. Verify mounted dependency hashes against those local files before implementation.

### 2. Test mixed-precision arithmetic in the custom LoRA branch

`code/arsenal_ocr.py:35` disables autocast, calls `x.float()`, and executes both adapter linear operations in FP32. This repeats across 128 adapted modules, while the frozen base uses BF16. It is a concrete source of extra conversions and higher-precision computation, although its share of runtime is not yet measured.

Test BF16 autocast for the adapter matrix operations while preserving FP32 trainable parameters and optimizer state. Keep checkpoint parameter names, rank 16 and alpha 32. Compare throughput, peak memory, finite gradients, validation loss and held-out OCR outputs. A precision change need not reproduce the old trajectory bit-for-bit; checkpoint replay within the new configuration must still pass. [PyTorch automatic mixed precision](https://docs.pytorch.org/docs/main/amp.html)

### 3. Group similar lengths and balance work across ranks

The current `collate` pads three consecutive examples to their longest sequence. Examples are not grouped by sequence length, and equal example counts do not guarantee equal text-token and image-token work across GPUs.

Log padding fraction, non-padding tokens, supervised target tokens, image tokens and forward/backward time per rank. Test grouping by text length and visual-token cost, ideally redistributing the same 384 examples within each existing global update. Preserve every example, split membership and full-pass coverage. Save any buffering/reordering state in checkpoints so resume cannot duplicate or skip data.

Batch grouping can also alter the current mean-of-microbatch-loss weighting. Establish the intended normalization and check gradients/validation; preserving the example count alone does not guarantee identical optimization. Add a token-weighted train-loss metric for comparison with evaluation before changing the objective.

### 4. Remove duplicated preprocessing; verify pinning after collation

`OCRLoader.encode` applies the multimodal processor twice to the same image, once for the prompt and once for prompt plus answer. Investigate reusing image processing and constructing the assistant mask without the repeated work. Require identical token IDs, pixel values, image grids and labels on a representative equivalence test.

The current stream pins individual examples before padding and concatenation create the final batch. Check final-batch `is_pinned()` and pin after collation in the producer if needed. Nonblocking device transfer is already present in the inspected Arsenal runtime; merely adding that flag again is not an optimization. A dedicated transfer stream is a later experiment if transfer latency is material. [PyTorch pinned-memory guidance](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)

Increase producer capacity or prefetch depth only if current data-wait measurements justify it. The historical optimized-prefetch benchmark recorded roughly 0.137 seconds of data waiting per 17.98-second update, so data loading is not established as the primary current bottleneck. That older benchmark used different samples and shorter sequences.

### 5. Test microbatch 4 only after measuring memory headroom

The next controlled candidate is **4 GPUs × microbatch 4 × accumulation 24 = global batch 384**, compared with the present 4 × 3 × 32. This keeps examples per optimizer update constant. Keep the learning-rate history and maximum sequence length unchanged.

GPU 2 reported about 134.29 GiB used out of 140.40 GiB. This includes possible cache, so it neither proves microbatch 4 will fail nor proves it will fit. Test real long-page and near-8,192-token batches, record allocated/reserved peaks, and retain a memory margin. The earlier matched benchmark reached only 2,808 tokens and is insufficient for this memory decision. Do not jump to microbatch 6 or 8 on the strength of average-length tests.

## Lower-priority changes

- Keep validation and verified checkpoints. Only about 152.25 of 20,279.82 recorded seconds fall outside measured update calls (0.75%). This rank-0 accounting is approximate, but it provides no evidence that reducing evaluation or checkpoint safety would deliver a large gain.
- Do not restart the learning-rate schedule or raise the learning rate simply to improve throughput; learning rate does not shorten a forward/backward pass.
- Do not reduce LoRA rank, maximum sequence length, or image resolution as a first speed fix. Those change learning capacity or the information available for OCR.
- Inspect the actual SDPA backend before proposing another attention package. SDPA may already select a fused backend.
- Consider compilation or selective activation checkpointing only after stabilizing shapes and measuring compute/memory costs. Fused AdamW is a lower priority: the old benchmark measured only milliseconds in optimizer work.
- Do not migrate to QLoRA/FSDP solely because NVIDIA reports high memory usage. First distinguish active tensors, allocator cache, variable-length activations and redundant casts.

## Quality safeguards

Retain the existing 256-example validation set for a comparable learning curve, and add a separately held-out, group-disjoint OCR evaluation covering dense Arabic pages, long targets, tables and varied layouts. Report CER/WER and page completeness, with the same decoding settings for original and fine-tuned models.

The 1,024-pixel thumbnail can remove fine text detail; that is a quality hypothesis to test on difficult pages, not a conclusion established by these loss logs. Keep the user's training sequence limit unchanged. Diagnose missing page endings using output-token limits, EOS behavior, input readability and held-out long-page performance rather than assuming a single cause.

## Acceptance plan

1. Start from a retained, verified checkpoint and record all runtime/module hashes, software versions and random states.
2. Trace a few representative updates on all four ranks, including a long-page batch, with CPU/GPU activity, memory, collectives and data-wait timing. Profiling is bounded; final speed measurements run without tracing. [PyTorch profiler](https://docs.pytorch.org/docs/main/profiler.html)
3. Change one factor at a time. Compare the same examples and length distribution, using five warm-up updates and at least 30 measured updates per candidate, plus a worst-length memory test. Repeat close results.
4. Compare examples/s and non-padding/supervised tokens/s, update median/p95, padding ratio, rank wait, data wait and memory peaks. Use at least a repeatable 10% gain as a practical adoption threshold, not a promised outcome.
5. Require no nonfinite loss/gradients, correct coverage, unchanged labels, successful distributed execution, checkpoint/replay validation and no material held-out regression. Explicitly evaluate objective or numerical changes from new batching/precision.
6. Roll out only the validated candidate through a checkpointed continuation. Preserve previous artifacts and rollback information.

## Time implications — scenarios, not predictions

At the saved snapshot, 2,250,248 examples remained. At 11.664 examples/s that was approximately 53.59 hours. This is not a newly measured remaining time.

| Hypothetical sustained end-to-end rate | Time for those remaining examples |
|---|---:|
| 14 examples/s | 44.65 hours |
| 16 examples/s | 39.07 hours |
| 18 examples/s | 34.73 hours |

No gain above the measured baseline is guaranteed. The next decision should use matched experiments on the actual long-page workload, not extrapolation from GPU model names or utilization percentages.

## Local evidence and reproducibility

Primary snapshot: `raw/metrics.jsonl`, `raw/status.json`, `raw/preflight.json`, `raw/latest.json`, `raw/best.json`, the raw GPU capture and `code/` inside this report directory. Derived statistics are saved in `optimization-analysis.json`.

The report's code subset omits several imported modules, including `pilot_train.py`, `large_batch.py`, `distributed_ocr.py` and `train_full.py`. This review additionally inspected their existing workspace copies under `outputs/qwen35-ocr/scripts`. It did not obtain a fresh mounted-ConfigMap snapshot. The launcher references these modules, but byte-for-byte deployed identity must be confirmed before applying dependency-specific changes. Include those dependencies and an environment lock/manifest in the next reproducibility package.
