> Document d’étape : pour le bilan consolidé du 30 octobre et les identités des modèles, lire [le dossier encadrant](encadrant/01-vue-ensemble.md).

# Qwen OCR throughput investigation — initial report, 24 September 2026

> Historical investigation. Its benchmark plan and pilot estimates predate the optimized four-H200 run. The latest measured state and updated recommendations are in [the 2 October review](optimization-review-2026-10-02.md) and [training snapshot](../metrics/training-snapshot-2026-10-02.json). Do not treat the earlier status, absent-kernel claim or proposed configuration as current.

## Findings backed by this project's code/logs

- Pilot: 53,504 examples / 31,617.05 seconds = 1.692 examples/s end to end. Median update excluding recorded validation updates:128.95s for256 examples, or1.985examples/s. This is a measured update-window rate, not a GPU-only rate.
- At the pilot end-to-end rate,3,683,080 training examples would require25.19days. This extrapolation assumes identical length mix, I/O and validation overhead. The pilot oversamples text-length strata; it is not an unbiased full-dataset time prediction. The earlier18-day figure is not a validated promise.
- Pilot log explicitly reports unavailable fast path and PyTorch fallback on all4ranks. Runtime probe: Torch2.11.0+cu130, Transformers5.8.0, Python3.13, Arsenal2026.7.11. Identify the exact missing kernels/import failures in this image before choosing compatible builds.
- Already DDP:4 complete replicas, disjoint rank examples, no model sharding. Merely switching to DDP cannot accelerate an already-DDP run.
- Each rank synchronously requests a byte range from a TAR in MinIO, decodes/resizes and runs chat-template processing twice (prompt and full text). There is no background prefetch. These are range requests, not millions of separate tiny S3 objects.
- Synchronization: finite-loss all-reduce every microstep, CPU scalar reads, and CUDA synchronization/barrier every16microsteps. These can limit overlap; the barriers were introduced after a real collective failure, so removing them requires correctness tests.
- Physical batch2/3 previously OOMed on real long samples despite passing short smoke tests. Do not repeat blind increases. Batch1 plus checkpointing is currently the proven safe configuration.
- Validation runs only on rank0 while other ranks wait;256 examples are evaluated frequently. Keep evaluation but shard its workload across ranks with exact global loss weighting.
- Labels/augmentation grouping: deterministic group splits already exist. Verify them, do not rebuild them unnecessarily. Table-link labels and limited test generation are separate quality problems.

## Changes prepared locally, not yet deployed

`scripts/ordered_prefetch.py`: bounded ordered CPU producer to overlap range reads/processing with GPU work; no sample reordering or label changes. Persists the consumed cursor, not the ahead-of-consumer producer position. Four CPU unit tests pass: order, bounded lookahead, exception propagation, exact sample-sequence resume. This is not yet a demonstrated GPU speedup or exact numerical training-resume test.

`scripts/audit_qwen_performance.py`: read-only installed-package/kernel-symbol/GPU/CPU-limit/topology audit for the actual runtime image.

No live training code or image was changed during the initial investigation. SSH timed out at that stage, so live profiling and package compatibility checks were temporarily blocked.

## Prioritized experiment, separate from full training

1. Reconnect and inspect project jobs, available GPUs and image dependencies. Preserve all checkpoints, old jobs and unrelated workloads.
2. Record baseline on a fixed, hashed sample manifest with the observed long-sequence tail. Use4GPU,global256,micro1,accum64 and unchanged LR/LoRA/image resolution. Separate warmup/compilation from steady-state measurements. Record token/visual-token distributions and allocated GPU memory.
3. Build a separate pinned dependency environment with compatible causal-conv1d and FLA training kernels; do not upgrade shared Torch/Transformers blindly. Verify actual executed kernels using profiler traces (package presence alone is insufficient), finite forward/backward, parameter updates, numerical agreement within explicit tolerances and checkpoint restoration. Keep the fallback baseline available. Fast kernels may not satisfy bit-exact deterministic requirements; disclose and test this rather than silently disabling determinism.
4. Compare synchronous input with bounded ordered prefetch at the same examples/globalbatch. Start modestly under the pod CPU/memory quota. Add pinned-memory async transfer after validating memory pressure and reproducibility. Measure data-wait, S3 latency, decode/tokenization and compute separately.
5. Use a bounded raw-image/shard cache. The raw dataset is398GiB and the PVC96GiB: staging everything on that PVC is impossible. Whole-shard staging is useful only with locality-aware sampling; the pilot's globally shuffled sample order can make it wasteful. Benchmark bounded range caching against sequential per-shard access without silently dropping examples.
6. Bucket by actual encoded length and visual tokens, retaining all long samples in an explicit bucket. Then test micro2/accum32 and micro4/accum16 with global256. Use one case per isolated bounded job; stop on OOM. Disable checkpointing only in a separate memory-tested case. Largest microbatch is not necessarily fastest.
7. Audit DDP synchronization separately; compare1GPU vs4GPU on matched global work. Scaling efficiency is T4/(4*T1); do not promise superlinear scaling. Profile the per-microstep finite check before considering coordinated checks at safe update boundaries; never remove failure coordination blindly.
8. Spread validation over all ranks, preserving the same256 examples and sum(loss*tokens)/sum(tokens). Keep checkpoint frequency defined in both elapsed time and updates. SHA checks remain mandatory.

For each case record wall examples/s, input/label/visual tokens/s, GPU utilization/power/memory, CPU use/quota, data wait, transfer, forward/backward/optimizer and NCCL time, max sequence length, failure status and checkpoint gates. Use coarse profiling separately: forced CUDA synchronization around every phase perturbs throughput.

First benchmark budget:30–60min, with individual cases bounded, benchmark weights isolated, no checkpoint pruning and no full-data launch. Keep exact corpus/order/configuration identifiers. Do not claim speedups until end-to-end matched measurements exist.

## Conditional planning, not measured speedups

| Sustained end-to-end speedup over pilot | Estimated one pass over3,683,080 examples |
|---|---:|
|1x|25.2days|
|2x|12.6days|
|4x|6.3days|
|8x|3.1days|

Actual optimized time may differ. Reducing dataset size changes coverage, not processing throughput. The full-data cap stays in place pending explicit authorization after optimization/quality review.

Primary reference: https://huggingface.co/docs/transformers/main/model_doc/qwen3_5 documents the slow/memory-hungry fallback without DeltaNet optional kernels. Its current version must not be assumed identical to the installed5.8.0; inspect installed source and pin compatible dependencies.
