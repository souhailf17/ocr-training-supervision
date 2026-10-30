> Document d’étape : pour le bilan consolidé du 30 octobre et les identités des modèles, lire [le dossier encadrant](encadrant/01-vue-ensemble.md).

# Multi-GPU training

## Recorded configuration

The 2 October 2026 run snapshot used four ranks with PyTorch DistributedDataParallel replication on four NVIDIA H200 GPUs. Each GPU processed a physical microbatch of three examples for 32 gradient-accumulation steps: `4 × 3 × 32 = 384` examples per optimizer update. The model was replicated; this was not full parameter sharding.

DDP, FLA and causal-convolution kernels were exercised by the saved real-update preflight. The report also recorded finite gradients, a parameter update, exact four-rank checkpoint replay and verified PVC/S3 checkpoint round trip. See `../metrics/training-snapshot-2026-10-02.json` for numerical details.

GPU utilization was sampled at approximately five-second intervals for about five minutes. Mean reported utilization by GPU was 83.77%, 80.93%, 75.32% and 76.65%. These are sampled kernel-activity readings, not a measure of theoretical compute efficiency. The per-sample data is in `../metrics/gpu-samples-2026-10-02.csv`.

## Correctness requirements

- Every rank must use the same model, optimizer, scheduler and update count.
- Each rank consumes a disjoint portion of the deterministic data stream.
- Intermediate accumulation microsteps use DDP `no_sync`; gradient synchronization occurs at the update boundary.
- Any explicit collective in the training loop must be called by every rank in the same order.
- Checkpoints preserve the cursor and per-rank random state so resumed training does not silently repeat or skip samples.
- A successful launch requires finite gradients, an actual parameter update, matching rank progress and successful checkpoint verification.

The launcher is fixed to four GPUs and the original OpenShift namespace. Review it before reuse, and let the cluster scheduler select nodes. The repository contains no access token or S3 key; runtime credentials must be supplied through an authorized Secret.

## Performance interpretation

GPU count alone does not establish speedup. Compare end-to-end examples/s and supervised non-padding tokens/s on the same sample distribution. Capture update median/p95, rank skew, data wait, GPU memory and a bounded profiler trace. Balance variable text and visual-token lengths without omitting long pages. See [`optimization-review-2026-10-02.md`](optimization-review-2026-10-02.md).
