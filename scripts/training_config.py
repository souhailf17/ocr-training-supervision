"""Pure helpers for keeping DDP global-batch math explicit and testable."""


def accumulation_steps(global_batch, world_size, microbatch_per_gpu):
    values = (global_batch, world_size, microbatch_per_gpu)
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("global batch, world size, and microbatch must be positive integers")
    samples_per_microstep = world_size * microbatch_per_gpu
    if global_batch % samples_per_microstep:
        raise ValueError("global batch must be divisible by world size × per-GPU microbatch")
    return global_batch // samples_per_microstep
