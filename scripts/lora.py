"""Small custom LoRA layer with an opt-in BF16 adapter compute path.

Trainable adapter parameters remain FP32 in both modes. The BF16 mode is an
experiment and must pass the distributed preflight and validation gates before
it is selected for a training run.
"""
import math

import torch
import torch.nn.functional as F


class LowRankLinear(torch.nn.Module):
    def __init__(self, base, rank=16, alpha=32, compute_dtype="fp32"):
        super().__init__()
        if compute_dtype not in {"fp32", "bf16"}:
            raise ValueError("LoRA compute_dtype must be 'fp32' or 'bf16'")
        self.base = base
        self.lora_a = torch.nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32)
        )
        self.lora_b = torch.nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32)
        )
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.compute_dtype = compute_dtype

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        result = self.base(x)
        if self.compute_dtype == "bf16":
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                update = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale
        else:
            with torch.autocast(device_type=x.device.type, enabled=False):
                update = F.linear(F.linear(x.float(), self.lora_a), self.lora_b) * self.scale
        return result + update.to(result.dtype)
