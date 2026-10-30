import copy
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from lora import LowRankLinear


@unittest.skipIf(torch is None, "PyTorch is supplied by the training runtime")
class LoRAPrecisionTests(unittest.TestCase):
    def test_bf16_adapter_compute_keeps_fp32_parameters_and_state_keys(self):
        torch.manual_seed(17)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base = torch.nn.Linear(16, 16, bias=False).to(device=device, dtype=torch.bfloat16)
        base.requires_grad_(False)
        fp32 = LowRankLinear(base, rank=4, alpha=8, compute_dtype="fp32")
        bf16 = LowRankLinear(copy.deepcopy(base), rank=4, alpha=8, compute_dtype="bf16")
        with torch.no_grad():
            bf16.lora_a.copy_(fp32.lora_a)
            fp32.lora_b.normal_(std=0.02)
            bf16.lora_b.copy_(fp32.lora_b)

        x = torch.randn(2, 5, 16, device=device, dtype=torch.bfloat16)
        y_fp32 = fp32(x)
        y_bf16 = bf16(x)
        self.assertEqual(y_fp32.dtype, torch.bfloat16)
        self.assertEqual(y_bf16.dtype, torch.bfloat16)
        self.assertTrue(torch.allclose(y_fp32.float(), y_bf16.float(), atol=0.05, rtol=0.05))

        y_fp32.float().square().mean().backward()
        y_bf16.float().square().mean().backward()
        for module in (fp32, bf16):
            self.assertEqual(module.lora_a.dtype, torch.float32)
            self.assertEqual(module.lora_b.dtype, torch.float32)
            self.assertTrue(torch.isfinite(module.lora_a.grad).all())
            self.assertTrue(torch.isfinite(module.lora_b.grad).all())
        self.assertEqual(set(fp32.state_dict()), set(bf16.state_dict()))

    def test_unknown_compute_mode_is_rejected(self):
        base = torch.nn.Linear(4, 4)
        with self.assertRaises(ValueError):
            LowRankLinear(base, compute_dtype="fp16")


if __name__ == "__main__":
    unittest.main()
