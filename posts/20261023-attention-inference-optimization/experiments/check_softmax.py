"""FP32 online normalizer checks; no GPU required (imports attention's deps)."""
import json
import torch
from attention import online_softmax


torch.manual_seed(42)
cases = {
    "increasing_max": torch.tensor([[1000., 1001., 1002.]]),
    "decreasing_max": torch.tensor([[1002., 1001., 1000.]]),
    "large_range": torch.tensor([[-1000., 0., 1000.]]),
    "random_tail": torch.randn(7, 257) * 20,
}
results = []
for name, x in cases.items():
    for block in (1, 2, 64):
        actual = online_softmax(x, block)
        expected = x.softmax(-1)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        results.append({"case": name, "block": block,
                        "max_abs_error": (actual - expected).abs().max().item(),
                        "max_sum_error": (actual.sum(-1) - 1).abs().max().item()})
print(json.dumps(results, indent=2))
