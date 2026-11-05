"""Run on an idle CUDA GPU; writes JSON to stdout, no model download."""
import argparse
import gc
import json
import platform
import statistics

import torch
import torch.nn.functional as F
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel

from attention import flash_attention, naive_attention, online_attention


def sdpa(q, k, v):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def measure(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    del out
    # Capture launches to avoid timing Python dispatch gaps on small kernels.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            out = fn()
    samples = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 20)
    return {"median_ms": statistics.median(samples), "samples_ms": samples,
            "peak_extra_mib": peak}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["naive", "flash64", "flash128"])
    args = parser.parse_args()
    torch.manual_seed(20261023)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.profile:
        q, k, v = [torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.float16)
                   for _ in range(3)]
        fn = {"naive": naive_attention,
              "flash64": lambda q, k, v: flash_attention(q, k, v, 64, 64),
              "flash128": lambda q, k, v: flash_attention(q, k, v, 64, 128)}[args.profile]
        for _ in range(3):
            fn(q, k, v)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        fn(q, k, v)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return
    result = {"environment": {"gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "triton": triton.__version__, "python": platform.python_version(),
              "seed": 20261023}, "correctness": [], "benchmarks": []}
    # Odd sequence lengths exercise tail masks; d=128 checks another head size.
    for n, d in [(1, 32), (17, 64), (127, 64), (257, 128), (1024, 64)]:
        q, k, v = [torch.randn(1, 2, n, d, device="cuda", dtype=torch.float16)
                   for _ in range(3)]
        ref = naive_attention(q.float(), k.float(), v.float())
        for name, fn in [("online", online_attention), ("naive_fp16", naive_attention),
                         ("flash64", lambda q, k, v: flash_attention(q, k, v, 64, 64)),
                         ("flash128", lambda q, k, v: flash_attention(q, k, v, 64, 128))]:
            out = fn(q, k, v).float()
            torch.testing.assert_close(out, ref, atol=0.005, rtol=0.005)
            result["correctness"].append({"n": n, "d": d, "method": name,
                "max_abs_error": (out-ref).abs().max().item(),
                "rms_error": (out-ref).square().mean().sqrt().item()})
    for n in (512, 1024, 2048, 4096):
        q, k, v = [torch.randn(1, 8, n, 64, device="cuda", dtype=torch.float16)
                   for _ in range(3)]
        ref = sdpa(q, k, v)
        methods = {"naive": lambda: naive_attention(q, k, v),
                   "flash64": lambda: flash_attention(q, k, v, 64, 64),
                   "flash128": lambda: flash_attention(q, k, v, 64, 128),
                   "torch_flash_sdpa": lambda: sdpa(q, k, v)}
        for name, fn in methods.items():
            out = fn()
            torch.testing.assert_close(out, ref, atol=0.005, rtol=0.005)
            error = (out.float() - ref.float()).abs().max().item()
            del out
            result["benchmarks"].append({"shape": [1, 8, n, 64], "method": name,
                "max_abs_vs_sdpa": error, **measure(fn)})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
