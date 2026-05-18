"""Microbenchmark — run inside a gpu-cosplay container to measure the slice.

Usage (inside container):
    python examples/bench.py
or:
    gpu-cosplay-apply -- python examples/bench.py

Reports FP32, BF16-tensor-core, and copy bandwidth alongside what the simulated
card *should* be doing per its datasheet.
"""

from __future__ import annotations

import json
import os
import time

import gpu_cosplay_inject  # noqa: F401  -- applies VRAM cap
import torch


def bench_matmul(dtype: torch.dtype, n: int = 4096, iters: int = 20) -> float:
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(5):
        _ = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return 2 * n**3 / dt / 1e12


def bench_bw(n_bytes: int = 1 << 30, iters: int = 20) -> float:
    n = n_bytes // 4
    src = torch.randn(n, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return 2 * n_bytes / dt / 1e9


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    fp32 = bench_matmul(torch.float32, n=4096)
    bf16 = bench_matmul(torch.bfloat16, n=4096)
    bw = bench_bw()
    free, total = torch.cuda.mem_get_info()
    out = {
        "device": torch.cuda.get_device_name(0),
        "physical_vram_gb": round(total / 1e9, 1),
        "measured": {
            "fp32_tflops": round(fp32, 2),
            "bf16_tc_tflops": round(bf16, 2),
            "copy_bw_gbs": round(bw, 1),
        },
        "target": {
            "card": os.environ.get("GPU_COSPLAY_PRETTY"),
            "vram_gb": float(os.environ.get("GPU_COSPLAY_VRAM_GB", 0)),
            "fp32_tflops": float(os.environ.get("GPU_COSPLAY_FP32_TFLOPS", 0)),
            "bf16_tc_tflops": float(os.environ.get("GPU_COSPLAY_BF16_TC_TFLOPS", 0)),
            "bw_gbs": float(os.environ.get("GPU_COSPLAY_BW_GBS", 0)),
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
