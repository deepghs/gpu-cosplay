"""In-container helper to apply the VRAM cap.

The host CLI sets `GPU_COSPLAY_VRAM_GB` in the container env. When a Python
script calls `gpu_cosplay.inject.apply()`, we ask PyTorch to enforce that cap
via `torch.cuda.set_per_process_memory_fraction`.

Users can also call `gpu-cosplay-apply` from CLI to print the cap, or have
their entrypoint `import gpu_cosplay_inject` automatically.
"""

from __future__ import annotations

import os
import sys


def apply(verbose: bool = True) -> None:
    cap_str = os.environ.get("GPU_COSPLAY_VRAM_GB")
    if not cap_str:
        return  # not in a cosplay container
    try:
        cap_gb = float(cap_str)
    except ValueError:
        return

    try:
        import torch
    except ImportError:
        if verbose:
            print(
                f"[gpu-cosplay] target VRAM cap = {cap_gb} GB "
                "(install torch to enforce via PyTorch caching allocator)",
                file=sys.stderr,
            )
        return

    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        frac = min(1.0, cap_gb * 1e9 / total)
        torch.cuda.set_per_process_memory_fraction(frac, i)
    if verbose:
        card = os.environ.get("GPU_COSPLAY_PRETTY", "<card>")
        print(
            f"[gpu-cosplay] PyTorch VRAM capped at {cap_gb} GB (simulating {card})",
            file=sys.stderr,
        )


if __name__ == "__main__":
    apply()
