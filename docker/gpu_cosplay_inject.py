"""Tiny in-container module: apply VRAM cap to PyTorch when imported.

Users add one line to their script:

    import gpu_cosplay_inject  # noqa: applies VRAM cap before any torch work

or call apply() explicitly.
"""
import os
import sys


def apply(verbose=True):
    cap_str = os.environ.get("GPU_COSPLAY_VRAM_GB")
    if not cap_str:
        return
    try:
        cap_gb = float(cap_str)
    except ValueError:
        return
    try:
        import torch
    except ImportError:
        if verbose:
            sys.stderr.write(
                "[gpu-cosplay] target VRAM cap = %s GB (install torch to enforce)\n"
                % cap_gb
            )
        return
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        _free, total = torch.cuda.mem_get_info(i)
        frac = min(1.0, cap_gb * 1e9 / total)
        torch.cuda.set_per_process_memory_fraction(frac, i)
    if verbose:
        pretty = os.environ.get("GPU_COSPLAY_PRETTY", "<card>")
        sys.stderr.write(
            "[gpu-cosplay] PyTorch VRAM capped at %s GB (simulating %s)\n"
            % (cap_gb, pretty)
        )


apply(verbose=False)
