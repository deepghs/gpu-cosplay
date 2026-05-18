"""Runtime hooks loaded automatically inside a gpu-cosplay container.

Triggered by `gpu_cosplay_runtime.pth` in site-packages. Installs:
  - Lazy VRAM cap apply the moment torch is first imported.
  - Monkey-patches on torch.cuda.* so name / total_memory reflect the target GPU.
  - Monkey-patches on pynvml so nvitop / nvtop / anything reading NVML via
    Python sees the target GPU's name and memory.

Reads target specs from env vars set by `gpu-cosplay up`. Does nothing if those
env vars are absent (= we're not in a cosplay container, so this is a no-op).
"""

from __future__ import annotations

import builtins
import os
import sys


_STATE = {"installed": False, "torch_patched": False, "pynvml_patched": False}


def _specs():
    name = (os.environ.get("GPU_COSPLAY_PRETTY") or "").strip()
    vram_str = os.environ.get("GPU_COSPLAY_VRAM_GB")
    if not name or not vram_str:
        return None
    try:
        vram_bytes = int(float(vram_str) * (1024**3))
    except (ValueError, TypeError):
        return None
    try:
        tdp = int(float(os.environ.get("GPU_COSPLAY_TDP_W", "0") or 0)) or None
    except (ValueError, TypeError):
        tdp = None
    try:
        bf16_tc = float(os.environ.get("GPU_COSPLAY_BF16_TC_TFLOPS", "0") or 0)
    except (ValueError, TypeError):
        bf16_tc = 0.0
    return {
        "name": name,
        "vram_bytes": vram_bytes,
        "tdp_w": tdp,
        "has_tensor_core": bf16_tc > 0,
    }


def _patch_torch():
    if _STATE["torch_patched"]:
        return
    torch = sys.modules.get("torch")
    if torch is None or not hasattr(torch, "cuda"):
        return
    try:
        if not torch.cuda.is_available():
            return
    except Exception:
        return
    specs = _specs()
    if not specs:
        return

    cap = specs["vram_bytes"]
    # Capture the ORIGINAL mem_get_info before any patch so the fraction
    # calc always sees real physical sizes, no matter when it runs.
    _orig_mem = torch.cuda.mem_get_info

    def _apply_fraction():
        """Set the per-process memory fraction. Safe to call multiple times.

        At import time, torch.cuda is loaded but the CUDA context may not be
        initialized yet, so device_count() can return 0. We hook torch.cuda's
        lazy-init to retry the moment CUDA is actually live.
        """
        if _STATE.get("fraction_set"):
            return
        try:
            n = torch.cuda.device_count()
        except Exception:
            return
        if n == 0:
            return
        try:
            for i in range(n):
                _free, phys = _orig_mem(i)
                if phys > 0 and cap < phys:
                    torch.cuda.set_per_process_memory_fraction(cap / phys, i)
            _STATE["fraction_set"] = True
        except Exception:
            pass

    # 1. Apply the fraction now if CUDA is initialised, otherwise hook the
    # lazy-init path so we get a second chance the moment it is.
    _apply_fraction()
    if not _STATE.get("fraction_set"):
        try:
            _orig_lazy = torch.cuda._lazy_init

            def _patched_lazy_init(*a, **kw):
                _orig_lazy(*a, **kw)
                _apply_fraction()

            torch.cuda._lazy_init = _patched_lazy_init
        except Exception:
            pass

    # 1b. Feature policy: if the target lacks Tensor Core (GTX 16-series),
    # turn off TF32 paths so cuBLAS/cuDNN don't quietly use the host's TC.
    # This is the only universal switch torch exposes; BF16/FP16 ops still
    # use TC when the user code asks for them (caller's choice).
    if not specs["has_tensor_core"]:
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

    # 2. Patch get_device_name.
    def get_device_name(device=None):
        return specs["name"]

    torch.cuda.get_device_name = get_device_name

    # 3. Patch mem_get_info -> report target total + free relative to cap.

    def mem_get_info(device=None):
        try:
            used = torch.cuda.memory_allocated(device)
        except Exception:
            used = 0
        free_phys, _ = _orig_mem(device)
        free_target = max(0, min(cap - used, free_phys))
        return free_target, cap

    torch.cuda.mem_get_info = mem_get_info

    # 4. Patch get_device_properties. The C struct it normally returns is
    # immutable in modern torch, so wrap it in a proxy that lies about
    # `name` and `total_memory` and forwards everything else.
    _orig_props = torch.cuda.get_device_properties
    target_name = specs["name"]

    class _CosplayDeviceProperties:
        __slots__ = ("_orig",)

        def __init__(self, orig):
            object.__setattr__(self, "_orig", orig)

        @property
        def name(self):
            return target_name

        @property
        def total_memory(self):
            return cap

        def __getattr__(self, k):
            return getattr(self._orig, k)

        def __repr__(self):
            return (
                f"_CudaDeviceProperties(name='{target_name}', "
                f"total_memory={cap // (1024 * 1024)}MB, "
                f"major={getattr(self._orig, 'major', '?')}, "
                f"minor={getattr(self._orig, 'minor', '?')})"
            )

    def get_device_properties(device=None):
        return _CosplayDeviceProperties(_orig_props(device))

    torch.cuda.get_device_properties = get_device_properties

    _STATE["torch_patched"] = True


def _patch_pynvml():
    """Idempotent. Returns without changing state if pynvml isn't fully loaded
    yet — this is important because the import hook can fire mid-load and at
    that point pynvml's symbols are still being defined. We only flip the
    "patched" flag once the patches are actually in place."""
    if _STATE["pynvml_patched"]:
        return
    pynvml = sys.modules.get("pynvml")
    if pynvml is None:
        return
    # Require the module to expose the symbols we depend on. If it doesn't,
    # the module is still in the middle of being executed; retry on the next
    # import event.
    if not (hasattr(pynvml, "nvmlDeviceGetName") and hasattr(pynvml, "nvmlDeviceGetMemoryInfo")):
        return
    specs = _specs()
    if not specs:
        return

    _orig_name = pynvml.nvmlDeviceGetName

    def nvmlDeviceGetName(handle):
        try:
            orig = _orig_name(handle)
        except Exception:
            orig = ""
        return specs["name"].encode() if isinstance(orig, bytes) else specs["name"]

    pynvml.nvmlDeviceGetName = nvmlDeviceGetName

    _orig_mem = pynvml.nvmlDeviceGetMemoryInfo

    def nvmlDeviceGetMemoryInfo(handle):
        try:
            orig = _orig_mem(handle)
            used = int(getattr(orig, "used", 0))
        except Exception:
            used = 0
        cap = specs["vram_bytes"]
        free = max(0, cap - used)

        class _Mem:
            def __init__(self_inner, used_, free_, total_):
                self_inner.used = used_
                self_inner.free = free_
                self_inner.total = total_

        return _Mem(used, free, cap)

    pynvml.nvmlDeviceGetMemoryInfo = nvmlDeviceGetMemoryInfo

    tdp = specs.get("tdp_w")
    if tdp and hasattr(pynvml, "nvmlDeviceGetPowerManagementLimit"):

        def nvmlDeviceGetPowerManagementLimit(handle):
            return tdp * 1000

        pynvml.nvmlDeviceGetPowerManagementLimit = nvmlDeviceGetPowerManagementLimit
        if hasattr(pynvml, "nvmlDeviceGetPowerManagementDefaultLimit"):
            pynvml.nvmlDeviceGetPowerManagementDefaultLimit = nvmlDeviceGetPowerManagementLimit

    _STATE["pynvml_patched"] = True


_orig_import = builtins.__import__


def _hook_import(name, globals=None, locals=None, fromlist=(), level=0):
    m = _orig_import(name, globals, locals, fromlist, level)
    try:
        if not _STATE["torch_patched"] and "torch" in sys.modules and "torch.cuda" in sys.modules:
            _patch_torch()
        if not _STATE["pynvml_patched"] and "pynvml" in sys.modules:
            _patch_pynvml()
    except Exception:
        pass
    return m


def install():
    if _STATE["installed"]:
        return
    # No-op outside a cosplay container.
    if not os.environ.get("GPU_COSPLAY_VRAM_GB"):
        return
    builtins.__import__ = _hook_import
    _STATE["installed"] = True


# Auto-install if loaded directly (also called by .pth).
install()
