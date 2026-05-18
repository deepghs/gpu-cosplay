#!/usr/bin/env python3
"""Self-check for a gpu-cosplay container.

Runs inside the container (via `gpu-cosplay-verify` or `gpu-cosplay verify`)
and reports whether each surface — env vars, nvidia-smi shim, Python runtime,
torch.cuda, pynvml, VRAM cap, feature flags — looks the way we expect for the
target GPU declared in env vars.

Exit code: 0 if all available checks pass, non-zero if any fail.
Tests requiring an absent dependency (torch, pynvml) are skipped, not failed.
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Callable, Optional


GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = YELLOW = RESET = ""

PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
SKIP = f"{YELLOW}SKIP{RESET}"

_results: list[tuple[str, str, str]] = []  # (name, status, detail)


def _record(name: str, status: str, detail: str = "") -> None:
    _results.append((name, status, detail))


def check(label: str):
    """Decorator: register a callable as a check."""

    def deco(fn: Callable[[], Optional[str]]):
        def runner():
            try:
                out = fn()
                if out is None:
                    _record(label, PASS)
                elif isinstance(out, tuple) and out[0] == "skip":
                    _record(label, SKIP, out[1])
                else:
                    _record(label, FAIL, str(out))
            except Exception as e:
                _record(label, FAIL, f"exception: {e}")

        runner.__name__ = fn.__name__
        return runner

    return deco


def _target():
    return {
        "name": (os.environ.get("GPU_COSPLAY_PRETTY") or "").strip(),
        "vram_gb": float(os.environ.get("GPU_COSPLAY_VRAM_GB") or 0),
        "tdp_w": float(os.environ.get("GPU_COSPLAY_TDP_W") or 0),
        "bf16_tc_tflops": float(os.environ.get("GPU_COSPLAY_BF16_TC_TFLOPS") or 0),
    }


# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------


@check("env: GPU_COSPLAY_* variables present")
def _env_present():
    needed = ["GPU_COSPLAY_PRETTY", "GPU_COSPLAY_VRAM_GB", "GPU_COSPLAY_CARD"]
    missing = [v for v in needed if not os.environ.get(v)]
    if missing:
        return f"missing: {missing}"


@check("env: user identity matches host")
def _env_user():
    expected_uid = os.environ.get("HOST_UID")
    expected_user = os.environ.get("HOST_USER")
    if not expected_uid:
        return ("skip", "HOST_UID not set")
    if os.getuid() != int(expected_uid):
        return f"uid={os.getuid()} != HOST_UID={expected_uid}"
    if expected_user and os.environ.get("USER", "") not in (expected_user, ""):
        return f"USER={os.environ.get('USER')!r} != HOST_USER={expected_user!r}"


# ---------------------------------------------------------------------------
# 2. nvidia-smi shim
# ---------------------------------------------------------------------------


@check("nvidia-smi: shim shadows the real binary")
def _shim_present():
    if not shutil.which("nvidia-smi"):
        return "no nvidia-smi on PATH"
    which_path = shutil.which("nvidia-smi")
    if which_path != "/usr/local/bin/nvidia-smi":
        return f"PATH resolves to {which_path}, not /usr/local/bin/nvidia-smi"


@check("nvidia-smi: --query-gpu=name returns target GPU")
def _shim_name():
    name = _target()["name"]
    if not name:
        return ("skip", "no GPU_COSPLAY_PRETTY")
    p = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if p.returncode != 0:
        return f"nvidia-smi rc={p.returncode}: {p.stderr.strip()[:100]}"
    got = p.stdout.strip()
    if got != name:
        return f"expected {name!r}, got {got!r}"


@check("nvidia-smi: --query-gpu=memory.total returns target VRAM")
def _shim_mem():
    vram_gb = _target()["vram_gb"]
    if not vram_gb:
        return ("skip", "no GPU_COSPLAY_VRAM_GB")
    p = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if p.returncode != 0:
        return f"nvidia-smi rc={p.returncode}"
    got_mib = int(float(p.stdout.strip()))
    expect_mib = int(round(vram_gb * 1024))
    if abs(got_mib - expect_mib) > 1:
        return f"expected ~{expect_mib} MiB, got {got_mib} MiB"


@check("nvidia-smi: --query-gpu=power.max_limit returns target TDP")
def _shim_power():
    tdp = _target()["tdp_w"]
    if not tdp:
        return ("skip", "no GPU_COSPLAY_TDP_W")
    p = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.max_limit", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if p.returncode != 0:
        return f"nvidia-smi rc={p.returncode}"
    got_w = float(p.stdout.strip())
    if abs(got_w - tdp) > 1:
        return f"expected ~{tdp} W, got {got_w} W"


@check("nvidia-smi: -L shows target name and no MIG sub-line")
def _shim_list():
    name = _target()["name"]
    if not name:
        return ("skip", "no GPU_COSPLAY_PRETTY")
    p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    if name not in p.stdout:
        return f"output does not contain {name!r}: {p.stdout.strip()[:200]}"
    if "MIG " in p.stdout and "Device" in p.stdout:
        return "MIG sub-line not stripped"


@check("nvidia-smi: real /usr/bin/nvidia-smi still untouched")
def _real_untouched():
    if not os.path.exists("/usr/bin/nvidia-smi"):
        return ("skip", "/usr/bin/nvidia-smi not present")
    p = subprocess.run(
        ["/usr/bin/nvidia-smi", "-L"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if p.returncode != 0:
        return f"rc={p.returncode}"
    # The real binary should still report the host name (e.g. "NVIDIA H200").
    if _target()["name"] and _target()["name"] in p.stdout:
        return "real binary appears to be modified (shows target name); shim should not touch it"


# ---------------------------------------------------------------------------
# 3. Python runtime hook
# ---------------------------------------------------------------------------


@check("python: gpu_cosplay_runtime auto-loaded")
def _py_runtime():
    p = subprocess.run(
        ["python3", "-c", "import gpu_cosplay_runtime as r; print(r._STATE['installed'])"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if p.returncode != 0:
        return f"import failed: {p.stderr.strip()[:200]}"
    if "True" not in p.stdout:
        return f"runtime state: {p.stdout.strip()!r}"


@check("python: .pth file is in site-packages")
def _py_pth():
    p = subprocess.run(
        [
            "python3",
            "-c",
            "import site, os; "
            "found = any(os.path.exists(os.path.join(d, 'gpu_cosplay_runtime.pth')) "
            "for d in site.getsitepackages()); print(found)",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if "True" not in p.stdout:
        return ".pth not found in any site-packages dir"


# ---------------------------------------------------------------------------
# 4. torch.cuda (skip if torch missing)
# ---------------------------------------------------------------------------


@check("torch: get_device_name reports target")
def _torch_name():
    name = _target()["name"]
    if not name:
        return ("skip", "no GPU_COSPLAY_PRETTY")
    p = subprocess.run(
        ["python3", "-c", "import torch; print(torch.cuda.get_device_name(0))"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if "ModuleNotFoundError" in p.stderr:
        return ("skip", "torch not installed")
    if p.returncode != 0:
        return f"rc={p.returncode}: {p.stderr.strip()[:200]}"
    got = p.stdout.strip().splitlines()[-1]
    if got != name:
        return f"expected {name!r}, got {got!r}"


@check("torch: mem_get_info total equals target VRAM")
def _torch_mem():
    vram_gb = _target()["vram_gb"]
    if not vram_gb:
        return ("skip", "no GPU_COSPLAY_VRAM_GB")
    p = subprocess.run(
        ["python3", "-c", "import torch; _,t=torch.cuda.mem_get_info(0); print(t)"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if "ModuleNotFoundError" in p.stderr:
        return ("skip", "torch not installed")
    if p.returncode != 0:
        return f"rc={p.returncode}"
    got = int(p.stdout.strip().splitlines()[-1])
    expect = int(vram_gb * (1024**3))
    if abs(got - expect) > 1024 * 1024:  # within 1 MB
        return f"expected ~{expect} bytes, got {got} bytes"


@check("torch: get_device_properties reports target")
def _torch_props():
    name = _target()["name"]
    vram_gb = _target()["vram_gb"]
    if not name or not vram_gb:
        return ("skip", "no target")
    p = subprocess.run(
        [
            "python3",
            "-c",
            "import torch; "
            "p=torch.cuda.get_device_properties(0); "
            "print(p.name); print(p.total_memory)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if "ModuleNotFoundError" in p.stderr:
        return ("skip", "torch not installed")
    if p.returncode != 0:
        return f"rc={p.returncode}: {p.stderr.strip()[:200]}"
    lines = p.stdout.strip().splitlines()
    got_name = lines[-2] if len(lines) >= 2 else ""
    got_mem = int(lines[-1])
    expect_mem = int(vram_gb * (1024**3))
    if got_name != name:
        return f"name: expected {name!r}, got {got_name!r}"
    if abs(got_mem - expect_mem) > 1024 * 1024:
        return f"total_memory: expected ~{expect_mem}, got {got_mem}"


@check("torch: allocate (cap - 2 GB) succeeds")
def _torch_alloc_under():
    vram_gb = _target()["vram_gb"]
    if not vram_gb or vram_gb < 4:
        return ("skip", "target too small to test allocation safely")
    under_bytes = int((vram_gb - 2) * 1e9)
    p = subprocess.run(
        [
            "python3",
            "-c",
            f"import torch; a=torch.empty({under_bytes // 4}, "
            "device='cuda', dtype=torch.float32); print(a.numel()*4)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if "ModuleNotFoundError" in p.stderr:
        return ("skip", "torch not installed")
    if p.returncode != 0:
        return f"alloc failed: {p.stderr.strip()[:200]}"


@check("torch: allocate beyond cap correctly OOMs")
def _torch_alloc_over():
    vram_gb = _target()["vram_gb"]
    if not vram_gb:
        return ("skip", "no GPU_COSPLAY_VRAM_GB")
    # Cap is GiB; ask for cap + 4 GiB, comfortably past any rounding edge.
    over_bytes = int(vram_gb * (1024**3) + 4 * (1024**3))
    p = subprocess.run(
        [
            "python3",
            "-c",
            f"import torch\n"
            f"try:\n"
            f"  a=torch.empty({over_bytes // 4}, device='cuda', dtype=torch.float32)\n"
            f"  print('OK')\n"
            f"except RuntimeError:\n"
            f"  print('OOM')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if "ModuleNotFoundError" in p.stderr:
        return ("skip", "torch not installed")
    if "OOM" not in p.stdout:
        return f"allocation beyond cap did not OOM: {p.stdout.strip()!r}"


@check("torch: TF32 disabled when target lacks Tensor Cores")
def _torch_tf32():
    has_tc = _target()["bf16_tc_tflops"] > 0
    if has_tc:
        return ("skip", "target has Tensor Cores; TF32 policy not enforced")
    p = subprocess.run(
        ["python3", "-c", "import torch; print(torch.backends.cuda.matmul.allow_tf32)"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if "ModuleNotFoundError" in p.stderr:
        return ("skip", "torch not installed")
    if "False" not in p.stdout:
        return f"allow_tf32 not disabled (got {p.stdout.strip()})"


# ---------------------------------------------------------------------------
# 5. pynvml (skip if missing)
# ---------------------------------------------------------------------------


@check("pynvml: nvmlDeviceGetName returns target")
def _pynvml_name():
    name = _target()["name"]
    if not name:
        return ("skip", "no GPU_COSPLAY_PRETTY")
    p = subprocess.run(
        [
            "python3",
            "-c",
            "import pynvml; pynvml.nvmlInit(); "
            "h=pynvml.nvmlDeviceGetHandleByIndex(0); "
            "n=pynvml.nvmlDeviceGetName(h); "
            "print(n.decode() if isinstance(n, bytes) else n)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if "ModuleNotFoundError" in p.stderr or "ImportError" in p.stderr:
        return ("skip", "pynvml not installed")
    if p.returncode != 0:
        return f"rc={p.returncode}: {p.stderr.strip()[:200]}"
    got = p.stdout.strip().splitlines()[-1]
    if got != name:
        return f"expected {name!r}, got {got!r}"


@check("pynvml: memory.total returns target VRAM")
def _pynvml_mem():
    vram_gb = _target()["vram_gb"]
    if not vram_gb:
        return ("skip", "no GPU_COSPLAY_VRAM_GB")
    p = subprocess.run(
        [
            "python3",
            "-c",
            "import pynvml; pynvml.nvmlInit(); "
            "h=pynvml.nvmlDeviceGetHandleByIndex(0); "
            "m=pynvml.nvmlDeviceGetMemoryInfo(h); "
            "print(m.total)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if "ModuleNotFoundError" in p.stderr or "ImportError" in p.stderr:
        return ("skip", "pynvml not installed")
    if p.returncode != 0:
        return f"rc={p.returncode}: {p.stderr.strip()[:200]}"
    got = int(p.stdout.strip().splitlines()[-1])
    expect = int(vram_gb * (1024**3))
    if abs(got - expect) > 1024 * 1024:
        return f"expected ~{expect}, got {got}"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _run_all():
    fns = [
        v
        for v in globals().values()
        if callable(v)
        and getattr(v, "__name__", "").startswith("_")
        and v.__module__ == "__main__"
        and v.__name__
        not in (
            "_target",
            "_record",
            "_run_all",
            "_print_report",
            "_results",
            "main",
        )
    ]
    # Cleaner: explicit list to keep ordering.
    fns = [
        _env_present,
        _env_user,
        _shim_present,
        _shim_name,
        _shim_mem,
        _shim_power,
        _shim_list,
        _real_untouched,
        _py_runtime,
        _py_pth,
        _torch_name,
        _torch_mem,
        _torch_props,
        _torch_alloc_under,
        _torch_alloc_over,
        _torch_tf32,
        _pynvml_name,
        _pynvml_mem,
    ]
    for fn in fns:
        fn()


def _print_report(use_json: bool):
    if use_json:
        out = [
            {
                "check": n,
                "status": s.replace(GREEN, "")
                .replace(RED, "")
                .replace(YELLOW, "")
                .replace(RESET, ""),
                "detail": d,
            }
            for n, s, d in _results
        ]
        print(json.dumps(out, indent=2))
    else:
        width = max(len(n) for n, _, _ in _results) + 2
        for name, status, detail in _results:
            line = f"  {status}  {name:<{width}}"
            if detail:
                line += f"  -- {detail}"
            print(line)
        n_pass = sum(1 for _, s, _ in _results if "PASS" in s)
        n_fail = sum(1 for _, s, _ in _results if "FAIL" in s)
        n_skip = sum(1 for _, s, _ in _results if "SKIP" in s)
        print(f"\n  Total: {len(_results)}  pass={n_pass}  fail={n_fail}  skip={n_skip}")

    failed = [n for n, s, _ in _results if "FAIL" in s]
    if failed:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(
        description="Verify a gpu-cosplay container's runtime is correctly set up."
    )
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()
    _run_all()
    _print_report(args.json)


if __name__ == "__main__":
    main()
