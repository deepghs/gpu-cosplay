"""Detect host GPU capabilities by parsing nvidia-smi."""

from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MigProfile:
    profile_id: int  # the integer used by nvidia-smi mig -cgi <id>
    name: str  # e.g. "1g.18gb"
    sm_count: int
    memory_gb: float
    instances_total: int
    instances_free: int


@dataclass
class HostGPU:
    index: int
    name: str  # e.g. "NVIDIA H200"
    uuid: str
    compute_cap: str  # e.g. "9.0"
    memory_total_gb: float
    power_min_w: Optional[int]
    power_max_w: Optional[int]
    power_default_w: Optional[int]
    clock_min_mhz: Optional[int]
    clock_max_mhz: Optional[int]
    mig_capable: bool
    mig_enabled: bool
    mig_profiles: list[MigProfile] = field(default_factory=list)


def _run(args: list[str], check: bool = True) -> str:
    p = subprocess.run(args, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(args)!r} failed: {p.stderr.strip() or p.stdout.strip()}")
    return p.stdout


def ensure_nvidia_smi() -> None:
    if not shutil.which("nvidia-smi"):
        raise SystemExit("nvidia-smi not found in PATH. Install the NVIDIA driver.")


def _parse_power_section(text: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (min, max, default) in W from a `nvidia-smi -q -d POWER -i X` block."""

    def grab(label: str) -> Optional[int]:
        m = re.search(rf"{label}\s*:\s*([\d.]+)\s*W", text)
        return int(float(m.group(1))) if m else None

    return grab(r"Min Power Limit"), grab(r"Max Power Limit"), grab(r"Default Power Limit")


def _parse_supported_clocks(text: str) -> tuple[Optional[int], Optional[int]]:
    """Return (min, max) graphics clock in MHz from supported clocks block."""
    gfx = [int(m.group(1)) for m in re.finditer(r"Graphics\s*:\s*(\d+)\s*MHz", text)]
    if not gfx:
        return None, None
    return min(gfx), max(gfx)


def _parse_mig_profiles_block(text: str, gpu_index: int) -> list[MigProfile]:
    """Parse `nvidia-smi mig -lgip` output for one GPU.

    The block looks like:
      |   0  MIG 1g.18gb         19     7/7        16.00      No     16     1     0   |
    """
    profiles: list[MigProfile] = []
    pattern = re.compile(
        r"\|\s*(\d+)\s+MIG\s+(\d+g\.\d+gb)\+?\w*\s+(\d+)\s+(\d+)/(\d+)\s+([\d.]+)\s+\S+\s+(\d+)"
    )
    for line in text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        idx, name, pid, free, total, mem, sms = m.groups()
        if int(idx) != gpu_index:
            continue
        # Skip "+me" variants — they have media engine; treat as the same profile name
        if "+" in line.split("MIG")[1].split()[0]:
            continue
        profiles.append(
            MigProfile(
                profile_id=int(pid),
                name=name,
                sm_count=int(sms),
                memory_gb=float(mem),
                instances_total=int(total),
                instances_free=int(free),
            )
        )
    return profiles


def list_host_gpus() -> list[HostGPU]:
    """Enumerate all host GPUs with their capabilities."""
    ensure_nvidia_smi()
    csv_out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,compute_cap,memory.total,mig.mode.current",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[HostGPU] = []
    for row in csv.reader(io.StringIO(csv_out)):
        if not row:
            continue
        idx, name, uuid, cc, mem_mib, mig_mode = [s.strip() for s in row]
        idx_i = int(idx)
        mem_gb = float(mem_mib) / 1024.0

        # Compute-cap >= 7.0 (Volta+) is what we care about for AI.
        # MIG is supported on data-center Ampere/Hopper (A100, H100, H200, A30).
        # We detect this from the name + capability to enumerate profiles.
        mig_cap = mig_mode.lower() != "n/a"
        mig_on = mig_mode.lower() == "enabled"

        # Power / clock range
        pwr = _run(["nvidia-smi", "-q", "-d", "POWER", "-i", str(idx_i)], check=False)
        pmin, pmax, pdef = _parse_power_section(pwr)

        clk = _run(["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS", "-i", str(idx_i)], check=False)
        cmin, cmax = _parse_supported_clocks(clk)

        # MIG profiles, if capable
        profs: list[MigProfile] = []
        if mig_cap:
            mig_text = _run(["nvidia-smi", "mig", "-lgip", "-i", str(idx_i)], check=False)
            profs = _parse_mig_profiles_block(mig_text, idx_i)

        gpus.append(
            HostGPU(
                index=idx_i,
                name=name,
                uuid=uuid,
                compute_cap=cc,
                memory_total_gb=mem_gb,
                power_min_w=pmin,
                power_max_w=pmax,
                power_default_w=pdef,
                clock_min_mhz=cmin,
                clock_max_mhz=cmax,
                mig_capable=mig_cap,
                mig_enabled=mig_on,
                mig_profiles=profs,
            )
        )
    return gpus


def pick_default_gpu(gpus: list[HostGPU], prefer_mig: bool = True) -> HostGPU:
    """Choose a sensible default host GPU.

    Preference order:
      1. MIG-capable, currently unused (or MIG already enabled with free instances)
      2. Any MIG-capable GPU
      3. The last GPU (to leave 0 free for ad-hoc use)
    """
    if not gpus:
        raise SystemExit("No NVIDIA GPUs detected.")
    if prefer_mig:
        for g in gpus:
            if g.mig_capable:
                return g
    return gpus[-1]
