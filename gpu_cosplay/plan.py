"""Match a target GPU to (MIG profile, clock lock, power, vram cap) on a host GPU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .cards import Card
from .host import HostGPU, MigProfile


@dataclass
class Plan:
    target: Card
    host: HostGPU
    mig_profile: Optional[MigProfile]  # None if we are using the full GPU
    clock_mhz: Optional[int]  # None if no clock lock needed
    power_limit_w: Optional[int]  # None to leave at default
    vram_cap_gb: float  # the cap to apply via memory_fraction
    expected_fp32: float
    expected_bf16: Optional[float]
    expected_bw_gbs: float
    warnings: list[str]


def _scale_for_clock(host_clock: int, host_max: int, base_value: float) -> float:
    return base_value * (host_clock / host_max)


def _pick_mig_profile(host: HostGPU, target: Card) -> Optional[MigProfile]:
    """Pick the smallest MIG profile that satisfies target VRAM.

    Rationale: we want headroom in physical VRAM so the cap is enforced by the
    PyTorch allocator rather than by the slice. If no profile fits, fall back
    to the largest available.
    """
    if not host.mig_capable or not host.mig_profiles:
        return None
    fit = [p for p in host.mig_profiles if p.memory_gb >= target.vram_gb]
    if fit:
        # Smallest profile that fits target VRAM, but prefer SM count close to target
        # If multiple profiles have the same memory tier, prefer the one with SM closer to target.
        fit.sort(key=lambda p: (p.memory_gb, abs(p.sm_count - target.sm_count)))
        return fit[0]
    # Target is bigger than any MIG slice → either use full GPU or biggest slice
    biggest = max(host.mig_profiles, key=lambda p: p.memory_gb)
    if biggest.memory_gb >= target.vram_gb * 0.9:
        return biggest
    return None  # caller will use full GPU


def _estimate_metrics(
    host: HostGPU,
    profile: Optional[MigProfile],
    clock_mhz: int,
) -> tuple[float, float, float]:
    """Estimate (FP32, BF16 TC, BW) at given (profile, clock).

    Anchored to per-SM throughput at boost clock for the host architecture,
    interpolated linearly with clock. BF16 TC numbers are dense / FP32-accum.
    """
    # Host-arch per-SM-per-cycle throughput. Values are TFLOPS per SM at 1 GHz.
    arch = _host_arch(host.name)
    fp32_per_sm = {
        "volta": 0.128,  # V100: ~15.7 TFLOPS / 80 SM / 1530 MHz = 0.128 per SM·GHz
        "turing": 0.128,
        "ampere": 0.256,  # GA10x doubled FP32 per SM
        "ada": 0.256,
        "hopper": 0.256,
    }.get(arch, 0.256)
    bf16_per_sm = {
        "volta": 0.0,  # No BF16 TC
        "turing": 0.0,  # Most Turing TUxxx has no BF16; we'll skip
        "ampere": 1.024,
        "ada": 2.048,
        "hopper": 3.84,  # ~989 TFLOPS / 132 SM / 1.98 GHz
    }.get(arch, 1.024)

    sm = profile.sm_count if profile else _full_sm_count(host)
    clk_ghz = clock_mhz / 1000.0
    fp32 = fp32_per_sm * sm * clk_ghz * 2  # FMA = 2 flops
    bf16 = bf16_per_sm * sm * clk_ghz

    # Bandwidth: HBM total is shared by MIG share; clock-locked GPU still emits
    # fewer requests so effective BW drops with clock. Empirical fit.
    if profile:
        share = profile.sm_count / max(_full_sm_count(host), 1)
        hbm_total = _host_total_bw(host)
        bw_peak = hbm_total * share
    else:
        bw_peak = _host_total_bw(host)
    # Below ~1 GHz, SMs cannot saturate HBM. Empirical knee around 1.2 GHz.
    bw_scale = min(1.0, clk_ghz / 1.2)
    bw = bw_peak * bw_scale
    return fp32, bf16, bw


def _host_arch(name: str) -> str:
    n = name.lower()
    if "h100" in n or "h200" in n or "gh200" in n:
        return "hopper"
    if "a100" in n or "a40" in n or "a30" in n or "a10" in n or "a16" in n or "ax00" in n:
        return "ampere"
    if " 30" in n or "rtx 30" in n:
        return "ampere"
    if "l4" in n or "l40" in n:
        return "ada"
    if " 40" in n or "rtx 40" in n or "ada" in n:
        return "ada"
    if "v100" in n:
        return "volta"
    if "t4" in n or "rtx 20" in n or "gtx 16" in n:
        return "turing"
    return "unknown"


def _full_sm_count(host: HostGPU) -> int:
    """Approximate SM count for the host GPU."""
    n = host.name.lower()
    table = {
        "h200": 132,
        "h100": 132,
        "h100 nvl": 132,
        "h100 pcie": 114,
        "a100": 108,
        "a30": 56,
        "a10": 72,
        "a40": 84,
        "v100": 80,
        "t4": 40,
        "l40": 142,
        "l40s": 142,
        "l4": 58,
        "rtx 6000 ada": 142,
    }
    for k, v in table.items():
        if k in n:
            return v
    # Unknown — fall back to compute-cap based guess
    return 80


def _host_total_bw(host: HostGPU) -> float:
    n = host.name.lower()
    if "h200" in n:
        return 4800
    if "h100 nvl" in n:
        return 3938
    if "h100 pcie" in n:
        return 2039
    if "h100" in n:
        return 3350
    if "a100" in n and "80" in n:
        return 2039
    if "a100" in n:
        return 1555
    if "v100" in n:
        return 900
    if "a40" in n:
        return 696
    if "a30" in n:
        return 933
    if "a10" in n:
        return 600
    if "l40" in n:
        return 864
    if "l4" in n:
        return 300
    if "t4" in n:
        return 320
    return 600.0


def plan(host: HostGPU, target: Card) -> Plan:
    warnings: list[str] = []

    profile = _pick_mig_profile(host, target)
    if profile is None and host.mig_capable:
        warnings.append(
            f"No MIG profile big enough for target VRAM {target.vram_gb} GB; will use full GPU."
        )

    # Find clock that matches target FP32 (or BF16 if target has TC and host has TC).
    # FP32 scales linearly with clock; we use it as the anchor.
    if profile:
        full_clk = host.clock_max_mhz or 1980
        fp32_full, _, _ = _estimate_metrics(host, profile, full_clk)
    else:
        full_clk = host.clock_max_mhz or 1980
        fp32_full, _, _ = _estimate_metrics(host, None, full_clk)

    if target.fp32_tflops >= fp32_full:
        # Can't reach target FP32 even at max clock; leave clock unlocked.
        clock_mhz = None
        if target.fp32_tflops > fp32_full * 1.1:
            warnings.append(
                f"Host slice maxes out at ~{fp32_full:.1f} TFLOPS FP32, "
                f"target wants {target.fp32_tflops:.1f}. Cosplay will be CPU-limited from the target's POV."
            )
    else:
        # Scale clock down
        ratio = target.fp32_tflops / fp32_full
        # Clock min/max from host caps
        cmax = host.clock_max_mhz or full_clk
        cmin = host.clock_min_mhz or 345
        clock_mhz = max(cmin, int(cmax * ratio))
        # Round to nearest 15 MHz (NVIDIA grid)
        clock_mhz = (clock_mhz // 15) * 15

    # Power: set to target TDP if within host range. Mostly cosmetic since
    # clock lock does the actual throttling.
    pwr = target.tdp_w
    if host.power_min_w is not None and pwr < host.power_min_w:
        pwr = host.power_min_w
    if host.power_max_w is not None and pwr > host.power_max_w:
        pwr = host.power_max_w

    actual_clk = clock_mhz or full_clk
    e_fp32, e_bf16, e_bw = _estimate_metrics(host, profile, actual_clk)

    # If target has no usable BF16 TC, report 0
    if target.bf16_tc_tflops is None:
        warnings.append(
            "Target has no BF16/FP16 Tensor Core — host hardware does, "
            "so disable mixed-precision in your framework to match."
        )

    if target.bf16_tc_tflops is not None and e_bf16 > target.bf16_tc_tflops * 1.3:
        warnings.append(
            f"BF16 TC will be ~{e_bf16:.0f} TFLOPS (target wants "
            f"{target.bf16_tc_tflops:.0f}). H100/H200 SMs have stronger TC; this "
            f"is a known limit of compute-cap-up-emulation."
        )

    return Plan(
        target=target,
        host=host,
        mig_profile=profile,
        clock_mhz=clock_mhz,
        power_limit_w=pwr,
        vram_cap_gb=target.vram_gb,
        expected_fp32=e_fp32,
        expected_bf16=e_bf16,
        expected_bw_gbs=e_bw,
        warnings=warnings,
    )
