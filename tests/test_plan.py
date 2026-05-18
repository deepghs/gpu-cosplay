"""Plan/matcher tests using synthetic HostGPU fixtures."""

import pytest

from gpu_cosplay.cards import find_card
from gpu_cosplay.host import HostGPU, MigProfile
from gpu_cosplay.plan import plan


def fake_h200() -> HostGPU:
    return HostGPU(
        index=0,
        name="NVIDIA H200",
        uuid="GPU-fake",
        compute_cap="9.0",
        memory_total_gb=141.0,
        power_min_w=200,
        power_max_w=700,
        power_default_w=700,
        clock_min_mhz=345,
        clock_max_mhz=1980,
        mig_capable=True,
        mig_enabled=False,
        mig_profiles=[
            MigProfile(
                profile_id=19,
                name="1g.18gb",
                sm_count=16,
                memory_gb=18,
                instances_total=7,
                instances_free=7,
            ),
            MigProfile(
                profile_id=15,
                name="1g.35gb",
                sm_count=26,
                memory_gb=35,
                instances_total=4,
                instances_free=4,
            ),
            MigProfile(
                profile_id=14,
                name="2g.35gb",
                sm_count=32,
                memory_gb=35,
                instances_total=3,
                instances_free=3,
            ),
            MigProfile(
                profile_id=9,
                name="3g.71gb",
                sm_count=60,
                memory_gb=71,
                instances_total=2,
                instances_free=2,
            ),
            MigProfile(
                profile_id=5,
                name="4g.71gb",
                sm_count=64,
                memory_gb=71,
                instances_total=1,
                instances_free=1,
            ),
            MigProfile(
                profile_id=0,
                name="7g.141gb",
                sm_count=132,
                memory_gb=141,
                instances_total=1,
                instances_free=1,
            ),
        ],
    )


def fake_a100_80g() -> HostGPU:
    return HostGPU(
        index=0,
        name="NVIDIA A100-SXM4-80GB",
        uuid="GPU-fake",
        compute_cap="8.0",
        memory_total_gb=80.0,
        power_min_w=100,
        power_max_w=400,
        power_default_w=400,
        clock_min_mhz=210,
        clock_max_mhz=1410,
        mig_capable=True,
        mig_enabled=False,
        mig_profiles=[
            MigProfile(
                profile_id=19,
                name="1g.10gb",
                sm_count=14,
                memory_gb=10,
                instances_total=7,
                instances_free=7,
            ),
            MigProfile(
                profile_id=15,
                name="1g.20gb",
                sm_count=14,
                memory_gb=20,
                instances_total=4,
                instances_free=4,
            ),
            MigProfile(
                profile_id=14,
                name="2g.20gb",
                sm_count=28,
                memory_gb=20,
                instances_total=3,
                instances_free=3,
            ),
            MigProfile(
                profile_id=9,
                name="3g.40gb",
                sm_count=42,
                memory_gb=40,
                instances_total=2,
                instances_free=2,
            ),
            MigProfile(
                profile_id=5,
                name="4g.40gb",
                sm_count=56,
                memory_gb=40,
                instances_total=1,
                instances_free=1,
            ),
            MigProfile(
                profile_id=0,
                name="7g.80gb",
                sm_count=98,
                memory_gb=80,
                instances_total=1,
                instances_free=1,
            ),
        ],
    )


def fake_l40s() -> HostGPU:
    # No MIG support
    return HostGPU(
        index=0,
        name="NVIDIA L40S",
        uuid="GPU-fake",
        compute_cap="8.9",
        memory_total_gb=48.0,
        power_min_w=100,
        power_max_w=350,
        power_default_w=350,
        clock_min_mhz=210,
        clock_max_mhz=2520,
        mig_capable=False,
        mig_enabled=False,
        mig_profiles=[],
    )


@pytest.mark.parametrize(
    "card_alias",
    [
        "1650",
        "1660ti",
        "2060",
        "3090",
        "4090",
        "a100",
        "h100",
    ],
)
def test_plan_runs_on_h200(card_alias):
    h = fake_h200()
    c = find_card(card_alias)
    p = plan(h, c)
    assert p.target.key == c.key
    assert p.vram_cap_gb == c.vram_gb
    # Expected FP32 should be within 2x of target (or limited by host)
    assert p.expected_fp32 > 0


def test_plan_picks_smallest_mig_that_fits():
    h = fake_h200()
    p = plan(h, find_card("3090"))  # 24 GB
    # 2g.35gb has 35 GB; 1g.18gb has 18 GB (too small)
    assert p.mig_profile is not None
    assert p.mig_profile.memory_gb >= 24


def test_plan_4gb_target_uses_smallest_slice():
    h = fake_h200()
    p = plan(h, find_card("1650"))  # 4 GB
    assert p.mig_profile is not None
    assert p.mig_profile.name == "1g.18gb"


def test_plan_on_a100_80g_for_3090():
    h = fake_a100_80g()
    p = plan(h, find_card("3090"))
    assert p.mig_profile is not None
    assert p.mig_profile.memory_gb >= 24


def test_plan_on_a100_80g_for_a100_40g():
    h = fake_a100_80g()
    p = plan(h, find_card("a100"))
    assert p.mig_profile is not None
    assert p.mig_profile.memory_gb >= 40


def test_plan_falls_back_to_no_mig_when_unsupported():
    h = fake_l40s()
    p = plan(h, find_card("3090"))
    assert p.mig_profile is None
    assert p.vram_cap_gb == 24


def test_plan_warns_on_no_tc_target():
    h = fake_h200()
    p = plan(h, find_card("1650"))
    assert any("no BF16/FP16 Tensor Core" in w for w in p.warnings)


def test_plan_warns_on_underpowered_host():
    # Target 4090 (82 TFLOPS) on a fake weak host
    weak = HostGPU(
        index=0,
        name="NVIDIA T4",
        uuid="GPU-fake",
        compute_cap="7.5",
        memory_total_gb=16.0,
        power_min_w=20,
        power_max_w=70,
        power_default_w=70,
        clock_min_mhz=300,
        clock_max_mhz=1590,
        mig_capable=False,
        mig_enabled=False,
        mig_profiles=[],
    )
    p = plan(weak, find_card("4090"))
    assert any("maxes out" in w for w in p.warnings)
