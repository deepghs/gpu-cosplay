"""Tests for the small pure helpers in apply.py — no docker required."""

import os
from unittest import mock

import gpu_cosplay.apply as apply_mod


def test_default_base_image_format():
    assert apply_mod.DEFAULT_BASE_IMAGE.startswith("nvidia/cuda:")
    assert apply_mod.DEFAULT_BASE_IMAGE.endswith(apply_mod.DEFAULT_CUDA_TAG)


def test_build_image_passes_base_arg():
    captured = []
    with (
        mock.patch.object(
            apply_mod, "_run", lambda args, **kw: captured.append(args) or mock.Mock()
        ),
        mock.patch.object(apply_mod, "_docker", lambda: ["docker"]),
    ):
        apply_mod.build_image("docker", base_image="myorg/pytorch:v3")
    cmd = captured[0]
    assert "--build-arg" in cmd
    i = cmd.index("--build-arg")
    assert cmd[i + 1] == "BASE_IMAGE=myorg/pytorch:v3"


def test_build_image_cuda_tag_translates_to_base():
    captured = []
    with (
        mock.patch.object(
            apply_mod, "_run", lambda args, **kw: captured.append(args) or mock.Mock()
        ),
        mock.patch.object(apply_mod, "_docker", lambda: ["docker"]),
    ):
        apply_mod.build_image("docker", cuda_tag="12.4.1-cudnn-devel-ubuntu22.04")
    cmd = captured[0]
    i = cmd.index("--build-arg")
    assert cmd[i + 1] == "BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04"


def test_build_image_base_overrides_cuda_tag():
    captured = []
    with (
        mock.patch.object(
            apply_mod, "_run", lambda args, **kw: captured.append(args) or mock.Mock()
        ),
        mock.patch.object(apply_mod, "_docker", lambda: ["docker"]),
    ):
        apply_mod.build_image("docker", base_image="my:img", cuda_tag="ignored")
    cmd = captured[0]
    i = cmd.index("--build-arg")
    assert cmd[i + 1] == "BASE_IMAGE=my:img"


def test_dockerfile_dir_exists_in_repo():
    df = apply_mod._dockerfile_dir()
    # In dev install, the docker/ dir should be present alongside the package.
    assert os.path.isfile(os.path.join(df, "Dockerfile"))
    assert os.path.isfile(os.path.join(df, "entrypoint.sh"))
    assert os.path.isfile(os.path.join(df, "gpu_cosplay_runtime.py"))
    assert os.path.isfile(os.path.join(df, "gpu_cosplay_runtime.pth"))
    assert os.path.isfile(os.path.join(df, "nvidia-smi"))


def test_phys_vram_mib_from_mig_profile():
    from gpu_cosplay.cards import find_card
    from gpu_cosplay.host import HostGPU, MigProfile

    h = HostGPU(
        index=0,
        name="NVIDIA H200",
        uuid="GPU-x",
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
                profile_id=14,
                name="2g.35gb",
                sm_count=32,
                memory_gb=35,
                instances_total=3,
                instances_free=3,
            ),
        ],
    )
    from gpu_cosplay.plan import plan as plan_fn

    p = plan_fn(h, find_card("3090"))
    assert apply_mod._phys_vram_mib(p) == int(round(35 * 1024))


def test_phys_vram_mib_no_mig_uses_full_gpu():
    from gpu_cosplay.cards import find_card
    from gpu_cosplay.host import HostGPU

    h = HostGPU(
        index=0,
        name="NVIDIA L40S",
        uuid="GPU-x",
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
    from gpu_cosplay.plan import plan as plan_fn

    p = plan_fn(h, find_card("3090"))
    assert apply_mod._phys_vram_mib(p) == int(round(48 * 1024))
