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
    assert os.path.isfile(os.path.join(df, "gpu_cosplay_inject.py"))
