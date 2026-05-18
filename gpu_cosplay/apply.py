"""Apply a Plan: mutate the host GPU, then launch the docker container."""

from __future__ import annotations

import os
import random
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

from . import ssh, state
from .host import list_host_gpus
from .plan import Plan

IMAGE_TAG = "gpu-cosplay:latest"


def _run(args: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    p = subprocess.run(args, capture_output=capture, text=True)
    if check and p.returncode != 0:
        out = (p.stderr or "") + (p.stdout or "")
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(args)}\n{out}")
    return p


def _need_sudo() -> list[str]:
    return [] if os.geteuid() == 0 else ["sudo", "-n"]


_DOCKER_PREFIX: Optional[list[str]] = None


def _docker() -> list[str]:
    """Return ['docker'] or ['sudo','-n','docker'] depending on access. Cached."""
    global _DOCKER_PREFIX
    if _DOCKER_PREFIX is not None:
        return _DOCKER_PREFIX
    # Try plain `docker version` first; if it works, no sudo needed.
    p = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True, text=True
    )
    if p.returncode == 0:
        _DOCKER_PREFIX = ["docker"]
    else:
        _DOCKER_PREFIX = _need_sudo() + ["docker"]
    return _DOCKER_PREFIX


def _pick_free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _container_exists(name: str) -> bool:
    p = subprocess.run(
        _docker() + ["ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    return name in p.stdout.split()


def _image_exists(tag: str) -> bool:
    p = subprocess.run(_docker() + ["images", "-q", tag], capture_output=True, text=True)
    return bool(p.stdout.strip())


DEFAULT_CUDA_TAG = "12.6.3-cudnn-devel-ubuntu24.04"


def build_image(
    dockerfile_dir: str,
    tag: str = IMAGE_TAG,
    no_cache: bool = False,
    cuda_tag: Optional[str] = None,
) -> None:
    args = _docker() + ["build", "-t", tag]
    if no_cache:
        args.insert(args.index("build") + 1, "--no-cache")
    if cuda_tag:
        args += ["--build-arg", f"CUDA_TAG={cuda_tag}"]
    args += [dockerfile_dir]
    print(
        f"[cosplay] building image {tag} from {dockerfile_dir} (CUDA_TAG={cuda_tag or DEFAULT_CUDA_TAG})"
    )
    _run(args)


def _ensure_image() -> None:
    if _image_exists(IMAGE_TAG):
        return
    # Locate docker/ relative to package
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(pkg_dir)
    df_dir = os.path.join(repo_root, "docker")
    if not os.path.isfile(os.path.join(df_dir, "Dockerfile")):
        raise SystemExit(
            f"Image {IMAGE_TAG} not built and Dockerfile not found at {df_dir}.\n"
            f"Run: gpu-cosplay build"
        )
    build_image(df_dir)


# ---------------------------------------------------------------------------
# MIG management
# ---------------------------------------------------------------------------


def _enable_mig(gpu_index: int) -> bool:
    """Enable MIG on a GPU. Returns True if we changed the state (so we can revert)."""
    gpus_before = {g.index: g.mig_enabled for g in list_host_gpus()}
    if gpus_before.get(gpu_index, False):
        return False  # already enabled, don't touch on cleanup
    _run(_need_sudo() + ["nvidia-smi", "-i", str(gpu_index), "-mig", "1"])
    # Verify
    time.sleep(0.5)
    for g in list_host_gpus():
        if g.index == gpu_index and g.mig_enabled:
            return True
    raise RuntimeError(f"failed to enable MIG on GPU {gpu_index}")


def _disable_mig(gpu_index: int) -> None:
    _run(_need_sudo() + ["nvidia-smi", "-i", str(gpu_index), "-mig", "0"], check=False)


def _create_mig_instance(gpu_index: int, profile_id: int) -> tuple[str, int, int]:
    """Create a GI+CI on the GPU with the given profile id.

    Returns (mig_uuid, gi_id, ci_id).
    """
    p = _run(
        _need_sudo() + ["nvidia-smi", "mig", "-i", str(gpu_index), "-cgi", str(profile_id), "-C"],
        capture=True,
    )
    text = p.stdout + p.stderr
    import re

    gi = re.search(r"GPU instance ID\s+(\d+)", text)
    ci = re.search(r"compute instance ID\s+(\d+)", text)
    if not gi or not ci:
        raise RuntimeError(f"failed to parse MIG creation output:\n{text}")
    gi_id, ci_id = int(gi.group(1)), int(ci.group(1))
    # Find MIG UUID for this GPU/GI
    pp = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    cur_gpu = None
    for line in pp.stdout.splitlines():
        gm = re.match(r"GPU (\d+):", line)
        if gm:
            cur_gpu = int(gm.group(1))
            continue
        if cur_gpu == gpu_index:
            um = re.search(r"\(UUID: (MIG-[a-f0-9-]+)\)", line)
            if um:
                return um.group(1), gi_id, ci_id
    raise RuntimeError("created MIG instance but failed to locate its UUID")


def _destroy_mig_instances(gpu_index: int) -> None:
    _run(_need_sudo() + ["nvidia-smi", "mig", "-i", str(gpu_index), "-dci"], check=False)
    _run(_need_sudo() + ["nvidia-smi", "mig", "-i", str(gpu_index), "-dgi"], check=False)


# ---------------------------------------------------------------------------
# clock + power
# ---------------------------------------------------------------------------


def _lock_clock(gpu_index: int, mhz: int) -> None:
    _run(
        _need_sudo()
        + [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            f"--lock-gpu-clocks={mhz},{mhz}",
        ]
    )


def _reset_clock(gpu_index: int) -> None:
    _run(_need_sudo() + ["nvidia-smi", "-i", str(gpu_index), "--reset-gpu-clocks"], check=False)


def _set_power(gpu_index: int, watts: int) -> None:
    _run(_need_sudo() + ["nvidia-smi", "-i", str(gpu_index), "-pl", str(watts)])


# ---------------------------------------------------------------------------
# Bring up
# ---------------------------------------------------------------------------


@dataclass
class UpResult:
    session: state.Session
    plan: Plan


def up(
    plan: Plan,
    name: Optional[str] = None,
    ssh_port: Optional[int] = None,
    extra_volumes: Optional[list[tuple[str, str]]] = None,
    workspace: Optional[str] = None,
    extra_env: Optional[dict[str, str]] = None,
    image: str = IMAGE_TAG,
    detach: bool = True,
) -> UpResult:
    """Apply the plan and launch a container."""
    if name is None:
        name = f"cosplay-{plan.target.key.replace('_', '-')}-{random.randint(1000, 9999)}"
    if _container_exists(name):
        raise SystemExit(f"container {name!r} already exists. Use `gpu-cosplay down {name}` first.")
    _ensure_image()

    gpu = plan.host
    original_power = gpu.power_default_w
    mig_changed = False
    mig_uuid = None
    gi_id = ci_id = None

    try:
        # Persistence mode (needed for clock lock to stick)
        _run(_need_sudo() + ["nvidia-smi", "-i", str(gpu.index), "-pm", "1"], check=False)

        # Power
        if plan.power_limit_w is not None:
            try:
                _set_power(gpu.index, plan.power_limit_w)
            except Exception as e:
                print(f"[cosplay] warning: failed to set power limit: {e}")

        # MIG
        if plan.mig_profile is not None:
            mig_changed = _enable_mig(gpu.index)
            _destroy_mig_instances(gpu.index)  # clean stale
            mig_uuid, gi_id, ci_id = _create_mig_instance(gpu.index, plan.mig_profile.profile_id)

        # Clock
        if plan.clock_mhz is not None:
            _lock_clock(gpu.index, plan.clock_mhz)

        # SSH port
        port = ssh_port or _pick_free_port()

        # GPU passthrough flag
        if mig_uuid:
            gpu_flag = ["--gpus", f'"device={mig_uuid}"']
        else:
            gpu_flag = ["--gpus", f'"device={gpu.index}"']

        # Workspace
        ws_host = os.path.abspath(workspace or os.getcwd())
        os.makedirs(ws_host, exist_ok=True)

        # User identity
        host_uid = os.getuid()
        host_gid = os.getgid()
        host_user = os.environ.get("USER", "ubuntu")

        # Public key
        pubkey = ssh.public_key()

        # Build docker run
        args = _docker() + [
            "run",
            "-d" if detach else "-it",
            "--name",
            name,
            "--hostname",
            f"cosplay-{plan.target.key.replace('_', '-')}",
            "-p",
            f"{port}:22",
            "-v",
            f"{ws_host}:/workspace",
            "--shm-size",
            "8g",
            "-e",
            f"HOST_USER={host_user}",
            "-e",
            f"HOST_UID={host_uid}",
            "-e",
            f"HOST_GID={host_gid}",
            "-e",
            f"GPU_COSPLAY_PUBKEY={pubkey}",
            "-e",
            f"GPU_COSPLAY_VRAM_GB={plan.vram_cap_gb}",
            "-e",
            f"GPU_COSPLAY_CARD={plan.target.key}",
            "-e",
            f"GPU_COSPLAY_PRETTY={plan.target.pretty}",
            "-e",
            f"GPU_COSPLAY_FP32_TFLOPS={plan.target.fp32_tflops}",
            "-e",
            f"GPU_COSPLAY_BF16_TC_TFLOPS={plan.target.bf16_tc_tflops or 0}",
            "-e",
            f"GPU_COSPLAY_BW_GBS={plan.target.bandwidth_gbs}",
        ]
        for k, v in (extra_env or {}).items():
            args += ["-e", f"{k}={v}"]
        for hp, cp in extra_volumes or []:
            args += ["-v", f"{os.path.abspath(hp)}:{cp}"]
        args += gpu_flag
        args += [image]

        print(f"[cosplay] docker run: {' '.join(shlex.quote(a) for a in args)}")
        cp = subprocess.run(args, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"docker run failed: {cp.stderr}")
        cid = cp.stdout.strip()

        sess = state.Session(
            name=name,
            card_key=plan.target.key,
            container_id=cid,
            container_name=name,
            gpu_index=gpu.index,
            mig_profile_name=plan.mig_profile.name if plan.mig_profile else None,
            mig_uuid=mig_uuid,
            mig_gi_id=gi_id,
            mig_ci_id=ci_id,
            clock_mhz=plan.clock_mhz,
            power_limit_w=plan.power_limit_w,
            ssh_port=port,
            vram_cap_gb=plan.vram_cap_gb,
            workspace_mount=ws_host,
            original_power_limit_w=original_power,
            original_mig_enabled=not mig_changed,  # True if MIG was already on
        )
        state.add(sess)
        return UpResult(session=sess, plan=plan)

    except Exception:
        # Best-effort revert
        if plan.clock_mhz is not None:
            _reset_clock(gpu.index)
        if plan.mig_profile is not None:
            _destroy_mig_instances(gpu.index)
            if mig_changed:
                _disable_mig(gpu.index)
        if plan.power_limit_w is not None and original_power is not None:
            try:
                _set_power(gpu.index, original_power)
            except Exception:
                pass
        raise


def down(name: str) -> None:
    sess = state.get(name)
    if sess is None:
        raise SystemExit(f"no such session: {name}. Use `gpu-cosplay ps`.")
    # Stop container
    subprocess.run(_docker() + ["rm", "-f", sess.container_name], capture_output=True)
    # Reset clock
    if sess.clock_mhz is not None:
        _reset_clock(sess.gpu_index)
    # Destroy MIG
    if sess.mig_profile_name is not None:
        _destroy_mig_instances(sess.gpu_index)
        if not sess.original_mig_enabled:
            _disable_mig(sess.gpu_index)
    # Restore power
    if sess.power_limit_w is not None and sess.original_power_limit_w is not None:
        try:
            _set_power(sess.gpu_index, sess.original_power_limit_w)
        except Exception:
            pass
    state.remove(name)


def down_all() -> int:
    n = 0
    for s in state.all_sessions():
        try:
            down(s.name)
            n += 1
        except Exception as e:
            print(f"[cosplay] warning: failed to clean up {s.name}: {e}", file=sys.stderr)
    return n
