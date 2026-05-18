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
# Default base ships torch + cuDNN preinstalled so the container is ready to
# run without `pip install torch`. Override via `--base IMAGE` to layer on top
# of your own image, or `--cuda-tag <tag>` for a nvidia/cuda variant.
DEFAULT_BASE_IMAGE = "pytorch/pytorch:2.12.0-cuda12.6-cudnn9-devel"


def _dockerfile_dir() -> str:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(pkg_dir), "docker")


def build_image(
    dockerfile_dir: Optional[str] = None,
    tag: str = IMAGE_TAG,
    no_cache: bool = False,
    base_image: Optional[str] = None,
    cuda_tag: Optional[str] = None,
) -> None:
    """Build the cosplay image. `base_image` wins over `cuda_tag` (compat)."""
    dockerfile_dir = dockerfile_dir or _dockerfile_dir()
    if base_image is None and cuda_tag is not None:
        base_image = f"nvidia/cuda:{cuda_tag}"
    args = _docker() + ["build", "-t", tag]
    if no_cache:
        args.insert(args.index("build") + 1, "--no-cache")
    if base_image:
        args += ["--build-arg", f"BASE_IMAGE={base_image}"]
    args += [dockerfile_dir]
    print(
        f"[cosplay] building image {tag} from {dockerfile_dir} "
        f"(BASE_IMAGE={base_image or DEFAULT_BASE_IMAGE})"
    )
    _run(args)


def _ensure_image(image: str = IMAGE_TAG) -> None:
    """If `image` is the default tag, build it on miss. Otherwise just check it exists locally."""
    if _image_exists(image):
        return
    if image == IMAGE_TAG:
        df_dir = _dockerfile_dir()
        if not os.path.isfile(os.path.join(df_dir, "Dockerfile")):
            raise SystemExit(
                f"Image {IMAGE_TAG} not built and Dockerfile not found at {df_dir}.\n"
                f"Run: gpu-cosplay build"
            )
        build_image(df_dir)
        return
    # User-specified image: try to pull, else fail loud.
    print(f"[cosplay] image {image} not present locally; attempting docker pull")
    p = subprocess.run(_docker() + ["pull", image], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(
            f"Image {image!r} not found locally and `docker pull` failed:\n{p.stderr.strip()}\n"
            f"Either pull/build it manually, or layer cosplay on top via:\n"
            f"  gpu-cosplay build --base {image} --tag my-cosplay"
        )


def _image_has_entrypoint(image: str) -> bool:
    """Check if image already has /entrypoint.sh baked in (a cosplay-built image)."""
    p = subprocess.run(
        _docker()
        + [
            "run",
            "--rm",
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            "test -x /entrypoint.sh && echo YES || echo NO",
        ],
        capture_output=True,
        text=True,
    )
    return "YES" in p.stdout


def _query_persistence_mode(gpu_index: int) -> Optional[bool]:
    """Returns True/False if known, None on parse failure."""
    p = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=persistence_mode",
            "--format=csv,noheader",
            "-i",
            str(gpu_index),
        ],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        return None
    out = p.stdout.strip().lower()
    if "enabled" in out:
        return True
    if "disabled" in out:
        return False
    return None


def _set_persistence(gpu_index: int, enabled: bool) -> None:
    _run(
        _need_sudo() + ["nvidia-smi", "-i", str(gpu_index), "-pm", "1" if enabled else "0"],
        check=False,
    )


def _phys_vram_mib(plan: Plan) -> int:
    """How many MiB the chosen MIG slice (or full GPU) physically has.

    The nvidia-smi shim uses this to know which number to substitute in the
    real binary's output. From the host side we know it exactly because we
    just picked the MIG profile (or are using the whole GPU).
    """
    if plan.mig_profile is not None:
        return int(round(plan.mig_profile.memory_gb * 1024))
    return int(round(plan.host.memory_total_gb * 1024))


def _image_has_sshd(image: str) -> bool:
    """Quick probe: does the image have sshd in PATH?"""
    p = subprocess.run(
        _docker()
        + [
            "run",
            "--rm",
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            "command -v sshd >/dev/null 2>&1 && echo YES || echo NO",
        ],
        capture_output=True,
        text=True,
    )
    return "YES" in p.stdout


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
    """Apply the plan and launch a container.

    When `image` is the default cosplay tag, behave as before. When it is a
    user-supplied image without our entrypoint baked in, bind-mount the
    entrypoint and inject helper at runtime so any Ubuntu/Debian-derived
    image can be used directly without rebuilding.
    """
    if name is None:
        name = f"cosplay-{plan.target.key.replace('_', '-')}-{random.randint(1000, 9999)}"
    if _container_exists(name):
        raise SystemExit(f"container {name!r} already exists. Use `gpu-cosplay down {name}` first.")
    _ensure_image(image)

    # Decide BYO vs cosplay-baked.
    is_byo = (image != IMAGE_TAG) and not _image_has_entrypoint(image)
    has_sshd = True
    if is_byo:
        has_sshd = _image_has_sshd(image)
        print(
            f"[cosplay] image {image!r} treated as BYO "
            f"(sshd={'present' if has_sshd else 'missing -> will use docker exec'})"
        )

    gpu = plan.host
    original_power = gpu.power_default_w
    original_persistence = _query_persistence_mode(gpu.index)
    mig_changed = False
    mig_uuid = None
    gi_id = ci_id = None

    try:
        # Persistence mode (needed for clock lock to stick). Remember the
        # prior state so `down` can restore it if we flipped it.
        _set_persistence(gpu.index, True)

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

        # SSH port (only if sshd is available)
        port = (ssh_port or _pick_free_port()) if has_sshd else 0

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
            # Authoritative tag for `gpu-cosplay reset` to find our containers
            # even when the user gave a non-standard --name.
            "--label",
            "gpu-cosplay.session=1",
            "--label",
            f"gpu-cosplay.target={plan.target.key}",
            "--label",
            f"gpu-cosplay.host-gpu={gpu.index}",
        ]
        if port:
            args += ["-p", f"{port}:22"]
        args += [
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
            "-e",
            f"GPU_COSPLAY_TDP_W={plan.target.tdp_w}",
            "-e",
            f"GPU_COSPLAY_PHYS_VRAM_MIB={_phys_vram_mib(plan)}",
        ]
        for k, v in (extra_env or {}).items():
            args += ["-e", f"{k}={v}"]
        for hp, cp in extra_volumes or []:
            args += ["-v", f"{os.path.abspath(hp)}:{cp}"]
        args += gpu_flag

        # BYO image: bind-mount entrypoint + Python runtime + nvidia-smi shim,
        # override entrypoint so we get user setup + env baking + symlink install.
        if is_byo:
            df_dir = _dockerfile_dir()
            mounts = {
                "entrypoint.sh": "/opt/gpu-cosplay/entrypoint.sh",
                "gpu_cosplay_runtime.py": "/opt/gpu-cosplay/python/gpu_cosplay_runtime.py",
                "gpu_cosplay_runtime.pth": "/opt/gpu-cosplay/python/gpu_cosplay_runtime.pth",
                "nvidia-smi": "/opt/gpu-cosplay/nvidia-smi",
                "gpu_cosplay_verify.py": "/opt/gpu-cosplay/gpu_cosplay_verify.py",
            }
            for src, dst in mounts.items():
                args += ["-v", f"{os.path.join(df_dir, src)}:{dst}:ro"]
            args += ["--entrypoint", "/opt/gpu-cosplay/entrypoint.sh"]
        args += [image]
        # CMD: sshd if available; else sleep infinity for docker-exec workflow.
        if not has_sshd:
            args += ["sleep", "infinity"]

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
            original_persistence_mode=original_persistence,
            image=image,
            ssh_available=has_sshd,
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
    """Tear down a session and restore the host GPU to its pre-up state.

    Every step is best-effort and continues on failure so that the state.json
    entry is always removed at the end — a stale entry is worse than a
    half-cleaned device because the latter is observable from `nvidia-smi`.
    """
    sess = state.get(name)
    if sess is None:
        raise SystemExit(f"no such session: {name}. Use `gpu-cosplay ps`.")

    # Stop container (idempotent: succeeds even if already gone).
    subprocess.run(_docker() + ["rm", "-f", sess.container_name], capture_output=True)

    # Reset clock lock if we set one.
    if sess.clock_mhz is not None:
        try:
            _reset_clock(sess.gpu_index)
        except Exception as e:
            print(
                f"[cosplay] warning: could not reset clock on GPU {sess.gpu_index}: {e}",
                file=sys.stderr,
            )

    # Destroy MIG instances we created.
    if sess.mig_profile_name is not None:
        try:
            _destroy_mig_instances(sess.gpu_index)
        except Exception as e:
            print(
                f"[cosplay] warning: could not destroy MIG on GPU {sess.gpu_index}: {e}",
                file=sys.stderr,
            )
        # Only flip MIG mode off if we were the one who flipped it on.
        if not sess.original_mig_enabled:
            try:
                _disable_mig(sess.gpu_index)
            except Exception as e:
                print(
                    f"[cosplay] warning: could not disable MIG on GPU {sess.gpu_index}: {e}",
                    file=sys.stderr,
                )

    # Restore power limit if we changed it.
    if sess.power_limit_w is not None and sess.original_power_limit_w is not None:
        try:
            _set_power(sess.gpu_index, sess.original_power_limit_w)
        except Exception as e:
            print(
                f"[cosplay] warning: could not restore power on GPU {sess.gpu_index}: {e}",
                file=sys.stderr,
            )

    # Restore persistence mode if we know what it was and flipped it.
    if sess.original_persistence_mode is False:
        try:
            _set_persistence(sess.gpu_index, False)
        except Exception:
            pass

    state.remove(name)


def reset(gpu_indices: Optional[list[int]] = None, purge_state: bool = False) -> dict:
    """Force-reset GPUs back to driver defaults. Safe to run any time; useful
    when `down` failed or state.json is corrupt.

    For each target GPU:
      - Reset clock lock
      - Reset power limit to its default value (queried from the GPU itself)
      - Destroy any MIG compute and GPU instances
      - Disable MIG mode if it was enabled

    Also removes any docker containers whose name starts with `cosplay-`. If
    `purge_state` is True, wipes ~/.cache/gpu-cosplay/state.json.

    Returns a dict report of what was done.
    """
    report: dict = {"containers_removed": [], "gpus": []}

    # 1. Remove all our containers. Try three sources so we catch everything:
    #    (a) docker label "gpu-cosplay.session=1" added at up() — authoritative.
    #    (b) name prefix "cosplay-" — handles legacy sessions without labels.
    #    (c) container_name from state.json — handles user-supplied --name.
    seen: set[str] = set()
    for filt in ("label=gpu-cosplay.session=1", "name=^cosplay-"):
        p = subprocess.run(
            _docker() + ["ps", "-a", "--filter", filt, "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        for cname in p.stdout.split():
            if cname in seen:
                continue
            seen.add(cname)
            rm = subprocess.run(_docker() + ["rm", "-f", cname], capture_output=True, text=True)
            report["containers_removed"].append(
                {"name": cname, "rc": rm.returncode, "err": rm.stderr.strip()}
            )
    for s in state.all_sessions():
        if s.container_name in seen:
            continue
        seen.add(s.container_name)
        rm = subprocess.run(
            _docker() + ["rm", "-f", s.container_name], capture_output=True, text=True
        )
        # Only report if the container actually existed; rm of a missing one is silent ok.
        if rm.returncode == 0 and rm.stdout.strip():
            report["containers_removed"].append({"name": s.container_name, "rc": 0, "err": ""})

    # 2. Iterate GPUs and reset.
    try:
        gpus = list_host_gpus()
    except Exception as e:
        gpus = []
        report["gpu_enum_error"] = str(e)

    if gpu_indices is None:
        targets = gpus
    else:
        targets = [g for g in gpus if g.index in gpu_indices]

    for g in targets:
        item: dict = {"index": g.index, "name": g.name, "actions": []}

        # Reset clock lock (no-op if no lock was set).
        rc = subprocess.run(
            _need_sudo() + ["nvidia-smi", "-i", str(g.index), "--reset-gpu-clocks"],
            capture_output=True,
            text=True,
        )
        item["actions"].append({"clock_reset": rc.returncode == 0, "err": rc.stderr.strip()})

        # Reset power limit to default (we read it from the GPU itself).
        if g.power_default_w is not None:
            rc = subprocess.run(
                _need_sudo() + ["nvidia-smi", "-i", str(g.index), "-pl", str(g.power_default_w)],
                capture_output=True,
                text=True,
            )
            item["actions"].append(
                {
                    "power_reset_to_w": g.power_default_w,
                    "rc": rc.returncode,
                    "err": rc.stderr.strip(),
                }
            )

        # Destroy MIG instances (CI before GI, as the API requires).
        if g.mig_capable:
            rc1 = subprocess.run(
                _need_sudo() + ["nvidia-smi", "mig", "-i", str(g.index), "-dci"],
                capture_output=True,
                text=True,
            )
            rc2 = subprocess.run(
                _need_sudo() + ["nvidia-smi", "mig", "-i", str(g.index), "-dgi"],
                capture_output=True,
                text=True,
            )
            item["actions"].append({"mig_destroyed": rc1.returncode == 0 or rc2.returncode == 0})

            # Disable MIG mode if currently enabled.
            if g.mig_enabled:
                rc = subprocess.run(
                    _need_sudo() + ["nvidia-smi", "-i", str(g.index), "-mig", "0"],
                    capture_output=True,
                    text=True,
                )
                item["actions"].append({"mig_mode_disabled": rc.returncode == 0})

        report["gpus"].append(item)

    # 3. Optionally purge state.json.
    if purge_state:
        report["state_purged"] = 0
        for s in state.all_sessions():
            state.remove(s.name)
            report["state_purged"] += 1

    return report


def down_all() -> int:
    n = 0
    for s in state.all_sessions():
        try:
            down(s.name)
            n += 1
        except Exception as e:
            print(f"[cosplay] warning: failed to clean up {s.name}: {e}", file=sys.stderr)
    return n
