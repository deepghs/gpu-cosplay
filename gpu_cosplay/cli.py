"""gpu-cosplay command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Optional

from . import __version__, apply, host, ssh, state
from . import plan as planmod
from .cards import find_card, load_cards


def cmd_ls(args: argparse.Namespace) -> int:
    cards = load_cards()
    if args.arch:
        cards = [c for c in cards if c.arch == args.arch]
    order = ["turing", "ampere", "ada", "hopper", "volta", "blackwell"]
    cards.sort(
        key=lambda c: (order.index(c.arch) if c.arch in order else 9, c.vram_gb, c.fp32_tflops)
    )
    if args.json:
        print(json.dumps([c.__dict__ for c in cards], indent=2))
        return 0
    print(f"{'CARD':<22} {'ARCH':<8} {'VRAM':>6} {'FP32':>6} {'BF16TC':>7} {'BW':>6} {'TDP':>5}")
    print("-" * 64)
    for c in cards:
        bf = f"{c.bf16_tc_tflops:.0f}" if c.bf16_tc_tflops else "  -"
        print(
            f"{c.pretty:<22} {c.arch:<8} {c.vram_gb:>4.0f}GB {c.fp32_tflops:>5.1f} {bf:>7} {c.bandwidth_gbs:>4.0f}GB/s {c.tdp_w:>3}W"
        )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    c = find_card(args.card)
    print(f"{c.pretty} ({c.key})")
    print(f"  arch:         {c.arch}")
    print(f"  SM count:     {c.sm_count}")
    print(f"  VRAM:         {c.vram_gb} GB")
    print(f"  FP32:         {c.fp32_tflops} TFLOPS")
    bf = f"{c.bf16_tc_tflops} TFLOPS dense" if c.bf16_tc_tflops else "-"
    print(f"  BF16 TC:      {bf}")
    print(f"  Bandwidth:    {c.bandwidth_gbs} GB/s")
    print(f"  TDP:          {c.tdp_w} W")
    if c.notes:
        print(f"  notes:        {c.notes}")
    print(f"  aliases:      {', '.join(c.aliases)}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    gpus = host.list_host_gpus()
    print(f"Host: {len(gpus)} GPU(s) detected")
    for g in gpus:
        print(f"  [{g.index}] {g.name}  cc={g.compute_cap}  vram={g.memory_total_gb:.0f}GB")
        if g.power_min_w is not None:
            print(f"       power: {g.power_min_w}-{g.power_max_w} W (default {g.power_default_w})")
        if g.clock_min_mhz is not None:
            print(f"       clock: {g.clock_min_mhz}-{g.clock_max_mhz} MHz")
        print(f"       MIG: capable={g.mig_capable}, enabled={g.mig_enabled}")
        for p in g.mig_profiles:
            print(
                f"         profile {p.profile_id} = {p.name:<10}  SMs={p.sm_count:<3}  mem={p.memory_gb:.0f}GB  free={p.instances_free}/{p.instances_total}"
            )
    have_docker_direct = subprocess.run(["docker", "version"], capture_output=True).returncode == 0
    have_docker_sudo = (
        subprocess.run(["sudo", "-n", "docker", "version"], capture_output=True).returncode == 0
    )
    have_docker = have_docker_direct or have_docker_sudo
    docker_mode = "direct" if have_docker_direct else ("sudo" if have_docker_sudo else "missing")
    have_nvc = subprocess.run(["nvidia-ctk", "--version"], capture_output=True).returncode == 0
    print(f"docker:           {'OK' if have_docker else 'MISSING'} ({docker_mode})")
    print(f"nvidia-container: {'OK' if have_nvc else 'MISSING - install nvidia-container-toolkit'}")
    sudo_ok = subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0
    print(f"passwordless sudo: {'OK' if sudo_ok else 'MISSING - required for nvidia-smi config'}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    c = find_card(args.card)
    gpus = host.list_host_gpus()
    g = (
        next((x for x in gpus if x.index == args.gpu), None)
        if args.gpu is not None
        else host.pick_default_gpu(gpus)
    )
    if g is None:
        print(f"GPU index {args.gpu} not found", file=sys.stderr)
        return 1
    p = planmod.plan(g, c)
    print(f"Cosplay plan: {c.pretty} on GPU {g.index} ({g.name})")
    print(f"  MIG profile:  {p.mig_profile.name if p.mig_profile else '(no MIG, whole GPU)'}")
    print(f"  Clock lock:   {f'{p.clock_mhz} MHz' if p.clock_mhz else '(unlocked)'}")
    print(f"  Power limit:  {f'{p.power_limit_w} W' if p.power_limit_w else '(default)'}")
    print(f"  VRAM cap:     {p.vram_cap_gb} GB (enforced via PyTorch memory fraction)")
    print(f"  Expected FP32: {p.expected_fp32:.1f} TFLOPS (target {c.fp32_tflops})")
    print(f"  Expected BF16 TC: {p.expected_bf16:.1f} TFLOPS (target {c.bf16_tc_tflops or 0})")
    print(f"  Expected BW:   {p.expected_bw_gbs:.0f} GB/s (target {c.bandwidth_gbs})")
    for w in p.warnings:
        print(f"  ! {w}")
    return 0


def cmd_up(args: argparse.Namespace) -> int:
    c = find_card(args.card)
    gpus = host.list_host_gpus()
    g = (
        next((x for x in gpus if x.index == args.gpu), None)
        if args.gpu is not None
        else host.pick_default_gpu(gpus)
    )
    if g is None:
        print(f"GPU index {args.gpu} not found", file=sys.stderr)
        return 1
    p = planmod.plan(g, c)
    extra_vols = [tuple(v.split(":", 1)) for v in (args.volume or [])]
    extra_env = {}
    for kv in args.env or []:
        k, _, v = kv.partition("=")
        extra_env[k] = v
    res = apply.up(
        p,
        name=args.name,
        ssh_port=args.ssh_port,
        extra_volumes=extra_vols,
        workspace=args.workspace,
        extra_env=extra_env,
        image=args.image,
        detach=True,
    )
    s = res.session
    privkey = ssh.private_key_path()
    print()
    print(f"  Cosplay session up: {s.name}")
    print(f"  Card:       {c.pretty}  on host GPU {g.index} ({g.name})")
    print(f"  MIG:        {s.mig_profile_name or '(whole GPU)'}")
    print(f"  Clock:      {s.clock_mhz or 'default'} MHz")
    print(f"  VRAM cap:   {s.vram_cap_gb} GB")
    print(f"  Workspace:  {s.workspace_mount} -> /workspace")
    print(
        f"  SSH:        ssh -i {privkey} -p {s.ssh_port} {os.environ.get('USER', 'ubuntu')}@localhost"
    )
    print()
    print(f"  Or:         gpu-cosplay ssh {s.name}")
    print(f"  Then:       gpu-cosplay down {s.name}")
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    if args.all:
        n = apply.down_all()
        print(f"removed {n} session(s)")
        return 0
    if not args.name:
        print("usage: gpu-cosplay down <name> | --all", file=sys.stderr)
        return 1
    apply.down(args.name)
    print(f"removed {args.name}")
    return 0


def cmd_ssh(args: argparse.Namespace) -> int:
    sessions = state.all_sessions()
    if not sessions:
        print("no running cosplay sessions", file=sys.stderr)
        return 1
    if args.name:
        s = next((x for x in sessions if x.name == args.name), None)
        if s is None:
            print(f"no such session: {args.name}", file=sys.stderr)
            return 1
    elif len(sessions) == 1:
        s = sessions[0]
    else:
        print("Multiple sessions; pick one with `gpu-cosplay ssh <name>`:", file=sys.stderr)
        for x in sessions:
            print(
                f"  {x.name}  ({x.card_key} on GPU {x.gpu_index}, ssh port {x.ssh_port})",
                file=sys.stderr,
            )
        return 1
    key = ssh.private_key_path()
    user = os.environ.get("USER", "ubuntu")
    cmd = [
        "ssh",
        "-i",
        key,
        "-p",
        str(s.ssh_port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        f"{user}@localhost",
    ]
    if args.command:
        cmd += args.command
    os.execvp(cmd[0], cmd)


def cmd_dexec(args: argparse.Namespace) -> int:
    if not args.command:
        print("usage: gpu-cosplay exec [name] -- <cmd> [args ...]", file=sys.stderr)
        return 1
    sessions = state.all_sessions()
    if not sessions:
        print("no running cosplay sessions", file=sys.stderr)
        return 1
    if args.name:
        s = next((x for x in sessions if x.name == args.name), None)
        if s is None:
            print(f"no such session: {args.name}", file=sys.stderr)
            return 1
    else:
        if len(sessions) > 1:
            print("multiple sessions; specify name", file=sys.stderr)
            return 1
        s = sessions[0]
    user = os.environ.get("USER", "ubuntu")
    cmd = apply._docker() + ["exec", "-i", "-u", user, s.container_name] + args.command
    os.execvp(cmd[0], cmd)


def cmd_ps(args: argparse.Namespace) -> int:
    sessions = state.all_sessions()
    if not sessions:
        print("no cosplay sessions running")
        return 0
    print(f"{'NAME':<32} {'CARD':<16} {'GPU':<4} {'MIG':<10} {'CLOCK':<7} {'VRAM':<6} {'SSH'}")
    for s in sessions:
        print(
            f"{s.name:<32} {s.card_key:<16} {s.gpu_index:<4} "
            f"{(s.mig_profile_name or '-'):<10} "
            f"{(str(s.clock_mhz) if s.clock_mhz else '-'):<7} "
            f"{s.vram_cap_gb:>4.0f}GB {s.ssh_port}"
        )
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(pkg_dir)
    df = os.path.join(repo_root, "docker")
    if not os.path.isfile(os.path.join(df, "Dockerfile")):
        print(f"Dockerfile not found at {df}", file=sys.stderr)
        return 1
    apply.build_image(df, tag=args.tag, no_cache=args.no_cache, cuda_tag=args.cuda_tag)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpu-cosplay",
        description="Make a beefy NVIDIA GPU pretend to be a smaller one.",
    )
    p.add_argument("--version", action="version", version=f"gpu-cosplay {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("ls", help="list supported cards")
    sp.add_argument("--arch", help="filter by architecture (turing/ampere/ada/hopper/volta)")
    sp.add_argument("--json", action="store_true", help="emit JSON")
    sp.set_defaults(func=cmd_ls)

    sp = sub.add_parser("info", help="show specs for a card")
    sp.add_argument("card")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("doctor", help="check host capabilities")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("plan", help="show how a target would be matched on this host")
    sp.add_argument("card")
    sp.add_argument("--gpu", type=int, default=None)
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("up", help="start a cosplay container")
    sp.add_argument("card", help="target card name (e.g. 3090, 4090, a100, 2060)")
    sp.add_argument("--name", help="container name (default: auto)")
    sp.add_argument("--gpu", type=int, default=None, help="host GPU index")
    sp.add_argument("--ssh-port", type=int, default=None, help="host port to map to container :22")
    sp.add_argument(
        "--volume", "-v", action="append", help="extra HOST:CONTAINER mount; repeatable"
    )
    sp.add_argument("--env", "-e", action="append", help="extra KEY=VALUE env; repeatable")
    sp.add_argument("--workspace", help="host dir to mount at /workspace (default: $PWD)")
    sp.add_argument("--image", default=apply.IMAGE_TAG)
    sp.set_defaults(func=cmd_up)

    sp = sub.add_parser("down", help="stop a cosplay container and revert GPU state")
    sp.add_argument("name", nargs="?")
    sp.add_argument("--all", action="store_true")
    sp.set_defaults(func=cmd_down)

    sp = sub.add_parser("ssh", help="ssh into a running cosplay container")
    sp.add_argument("name", nargs="?")
    sp.add_argument("command", nargs="*", help="optional command to run instead of shell")
    sp.set_defaults(func=cmd_ssh)

    sp = sub.add_parser("exec", help="docker exec into a running cosplay container")
    sp.add_argument("name", nargs="?")
    sp.add_argument("command", nargs=argparse.REMAINDER, help="-- cmd args ...")
    sp.set_defaults(func=cmd_dexec)

    sp = sub.add_parser("ps", help="list running sessions")
    sp.set_defaults(func=cmd_ps)

    sp = sub.add_parser(
        "build",
        help="build the cosplay docker image (one-time; `up` calls this automatically on first use)",
    )
    sp.add_argument(
        "--tag", default=apply.IMAGE_TAG, help="image tag to write (default: gpu-cosplay:latest)"
    )
    sp.add_argument("--no-cache", action="store_true", help="ignore docker build cache")
    sp.add_argument(
        "--cuda-tag",
        default=None,
        help=f"override the nvidia/cuda base tag (default: {apply.DEFAULT_CUDA_TAG}). "
        f"Use e.g. '12.4.1-cudnn-devel-ubuntu22.04' for older drivers, "
        f"or '12.6.3-base-ubuntu24.04' for a leaner image without cuDNN.",
    )
    sp.set_defaults(func=cmd_build)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    try:
        return args.func(args)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
