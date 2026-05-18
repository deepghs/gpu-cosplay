# gpu-cosplay

[![CI](https://github.com/deepghs/gpu-cosplay/actions/workflows/ci.yml/badge.svg)](https://github.com/deepghs/gpu-cosplay/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

> Make your beefy datacenter GPU pretend to be a smaller consumer GPU.

`gpu-cosplay` is a tool to **simulate the resource envelope of one NVIDIA GPU on
another**. Have an H200 but need to know if your model fits on an RTX 3090? Got
an A100 cluster and want to develop against a 2060? `gpu-cosplay` carves out a
slice of your real GPU and pretends to be the target GPU you ask for — matching
VRAM, FP32 throughput, BF16 Tensor Core throughput, and memory bandwidth as
closely as the underlying hardware allows.

## What it does

Given a target GPU name like `3090`, `4090`, `2060`, or `a100`, it:

1. Picks a **MIG profile** with appropriate SM count and memory (on Ampere/Hopper data-center hosts).
2. **Locks the GPU clock** to scale FP32 throughput to the target.
3. **Limits the power cap** to the target's TDP envelope.
4. Launches a **Docker container** with:
   - SSH server, your host SSH pubkey installed (passwordless login).
   - A user with the **same UID/GID/name** as the host user.
   - The selected MIG slice as the only visible GPU.
   - Your working dir mounted at `/workspace`.
   - `gpu_cosplay_inject` module that auto-applies the VRAM cap via
     `torch.cuda.set_per_process_memory_fraction` when imported.

When you're done, one command tears it all down: removes the container, destroys
the MIG instance, restores clocks and power. The host GPU is left exactly as it
was found.

## Why "cosplay"?

Because we are not emulating the target GPU. We are dressing the H200 up in a
costume. The bones underneath are still Hopper. See [Honest limits](#honest-limits)
below.

## Quick start

```bash
# 1. Prerequisites (Linux + NVIDIA driver + Docker + nvidia-container-toolkit + sudo for nvidia-smi).
gpu-cosplay doctor

# 2. List the target GPUs you can cosplay as.
gpu-cosplay ls

# 3. Preview what would happen.
gpu-cosplay plan 3090

# 4. Bring up the cosplay.
gpu-cosplay up 3090 --workspace ~/my-experiment

# 5. SSH in (the command line is printed for you).
gpu-cosplay ssh

# 6. Inside the container, your code Just Works:
python -c "import gpu_cosplay_inject; import torch; print(torch.cuda.get_device_name(0))"

# 7. Tear it down.
gpu-cosplay down --all
```

## Install

```bash
git clone https://github.com/deepghs/gpu-cosplay
cd gpu-cosplay
pip install -e .
gpu-cosplay build       # build the docker image (one-time, ~5 GB on disk)
```

The `build` step is optional — the first `gpu-cosplay up` will auto-build the
image if it's missing. You only need to call `build` explicitly when you want
to **rebuild from scratch**, **use a different CUDA base**, or **prepare a
machine for offline use** in advance.

Requirements:
- **Linux host** (Ubuntu/Debian/RHEL/etc.).
- **NVIDIA driver** ≥ R515 (for MIG on H100/H200, R535+).
- **CUDA-capable host GPU**. For full functionality (MIG), an Ampere or Hopper
  data-center GPU: A100, A30, H100, H200, GH200. Other GPUs (consumer, L40,
  V100) still work via clock-lock + VRAM cap only, without SM slicing.
- **Docker** with the [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **passwordless `sudo`** to invoke `nvidia-smi mig` / `nvidia-smi -pl` etc.
- **Python ≥ 3.9** on the host.

Run `gpu-cosplay doctor` to verify each requirement.

## The cosplay container

When you run `gpu-cosplay up <GPU>`, your code runs inside a Docker image built
from [`docker/Dockerfile`](docker/Dockerfile). It is **not** a pre-baked DL
environment — it is a thin, configurable shell around the official NVIDIA CUDA
image. You decide what ML stack goes inside.

**Base image.** `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04` (Ubuntu 24.04,
CUDA 12.6.3 + cuDNN + headers, ~5 GB compressed). Chosen because:
- It runs against NVIDIA driver ≥ R520 (covers any host that supports H100/H200
  MIG, plus all current A100/A30/L40 systems).
- `cudnn-devel` ships headers + libraries so `pip install torch` finds cuDNN
  symbols and JIT-compiled kernels link cleanly.

Override with `--cuda-tag` if you need something else:

```bash
# Older driver: use CUDA 12.4 + Ubuntu 22.04
gpu-cosplay build --cuda-tag 12.4.1-cudnn-devel-ubuntu22.04

# Lean image without cuDNN (~3 GB smaller)
gpu-cosplay build --cuda-tag 12.6.3-base-ubuntu24.04

# CUDA 11.8 (for older PyTorch wheels)
gpu-cosplay build --cuda-tag 11.8.0-cudnn8-devel-ubuntu22.04
```

Any tag from [hub.docker.com/r/nvidia/cuda](https://hub.docker.com/r/nvidia/cuda/tags) works.

**What's pre-installed.** Apart from CUDA/cuDNN from the base, just utilities:
`openssh-server`, `sudo`, `tini`, `python3` + `pip` + `venv`, `build-essential`,
`pkg-config`, `git`, `curl`/`wget`, `vim`, `tmux`, `htop`, `less`. No PyTorch,
no transformers, no diffusers — install what you need via `pip` inside.

**What runs at start.** The entrypoint (`docker/entrypoint.sh`) is the
interesting part — for each container:

1. Creates a user with the host's `$USER` / `$UID` / `$GID` so file ownership
   on the bind-mounted workspace stays consistent on both sides.
2. Installs the host's SSH public key into that user's `authorized_keys`.
3. Bakes the cosplay env vars into `/etc/environment` and a
   `/etc/profile.d/gpu-cosplay.sh` so they're visible to both login and
   non-login shells (including `ssh user@host CMD`).
4. Sets `PIP_BREAK_SYSTEM_PACKAGES=1` so `pip install` works against the
   system Python (this is a disposable, single-purpose container — PEP 668
   protections aren't useful here).
5. Generates sshd host keys if missing and launches sshd in foreground.

Once `up` returns, you can:

```bash
gpu-cosplay ssh                    # interactive shell as your user
gpu-cosplay ssh -- python train.py # one-off command
gpu-cosplay exec NAME -- bash      # docker exec, bypassing sshd
```

**Image lifecycle.** The image is named `gpu-cosplay:latest` and lives only on
your host's local Docker daemon (we don't push to any registry). Rebuild it
with `gpu-cosplay build --no-cache` after editing the Dockerfile; multiple
versions can coexist via `--tag` and selected per-`up` with `--image`.

## Bring your own image

You probably already have a PyTorch/training image with all your wheels and
datasets pre-baked. `gpu-cosplay` gives you **three ways** to use it:

### Option 1: layer cosplay on top of your image (recommended)

```bash
gpu-cosplay build --base my-org/pytorch:v3 --tag my-cosplay-pt
gpu-cosplay up 3090 --image my-cosplay-pt
```

Your image keeps everything it had; we just add sshd, sudo, and the cosplay
entrypoint on top — usually ~200 MB of layers. Requires the base to be
Ubuntu/Debian-derived (we use apt).

### Option 2: use your image directly, no rebuild

```bash
gpu-cosplay up 3090 --image my-org/pytorch:v3
```

When `--image` is anything other than `gpu-cosplay:latest`, we run in
**BYO mode**:
- The entrypoint and inject helper are bind-mounted into the container at
  `/opt/gpu-cosplay/`.
- User creation, env baking, and workspace ownership still work.
- If your image has `sshd`, full SSH access is wired up.
- If your image doesn't have sshd (most DL images don't), `gpu-cosplay ssh`
  transparently falls back to `docker exec`. Everything else is unchanged.

This is the lowest-friction path when your image already has the runtime you
want. You don't need to rebuild anything.

### Option 3: a custom base CUDA tag

If you just want a different CUDA version under the default cosplay layers:

```bash
gpu-cosplay build --cuda-tag 12.4.1-cudnn-devel-ubuntu22.04
gpu-cosplay build --cuda-tag 12.6.3-base-ubuntu24.04   # lean, no cuDNN
gpu-cosplay build --cuda-tag 11.8.0-cudnn8-devel-ubuntu22.04
```

This is a shortcut for `--base nvidia/cuda:<TAG>`. Any tag from
[hub.docker.com/r/nvidia/cuda](https://hub.docker.com/r/nvidia/cuda/tags) works.

## Supported GPUs

`gpu-cosplay ls` shows the full catalog. Highlights:

| Family       | GPUs                                                              |
|--------------|--------------------------------------------------------------------|
| GTX 16-series | 1650, 1660, 1660 Ti, 1660 Super                                   |
| RTX 20-series | 2060 (6G/12G), 2070, 2070 Super, 2080, 2080 Super, 2080 Ti        |
| RTX 30-series | 3060, 3060 Ti, 3070, 3070 Ti, 3080 (10G/12G), 3080 Ti, 3090, 3090 Ti |
| RTX 40-series | 4060, 4060 Ti (8G/16G), 4070 / Super / Ti / Ti Super, 4080 / Super, 4090 |
| Datacenter   | A10, A30, A40, A100 (40G/80G), L4, L40, L40S, V100 (16G/32G), T4, H100 (PCIe/SXM/NVL), H200 |

Aliases are forgiving: `3090`, `rtx3090`, `RTX_3090`, `rtx-3090`, `RTX 3090` all
resolve to the same GPU.

## How it picks a configuration

Given a host GPU (e.g. H200) and a target GPU (e.g. RTX 3090):

1. **MIG profile selection**: choose the smallest profile whose memory ≥ target
   VRAM. For RTX 3090 (24 GB) on H200, that's `2g.35gb` (35 GB).
2. **Clock lock**: scale boost clock down so per-SM × clock × 2 ≈ target FP32.
   For 3090 on H200 with `2g.35gb` (32 SMs × Hopper FP32-per-SM), FP32 at full
   clock already overshoots the target, so the clock stays unlocked.
3. **Power cap**: set to the target's TDP (within the host's min/max range).
4. **VRAM cap**: enforced in-container via PyTorch's
   `set_per_process_memory_fraction` so allocations beyond target VRAM OOM.

Run `gpu-cosplay plan <GPU>` to see exactly what it would do — including
warnings about any failure to match.

Example:
```
$ gpu-cosplay plan a100
Cosplay plan: A100 (40GB) on GPU 0 (NVIDIA H200)
  MIG profile:  4g.71gb
  Clock lock:   585 MHz
  Power limit:  400 W
  VRAM cap:     40.0 GB (enforced via PyTorch memory fraction)
  Expected FP32: 19.2 TFLOPS (target 19.5)
  Expected BF16 TC: 143.8 TFLOPS (target 312.0)
  Expected BW:   1135 GB/s (target 1555.0)
```

## Container interface

When you `gpu-cosplay up <GPU>`, the container is set up so that:

- **User identity**: an in-container user with the same `$USER`, `uid`, and
  `gid` as the host runs sshd and owns `/workspace`. Files you create in
  `/workspace` are owned identically on host and inside.
- **Workspace**: host `$PWD` (or `--workspace DIR`) is mounted at `/workspace`.
  Extra mounts: `--volume HOST:CONTAINER` (repeatable).
- **SSH**: a free host port is mapped to `:22`. Login uses your host SSH key,
  no password. `gpu-cosplay ssh [name]` wraps the connection.
- **Env vars** baked into `/etc/environment` so they're visible to non-login
  shells too:
  - `GPU_COSPLAY_CARD`, `GPU_COSPLAY_PRETTY`
  - `GPU_COSPLAY_VRAM_GB`, `GPU_COSPLAY_FP32_TFLOPS`, `GPU_COSPLAY_BF16_TC_TFLOPS`, `GPU_COSPLAY_BW_GBS`
  - `PIP_BREAK_SYSTEM_PACKAGES=1` (the container is single-purpose, pip-installing is safe)

### Enforcing the VRAM cap in your code

Two options:

**Option A: import the helper module at the top of your script.**

```python
import gpu_cosplay_inject  # this single line caps VRAM
import torch
# ...
```

`gpu_cosplay_inject` is pre-installed in the image and calls
`torch.cuda.set_per_process_memory_fraction(target_gb / device_total, i)` for
every visible CUDA device.

**Option B: use the wrapper CLI.**

```bash
gpu-cosplay-apply -- python train.py
```

Equivalent to option A but for code you can't (or don't want to) modify.

## Examples

### Pretend to be an RTX 3090 on an H200

```bash
gpu-cosplay up 3090 --workspace ~/proj
gpu-cosplay ssh -- python train.py
gpu-cosplay down
```

### Run multiple cosplays in parallel

H200 with 7× 1g.18gb slices: simulate up to 7 RTX 2060s on one physical GPU.

```bash
for i in 1 2 3 4; do
  gpu-cosplay up 2060 --gpu 0 --name worker-$i &
done
wait
gpu-cosplay ps
```

### One-shot exec (no shell)

```bash
gpu-cosplay up 4060 --name oneshot
gpu-cosplay exec oneshot -- nvidia-smi -L
gpu-cosplay down oneshot
```

### Use a specific host GPU index

```bash
gpu-cosplay up a100 --gpu 7
```

## Honest limits

`gpu-cosplay` is a *useful approximation*, not a faithful emulator. The host's
silicon shows through in several ways:

1. **Architectural features cannot be removed.** If the host is Hopper, FP8
   Tensor Core, TMA, async copy, and the Hopper L2 are all *present*. The
   tool can throttle the host's throughput, but it cannot make the GPU lack a
   feature. To match a GPU without BF16 TC (GTX 16-series, V100), you must
   disable mixed-precision in your framework.
2. **BF16 Tensor Cores are stronger per-SM on the host.** A Hopper SM does
   ~6 BF16 TFLOPS·GHz⁻¹ vs ~1 on Ampere and ~2 on Ada. Matching SM count and
   clock still leaves BF16 TC throughput 2–3× too strong. The planner
   prints a warning when this happens.
3. **HBM ≠ GDDR.** MIG slices still see HBM3e latency, even when bandwidth
   is throttled to match a GDDR6X-class number. Random-access workloads
   on the host are unrealistically fast.
4. **`set_per_process_memory_fraction` is PyTorch-only.** Bare `cudaMalloc`
   from custom CUDA code or other frameworks bypasses it. For most
   PyTorch / HF Transformers / vLLM / Diffusers workloads it's sufficient.
5. **MIG slices have no P2P.** Multi-GPU NCCL training on MIG instances
   doesn't work; for that, use whole GPUs without `--mig`.
6. **CUDA 11+ limits one MIG per process.** That means parallel multi-GPU
   training within a single Python process needs whole GPUs.

Use this tool to **estimate fit and relative performance**. Don't quote its
wall-clock numbers as "RTX 3090 performance" in a paper without a disclaimer.

## CLI reference

```
gpu-cosplay ls [--arch ARCH] [--json]            list supported target GPUs
gpu-cosplay info GPU                             show specs for a target GPU
gpu-cosplay doctor                               check host environment
gpu-cosplay plan GPU [--host-gpu N]              show what would be applied
gpu-cosplay up GPU [opts...]                     bring up a cosplay container
gpu-cosplay ssh [NAME] [CMD...]                  shell into a session (sshd or docker exec)
gpu-cosplay exec [NAME] -- CMD...                docker exec into a session
gpu-cosplay ps                                   list running sessions
gpu-cosplay down NAME | --all                    tear down and revert GPU state
gpu-cosplay build [--tag T] [--no-cache]         (re)build the docker image
                  [--base IMAGE | --cuda-tag T]
```

`up` options:
- `--name NAME` — container/session name (default: auto)
- `--host-gpu N` — host GPU index (default: pick a MIG-capable one). `--gpu N` accepted as alias.
- `--ssh-port PORT` — host port mapped to `:22` (default: random free)
- `--volume HOST:CONTAINER` — extra volume mount (repeatable)
- `--env KEY=VALUE` — extra container env (repeatable)
- `--workspace DIR` — host dir mounted at `/workspace` (default: `$PWD`)
- `--image IMAGE` — custom docker image. Any image works; when it's not our
  default cosplay image, we bind-mount the entrypoint and inject helper in,
  and fall back to `docker exec` if it has no sshd.

## Development

```bash
git clone https://github.com/deepghs/gpu-cosplay
cd gpu-cosplay
pip install -e ".[dev]"
pytest              # unit tests (no GPU required)
ruff check .
ruff format .
```

Tests run on Linux and macOS (GitHub Actions matrix). The GPU-touching code
paths are tested via mock host fixtures so the suite is fully hardware-free.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Acknowledgements

Built by [DeepGHS](https://github.com/deepghs). The GPU spec database draws
on NVIDIA's official architecture whitepapers (Turing, Ampere, Ada Lovelace,
Hopper) and the [Epoch AI ML hardware database](https://epoch.ai/data/ml_hardware.csv).
