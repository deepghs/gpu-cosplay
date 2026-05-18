# gpu-cosplay

[![CI](https://github.com/deepghs/gpu-cosplay/actions/workflows/ci.yml/badge.svg)](https://github.com/deepghs/gpu-cosplay/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

> Make your beefy datacenter GPU pretend to be a smaller consumer card.

`gpu-cosplay` is a tool to **simulate the resource envelope of one NVIDIA GPU on
another**. Have an H200 but need to know if your model fits on an RTX 3090? Got
an A100 cluster and want to develop against a 2060? `gpu-cosplay` carves out a
slice of your real GPU and pretends to be the card you ask for — matching VRAM
capacity, FP32 throughput, BF16 Tensor Core throughput, and memory bandwidth as
closely as the underlying hardware allows.

## What it does

Given a target card name like `3090`, `4090`, `2060`, or `a100`, it:

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

Because we are not emulating the card. We are dressing the H200 up in a costume.
The bones underneath are still Hopper. See [Honest limits](#honest-limits) below.

## Quick start

```bash
# 1. Prerequisites (Linux + NVIDIA driver + Docker + nvidia-container-toolkit + sudo for nvidia-smi).
gpu-cosplay doctor

# 2. List the cards you can cosplay as.
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
gpu-cosplay build       # builds the docker image (one-time, ~5 GB)
```

Requirements:
- **Linux host** (Ubuntu/Debian/RHEL/etc.).
- **NVIDIA driver** ≥ R515 (for MIG on H100/H200, R535+).
- **CUDA-capable host GPU**. For full functionality (MIG), an Ampere or Hopper
  data-center card: A100, A30, H100, H200, GH200. Other cards (consumer, L40,
  V100) still work via clock-lock + VRAM cap only, without SM slicing.
- **Docker** with the [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **passwordless `sudo`** to invoke `nvidia-smi mig` / `nvidia-smi -pl` etc.
- **Python ≥ 3.9**.

Run `gpu-cosplay doctor` to verify each requirement.

## Supported cards

`gpu-cosplay ls` shows the full catalog. Highlights:

| Family       | Cards                                                              |
|--------------|--------------------------------------------------------------------|
| GTX 16-series | 1650, 1660, 1660 Ti, 1660 Super                                   |
| RTX 20-series | 2060 (6G/12G), 2070, 2070 Super, 2080, 2080 Super, 2080 Ti        |
| RTX 30-series | 3060, 3060 Ti, 3070, 3070 Ti, 3080 (10G/12G), 3080 Ti, 3090, 3090 Ti |
| RTX 40-series | 4060, 4060 Ti (8G/16G), 4070 / Super / Ti / Ti Super, 4080 / Super, 4090 |
| Datacenter   | A10, A30, A40, A100 (40G/80G), L4, L40, L40S, V100 (16G/32G), T4, H100 (PCIe/SXM/NVL), H200 |

Aliases are forgiving: `3090`, `rtx3090`, `RTX_3090`, `rtx-3090`, `RTX 3090` all
resolve to the same card.

## How it picks a configuration

Given a host GPU (e.g. H200) and a target card (e.g. RTX 3090):

1. **MIG profile selection**: choose the smallest profile whose memory ≥ target
   VRAM. For RTX 3090 (24 GB) on H200, that's `2g.35gb` (35 GB).
2. **Clock lock**: scale boost clock down so per-SM × clock × 2 ≈ target FP32.
   For 3090 on H200 with `2g.35gb` (32 SMs × Hopper FP32-per-SM), FP32 at full
   clock already overshoots the target, so the clock stays unlocked.
3. **Power cap**: set to the target's TDP (within the host's min/max range).
4. **VRAM cap**: enforced in-container via PyTorch's
   `set_per_process_memory_fraction` so allocations beyond target VRAM OOM.

Run `gpu-cosplay plan <card>` to see exactly what it would do — including
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

When you `gpu-cosplay up <card>`, the container is set up so that:

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

H200 with 7× 1g.18gb slices: simulate up to 7 RTX 2060s on one card.

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
   feature. To match a card without BF16 TC (GTX 16-series, V100), you must
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
5. **MIG slices have no P2P.** Multi-card NCCL training on MIG instances
   doesn't work; for that, use whole GPUs without `--mig`.
6. **CUDA 11+ limits one MIG per process.** That means parallel multi-GPU
   training within a single Python process needs whole GPUs.

Use this tool to **estimate fit and relative performance**. Don't quote its
wall-clock numbers as "RTX 3090 performance" in a paper without a disclaimer.

## CLI reference

```
gpu-cosplay ls [--arch ARCH] [--json]    list supported cards
gpu-cosplay info CARD                    show card specs and aliases
gpu-cosplay doctor                       check host environment
gpu-cosplay plan CARD [--gpu N]          show what would be applied
gpu-cosplay up CARD [opts...]            bring up a cosplay container
gpu-cosplay ssh [NAME] [CMD...]          ssh into a session
gpu-cosplay exec [NAME] -- CMD...        docker exec into a session
gpu-cosplay ps                           list running sessions
gpu-cosplay down NAME | --all            tear down and revert GPU state
gpu-cosplay build [--tag T] [--no-cache] (re)build the docker image
```

`up` options:
- `--name NAME` — container/session name (default: auto)
- `--gpu N` — host GPU index (default: pick a MIG-capable one)
- `--ssh-port PORT` — host port mapped to `:22` (default: random free)
- `--volume HOST:CONTAINER` — extra volume mount (repeatable)
- `--env KEY=VALUE` — extra container env (repeatable)
- `--workspace DIR` — host dir mounted at `/workspace` (default: `$PWD`)
- `--image IMAGE` — custom docker image tag

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

Built by [DeepGHS](https://github.com/deepghs). The card spec database draws
on NVIDIA's official architecture whitepapers (Turing, Ampere, Ada Lovelace,
Hopper) and the [Epoch AI ML hardware database](https://epoch.ai/data/ml_hardware.csv).
