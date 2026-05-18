# AGENTS.md — guide for AI coding agents (Codex, Claude Code, etc.)

> Read this end-to-end before changing anything. It tells you the **mental
> model**, the **invariants**, the **smoke tests** to run after every change,
> and the **failure modes** unique to this project.
>
> Linked from CLAUDE.md → AGENTS.md (symlink). Codex looks for AGENTS.md.

## What this repo is, in 60 seconds

`gpu-cosplay` is a CLI that takes a real NVIDIA GPU (typically H100/H200/A100
on a cluster) and makes it **pretend** to be a smaller consumer GPU (RTX
3090, RTX 4090, GTX 1660 Ti, …) on four axes:

| Axis | How it's done |
|---|---|
| **VRAM** | MIG slice (smallest profile that fits target VRAM) + PyTorch `set_per_process_memory_fraction` for the precise cap |
| **FP32 throughput** | `nvidia-smi --lock-gpu-clocks` scaled to hit target FP32 |
| **Power envelope** | `nvidia-smi -pl` to target TDP |
| **What user-space sees** | Python `.pth` runtime hook patches `torch.cuda.*` and `pynvml.*`; a `/usr/local/bin/nvidia-smi` shim rewrites the real binary's output |

The output is a **Docker container** with SSH, a UID-matched user, your
workspace bind-mounted, and the runtime hook + shim automatically active. To
the user code inside, the GPU **is** an RTX 3090 (or whatever they asked for).

## Code map

```
gpu_cosplay/
  cli.py                       argparse subcommands; ~30 lines per command
  cards.py                     load + alias-resolve data/gpus.yaml
  data/gpus.yaml               ~45 GPUs: spec database (single source of truth)
  host.py                      parse nvidia-smi output (POWER, SUPPORTED_CLOCKS, mig -lgip)
  plan.py                      match target Card to (MigProfile, clock_mhz, power_w, vram_cap)
  apply.py                     execute the Plan: nvidia-smi config + docker run + state tracking + reset
  state.py                     ~/.cache/gpu-cosplay/state.json session tracker
  ssh.py                       locate or generate the host's SSH keypair
  inject.py                    (unused; legacy placeholder, do not import)
docker/
  Dockerfile                   layered on pytorch/pytorch by default
  entrypoint.sh                creates UID-matched user, sets up sshd, bakes env, installs runtime + shim
  gpu_cosplay_runtime.py       the Python .pth-loaded hook (auto-applies VRAM cap + monkey-patches torch/pynvml)
  gpu_cosplay_runtime.pth      one-liner: import gpu_cosplay_runtime; gpu_cosplay_runtime.install()
  nvidia-smi                   shim that wraps /usr/bin/nvidia-smi, rewrites output, preserves table column widths
  gpu_cosplay_verify.py        18-check self-test, invoked by `gpu-cosplay verify`
tests/                         pytest, ~48 tests; all GPU-free (mock fixtures only)
.github/workflows/             ci.yml (lint + tests on py3.9-3.12 + docker build smoke), lint.yml
```

## Where to make a change

| You want to … | Edit |
|---|---|
| Add a new target GPU (e.g. RTX 5070) | `gpu_cosplay/data/gpus.yaml` + a parametrized alias test in `tests/test_cards.py` |
| Teach the planner about a new host GPU family (e.g. B100) | `gpu_cosplay/plan.py::_host_arch`, `_full_sm_count`, `_host_total_bw`, plus per-SM throughput in `_estimate_metrics`; add a `fake_<host>()` fixture in `tests/test_plan.py` |
| Change what gets restored on `down` | `gpu_cosplay/apply.py::down`; the matching tracking goes in `up` and the `Session` dataclass in `gpu_cosplay/state.py` |
| Add a new self-check to `verify` | `docker/gpu_cosplay_verify.py` — add a `@check("...")` decorated function and list it in `_run_all` |
| Add or change a `nvidia-smi` rewrite rule | `docker/nvidia-smi`. Read the docstring at the top first — there are subtle details about column-width preservation and streaming |
| Patch a new `torch.cuda` API to lie | `docker/gpu_cosplay_runtime.py::_patch_torch` |
| Patch a new `pynvml` symbol so nvitop sees the target | `docker/gpu_cosplay_runtime.py::_patch_pynvml` |
| Add a CLI command | `gpu_cosplay/cli.py`: write a `cmd_<name>` function, then register it in `build_parser` |

## Invariants you must not break

1. **No code at import time touches `nvidia-smi`, `docker`, or sudo.** Modules
   must import cleanly on a host without GPU (CI runs there). All hardware
   probes live inside function bodies. Tests rely on this.
2. **Tests are GPU-free.** New tests must use mocked host fixtures
   (`tests/test_plan.py::fake_h200()` etc.) — never call `nvidia-smi`.
3. **`down` always reverts.** Power limit, clock lock, MIG instance, MIG mode,
   persistence mode must all be restored to pre-`up` state. The tracking
   fields exist on `state.Session`:
   - `original_power_limit_w`
   - `original_mig_enabled`
   - `original_persistence_mode`
   - (`clock_mhz` and `mig_profile_name` being non-None tell down what it set)
   Add a new tracking field whenever you add new host mutation in `up`.
4. **`reset` is the fallback.** It must work even when state.json is corrupt,
   missing, or out of sync with reality. It enumerates containers by docker
   label `gpu-cosplay.session=1` and by `name=^cosplay-` and by state.json
   (union), then resets every visible GPU.
5. **Sudo only when necessary.** `nvidia-smi` config commands need sudo; reads
   do not. Docker access auto-falls-back to `sudo -n docker` if the user
   isn't in the docker group (see `_docker()`).
6. **User identity is preserved across the boundary.** Entrypoint creates a
   user with the host's `$USER` / `$UID` / `$GID`. Files in `/workspace` must
   end up identically owned on both sides.
7. **VRAM cap is `torch.cuda.set_per_process_memory_fraction`.** That's the
   only mechanism. It's a PyTorch-caching-allocator-only constraint —
   document its scope, don't claim a hard cap.
8. **The real `/usr/bin/nvidia-smi` is never renamed or replaced.** Our shim
   lives at `/usr/local/bin/nvidia-smi` (PATH precedence). The shim finds the
   real binary via `shutil.which` after dropping its own directory from PATH.
9. **The `.pth` runtime hook must be idempotent and crash-free.** It runs on
   every Python startup inside the container. If a user has no torch / no
   pynvml installed, the hook must be a no-op.
10. **Column widths in the `nvidia-smi` shim.** Inside `|`-delimited tables,
    substitutions must preserve the column width. The current code does this
    by greedy-matching trailing whitespace and re-padding to the original
    match length. If you change the regex, run `gpu-cosplay verify` and
    visually diff `nvidia-smi` output against the host's.

## Workflows you'll run constantly

```bash
# Develop
pip install -e ".[dev]"
ruff check gpu_cosplay tests
ruff format --check gpu_cosplay tests
pytest -v                                # 48 tests, ~1s, no GPU needed

# Build the docker image after editing Dockerfile/entrypoint/shim/runtime
sudo docker build -q -t gpu-cosplay:latest docker/

# End-to-end on a real H100/H200/A100 host (requires GPU + sudo)
gpu-cosplay doctor                       # confirm host is set up
gpu-cosplay plan 3090                    # preview the configuration
gpu-cosplay up 3090 --host-gpu 7         # bring up
gpu-cosplay verify                       # 18-check self-test inside container
gpu-cosplay ssh -- bash -c 'nvidia-smi'  # eyeball it
gpu-cosplay down --all                   # tear down
nvidia-smi --query-gpu=index,persistence_mode,mig.mode.current,clocks.applications.graphics,power.limit --format=csv -i 7
                                         # ^ confirm pristine

# When `down` got stuck (or you panicked):
gpu-cosplay reset --gpu 7 -y --purge-state
```

## Add a new GPU to the catalog

```yaml
# In gpu_cosplay/data/gpus.yaml
  - key: rtx_5070
    pretty: "RTX 5070"
    aliases: ["5070", "rtx5070"]
    arch: blackwell           # add to _host_arch in plan.py if it's also a host
    sm_count: 48
    vram_gb: 12
    fp32_tflops: 30.5         # NVIDIA datasheet, dense
    bf16_tc_tflops: 244.0     # FP16/BF16 with FP32 accumulate, dense (no sparsity)
    bandwidth_gbs: 672
    tdp_w: 250
```

Always source numbers from the relevant NVIDIA architecture whitepaper
(Turing, Ampere, Ada, Hopper, Blackwell) or
[Epoch AI's ML hardware database](https://epoch.ai/data/ml_hardware.csv).
Avoid third-party tables that quietly fold in 2:4 sparsity 2× factors.

Add an alias test:

```python
# tests/test_cards.py — extend the parametrized test
@pytest.mark.parametrize("alias,expected_key", [
    ...,
    ("5070", "rtx_5070"),
    ("rtx5070", "rtx_5070"),
])
```

Verify locally:

```bash
gpu-cosplay info 5070
gpu-cosplay plan 5070 --host-gpu 0
```

## Add a new host GPU family

If a new generation (B100, B200, …) ships:

1. `gpu_cosplay/plan.py::_host_arch`: map a name substring to an arch tag.
2. `_full_sm_count`: full-die SM count for the new GPU.
3. `_host_total_bw`: peak HBM bandwidth in GB/s.
4. `_estimate_metrics`: per-SM FP32 and BF16-TC throughput at 1 GHz for the
   new arch (entry in `fp32_per_sm` / `bf16_per_sm` dicts).
5. Add a `fake_<host>()` fixture and a couple of `test_plan_on_<host>_for_<target>` cases in `tests/test_plan.py`.

## Add a new self-check to `verify`

```python
# docker/gpu_cosplay_verify.py
@check("torch: my new check description")
def _torch_my_check():
    if not _target()["vram_gb"]:
        return ("skip", "no GPU_COSPLAY_VRAM_GB")
    p = subprocess.run(["python3", "-c", "..."], capture_output=True, text=True)
    if "ModuleNotFoundError" in p.stderr:
        return ("skip", "torch not installed")
    if p.returncode != 0:
        return f"rc={p.returncode}"
    # Return None on success; a string on failure; ("skip", reason) to skip.
```

Add the function to the explicit `fns = [...]` list in `_run_all()` so it
shows up in `gpu-cosplay verify`.

## Common pitfalls (these all bit us, learn from them)

- **`_patch_torch` runs too early.** When invoked from the import hook during
  `import torch`, `torch.cuda.device_count()` can return 0 because the CUDA
  context isn't initialised yet. **Fix already in place**: we hook
  `torch.cuda._lazy_init` and re-apply the fraction the moment CUDA is live.
  Don't undo this.
- **`_patch_pynvml` runs mid-load.** `pynvml.py` is 7000+ lines and our hook
  fires recursively during its own internal imports. At that point the
  symbols we want to patch don't exist yet. **Fix in place**: only flip
  `_STATE["pynvml_patched"] = True` when the patches actually landed.
- **`set_per_process_memory_fraction` is computed from the patched
  `mem_get_info`.** If you call it after the patch is installed,
  `phys_total` becomes the cap itself and the fraction collapses to 1.0
  (no limit). **Fix in place**: capture `_orig_mem` before the patch.
- **nvidia-smi table column widths.** Naive `re.sub("NVIDIA H200", "RTX 3090")`
  shifts every column right of the name 3 chars left. The shim eats trailing
  whitespace as part of the match and re-pads to the original match length.
- **Streaming nvidia-smi subcommands** (`dmon`, `pmon`, `-l N`) need
  `subprocess.Popen` + line-by-line pump threads, not `subprocess.run`. The
  shim does this. Don't break it.
- **Docker user not in group.** Many cluster hosts require `sudo` to talk to
  the docker daemon. `_docker()` auto-falls-back; don't hardcode `["docker", ...]`.
- **MIG slice memory queries hit `[Insufficient Permissions]`.** The shim's
  `--query-gpu` rewriter substitutes target values for these specifically.
  See `_query_gpu_rewriter` in `docker/nvidia-smi`.

## Pitfall map: which surface relies on what

```
User runs `python my_script.py` inside container
  └─ python loads .pth files → gpu_cosplay_runtime.install() → hooks __import__
       └─ user imports torch → hook detects "torch.cuda" in sys.modules
            └─ _patch_torch monkey-patches torch.cuda.* + hooks _lazy_init
                 └─ first allocation → _lazy_init runs → fraction set, OOM enforced
       └─ user imports pynvml (e.g. nvitop) → hook detects "pynvml"
            └─ _patch_pynvml monkey-patches nvmlDeviceGetName / GetMemoryInfo

User runs `nvidia-smi` in shell
  └─ PATH lookup finds /usr/local/bin/nvidia-smi (our shim) first
       └─ shim shutil.which's the real /usr/bin/nvidia-smi (excluding self)
            └─ runs real binary, streams output through _rewrite_line per line
                 └─ name regex + phys_mib regex + MIG sub-line strip
                      └─ if --query-gpu was used, also rewrite CSV fields

User runs `gpu-cosplay down <name>` from host
  └─ docker rm -f <container>
  └─ nvidia-smi --reset-gpu-clocks
  └─ nvidia-smi mig -dci && -dgi
  └─ nvidia-smi -mig 0  (only if up() flipped it on)
  └─ nvidia-smi -pl <original>
  └─ nvidia-smi -pm <original>
  └─ state.remove(name)

User runs `gpu-cosplay reset --gpu N`
  └─ docker rm -f every container with label gpu-cosplay.session=1
                   OR name starting with cosplay-
                   OR named in state.json
  └─ For each target GPU: same nvidia-smi reset steps as down, brute-force
```

## CI

`.github/workflows/ci.yml`:
- Linux × py3.9–3.12 matrix: install, lint, run pytest, smoke-test CLI (no GPU).
- `docker-build` job builds the image with `BASE_IMAGE=nvidia/cuda:12.6.3-base-ubuntu24.04`
  (lean, fits on the GHA disk budget) and smoke-tests the entrypoint.

`.github/workflows/lint.yml`:
- Just `ruff check` + `ruff format --check`.

CI runs are short (~3 min). Always wait for them to pass before merging.

## Conventions

- **Python 3.9+** syntax. `from __future__ import annotations` at the top of
  every module so PEP 604 union types and forward references work.
- **`subprocess.run([...], capture_output=True, text=True)`** — never
  `shell=True`. Argv lists everywhere.
- **Paths**: always absolute, via `os.path.abspath` / `os.path.expanduser`.
  Never hardcode `/home/...` or `/opt/conda/...`.
- **User-facing strings**: ASCII-safe, no emoji, no markdown in CLI output.
- **Errors**: `SystemExit` for end-user mistakes (bad GPU name, container
  exists). `RuntimeError` for "command we ran failed unexpectedly".
- **Internal name**: code still uses `Card` / `card_key` / `find_card` for the
  dataclass and lookup function. User-facing strings say "GPU". Don't rename
  the internal type — it'd be a noisy diff and a compatibility break for the
  state.json schema.

## When AI agents should pause and ask

- Adding a new top-level runtime dependency. The project deliberately has one
  (PyYAML). Justify additions.
- Pre-baking heavyweight things (torch wheels of a specific version,
  transformers, datasets) into the image. The image is intentionally lean
  on top of the `pytorch/pytorch` base.
- Changing the planner's match criteria. That's a user-facing behavior change
  — show before/after `gpu-cosplay plan <X>` output for several common
  targets.
- Adding any GHA secret, push-protected workflow, or token-using job.
- Changing the `Session` schema in `state.py` in a way that's not
  forward-compatible with existing state.json files (use defaults on new
  fields; see `test_session_compat.py`).

## Releasing

```bash
# Bump __version__ in gpu_cosplay/__init__.py and pyproject.toml
ruff check . && ruff format --check . && pytest
git tag vX.Y.Z && git push --tags
```

No PyPI publish yet — install via `pip install -e .` from a clone. When
adding one, build wheel + sdist with `python -m build` and use GH Actions
OIDC trusted publishing.

## Quick reference: every CLI subcommand

| Command | Purpose |
|---|---|
| `gpu-cosplay ls [--arch X] [--json]` | List supported target GPUs |
| `gpu-cosplay info <GPU>` | Show specs + aliases for a target |
| `gpu-cosplay doctor` | Check host: GPUs, MIG profiles, docker, sudo |
| `gpu-cosplay plan <GPU> [--host-gpu N]` | Preview what `up` would do |
| `gpu-cosplay up <GPU> [opts]` | Bring up a cosplay container |
| `gpu-cosplay ssh [name] [CMD...]` | Shell into the container (sshd or docker exec) |
| `gpu-cosplay exec [name] -- CMD...` | One-shot `docker exec` |
| `gpu-cosplay verify [name] [--json]` | Run the 18-check self-test inside container |
| `gpu-cosplay ps` | List running sessions |
| `gpu-cosplay down <name> \| --all` | Tear down + restore GPU state |
| `gpu-cosplay reset [--gpu N] [--purge-state] [-y]` | Force-reset host GPUs (fallback when `down` fails) |
| `gpu-cosplay build [--base IMG \| --cuda-tag TAG] [--no-cache]` | Build the cosplay docker image |
