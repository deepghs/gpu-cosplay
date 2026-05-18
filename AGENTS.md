# AGENTS.md — engineering notes for AI coding agents

This file is the contract between human maintainers and AI agents working in
this repository. Read it before making changes.

## What gpu-cosplay is, in one paragraph

A CLI that combines NVIDIA MIG slicing, `nvidia-smi --lock-gpu-clocks`,
`nvidia-smi -pl`, and PyTorch's `set_per_process_memory_fraction` to make one
GPU behave roughly like another. It launches a Docker container with the
selected slice, sshd, and a UID-matched user. State for each running "cosplay"
is persisted to `~/.cache/gpu-cosplay/state.json`.

## Code layout

```
gpu_cosplay/
  cli.py          argparse subcommands
  cards.py        load+resolve cards.yaml
  data/cards.yaml ~50 cards from GTX 1650 to H200
  host.py        parse nvidia-smi output (POWER, SUPPORTED_CLOCKS, mig -lgip)
  plan.py        match target Card to (MigProfile, clock_mhz, power_w, vram_gb)
  apply.py       apply Plan: nvidia-smi config + docker run
  state.py       JSON-on-disk session tracker
  ssh.py         locate/generate host SSH keypair
  inject.py      in-host module mirroring docker/gpu_cosplay_inject.py
docker/
  Dockerfile                CUDA + sshd + tini base
  entrypoint.sh             user creation, key install, env baking
  gpu_cosplay_inject.py     imported by user code to apply VRAM cap
  gpu_cosplay_apply.py      wrapper CLI inside the container
tests/
  test_cards.py             alias resolution, schema sanity
  test_host_parsing.py      regex on canned nvidia-smi text
  test_plan.py              synthetic HostGPU fixtures for many targets
  test_state.py             persistence round-trip
.github/workflows/
  ci.yml                    pytest + ruff + image build, ubuntu+macos matrix
  lint.yml                  ruff check + ruff format --check
```

## Invariants you must not break

1. **No code at import time touches `nvidia-smi`, `docker`, or sudo.**
   Imports must succeed on a laptop without a GPU. All hardware probes live
   inside function bodies.
2. **Tests are GPU-free.** Anything that reads `nvidia-smi` output is fed via a
   captured-text fixture. New tests must follow this rule so CI stays green.
3. **`down` always reverts.** Power limit and clock lock are restored, MIG
   instance is destroyed, MIG mode is disabled iff this session enabled it.
   `state.Session.original_*` fields exist exactly to support this — keep
   them populated.
4. **Sudo only when necessary.** `nvidia-smi` config commands need sudo; reads
   do not. Docker uses `_docker()` which falls back to `sudo -n docker` when
   the user isn't in the docker group.
5. **User identity is preserved across the boundary.** Entrypoint creates a
   user with the host's `$USER` / `$UID` / `$GID`. Files in `/workspace`
   must end up identically owned on both sides.
6. **VRAM cap is enforced by `torch.cuda.set_per_process_memory_fraction`.**
   That's the only mechanism. Document its scope (PyTorch caching allocator
   only) prominently — don't claim a hard cap.

## Adding a new card

1. Edit `gpu_cosplay/data/cards.yaml`. Required fields: `key, pretty,
   aliases, arch, sm_count, vram_gb, fp32_tflops, bf16_tc_tflops (or null),
   bandwidth_gbs, tdp_w`.
2. Use NVIDIA's official architecture whitepaper for `fp32_tflops` and
   `bf16_tc_tflops` (dense, FP32-accumulate). Avoid third-party numbers
   that quietly fold in 2:4 sparsity 2× factors.
3. Add an alias test in `tests/test_cards.py`.
4. Pass `gpu-cosplay info <new_card>` locally; the output is the user-facing
   contract.

## Adding a new host GPU class

1. Extend `_full_sm_count()` and `_host_total_bw()` in `plan.py`. Use the
   official die-level numbers, not the maximum-SKU number.
2. Extend `_host_arch()` to map the new GPU's `name` substring to one of
   `volta | turing | ampere | ada | hopper`.
3. Update `bf16_per_sm` and `fp32_per_sm` tables if the new arch has
   different per-SM throughput.
4. Add a fixture in `tests/test_plan.py` and a few `test_plan_on_<host>_*`
   cases.

## When the planner suggests something weird

The planner is intentionally simple — see `gpu_cosplay/plan.py`. It picks the
smallest MIG profile that fits target VRAM, then linearly scales the clock to
match FP32. If you find a case where this clearly produces the wrong answer
(off by >30% on the dominant axis), the right fix is usually:

- Adjusting `_pick_mig_profile` tie-break (currently: smaller memory first,
  then closer SM count).
- Adding a special case (e.g. for cards where memory bandwidth, not FP32, is
  the headline number — like A100 vs L40).

Don't try to make the planner extensible until there are >3 concrete special
cases. Premature abstraction here will obscure the math.

## Conventions

- Python 3.9+ syntax. Use `from __future__ import annotations` so PEP 604
  union types work.
- `subprocess.run([...], capture_output=True, text=True)` — never
  `shell=True`. Use list-form argv everywhere.
- Path handling: always absolute, use `os.path.abspath`/`os.path.expanduser`.
  Never hardcode `/home/...`.
- User-facing strings: no emoji; ASCII-safe; no markdown in CLI output.
- Errors: raise `SystemExit` for end-user mistakes (bad card name, container
  exists), `RuntimeError` for "command we ran failed unexpectedly".

## Releasing

```bash
# Bump __version__ in gpu_cosplay/__init__.py and pyproject.toml
pytest && ruff check . && ruff format --check .
git tag vX.Y.Z && git push --tags
```

There is no PyPI publish step yet. (Most users `pip install -e .` from a
clone.) When adding one, build wheel + sdist with `python -m build` and use
GH Actions OIDC trusted publishing.

## When AI agents should ask first

- Adding new dependencies (the project deliberately has one runtime dep, PyYAML).
- Pre-baking heavy stuff (torch, transformers) into the docker image — the
  image is intentionally lean.
- Changing the matcher's selection criteria — that's a user-facing behaviour
  change.
- Adding a GHA secret or any push-protected workflow.
