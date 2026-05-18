"""Track running cosplay sessions across CLI invocations."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

_STATE_DIR = os.path.expanduser("~/.cache/gpu-cosplay")
_STATE_FILE = os.path.join(_STATE_DIR, "state.json")


@dataclass
class Session:
    name: str
    card_key: str
    container_id: str
    container_name: str
    gpu_index: int
    mig_profile_name: Optional[str]  # e.g. "1g.18gb"
    mig_uuid: Optional[str]
    mig_gi_id: Optional[int]
    mig_ci_id: Optional[int]
    clock_mhz: Optional[int]
    power_limit_w: Optional[int]
    ssh_port: int
    vram_cap_gb: float
    workspace_mount: str
    original_power_limit_w: Optional[int] = None
    original_mig_enabled: bool = False
    extra: dict = field(default_factory=dict)


def _load() -> dict[str, Session]:
    if not os.path.exists(_STATE_FILE):
        return {}
    with open(_STATE_FILE) as f:
        raw = json.load(f)
    return {k: Session(**v) for k, v in raw.items()}


def _save(d: dict[str, Session]) -> None:
    os.makedirs(_STATE_DIR, exist_ok=True)
    tmp = _STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({k: asdict(v) for k, v in d.items()}, f, indent=2)
    os.replace(tmp, _STATE_FILE)


def add(session: Session) -> None:
    d = _load()
    d[session.name] = session
    _save(d)


def remove(name: str) -> Optional[Session]:
    d = _load()
    s = d.pop(name, None)
    _save(d)
    return s


def get(name: str) -> Optional[Session]:
    return _load().get(name)


def all_sessions() -> list[Session]:
    return list(_load().values())
