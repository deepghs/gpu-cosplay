"""Card database: load cards.yaml and resolve aliases."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

try:
    import yaml
except ImportError as e:
    raise SystemExit("PyYAML is required: pip install pyyaml (or pip install gpu-cosplay)") from e


@dataclass
class Card:
    key: str
    pretty: str
    aliases: list[str]
    arch: str
    sm_count: int
    vram_gb: float
    fp32_tflops: float
    bf16_tc_tflops: Optional[float]
    bandwidth_gbs: float
    tdp_w: int
    notes: str = ""


_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "cards.yaml")


def _normalize(name: str) -> str:
    """Lowercase, strip non-alnum so '3090 Ti' and 'rtx-3090-ti' match."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load_cards(path: str = _DATA_PATH) -> list[Card]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    out: list[Card] = []
    for c in raw["cards"]:
        out.append(
            Card(
                key=c["key"],
                pretty=c["pretty"],
                aliases=c.get("aliases", []),
                arch=c["arch"],
                sm_count=c["sm_count"],
                vram_gb=float(c["vram_gb"]),
                fp32_tflops=float(c["fp32_tflops"]),
                bf16_tc_tflops=(
                    float(c["bf16_tc_tflops"]) if c.get("bf16_tc_tflops") is not None else None
                ),
                bandwidth_gbs=float(c["bandwidth_gbs"]),
                tdp_w=int(c["tdp_w"]),
                notes=c.get("notes", ""),
            )
        )
    return out


def find_card(name: str, cards: Optional[list[Card]] = None) -> Card:
    """Resolve a human name to a Card. Raises KeyError on miss."""
    cards = cards or load_cards()
    norm = _normalize(name)
    for c in cards:
        if _normalize(c.key) == norm or _normalize(c.pretty) == norm:
            return c
        for a in c.aliases:
            if _normalize(a) == norm:
                return c
    raise KeyError(f"unknown card: {name!r}. Try `gpu-cosplay ls` for the catalog.")
