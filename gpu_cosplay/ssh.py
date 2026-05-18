"""Locate or generate a usable SSH keypair for cosplay containers."""

from __future__ import annotations

import os
import subprocess

KEY_DIR = os.path.expanduser("~/.cache/gpu-cosplay/keys")
PRIV = os.path.join(KEY_DIR, "id_ed25519")
PUB = PRIV + ".pub"


def public_key() -> str:
    """Return public key contents. Prefers the user's existing key, else creates one."""
    candidates = [
        os.path.expanduser("~/.ssh/id_ed25519.pub"),
        os.path.expanduser("~/.ssh/id_rsa.pub"),
        os.path.expanduser("~/.ssh/id_ecdsa.pub"),
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c) as f:
                return f.read().strip()
    # No host key: generate a dedicated one under ~/.cache/gpu-cosplay/keys
    if not os.path.exists(PUB):
        os.makedirs(KEY_DIR, exist_ok=True)
        os.chmod(KEY_DIR, 0o700)
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", PRIV, "-q", "-C", "gpu-cosplay"],
            check=True,
        )
    with open(PUB) as f:
        return f.read().strip()


def private_key_path() -> str:
    """Path to the private key that pairs with public_key().

    If we generated one, that's the cache path; otherwise we point at the
    user's existing identity (the first one we used).
    """
    candidates = [
        os.path.expanduser("~/.ssh/id_ed25519"),
        os.path.expanduser("~/.ssh/id_rsa"),
        os.path.expanduser("~/.ssh/id_ecdsa"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return PRIV
