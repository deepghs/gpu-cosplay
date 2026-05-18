"""Session field additions must remain backward-compatible with older state.json."""

import json
import os
import tempfile

import gpu_cosplay.state as state


def test_legacy_session_loads_without_new_fields(monkeypatch):
    # Simulate a state.json written by v0.1.0 (before image/ssh_available fields).
    legacy = {
        "foo": {
            "name": "foo",
            "card_key": "rtx_3090",
            "container_id": "abc",
            "container_name": "foo",
            "gpu_index": 0,
            "mig_profile_name": "2g.35gb",
            "mig_uuid": "MIG-xxx",
            "mig_gi_id": 1,
            "mig_ci_id": 0,
            "clock_mhz": 1380,
            "power_limit_w": 350,
            "ssh_port": 2222,
            "vram_cap_gb": 24.0,
            "workspace_mount": "/tmp/ws",
            "original_power_limit_w": 700,
            "original_mig_enabled": False,
            "extra": {},
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(state, "_STATE_DIR", tmp)
        monkeypatch.setattr(state, "_STATE_FILE", os.path.join(tmp, "state.json"))
        with open(state._STATE_FILE, "w") as f:
            json.dump(legacy, f)
        s = state.get("foo")
        assert s is not None
        # New fields take their defaults.
        assert s.image == "gpu-cosplay:latest"
        assert s.ssh_available is True
