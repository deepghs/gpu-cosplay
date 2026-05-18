"""Persistence state tests."""

import os
import tempfile

import gpu_cosplay.state as state


def test_state_roundtrip(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(state, "_STATE_DIR", tmp)
        monkeypatch.setattr(state, "_STATE_FILE", os.path.join(tmp, "state.json"))
        s = state.Session(
            name="foo",
            card_key="rtx_3090",
            container_id="abc",
            container_name="foo",
            gpu_index=0,
            mig_profile_name="2g.35gb",
            mig_uuid="MIG-xxx",
            mig_gi_id=1,
            mig_ci_id=0,
            clock_mhz=1380,
            power_limit_w=350,
            ssh_port=2222,
            vram_cap_gb=24.0,
            workspace_mount="/tmp/ws",
            original_power_limit_w=700,
            original_mig_enabled=False,
        )
        state.add(s)
        assert state.get("foo") is not None
        assert state.get("foo").container_id == "abc"
        all_s = state.all_sessions()
        assert len(all_s) == 1
        removed = state.remove("foo")
        assert removed is not None
        assert state.get("foo") is None
        assert state.all_sessions() == []
