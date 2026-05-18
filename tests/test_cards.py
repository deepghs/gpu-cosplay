"""Card database tests."""

import pytest

from gpu_cosplay.cards import find_card, load_cards


def test_load_cards_nonempty():
    cards = load_cards()
    assert len(cards) >= 25
    keys = {c.key for c in cards}
    for k in ["rtx_3090", "rtx_4090", "rtx_2060", "a100_40g", "h200_sxm_141g"]:
        assert k in keys, f"missing canonical key {k}"


def test_required_fields_present():
    for c in load_cards():
        assert c.key and c.pretty and c.arch
        assert c.sm_count > 0
        assert c.vram_gb > 0
        assert c.fp32_tflops > 0
        assert c.bandwidth_gbs > 0
        assert c.tdp_w > 0


@pytest.mark.parametrize(
    "alias,expected_key",
    [
        ("3090", "rtx_3090"),
        ("rtx3090", "rtx_3090"),
        ("RTX 3090", "rtx_3090"),
        ("rtx-3090", "rtx_3090"),
        ("4090", "rtx_4090"),
        ("RTX_4090", "rtx_4090"),
        ("2060", "rtx_2060"),
        ("2060_12g", "rtx_2060_12g"),
        ("1660ti", "gtx_1660_ti"),
        ("1660 Ti", "gtx_1660_ti"),
        ("a100", "a100_40g"),
        ("A100_80g", "a100_80g"),
        ("h100", "h100_sxm_80g"),
        ("h200", "h200_sxm_141g"),
        ("L40S", "l40s"),
        ("t4", "t4"),
    ],
)
def test_alias_resolution(alias, expected_key):
    c = find_card(alias)
    assert c.key == expected_key


def test_unknown_card_raises():
    with pytest.raises(KeyError):
        find_card("rtx_99999")


def test_consumer_cards_have_known_vram():
    by_key = {c.key: c for c in load_cards()}
    assert by_key["rtx_3090"].vram_gb == 24
    assert by_key["rtx_4090"].vram_gb == 24
    assert by_key["rtx_2060"].vram_gb == 6
    assert by_key["gtx_1650"].vram_gb == 4
