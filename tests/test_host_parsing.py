"""Host parsing tests — work without GPU."""

from gpu_cosplay import host


def test_parse_power_section():
    text = """
        Min Power Limit                                : 200.00 W
        Max Power Limit                                : 700.00 W
        Current Power Limit                            : 600.00 W
        Default Power Limit                            : 700.00 W
    """
    pmin, pmax, pdef = host._parse_power_section(text)
    assert (pmin, pmax, pdef) == (200, 700, 700)


def test_parse_power_section_missing():
    pmin, pmax, pdef = host._parse_power_section("Min Power Limit : N/A")
    assert pmin is None and pmax is None and pdef is None


def test_parse_supported_clocks():
    text = """
        Supported Clocks
            Graphics                                   : 1980 MHz
            Graphics                                   : 1965 MHz
            Graphics                                   : 1500 MHz
            Graphics                                   : 345 MHz
    """
    cmin, cmax = host._parse_supported_clocks(text)
    assert (cmin, cmax) == (345, 1980)


def test_parse_mig_profiles_block_h200():
    text = """\
+-------------------------------------------------------------------------------+
| GPU instance profiles:                                                        |
| GPU   Name               ID    Instances   Memory     P2P    SM    DEC   ENC  |
|                                Free/Total   GiB              CE    JPEG  OFA  |
|===============================================================================|
|   0  MIG 1g.18gb         19     7/7        16.00      No     16     1     0   |
|                                                               1     1     0   |
+-------------------------------------------------------------------------------+
|   0  MIG 1g.18gb+me      20     1/1        16.00      No     16     1     0   |
|                                                               1     1     1   |
+-------------------------------------------------------------------------------+
|   0  MIG 2g.35gb         14     3/3        32.50      No     32     2     0   |
|                                                               2     2     0   |
+-------------------------------------------------------------------------------+
|   0  MIG 7g.141gb         0     1/1        140.00     No     132    7     0   |
|                                                               8     7     1   |
+-------------------------------------------------------------------------------+
"""
    profiles = host._parse_mig_profiles_block(text, gpu_index=0)
    by_name = {p.name: p for p in profiles}
    # +me variants are filtered
    assert "1g.18gb" in by_name and by_name["1g.18gb"].profile_id == 19
    assert by_name["1g.18gb"].sm_count == 16
    assert by_name["2g.35gb"].sm_count == 32
    assert by_name["7g.141gb"].sm_count == 132


def test_parse_mig_profiles_block_a100_40g():
    text = """\
|===============================================================================|
|   0  MIG 1g.5gb          19     7/7        4.75       No     14     0     0   |
|                                                               1     0     0   |
+-------------------------------------------------------------------------------+
|   0  MIG 2g.10gb         14     3/3        9.75       No     28     1     0   |
|                                                               2     0     0   |
+-------------------------------------------------------------------------------+
|   0  MIG 3g.20gb          9     2/2        19.62      No     42     2     0   |
|                                                               3     0     0   |
+-------------------------------------------------------------------------------+
|   0  MIG 7g.40gb          0     1/1        39.50      No     98     5     0   |
|                                                               7     1     1   |
+-------------------------------------------------------------------------------+
"""
    profiles = host._parse_mig_profiles_block(text, gpu_index=0)
    by_name = {p.name: p for p in profiles}
    assert "1g.5gb" in by_name and by_name["1g.5gb"].sm_count == 14
    assert by_name["7g.40gb"].sm_count == 98
