"""tests/unit/test_mapping_profiles.py

Unit tests for modulation mapping profiles (8-PSK and 16-QAM).
"""

from __future__ import annotations
import numpy as np
import pytest

from spectralq.demod import (
    demap_8psk,
    demap_qam,
    demap_bpsk,
    demap_psk,
    MAPPING_PROFILES_8PSK,
    MAPPING_PROFILES_16QAM,
    MAPPING_PROFILES_BPSK,
)


def test_8psk_mapping_profiles():
    """Verify DEFAULT vs OCTAVE mapping for 8-PSK sectors."""
    # 8 constellation points at angles k * pi/4:
    angles = np.array([k * (np.pi / 4.0) for k in range(8)])
    symbols = np.exp(1j * angles)

    bits_default, _ = demap_8psk(symbols, profile="DEFAULT")
    bits_octave, _ = demap_8psk(symbols, profile="OCTAVE")

    # Sectors 0-3 are identical across both mappings:
    # 0: 000, 1: 001, 2: 011, 3: 010
    assert np.array_equal(bits_default[:12], bits_octave[:12])

    # Sectors 4-7 differ:
    # In DEFAULT:
    # 4: 110, 5: 111, 6: 101, 7: 100
    expected_default_tail = [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0]
    assert bits_default[12:].tolist() == expected_default_tail

    # In OCTAVE:
    # 4: 111, 5: 110, 6: 100, 7: 101
    expected_octave_tail = [1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1]
    assert bits_octave[12:].tolist() == expected_octave_tail


def test_16qam_mapping_profiles():
    """Verify DEFAULT vs OCTAVE mapping for 16-QAM PAM-4 decision regions."""
    # Symbols at +3+3j, +1+1j, -1-1j, -3-3j
    syms = np.array([3.0 + 3.0j, 1.0 + 1.0j, -1.0 - 1.0j, -3.0 - 3.0j], dtype=complex)

    bits_default, _ = demap_qam(syms, constellation_order=16, profile="DEFAULT")
    bits_octave, _ = demap_qam(syms, constellation_order=16, profile="OCTAVE")

    # DEFAULT mapping:
    # +3: (1, 0), +1: (1, 1), -1: (0, 1), -3: (0, 0)
    assert bits_default[:4].tolist() == [1, 0, 1, 0]    # +3 + 3j
    assert bits_default[4:8].tolist() == [1, 1, 1, 1]   # +1 + 1j
    assert bits_default[8:12].tolist() == [0, 1, 0, 1]  # -1 - 1j
    assert bits_default[12:16].tolist() == [0, 0, 0, 0] # -3 - 3j

    # OCTAVE mapping:
    # +3: (0, 0), +1: (0, 1), -1: (1, 1), -3: (1, 0)
    assert bits_octave[:4].tolist() == [0, 0, 0, 0]     # +3 + 3j
    assert bits_octave[4:8].tolist() == [0, 1, 0, 1]    # +1 + 1j
    assert bits_octave[8:12].tolist() == [1, 1, 1, 1]   # -1 - 1j
    assert bits_octave[12:16].tolist() == [1, 0, 1, 0]  # -3 - 3j


def test_unsupported_mapping_profile():
    """Verify that unsupported mapping profiles raise a clear ValueError."""
    syms = np.array([1.0 + 0j], dtype=complex)
    with pytest.raises(ValueError, match="Unsupported 8-PSK mapping profile"):
        demap_8psk(syms, profile="NON_EXISTENT")

    with pytest.raises(ValueError, match="Unsupported 16-QAM mapping profile"):
        demap_qam(syms, constellation_order=16, profile="NON_EXISTENT")
