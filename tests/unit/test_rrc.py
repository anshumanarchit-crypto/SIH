"""
tests/unit/test_rrc.py

Unit tests for Root-Raised-Cosine (RRC) matched filter implementation:
1. Energy normalization: sum(h^2) == 1.0
2. Filter symmetry: h[-t] == h[t]
3. Singular tap points: t = 0 and t = +- 1 / (4 * alpha)
4. Sinc limit when alpha = 0.0
5. Impulse response convolution and group delay
6. Deterministic repeatability
"""

import numpy as np
import pytest

from spectralq.demod import rrc_filter_taps, apply_rrc_filter


def test_rrc_energy_normalization():
    """Verify RRC filter taps are normalized to unit energy across alpha values."""
    for sps in [2, 4, 8]:
        for alpha in [0.0, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0]:
            h = rrc_filter_taps(sps=sps, alpha=alpha, span=8)
            energy = np.sum(h ** 2)
            assert np.isclose(energy, 1.0, atol=1e-10), f"Energy {energy} != 1.0 for sps={sps}, alpha={alpha}"


def test_rrc_symmetry():
    """Verify RRC impulse response is perfectly symmetric around the center tap."""
    for sps in [2, 4, 8]:
        for alpha in [0.25, 0.35, 0.5]:
            h = rrc_filter_taps(sps=sps, alpha=alpha, span=8)
            assert np.allclose(h, h[::-1], atol=1e-12), "RRC filter taps must be symmetric"


def test_rrc_singular_tap_handling():
    """Verify exact numerical stability at singular tap points t = 0 and t = +- 1 / (4 * alpha)."""
    # When alpha = 0.25 and sps = 4, t = 1/(4*alpha) = 1.0 symbol, which corresponds to index 4
    h_singular = rrc_filter_taps(sps=4, alpha=0.25, span=8)
    assert not np.any(np.isnan(h_singular)), "Singular points produced NaN"
    assert not np.any(np.isinf(h_singular)), "Singular points produced Inf"

    # Also test alpha = 0.5 and sps = 2: t = 1/(4*0.5) = 0.5 symbol -> index 1
    h_singular2 = rrc_filter_taps(sps=2, alpha=0.5, span=8)
    assert not np.any(np.isnan(h_singular2))
    assert not np.any(np.isinf(h_singular2))


def test_rrc_sinc_limit_zero_alpha():
    """Verify RRC converges to sinc filter when alpha = 0.0."""
    h_zero = rrc_filter_taps(sps=4, alpha=0.0, span=8)
    center = (len(h_zero) - 1) // 2
    # Zero crossings of sinc at integer symbol intervals: index center +- k * sps
    for k in [1, 2, 3]:
        assert abs(h_zero[center + k * 4]) < 1e-10
        assert abs(h_zero[center - k * 4]) < 1e-10


def test_rrc_impulse_response_and_group_delay():
    """Verify convolution of an impulse with RRC matched filter yields RRC shape centered at group delay."""
    sps = 4
    span = 8
    h = rrc_filter_taps(sps=sps, alpha=0.35, span=span)
    group_delay = (len(h) - 1) // 2
    assert group_delay == (span * sps) // 2

    # Create delta impulse at index 50
    signal = np.zeros(100, dtype=complex)
    signal[50] = 1.0 + 0.0j

    # Convolve with apply_rrc_filter (mode='same')
    filtered = apply_rrc_filter(signal, sps=sps, alpha=0.35, span=span)
    peak_idx = int(np.argmax(np.abs(filtered)))
    assert peak_idx == 50, f"Mode='same' should preserve peak alignment, got {peak_idx}"


def test_rrc_cascaded_raised_cosine_property():
    """Verify that cascading TX RRC and RX RRC filters forms a full Raised Cosine with zero ISI at T_s."""
    sps = 8
    span = 12
    h = rrc_filter_taps(sps=sps, alpha=0.35, span=span)

    # Full convolution of two RRC filters gives Raised Cosine (RC)
    rc = np.convolve(h, h, mode="same")
    center = len(rc) // 2

    # Normalized peak
    rc_norm = rc / rc[center]

    # At non-zero integer symbol multiples, RC must have near-zero ISI
    for k in [-3, -2, -1, 1, 2, 3]:
        idx = center + k * sps
        assert abs(rc_norm[idx]) < 0.03, f"ISI at symbol {k} is too large: {rc_norm[idx]}"


def test_rrc_deterministic_repeatability():
    """Verify that multiple invocations produce bit-identical tap values."""
    h1 = rrc_filter_taps(sps=4, alpha=0.35, span=8)
    h2 = rrc_filter_taps(sps=4, alpha=0.35, span=8)
    assert np.array_equal(h1, h2)
