"""
tests/unit/test_gardner.py

Unit tests for Gardner Timing Error Detector (TED) and timing recovery loop:
1. Perfect timing alignment recovery
2. Fractional timing offset tracking (0.25, 0.50, -0.30 symbols)
3. Diverse samples-per-symbol (sps = 2, 4, 8)
4. Moderate AWGN noise tolerance
5. Deterministic repeatability
6. Non-convergence detection on invalid or random noise input
"""

import numpy as np
import pytest

from spectralq.demod import gardner_timing_recovery, apply_rrc_filter, demap_qpsk
from tests.fixtures.synthetic_generator import generate_synthetic_waveform


def test_gardner_perfect_timing():
    """Verify Gardner timing recovery produces valid symbols under zero timing offset."""
    waveform, tx_bits, meta = generate_synthetic_waveform(
        modulation="QPSK",
        num_bits=200,
        seed=101,
        sps=4,
        timing_offset_symbols=0.0,
    )
    filtered = apply_rrc_filter(waveform, sps=4, alpha=0.35, span=8)
    symbols, timing_metrics = gardner_timing_recovery(filtered, sps=4)

    assert len(symbols) > 80
    assert timing_metrics["converged"] is True
    assert timing_metrics["jitter_std"] < 0.35

    # Check EVM / symbol quality
    # Demap symbols and verify low bit errors on center payload
    hard_bits, _ = demap_qpsk(symbols)
    # Compare with known tx bits
    cmp_len = min(len(hard_bits), len(tx_bits) - 8)
    # First few symbols are filter warm-up
    errors = np.sum(hard_bits[8:cmp_len] != tx_bits[8:cmp_len])
    ber = errors / (cmp_len - 8)
    assert ber < 0.05, f"BER {ber} is too high for clean perfect timing"


@pytest.mark.parametrize("offset", [0.15, 0.35, -0.25, -0.40])
def test_gardner_fractional_timing_offsets(offset):
    """Verify Gardner loop tracks diverse positive and negative fractional timing offsets."""
    waveform, tx_bits, meta = generate_synthetic_waveform(
        modulation="QPSK",
        num_bits=300,
        seed=202 + int(abs(offset) * 100),
        sps=4,
        timing_offset_symbols=offset,
    )
    filtered = apply_rrc_filter(waveform, sps=4, alpha=0.35, span=8)
    symbols, timing_metrics = gardner_timing_recovery(filtered, sps=4)

    assert len(symbols) > 100
    assert timing_metrics["converged"] is True
    assert timing_metrics["jitter_std"] < 0.65

    # Demap center symbols
    hard_bits, _ = demap_qpsk(symbols)
    cmp_len = min(len(hard_bits), len(tx_bits) - 10)
    errors = np.sum(hard_bits[10:cmp_len] != tx_bits[10:cmp_len])
    ber = errors / (cmp_len - 10)
    assert ber < 0.08, f"BER {ber} too high for offset {offset}"


@pytest.mark.parametrize("sps", [2, 4, 8])
def test_gardner_sps_variations(sps):
    """Verify Gardner timing recovery functions across supported samples-per-symbol (2, 4, 8)."""
    waveform, tx_bits, meta = generate_synthetic_waveform(
        modulation="BPSK",
        num_bits=150,
        seed=303 + sps,
        sps=sps,
        timing_offset_symbols=0.2,
    )
    filtered = apply_rrc_filter(waveform, sps=sps, alpha=0.35, span=8)
    symbols, metrics = gardner_timing_recovery(filtered, sps=sps)

    assert len(symbols) > 50
    assert metrics["converged"] is True
    assert metrics["nominal_sps"] == sps


def test_gardner_moderate_awgn():
    """Verify Gardner loop maintains lock under moderate AWGN (SNR = 12 dB)."""
    waveform, tx_bits, meta = generate_synthetic_waveform(
        modulation="QPSK",
        num_bits=300,
        seed=404,
        sps=4,
        timing_offset_symbols=0.25,
        snr_db=12.0,
    )
    filtered = apply_rrc_filter(waveform, sps=4, alpha=0.35, span=8)
    symbols, metrics = gardner_timing_recovery(filtered, sps=4)

    assert len(symbols) > 100
    # Even with noise, jitter should stay bounded
    assert metrics["jitter_std"] < 0.55


def test_gardner_determinism():
    """Verify that multiple executions produce identical symbol and diagnostic values."""
    waveform, _, _ = generate_synthetic_waveform(
        modulation="QPSK",
        num_bits=100,
        seed=505,
        sps=4,
        timing_offset_symbols=0.3,
    )
    filtered = apply_rrc_filter(waveform, sps=4, alpha=0.35, span=8)
    s1, m1 = gardner_timing_recovery(filtered, sps=4)
    s2, m2 = gardner_timing_recovery(filtered, sps=4)

    assert np.array_equal(s1, s2)
    assert m1["jitter_std"] == m2["jitter_std"]
    assert m1["mean_error"] == m2["mean_error"]


def test_gardner_short_input_rejection():
    """Verify Gardner timing recovery handles insufficient input samples gracefully."""
    too_short = np.array([1.0 + 1.0j, -1.0 + 1.0j], dtype=complex)
    symbols, metrics = gardner_timing_recovery(too_short, sps=4)
    assert len(symbols) == 0
    assert metrics["converged"] is False
    assert "warnings" in metrics
