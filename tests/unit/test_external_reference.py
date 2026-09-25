"""tests/unit/test_external_reference.py

Unit tests for externally supplied reference bit handling and diagnostic reporting.
"""

from __future__ import annotations
import numpy as np
import pytest

from spectralq.demod import demodulate, DemodConfig, DemodStatus


def test_reference_bits_unavailable_default():
    """Verify that omitting external reference bits leaves reference_status as UNAVAILABLE and BER as None."""
    iq = np.array([1.0 + 0j, -1.0 + 0j, 1.0 + 0j], dtype=complex)
    cfg = DemodConfig(modulation="BPSK")
    res = demodulate(iq, cfg)

    assert res.reference_status == "REFERENCE_BITS_UNAVAILABLE"
    assert res.bit_error_rate is None
    assert res.bit_errors is None


def test_external_reference_bits_evaluation():
    """Verify that supplying external reference bits computes exact BER and updates reference status."""
    # BPSK: +1 -> 0, -1 -> 1
    iq = np.repeat([1.0, -1.0, 1.0, -1.0], 4).astype(complex)
    cfg = DemodConfig(modulation="BPSK", samples_per_symbol=4)
    res_base = demodulate(iq, cfg)

    # Reference bits with exactly 1 mismatch against demodulated bits
    ref_bits = res_base.hard_bits.copy()
    ref_bits[0] = 1 - ref_bits[0]

    cfg_ref = DemodConfig(modulation="BPSK", samples_per_symbol=4, external_reference_bits=ref_bits)
    res = demodulate(iq, cfg_ref)

    assert res.reference_status == "EVALUATED_AGAINST_EXTERNAL_REFERENCE"
    assert res.bit_errors == 1
    assert res.bit_error_rate == 1.0 / len(ref_bits)


def test_g7_diagnostic_reporting_contract():
    """Verify that demodulator generates rich diagnostics suitable for G7 analysis."""
    # Create noisy near-threshold signal
    rng = np.random.default_rng(2026)
    noise = (rng.normal(0, 0.7, 100) + 1j * rng.normal(0, 0.7, 100))
    signal = np.repeat([1+1j, -1+1j, -1-1j, 1-1j], 25) / np.sqrt(2.0)
    noisy = signal + noise

    cfg = DemodConfig(modulation="QPSK", samples_per_symbol=4)
    res = demodulate(noisy, cfg)

    diag = res.diagnostics
    assert "rrc_filter_span" in diag
    assert "timing_metrics" in diag
    assert "carrier_metrics" in diag
    assert "ambiguity_info" in diag
    assert "mean_error" in res.timing_status
    assert "jitter_std" in res.timing_status
    assert "phase_jitter_std" in res.carrier_status
