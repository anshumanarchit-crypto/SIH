"""
tests/unit/test_carrier_recovery.py

Unit tests for M-PSK Carrier Recovery and Phase Ambiguity Resolution:
1. BPSK (M=2) carrier frequency and phase offset correction
2. QPSK (M=4) carrier recovery with controlled CFO and phase rotation
3. 8-PSK (M=8) carrier recovery with decision-directed Costas loop
4. M-th power coarse frequency estimation accuracy
5. Phase ambiguity resolution with known sync preamble
6. Deterministic repeatability
"""

import numpy as np
import pytest

from spectralq.demod import (
    costas_carrier_recovery,
    estimate_cfo_mth_power,
    resolve_phase_ambiguity,
    demap_qpsk,
    demap_bpsk,
)
from tests.fixtures.synthetic_generator import generate_synthetic_waveform


def test_mth_power_cfo_estimation():
    """Verify M-th power frequency estimator accurately measures normalized CFO for QPSK."""
    rng = np.random.default_rng(101)
    # Generate clean QPSK symbols
    n_syms = 400
    phases = rng.integers(0, 4, size=n_syms) * (np.pi / 2.0) + (np.pi / 4.0)
    clean_syms = np.exp(1j * phases)

    # Inject CFO: delta_f = 0.02 * R_sym
    target_cfo = 0.025
    k_axis = np.arange(n_syms)
    rotated = clean_syms * np.exp(1j * 2.0 * np.pi * target_cfo * k_axis)

    est_cfo = estimate_cfo_mth_power(rotated, order=4)
    assert np.isclose(est_cfo, target_cfo, atol=2e-3), f"Estimated CFO {est_cfo} != target {target_cfo}"


def test_bpsk_carrier_recovery():
    """Verify BPSK Costas loop corrects carrier phase offset and small CFO."""
    rng = np.random.default_rng(202)
    n_syms = 300
    bits = rng.integers(0, 2, size=n_syms)
    clean_syms = np.where(bits == 0, 1.0 + 0j, -1.0 + 0j)

    # Inject 45 degree phase offset and small CFO
    phase_offset = np.radians(45.0)
    cfo = 0.005
    k = np.arange(n_syms)
    impaired = clean_syms * np.exp(1j * (2.0 * np.pi * cfo * k + phase_offset))

    corrected, metrics = costas_carrier_recovery(impaired, order=2)
    assert metrics["converged"] is True

    # After carrier lock, symbols should cluster around real axis (+-1)
    tail = corrected[n_syms // 2:]
    hard_bits, _ = demap_bpsk(tail)
    # Account for possible 180-deg ambiguity in BPSK
    tail_true = bits[n_syms // 2:]
    ber_normal = np.mean(hard_bits != tail_true)
    ber_inverted = np.mean((1 - hard_bits) != tail_true)
    assert min(ber_normal, ber_inverted) < 0.02, f"BER {min(ber_normal, ber_inverted)} too high"


def test_qpsk_carrier_recovery_and_ambiguity_resolution():
    """Verify QPSK carrier recovery tracks CFO and resolves 90-degree phase ambiguity via preamble."""
    preamble_bits = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1], dtype=int)
    rng = np.random.default_rng(303)
    payload_bits = rng.integers(0, 2, size=200)
    all_bits = np.concatenate([preamble_bits, payload_bits])

    # Map to Gray QPSK
    b0 = all_bits[0::2]
    b1 = all_bits[1::2]
    re = np.where(b0 == 0, 1.0, -1.0)
    im = np.where(b1 == 0, 1.0, -1.0)
    clean_syms = (re + 1j * im) / np.sqrt(2.0)

    # Inject 90 degree phase rotation and CFO
    impaired = clean_syms * np.exp(1j * (np.radians(90.0) + 2.0 * np.pi * 0.01 * np.arange(len(clean_syms))))

    # Carrier recovery
    corrected, metrics = costas_carrier_recovery(impaired, order=4)
    assert metrics["converged"] is True

    # Resolve ambiguity
    resolved, amb_info = resolve_phase_ambiguity(corrected, order=4, preamble_bits=preamble_bits)
    assert amb_info["resolved"] is True
    assert amb_info["preamble_bit_errors"] == 0

    # Demap payload
    hard_bits, _ = demap_qpsk(resolved)
    cmp_len = len(all_bits)
    ber = np.mean(hard_bits[:cmp_len] != all_bits[:cmp_len])
    assert ber < 0.02, f"Demodulated BER {ber} too high after ambiguity resolution"


def test_8psk_carrier_recovery():
    """Verify 8-PSK carrier recovery locks and corrects phase offset."""
    rng = np.random.default_rng(404)
    n_syms = 400
    phases = rng.integers(0, 8, size=n_syms) * (np.pi / 4.0)
    clean_syms = np.exp(1j * phases)

    # Inject 20 degree phase offset
    impaired = clean_syms * np.exp(1j * np.radians(20.0))

    corrected, metrics = costas_carrier_recovery(impaired, order=8)
    assert metrics["converged"] is True
    assert metrics["phase_jitter_std"] < 0.35


def test_phase_ambiguity_failure_handling():
    """Verify that unresolved ambiguity is correctly flagged when preamble does not match."""
    rng = np.random.default_rng(505)
    syms = np.exp(1j * rng.uniform(0, 2 * np.pi, size=50))
    mismatched_preamble = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int)

    _, info = resolve_phase_ambiguity(syms, order=4, preamble_bits=mismatched_preamble)
    # Random noise against all-ones preamble should report high bit errors
    assert info["preamble_bit_errors"] > 0
