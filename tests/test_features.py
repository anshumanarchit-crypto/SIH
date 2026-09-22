"""Unit tests for core/features.py."""

import numpy as np
import pytest

from core.features import (
    compute_higher_order_cumulants,
    compute_instantaneous_statistics,
    estimate_snr_m2m4,
    estimate_symbol_rate,
    extract_all_features,
)
from core.preprocessing import apply_rrc_filter


def test_hoc_bpsk():
    np.random.seed(42)
    n = 20000
    bits = np.random.randint(0, 2, n)
    bpsk_syms = 2.0 * bits - 1.0 + 0j  # in {-1, +1}

    hoc = compute_higher_order_cumulants(bpsk_syms)
    # For unit-power BPSK: |C20|=1, C21=1, |C40|=2, |C42|=2
    assert abs(hoc["c20"]) == pytest.approx(1.0, rel=0.05)
    assert hoc["c21"] == pytest.approx(1.0, rel=0.05)
    assert abs(hoc["c40"]) == pytest.approx(2.0, rel=0.05)
    assert abs(hoc["c42"]) == pytest.approx(2.0, rel=0.05)


def test_hoc_qpsk():
    np.random.seed(42)
    n = 20000
    bits_i = 2.0 * np.random.randint(0, 2, n) - 1.0
    bits_q = 2.0 * np.random.randint(0, 2, n) - 1.0
    qpsk_syms = (bits_i + 1j * bits_q) / np.sqrt(2.0)

    hoc = compute_higher_order_cumulants(qpsk_syms)
    # For unit-power QPSK: |C20|=0, C21=1, |C40|=1, |C42|=1
    assert abs(hoc["c20"]) == pytest.approx(0.0, abs=0.05)
    assert hoc["c21"] == pytest.approx(1.0, rel=0.05)
    assert abs(hoc["c40"]) == pytest.approx(1.0, rel=0.05)
    assert abs(hoc["c42"]) == pytest.approx(1.0, rel=0.05)


def test_hoc_16qam():
    np.random.seed(42)
    n = 30000
    levels = np.array([-3, -1, 1, 3])
    i_syms = np.random.choice(levels, size=n)
    q_syms = np.random.choice(levels, size=n)
    qam_syms = (i_syms + 1j * q_syms) / np.sqrt(10.0)  # average power = (9+1+1+9)/2 = 10

    hoc = compute_higher_order_cumulants(qam_syms)
    # For 16-QAM: |C20|=0, |C40| ~= 0.68, |C42| ~= 0.68
    assert abs(hoc["c20"]) == pytest.approx(0.0, abs=0.05)
    assert abs(hoc["c40"]) == pytest.approx(0.68, rel=0.08)
    assert abs(hoc["c42"]) == pytest.approx(0.68, rel=0.08)


def test_hoc_awgn():
    np.random.seed(42)
    n = 20000
    noise = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2.0)

    hoc = compute_higher_order_cumulants(noise)
    # Gaussian noise has cumulants of order >= 3 identically zero
    assert abs(hoc["c40"]) < 0.1
    assert abs(hoc["c42"]) < 0.1


def test_estimate_snr_m2m4():
    np.random.seed(42)
    n = 20000
    bits_i = 2.0 * np.random.randint(0, 2, n) - 1.0
    bits_q = 2.0 * np.random.randint(0, 2, n) - 1.0
    signal = (bits_i + 1j * bits_q) / np.sqrt(2.0)

    # Add AWGN with target SNR = 15 dB
    target_snr_db = 15.0
    noise_power = 10.0 ** (-target_snr_db / 10.0)
    noise = (np.random.randn(n) + 1j * np.random.randn(n)) * np.sqrt(noise_power / 2.0)
    rx_signal = signal + noise

    est_snr = estimate_snr_m2m4(rx_signal, assumed_ka=1.0)
    assert est_snr == pytest.approx(target_snr_db, abs=1.5)


def test_estimate_symbol_rate():
    fs = 100_000.0
    target_baud = 12_500.0  # sps = 8
    sps = 8
    n_syms = 2000
    np.random.seed(42)
    syms = 2.0 * np.random.randint(0, 2, n_syms) - 1.0

    upsampled = np.zeros(n_syms * sps, dtype=np.complex64)
    upsampled[::sps] = syms
    tx = apply_rrc_filter(upsampled, sps=sps, beta=0.35)

    baud_est, sps_est = estimate_symbol_rate(tx, sample_rate=fs)
    assert baud_est == pytest.approx(target_baud, rel=0.1)
    assert sps_est == pytest.approx(sps, rel=0.1)


def test_extract_all_features():
    samples = (np.random.randn(1000) + 1j * np.random.randn(1000)).astype(np.complex64)
    feats = extract_all_features(samples, sample_rate=48000.0, center_freq_offset_hz=0.0)
    assert feats.c21 == pytest.approx(1.0, rel=0.05)
    assert isinstance(feats.raw_metrics, dict)
    assert "gamma_max" in feats.raw_metrics
