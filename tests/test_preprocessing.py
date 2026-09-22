"""Unit tests for core/preprocessing.py."""

import numpy as np
import pytest

from core.preprocessing import (
    remove_dc_offset,
    correct_iq_imbalance,
    normalize_power,
    design_rrc_filter,
    apply_rrc_filter,
    estimate_and_correct_cfo,
    automatic_gain_control,
)


def test_remove_dc_offset():
    s = np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j], dtype=np.complex64)
    cleaned = remove_dc_offset(s)
    assert np.mean(cleaned) == pytest.approx(0.0, abs=1e-6)


def test_correct_iq_imbalance():
    np.random.seed(42)
    n = 20000
    # True balanced I/Q
    i_orig = np.random.randn(n)
    q_orig = np.random.randn(n)

    # Introduce 3.0 dB amplitude imbalance and 20 degree phase error
    alpha = 10.0 ** (3.0 / 20.0)  # ~1.412
    phi = np.deg2rad(20.0)

    i_imb = i_orig
    q_imb = alpha * (np.sin(phi) * i_orig + np.cos(phi) * q_orig)
    sig_imb = (i_imb + 1j * q_imb).astype(np.complex64)

    corrected, metrics = correct_iq_imbalance(sig_imb)
    assert abs(metrics["gain_imbalance_db"]) > 2.5
    assert abs(metrics["phase_error_deg"]) > 15.0

    # Verify corrected signal is orthogonal and balanced
    i_corr = np.real(corrected)
    q_corr = np.imag(corrected)
    p_i = np.mean(i_corr ** 2)
    p_q = np.mean(q_corr ** 2)
    corr_coeff = np.mean(i_corr * q_corr) / np.sqrt(p_i * p_q)

    assert p_q / p_i == pytest.approx(1.0, rel=0.05)
    assert corr_coeff == pytest.approx(0.0, abs=0.03)


def test_normalize_power():
    s = np.array([10.0 + 10j, -10.0 + 10j], dtype=np.complex64)
    norm = normalize_power(s, target_power=1.0)
    p = np.mean(np.abs(norm) ** 2)
    assert p == pytest.approx(1.0, rel=1e-5)


def test_design_rrc_filter():
    h = design_rrc_filter(sps=8, beta=0.35, span=10)
    assert len(h) == 8 * 10 + 1
    # Check unit energy sum(h^2) == 1.0
    assert np.sum(h ** 2) == pytest.approx(1.0, rel=1e-5)


def test_apply_rrc_filter():
    # Double RRC application produces Nyquist Raised-Cosine with ISI=0 at symbol center
    sps = 8
    num_syms = 20
    np.random.seed(42)
    syms = 2 * np.random.randint(0, 2, num_syms) - 1.0  # BPSK symbols +/-1

    # Upsample
    upsampled = np.zeros(num_syms * sps, dtype=np.complex64)
    upsampled[::sps] = syms

    # TX RRC
    tx = apply_rrc_filter(upsampled, sps=sps, beta=0.35, span=8)
    # RX RRC (Matched filter)
    rx = apply_rrc_filter(tx, sps=sps, beta=0.35, span=8)

    # Eye diagram center should reconstruct transmitted symbols
    mid_syms = np.real(rx[4 * sps : 16 * sps : sps])
    orig_mid_syms = syms[4:16]
    # Check sign matches perfectly
    np.testing.assert_array_equal(np.sign(mid_syms), orig_mid_syms)


def test_estimate_and_correct_cfo():
    fs = 100_000.0
    n = 10000
    t = np.arange(n) / fs
    cfo_true = 2500.0  # 2.5 kHz offset

    # Tone signal
    sig = np.exp(1j * 2 * np.pi * cfo_true * t).astype(np.complex64)

    corrected, cfo_est = estimate_and_correct_cfo(sig, sample_rate=fs, m_order=1)
    assert cfo_est == pytest.approx(cfo_true, abs=10.0)

    # Verify residual frequency is near zero
    phase_diff = np.diff(np.unwrap(np.angle(corrected)))
    mean_residual_freq = np.mean(phase_diff) * fs / (2 * np.pi)
    assert mean_residual_freq == pytest.approx(0.0, abs=5.0)


def test_automatic_gain_control():
    # Signal with sudden step amplitude drop
    n = 5000
    t = np.arange(n)
    envelope = np.ones(n)
    envelope[2500:] = 0.2  # 14 dB drop
    sig = envelope * np.exp(1j * 2 * np.pi * 0.05 * t)

    leveled, gains = automatic_gain_control(sig, target_level=1.0, mu=0.01)
    # AGC should adjust gain and restore envelope towards 1.0
    assert np.mean(np.abs(leveled[4000:])) == pytest.approx(1.0, abs=0.1)
