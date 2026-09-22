"""Unit tests for core/modulation.py."""

import numpy as np
import pytest

from core.features import extract_all_features
from core.modulation import HybridModulationClassifier, ModulationType


def test_classify_bpsk():
    np.random.seed(42)
    n = 20000
    bits = np.random.randint(0, 2, n)
    syms = (2.0 * bits - 1.0 + 0j).astype(np.complex64)
    # Add mild AWGN (20 dB)
    noise = (np.random.randn(n) + 1j * np.random.randn(n)) * 0.07
    sig = syms + noise

    feats = extract_all_features(sig, sample_rate=100_000.0)
    clf = HybridModulationClassifier()
    res = clf.classify(feats)

    assert res.modulation == ModulationType.BPSK
    assert res.confidence > 0.85
    assert not res.is_unknown


def test_classify_qpsk():
    np.random.seed(42)
    n = 20000
    bits_i = 2.0 * np.random.randint(0, 2, n) - 1.0
    bits_q = 2.0 * np.random.randint(0, 2, n) - 1.0
    syms = ((bits_i + 1j * bits_q) / np.sqrt(2.0)).astype(np.complex64)
    noise = (np.random.randn(n) + 1j * np.random.randn(n)) * 0.07
    sig = syms + noise

    feats = extract_all_features(sig, sample_rate=100_000.0)
    clf = HybridModulationClassifier()
    res = clf.classify(feats)

    assert res.modulation == ModulationType.QPSK
    assert res.confidence > 0.85
    assert not res.is_unknown


def test_classify_16qam():
    np.random.seed(42)
    n = 30000
    levels = np.array([-3, -1, 1, 3])
    i_syms = np.random.choice(levels, size=n)
    q_syms = np.random.choice(levels, size=n)
    syms = ((i_syms + 1j * q_syms) / np.sqrt(10.0)).astype(np.complex64)
    noise = (np.random.randn(n) + 1j * np.random.randn(n)) * 0.05
    sig = syms + noise

    feats = extract_all_features(sig, sample_rate=100_000.0)
    clf = HybridModulationClassifier()
    res = clf.classify(feats)

    assert res.modulation == ModulationType.QAM16
    assert res.confidence > 0.80
    assert not res.is_unknown


def test_rejection_of_pure_noise():
    np.random.seed(42)
    n = 20000
    # Pure AWGN with no signal
    noise = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)

    feats = extract_all_features(noise, sample_rate=100_000.0)
    clf = HybridModulationClassifier(min_snr_threshold_db=3.0)
    res = clf.classify(feats)

    # Engineering Rule 7: Must report Unknown / Insufficient Evidence
    assert res.modulation == ModulationType.UNKNOWN
    assert res.is_unknown is True
    assert "Insufficient evidence" in res.rationale or res.confidence == 0.0


def test_classify_2fsk():
    np.random.seed(42)
    fs = 100_000.0
    baud = 5000.0
    sps = int(fs / baud)
    n_syms = 1000
    bits = np.random.randint(0, 2, n_syms)

    f_dev = 10_000.0  # +/- 10 kHz deviation
    freqs = np.repeat((2.0 * bits - 1.0) * f_dev, sps)
    phase = 2.0 * np.pi * np.cumsum(freqs) / fs
    sig = np.exp(1j * phase).astype(np.complex64)

    feats = extract_all_features(sig, sample_rate=fs)
    clf = HybridModulationClassifier()
    res = clf.classify(feats)

    assert res.modulation in (ModulationType.FSK2, ModulationType.FM)
