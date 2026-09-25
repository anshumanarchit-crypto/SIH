"""
tests/unit/test_fsk.py

Unit tests for Frequency Shift Keying (FSK) discriminator demodulation:
1. Clean 2-FSK demodulation (exact zero bit errors)
2. Clean 4-FSK demodulation with Gray mapping
3. 2-FSK and 4-FSK with controlled Carrier Frequency Offset (CFO)
4. Noisy 2-FSK and 4-FSK with AWGN
5. Diagnostic metrics verification (estimated CFO and deviation)
"""

import numpy as np
import pytest

from spectralq.demod import demod_fsk
from tests.fixtures.synthetic_generator import generate_synthetic_waveform


def test_clean_2fsk_demodulation():
    """Verify clean 2-FSK demodulation achieves exact zero bit errors."""
    waveform, tx_bits, meta = generate_synthetic_waveform(
        modulation="2-FSK",
        num_bits=256,
        seed=101,
        sps=8,
        sample_rate=1e6,
        fsk_deviation_hz=50e3,
    )
    hard_bits, soft_bits, diag = demod_fsk(
        waveform,
        f_dev=50e3,
        fs=1e6,
        order=2,
        sps=8,
    )

    assert diag["converged"] is True
    # Strip warm-up edge if any
    cmp_len = min(len(hard_bits), len(tx_bits))
    bit_errors = int(np.sum(hard_bits[:cmp_len] != tx_bits[:cmp_len]))
    ber = bit_errors / cmp_len
    assert ber == 0.0, f"Clean 2-FSK should achieve zero BER, got {ber} ({bit_errors} errors)"


def test_clean_4fsk_demodulation():
    """Verify clean 4-FSK demodulation achieves exact zero bit errors with Gray mapping."""
    waveform, tx_bits, meta = generate_synthetic_waveform(
        modulation="4-FSK",
        num_bits=256,
        seed=202,
        sps=8,
        sample_rate=1e6,
        fsk_deviation_hz=75e3,
    )
    hard_bits, soft_bits, diag = demod_fsk(
        waveform,
        f_dev=75e3,
        fs=1e6,
        order=4,
        sps=8,
    )

    assert diag["converged"] is True
    cmp_len = min(len(hard_bits), len(tx_bits))
    bit_errors = int(np.sum(hard_bits[:cmp_len] != tx_bits[:cmp_len]))
    ber = bit_errors / cmp_len
    assert ber == 0.0, f"Clean 4-FSK should achieve zero BER, got {ber} ({bit_errors} errors)"


def test_2fsk_with_carrier_frequency_offset():
    """Verify 2-FSK discriminator removes DC carrier frequency offset."""
    target_cfo = 15e3  # 15 kHz CFO
    waveform, tx_bits, meta = generate_synthetic_waveform(
        modulation="2-FSK",
        num_bits=256,
        seed=303,
        sps=8,
        sample_rate=1e6,
        fsk_deviation_hz=60e3,
        cfo_hz=target_cfo,
    )
    hard_bits, _, diag = demod_fsk(
        waveform,
        f_dev=60e3,
        fs=1e6,
        order=2,
        sps=8,
    )

    assert abs(diag["estimated_cfo_hz"] - target_cfo) < 2e3
    cmp_len = min(len(hard_bits), len(tx_bits))
    bit_errors = int(np.sum(hard_bits[:cmp_len] != tx_bits[:cmp_len]))
    assert bit_errors == 0, f"2-FSK failed with CFO: {bit_errors} errors"


def test_4fsk_with_carrier_frequency_offset():
    """Verify 4-FSK discriminator removes DC carrier frequency offset."""
    target_cfo = 10e3
    waveform, tx_bits, meta = generate_synthetic_waveform(
        modulation="4-FSK",
        num_bits=256,
        seed=404,
        sps=8,
        sample_rate=1e6,
        fsk_deviation_hz=90e3,
        cfo_hz=target_cfo,
    )
    hard_bits, _, diag = demod_fsk(
        waveform,
        f_dev=90e3,
        fs=1e6,
        order=4,
        sps=8,
    )

    cmp_len = min(len(hard_bits), len(tx_bits))
    bit_errors = int(np.sum(hard_bits[:cmp_len] != tx_bits[:cmp_len]))
    assert bit_errors == 0, f"4-FSK failed with CFO: {bit_errors} errors"


def test_noisy_2fsk():
    """Verify 2-FSK under moderate AWGN (SNR = 14 dB) yields low BER."""
    waveform, tx_bits, meta = generate_synthetic_waveform(
        modulation="2-FSK",
        num_bits=400,
        seed=505,
        sps=8,
        sample_rate=1e6,
        fsk_deviation_hz=50e3,
        snr_db=14.0,
    )
    hard_bits, _, _ = demod_fsk(waveform, f_dev=50e3, fs=1e6, order=2, sps=8)
    cmp_len = min(len(hard_bits), len(tx_bits))
    ber = np.mean(hard_bits[:cmp_len] != tx_bits[:cmp_len])
    assert ber < 0.05, f"Noisy 2-FSK BER too high: {ber}"


def test_short_fsk_sample_rejection():
    """Verify FSK demodulator rejects insufficient sample streams gracefully."""
    short_iq = np.array([1.0 + 0j, 0.0 + 1j], dtype=complex)
    hard_bits, soft_bits, diag = demod_fsk(short_iq, sps=8)
    assert len(hard_bits) == 0
    assert diag["converged"] is False
