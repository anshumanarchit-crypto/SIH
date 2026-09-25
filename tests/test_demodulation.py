"""Unit tests for core/demodulation.py (CSE-3)."""

import numpy as np
import pytest

from core.contracts import (
    DemodulationResult,
    ModulationType,
    ResultStatus,
    SignalData,
)
from core.demodulation import (
    calculate_evm_and_mer,
    costas_carrier_sync,
    demodulate,
    demodulate_bpsk,
    demodulate_qpsk,
    demodulate_16qam,
    demodulate_2fsk,
    demodulate_signal,
    recover_timing_gardner,
)
from core.preprocessing import apply_rrc_filter


def test_evm_and_mer():
    ref = np.array([1.0 + 1j, -1.0 + 1j, -1.0 - 1j, 1.0 - 1j], dtype=np.complex64) / np.sqrt(2.0)
    # 0 error
    evm, mer = calculate_evm_and_mer(ref, ref)
    assert evm == pytest.approx(0.0, abs=1e-5)
    assert mer > 100.0

    # 10% error
    rx = ref * 1.1
    evm10, mer10 = calculate_evm_and_mer(rx, ref)
    assert evm10 == pytest.approx(10.0, rel=0.01)
    assert mer10 == pytest.approx(20.0, rel=0.01)


def test_demodulate_bpsk_clean_and_noisy():
    np.random.seed(42)
    tx_bits = np.random.randint(0, 2, 500, dtype=np.uint8)
    tx_syms = (2.0 * tx_bits - 1.0 + 0j).astype(np.complex64)

    # Clean BPSK
    ref_syms, rx_bits, soft_llrs = demodulate_bpsk(tx_syms)
    np.testing.assert_array_equal(rx_bits, tx_bits)
    assert np.all((soft_llrs > 0) == (tx_bits == 1))

    # Add mild noise
    rx_syms = tx_syms + (np.random.randn(500) + 1j * np.random.randn(500)) * 0.1
    ref_syms, rx_bits, soft_llrs = demodulate_bpsk(rx_syms)
    np.testing.assert_array_equal(rx_bits, tx_bits)


def test_demodulate_bpsk_differential():
    np.random.seed(42)
    # DBPSK test: bit 1 is phase transition, bit 0 is no phase transition
    tx_bits = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    # Generate DBPSK symbols
    syms = np.empty(len(tx_bits), dtype=np.complex64)
    curr_phase = 1.0
    for i, b in enumerate(tx_bits):
        if b == 1:
            curr_phase = -curr_phase
        syms[i] = curr_phase

    ref_syms, rx_bits, soft_llrs = demodulate_bpsk(syms, differential=True)
    # Recovered bits should match transmitted sequence
    np.testing.assert_array_equal(rx_bits[1:], tx_bits[1:])


def test_demodulate_qpsk_gray_mapping():
    np.random.seed(42)
    tx_bits = np.random.randint(0, 2, 1000, dtype=np.uint8)
    b0 = tx_bits[0::2]
    b1 = tx_bits[1::2]
    tx_syms = ((2.0 * b0 - 1.0) + 1j * (2.0 * b1 - 1.0)) / np.sqrt(2.0)

    # Add mild noise
    rx_syms = tx_syms + (np.random.randn(len(tx_syms)) + 1j * np.random.randn(len(tx_syms))) * 0.05

    ref_syms, rx_bits, soft_llrs = demodulate_qpsk(rx_syms.astype(np.complex64))
    np.testing.assert_array_equal(rx_bits, tx_bits)
    assert len(soft_llrs) == 1000


def test_demodulate_2fsk_matched_filter_and_discriminator():
    np.random.seed(42)
    n_syms = 80
    sps = 32
    fs = 32000.0
    baud = fs / sps
    f_mark = 1000.0
    f_space = -1000.0

    tx_bits = np.random.randint(0, 2, n_syms, dtype=np.uint8)
    samples = np.empty(n_syms * sps, dtype=np.complex64)
    t_sym = np.arange(sps) / fs

    for idx, b in enumerate(tx_bits):
        freq = f_mark if b == 1 else f_space
        samples[idx * sps : (idx + 1) * sps] = np.exp(1j * 2.0 * np.pi * freq * t_sym)

    # Add moderate noise (SNR ~ 15 dB)
    noisy_samples = samples + (np.random.randn(len(samples)) + 1j * np.random.randn(len(samples))) * 0.2

    # Noncoherent matched filter tone bank
    ref_syms, rx_bits, soft_llrs, diag = demodulate_2fsk(
        noisy_samples, sample_rate=fs, sps=sps, mark_freq=f_mark, space_freq=f_space, method="matched_filter"
    )
    np.testing.assert_array_equal(rx_bits, tx_bits)
    assert diag["freq_separation_hz"] == pytest.approx(2000.0)

    # Frequency discriminator method
    ref_syms_d, rx_bits_d, soft_llrs_d, diag_d = demodulate_2fsk(
        samples, sample_rate=fs, sps=sps, method="discriminator"
    )
    assert len(rx_bits_d) == n_syms
    assert len(ref_syms_d) == n_syms
    assert len(soft_llrs_d) == n_syms
    assert diag_d["method"] == "instantaneous_frequency_discriminator"
    np.testing.assert_array_equal(rx_bits_d, tx_bits)


def test_demodulate_unified_interface_bpsk_and_qpsk():
    np.random.seed(42)
    tx_bits = np.random.randint(0, 2, 200, dtype=np.uint8)
    tx_syms = (2.0 * tx_bits - 1.0 + 0j).astype(np.complex64)

    # Demodulate with string name
    res = demodulate(tx_syms, modulation="BPSK", sample_rate=10000.0, sps=1.0)
    assert isinstance(res, DemodulationResult)
    assert res.status == ResultStatus.CONFIRMED
    assert res.modulation == ModulationType.BPSK
    np.testing.assert_array_equal(res.bits, tx_bits)
    assert res.metrics["symbol_count"] == 200
    assert res.metrics["bit_count"] == 200

    # Demodulate with SignalData object
    sig = SignalData(samples=tx_syms, sample_rate=10000.0)
    res_sig = demodulate(sig, modulation=ModulationType.BPSK)
    assert res_sig.status == ResultStatus.CONFIRMED
    np.testing.assert_array_equal(res_sig.bits, tx_bits)


def test_demodulate_phase_offset():
    np.random.seed(42)
    tx_bits = np.random.randint(0, 2, 300, dtype=np.uint8)
    b0 = tx_bits[0::2]
    b1 = tx_bits[1::2]
    tx_syms = ((2.0 * b0 - 1.0) + 1j * (2.0 * b1 - 1.0)) / np.sqrt(2.0)

    # Apply 45 degree phase rotation
    theta_deg = 45.0
    rot_syms = tx_syms * np.exp(1j * np.deg2rad(theta_deg))

    # Demodulate with phase_offset_deg compensation
    res = demodulate(rot_syms, modulation="QPSK", phase_offset_deg=45.0, sps=1.0)
    assert res.status == ResultStatus.CONFIRMED
    np.testing.assert_array_equal(res.bits, tx_bits)


def test_costas_carrier_sync():
    np.random.seed(42)
    n = 2000
    bits = np.random.randint(0, 2, n)
    tx_syms = (2.0 * bits - 1.0 + 0j).astype(np.complex64)

    # Introduce static phase offset of 40 degrees
    theta = np.deg2rad(40.0)
    rx_syms = tx_syms * np.exp(1j * theta)

    derotated, phases = costas_carrier_sync(rx_syms, modulation=ModulationType.BPSK, kp=0.05, ki=0.001)

    # In steady-state (last 500 samples), phase error should be locked
    steady_syms = derotated[1000:]
    hard_bits = (np.real(steady_syms) > 0).astype(np.uint8)
    ber_direct = np.mean(hard_bits != bits[1000:])
    ber_inv = np.mean((1 - hard_bits) != bits[1000:])
    assert min(ber_direct, ber_inv) < 0.01


def test_recover_timing_gardner():
    sps = 8
    n_syms = 300
    np.random.seed(42)
    bits = np.random.randint(0, 2, n_syms)
    syms = (2.0 * bits - 1.0 + 0j).astype(np.complex64)

    upsampled = np.zeros(n_syms * sps, dtype=np.complex64)
    upsampled[::sps] = syms
    tx = apply_rrc_filter(upsampled, sps=sps, beta=0.35, span=6)
    rx_matched = apply_rrc_filter(tx, sps=sps, beta=0.35, span=6)

    recovered_syms, timing_errs = recover_timing_gardner(rx_matched, sps=float(sps))
    assert len(recovered_syms) > 200


def test_demodulate_degenerate_and_error_handling():
    # 1. Empty samples
    res_empty = demodulate(np.array([], dtype=np.complex64), modulation="BPSK")
    assert res_empty.status == ResultStatus.FAILED
    assert len(res_empty.warnings) > 0
    assert res_empty.warnings[0]["code"] == "EMPTY_INPUT"
    assert len(res_empty.bits) == 0

    # 2. All-NaN samples
    nan_arr = np.full(100, np.nan + 1j * np.nan, dtype=np.complex64)
    res_nan = demodulate(nan_arr, modulation="QPSK")
    assert res_nan.status == ResultStatus.FAILED
    assert res_nan.warnings[0]["code"] == "INVALID_SIGNAL_DATA"

    # 3. Insufficient samples (length < sps)
    short_arr = np.array([1.0 + 1j], dtype=np.complex64)
    res_short = demodulate(short_arr, modulation="BPSK", sps=8.0)
    assert res_short.status == ResultStatus.INSUFFICIENT_EVIDENCE
    assert res_short.warnings[0]["code"] == "INSUFFICIENT_SAMPLES"

    # 4. Unsupported modulation scheme
    valid_syms = np.array([1.0, -1.0, 1.0], dtype=np.complex64)
    res_unsupported = demodulate(valid_syms, modulation="UNSUPPORTED_MOD_XYZ")
    assert res_unsupported.status == ResultStatus.UNSUPPORTED
    assert res_unsupported.warnings[0]["code"] == "UNSUPPORTED_MODULATION"


def test_demodulate_signal_backwards_compatibility():
    tx_bits = np.random.randint(0, 2, 400, dtype=np.uint8)
    b0 = tx_bits[0::2]
    b1 = tx_bits[1::2]
    syms = ((2.0 * b0 - 1.0) + 1j * (2.0 * b1 - 1.0)) / np.sqrt(2.0)

    res = demodulate_signal(syms.astype(np.complex64), sample_rate=10000.0, modulation=ModulationType.QPSK, sps=1.0)
    assert res.modulation == ModulationType.QPSK
    assert res.evm_percent < 1.0
    assert len(res.hard_bits) == 400
    np.testing.assert_array_equal(res.hard_bits, tx_bits)
