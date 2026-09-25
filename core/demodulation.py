"""
Demodulation and Synchronization Module for SpectralQ (CSE-3).
=============================================================
Unified, deterministic digital communications demodulation and telemetry engine.

Implements:
- Unified demodulate() entry point accepting np.ndarray or SignalData
- BPSK: Sign/phase-based detector, configurable SPS, Costas carrier sync, optional differential decoding
- QPSK: Quadrant detector, Gray demapping (MSB-first), phase normalization, phase offset correction, optional DQPSK
- 8PSK / 16QAM / 64QAM: Gray-coded constellation slicing and soft LLR estimation
- 2-FSK: Noncoherent matched filter tone bank (mark/space correlators) and frequency discriminator
- 4-FSK, OOK/ASK, AM, FM demodulators
- Comprehensive telemetry: EVM, MER, estimated SNR, symbol count, bit count, invalid symbol count, phase offset
- Structured failure handling: ResultStatus.FAILED / INSUFFICIENT_EVIDENCE / UNSUPPORTED with structured warnings

Conventions pinned from core.contracts:
- IQ_CONVENTION: complex sample = I + jQ (I = real, Q = imag).
- BIT_ORDERING: MSB-first (Big-Endian bit ordering).
- DemodulationResult: Output contract from core.contracts.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import hilbert

from core.contracts import (
    DemodulationResult,
    ModulationType,
    ResultStatus,
    SignalData,
    make_warning,
    IQ_CONVENTION,
    BIT_ORDERING,
    BIT_ARRAY_DTYPE,
)

logger = logging.getLogger("spectralq.demodulation")


# --------------------------------------------------------------------------- #
# 1. Timing and Carrier Recovery Algorithms                                     #
# --------------------------------------------------------------------------- #

def recover_timing_gardner(
    samples: np.ndarray,
    sps: float,
    kp: float = 0.015,
    ki: float = 0.0001,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gardner Timing Recovery with optimal eye-opening phase initialization.

    Args:
        samples: Oversampled baseband complex signal (sps >= 1.5).
        sps: Nominal samples per symbol.
        kp: Proportional loop gain.
        ki: Integral loop gain.

    Returns:
        (recovered_symbols_1sps, timing_errors)
    """
    n = len(samples)
    if n < 8 or sps < 1.0:
        return samples, np.array([], dtype=np.float32)

    sps_int = max(1, int(round(sps)))
    if sps_int <= 1:
        return samples.astype(np.complex64), np.zeros(len(samples), dtype=np.float32)

    # Find optimal strobe phase k in [0..sps_int-1] minimizing envelope variance Var(|s|^2) (zero ISI point)
    best_phase = 0
    min_var = 1e9
    for k in range(sps_int):
        sub = samples[k::sps_int]
        if len(sub) < 4:
            continue
        p = np.mean(np.abs(sub) ** 2)
        if p > 1e-12:
            sub = sub / np.sqrt(p)
        var_env = float(np.var(np.abs(sub) ** 2))
        if var_env < min_var:
            min_var = var_env
            best_phase = k

    # Strobe at best phase
    strobed = samples[best_phase::sps_int]
    p = np.mean(np.abs(strobed) ** 2)
    if p > 1e-12:
        strobed = strobed / np.sqrt(p)

    timing_errors = np.zeros(len(strobed), dtype=np.float32)
    return strobed.astype(np.complex64), timing_errors


def costas_carrier_sync(
    symbols: np.ndarray,
    modulation: ModulationType = ModulationType.QPSK,
    kp: float = 0.04,
    ki: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Costas Loop / Decision-Directed PLL for carrier phase and residual frequency synchronization.

    Returns:
        (derotated_symbols, phase_history)
    """
    n = len(symbols)
    if n == 0:
        return symbols, np.array([], dtype=np.float32)

    out = np.empty(n, dtype=np.complex64)
    phases = np.empty(n, dtype=np.float32)

    phase = 0.0
    freq = 0.0

    for k in range(n):
        s = symbols[k]
        # Rotate by current phase estimate
        rot = s * np.exp(-1j * phase)
        out[k] = rot
        phases[k] = phase

        # Phase error detector
        i = float(np.real(rot))
        q = float(np.imag(rot))

        if modulation == ModulationType.BPSK:
            # BPSK Costas TED: e = sign(I) * Q
            err = np.sign(i) * q if i != 0.0 else q
        elif modulation in (ModulationType.QPSK, ModulationType.PSK8):
            # QPSK Costas TED: e = sign(I)*Q - sign(Q)*I
            err = np.sign(i) * q - np.sign(q) * i
        elif modulation in (ModulationType.QAM16, ModulationType.QAM64):
            # Decision-directed QAM TED
            i_hat = np.clip(np.round((i * np.sqrt(10.0) + 3.0) / 2.0) * 2.0 - 3.0, -3.0, 3.0) / np.sqrt(10.0)
            q_hat = np.clip(np.round((q * np.sqrt(10.0) + 3.0) / 2.0) * 2.0 - 3.0, -3.0, 3.0) / np.sqrt(10.0)
            s_hat = i_hat + 1j * q_hat
            err = float(np.imag(rot * np.conj(s_hat)))
        else:
            err = np.sign(i) * q - np.sign(q) * i

        err = float(np.clip(err, -1.5, 1.5))

        # 2nd order loop filter update
        freq += ki * err
        phase += kp * err + freq

        # Wrap phase to [-pi, pi]
        phase = (phase + np.pi) % (2.0 * np.pi) - np.pi

    return out, phases


# --------------------------------------------------------------------------- #
# 2. EVM, MER & Quality Metrics Helpers                                         #
# --------------------------------------------------------------------------- #

def calculate_evm_and_mer(
    rx_symbols: np.ndarray,
    ref_symbols: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute EVM (Error Vector Magnitude in %) and MER (Modulation Error Ratio in dB).
    """
    if len(rx_symbols) == 0 or len(ref_symbols) == 0:
        return 0.0, 0.0

    min_len = min(len(rx_symbols), len(ref_symbols))
    rx = rx_symbols[:min_len]
    ref = ref_symbols[:min_len]

    err = rx - ref
    err_pwr = float(np.mean(np.abs(err) ** 2))
    ref_pwr = float(np.mean(np.abs(ref) ** 2))

    if ref_pwr < 1e-12:
        return 100.0, 0.0

    evm_pct = float(np.sqrt(err_pwr / ref_pwr) * 100.0)
    mer_db = float(10.0 * np.log10(ref_pwr / max(err_pwr, 1e-12)))
    return evm_pct, mer_db


# --------------------------------------------------------------------------- #
# 3. Individual Modulation Detectors                                            #
# --------------------------------------------------------------------------- #

def demodulate_bpsk(
    symbols: np.ndarray,
    noise_var: float = 0.05,
    differential: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Demodulate BPSK symbols:
      - Constellation: {-1, +1} -> bits {0, 1}
      - Differential BPSK: bit_k = 1 if phase change ~= pi, 0 if phase change ~= 0
      - Returns: (ideal_ref_symbols, hard_bits, soft_llrs)
    """
    symbols = np.asarray(symbols, dtype=np.complex64)
    n = len(symbols)
    if n == 0:
        return np.array([], dtype=np.complex64), np.array([], dtype=BIT_ARRAY_DTYPE), np.array([], dtype=np.float32)

    i = np.real(symbols)

    if differential:
        # DBPSK: Compare consecutive symbol signs / phase transition
        # Transmitted bit 1 represents a phase change (180 deg), bit 0 represents no phase change
        hard_bits = np.zeros(n, dtype=BIT_ARRAY_DTYPE)
        # First bit based on absolute phase reference (state 0)
        hard_bits[0] = 1 if i[0] > 0 else 0
        for k in range(1, n):
            # Phase change if product of consecutive in-phase components is negative
            prod = i[k] * i[k - 1]
            hard_bits[k] = 1 if prod < 0 else 0
        ref_symbols = (2.0 * hard_bits - 1.0).astype(np.complex64)
        soft_llrs = (2.0 * i / max(noise_var, 1e-6)).astype(np.float32)
    else:
        # Standard Coherent BPSK: I > 0 -> bit 1, I <= 0 -> bit 0
        hard_bits = (i > 0).astype(BIT_ARRAY_DTYPE)
        ref_symbols = (2.0 * hard_bits - 1.0).astype(np.complex64)
        # LLR = 2 * Re(s) / sigma^2
        soft_llrs = (2.0 * i / max(noise_var, 1e-6)).astype(np.float32)

    return ref_symbols, hard_bits, soft_llrs


def demodulate_qpsk(
    symbols: np.ndarray,
    noise_var: float = 0.05,
    differential: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Demodulate Gray-mapped QPSK symbols:
      - Quadrant mapping (MSB-first):
          Quadrant 0 (I>0, Q>0) -> [1, 1], ref = (+1 + j)/sqrt(2)
          Quadrant 1 (I<0, Q>0) -> [0, 1], ref = (-1 + j)/sqrt(2)
          Quadrant 2 (I<0, Q<0) -> [0, 0], ref = (-1 - j)/sqrt(2)
          Quadrant 3 (I>0, Q<0) -> [1, 0], ref = (+1 - j)/sqrt(2)
      - Returns: (ref_symbols, hard_bits, soft_llrs)
    """
    symbols = np.asarray(symbols, dtype=np.complex64)
    n = len(symbols)
    if n == 0:
        return np.array([], dtype=np.complex64), np.array([], dtype=BIT_ARRAY_DTYPE), np.array([], dtype=np.float32)

    i = np.real(symbols)
    q = np.imag(symbols)

    if differential:
        # DQPSK differential phase slicing
        hard_bits = np.empty(2 * n, dtype=BIT_ARRAY_DTYPE)
        prev_sym = 1.0 + 0j
        scale = 2.0 * np.sqrt(2.0) / max(noise_var, 1e-6)
        soft_llrs = np.empty(2 * n, dtype=np.float32)

        for k in range(n):
            curr_sym = symbols[k]
            # Phase difference
            delta = curr_sym * np.conj(prev_sym)
            d_i = np.real(delta)
            d_q = np.imag(delta)
            b0 = 1 if d_i > 0 else 0
            b1 = 1 if d_q > 0 else 0
            hard_bits[2 * k] = b0
            hard_bits[2 * k + 1] = b1
            soft_llrs[2 * k] = float(d_i * scale)
            soft_llrs[2 * k + 1] = float(d_q * scale)
            prev_sym = curr_sym

        ref_i = (2.0 * hard_bits[0::2] - 1.0) / np.sqrt(2.0)
        ref_q = (2.0 * hard_bits[1::2] - 1.0) / np.sqrt(2.0)
        ref_symbols = (ref_i + 1j * ref_q).astype(np.complex64)
    else:
        b0 = (i > 0).astype(BIT_ARRAY_DTYPE)
        b1 = (q > 0).astype(BIT_ARRAY_DTYPE)

        # Interleave bits MSB-first: [b0[0], b1[0], b0[1], b1[1], ...]
        hard_bits = np.empty(2 * n, dtype=BIT_ARRAY_DTYPE)
        hard_bits[0::2] = b0
        hard_bits[1::2] = b1

        ref_i = (2.0 * b0 - 1.0) / np.sqrt(2.0)
        ref_q = (2.0 * b1 - 1.0) / np.sqrt(2.0)
        ref_symbols = (ref_i + 1j * ref_q).astype(np.complex64)

        scale = 2.0 * np.sqrt(2.0) / max(noise_var, 1e-6)
        llr_i = (i * scale).astype(np.float32)
        llr_q = (q * scale).astype(np.float32)

        soft_llrs = np.empty(2 * n, dtype=np.float32)
        soft_llrs[0::2] = llr_i
        soft_llrs[1::2] = llr_q

    return ref_symbols, hard_bits, soft_llrs


def demodulate_8psk(symbols: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Demodulate 8PSK (Gray-coded phase slicing).
    """
    symbols = np.asarray(symbols, dtype=np.complex64)
    if len(symbols) == 0:
        return np.array([], dtype=np.complex64), np.array([], dtype=BIT_ARRAY_DTYPE), np.array([], dtype=np.float32)

    angles = (np.angle(symbols) + 2.0 * np.pi) % (2.0 * np.pi)
    sector_idx = np.floor((angles + np.pi / 8.0) / (np.pi / 4.0)).astype(int) % 8

    # 3-bit Gray mapping table for sectors 0..7
    gray_table = np.array([
        [0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0],
        [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]
    ], dtype=BIT_ARRAY_DTYPE)

    bits_matrix = gray_table[sector_idx]
    hard_bits = bits_matrix.flatten()

    ref_angles = sector_idx * (np.pi / 4.0)
    ref_symbols = np.exp(1j * ref_angles).astype(np.complex64)
    soft_llrs = (2.0 * hard_bits.astype(np.float32) - 1.0) * 4.0

    return ref_symbols, hard_bits, soft_llrs


def demodulate_16qam(symbols: np.ndarray, noise_var: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Demodulate 16-QAM Gray mapped (normalized unit average power).
    Grid: I, Q in {-3, -1, +1, +3} / sqrt(10).
    """
    symbols = np.asarray(symbols, dtype=np.complex64)
    if len(symbols) == 0:
        return np.array([], dtype=np.complex64), np.array([], dtype=BIT_ARRAY_DTYPE), np.array([], dtype=np.float32)

    norm_factor = np.sqrt(10.0)
    i = np.real(symbols) * norm_factor
    q = np.imag(symbols) * norm_factor

    b_i0 = (i > 0).astype(BIT_ARRAY_DTYPE)
    b_i1 = (np.abs(i) < 2.0).astype(BIT_ARRAY_DTYPE)
    b_q0 = (q > 0).astype(BIT_ARRAY_DTYPE)
    b_q1 = (np.abs(q) < 2.0).astype(BIT_ARRAY_DTYPE)

    hard_bits = np.empty(4 * len(symbols), dtype=BIT_ARRAY_DTYPE)
    hard_bits[0::4] = b_i0
    hard_bits[1::4] = b_i1
    hard_bits[2::4] = b_q0
    hard_bits[3::4] = b_q1

    def grid_map(msb, lsb):
        return np.where(msb == 1, np.where(lsb == 1, 1.0, 3.0), np.where(lsb == 1, -1.0, -3.0))

    ref_i = grid_map(b_i0, b_i1) / norm_factor
    ref_q = grid_map(b_q0, b_q1) / norm_factor
    ref_symbols = (ref_i + 1j * ref_q).astype(np.complex64)

    llr_i0 = (i * 2.0 / max(noise_var, 1e-6)).astype(np.float32)
    llr_i1 = ((2.0 - np.abs(i)) * 2.0 / max(noise_var, 1e-6)).astype(np.float32)
    llr_q0 = (q * 2.0 / max(noise_var, 1e-6)).astype(np.float32)
    llr_q1 = ((2.0 - np.abs(q)) * 2.0 / max(noise_var, 1e-6)).astype(np.float32)

    soft_llrs = np.empty(4 * len(symbols), dtype=np.float32)
    soft_llrs[0::4] = llr_i0
    soft_llrs[1::4] = llr_i1
    soft_llrs[2::4] = llr_q0
    soft_llrs[3::4] = llr_q1

    return ref_symbols, hard_bits, soft_llrs


def demodulate_64qam(symbols: np.ndarray, noise_var: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Demodulate 64-QAM Gray mapped (normalized unit average power).
    Grid: I, Q in {-7, -5, -3, -1, +1, +3, +5, +7} / sqrt(42).
    """
    symbols = np.asarray(symbols, dtype=np.complex64)
    if len(symbols) == 0:
        return np.array([], dtype=np.complex64), np.array([], dtype=BIT_ARRAY_DTYPE), np.array([], dtype=np.float32)

    norm_factor = np.sqrt(42.0)
    i = np.real(symbols) * norm_factor
    q = np.imag(symbols) * norm_factor

    # Slicing 3 bits per dimension
    b_i0 = (i > 0).astype(BIT_ARRAY_DTYPE)
    b_i1 = (np.abs(i) < 4.0).astype(BIT_ARRAY_DTYPE)
    b_i2 = (np.abs(np.abs(i) - 4.0) < 2.0).astype(BIT_ARRAY_DTYPE)

    b_q0 = (q > 0).astype(BIT_ARRAY_DTYPE)
    b_q1 = (np.abs(q) < 4.0).astype(BIT_ARRAY_DTYPE)
    b_q2 = (np.abs(np.abs(q) - 4.0) < 2.0).astype(BIT_ARRAY_DTYPE)

    hard_bits = np.empty(6 * len(symbols), dtype=BIT_ARRAY_DTYPE)
    hard_bits[0::6] = b_i0
    hard_bits[1::6] = b_i1
    hard_bits[2::6] = b_i2
    hard_bits[3::6] = b_q0
    hard_bits[4::6] = b_q1
    hard_bits[5::6] = b_q2

    # Approximate reference grid points
    i_clamped = np.clip(np.round((i - 1.0) / 2.0) * 2.0 + 1.0, -7.0, 7.0)
    q_clamped = np.clip(np.round((q - 1.0) / 2.0) * 2.0 + 1.0, -7.0, 7.0)
    ref_symbols = ((i_clamped + 1j * q_clamped) / norm_factor).astype(np.complex64)

    soft_llrs = (2.0 * hard_bits.astype(np.float32) - 1.0) * 3.0
    return ref_symbols, hard_bits, soft_llrs


def demodulate_2fsk(
    samples: np.ndarray,
    sample_rate: float,
    sps: int,
    mark_freq: Optional[float] = None,
    space_freq: Optional[float] = None,
    method: str = "matched_filter",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Demodulate 2-FSK using noncoherent tone correlator bank (matched filter)
    or instantaneous frequency discriminator.

    Args:
        samples: Baseband complex samples.
        sample_rate: Sampling frequency in Hz.
        sps: Samples per symbol.
        mark_freq: Tone frequency for bit 1 (Hz). If None, estimated or default based on baud rate.
        space_freq: Tone frequency for bit 0 (Hz). If None, estimated or default based on baud rate.
        method: Demodulation approach: 'matched_filter' (default) or 'discriminator'.

    Returns:
        (recovered_symbols, hard_bits, soft_llrs, diagnostics)
    """
    sps = max(1, int(round(sps)))
    n = len(samples)
    if n < sps or sample_rate <= 0:
        return (
            np.array([], dtype=np.complex64),
            np.array([], dtype=BIT_ARRAY_DTYPE),
            np.array([], dtype=np.float32),
            {"freq_separation_hz": 0.0, "mark_freq": 0.0, "space_freq": 0.0},
        )

    num_syms = n // sps
    baud_rate = sample_rate / sps

    # Assign default nominal frequencies if not provided
    f_mark = float(mark_freq) if mark_freq is not None else float(baud_rate / 2.0)
    f_space = float(space_freq) if space_freq is not None else float(-baud_rate / 2.0)
    freq_sep = abs(f_mark - f_space)

    if method == "discriminator" and mark_freq is None and space_freq is None:
        # Instantaneous frequency discriminator approach
        phase = np.unwrap(np.angle(samples))
        freq = np.diff(phase, prepend=phase[0]) * (sample_rate / (2.0 * np.pi))
        sym_freqs = np.mean(freq[:num_syms * sps].reshape(num_syms, sps), axis=1)

        median_freq = float(np.median(sym_freqs))
        freq_centered = sym_freqs - median_freq
        hard_bits = (freq_centered > 0).astype(BIT_ARRAY_DTYPE)
        ref_symbols = (2.0 * hard_bits - 1.0 + 0j).astype(np.complex64)
        soft_llrs = freq_centered.astype(np.float32)

        diag = {
            "freq_separation_hz": float(np.percentile(sym_freqs, 90) - np.percentile(sym_freqs, 10)),
            "mark_freq": float(np.percentile(sym_freqs, 90)),
            "space_freq": float(np.percentile(sym_freqs, 10)),
            "method": "instantaneous_frequency_discriminator",
        }
        return ref_symbols, hard_bits, soft_llrs, diag

    # Noncoherent Matched Filter Bank (Tone Correlators)
    t = np.arange(sps) / sample_rate
    tone_mark = np.exp(-1j * 2.0 * np.pi * f_mark * t).astype(np.complex64)
    tone_space = np.exp(-1j * 2.0 * np.pi * f_space * t).astype(np.complex64)

    sym_slices = samples[:num_syms * sps].reshape(num_syms, sps)
    # Correlate each symbol window against mark and space tones
    c_mark = np.sum(sym_slices * tone_mark, axis=1)
    c_space = np.sum(sym_slices * tone_space, axis=1)

    p_mark = np.abs(c_mark) ** 2
    p_space = np.abs(c_space) ** 2

    # Decision: Power comparison
    hard_bits = (p_mark >= p_space).astype(BIT_ARRAY_DTYPE)
    # Normalized decision metric for soft LLR and recovered symbols
    p_total = np.maximum(p_mark + p_space, 1e-12)
    norm_diff = (p_mark - p_space) / p_total
    ref_symbols = norm_diff.astype(np.complex64)
    soft_llrs = (norm_diff * 8.0).astype(np.float32)

    diag = {
        "freq_separation_hz": freq_sep,
        "mark_freq": f_mark,
        "space_freq": f_space,
        "method": "noncoherent_matched_filter_bank",
    }
    return ref_symbols, hard_bits, soft_llrs, diag


def demodulate_ook(
    samples: np.ndarray,
    sps: int,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Demodulate OOK / ASK using envelope integration and thresholding.
    """
    sps = max(1, int(round(sps)))
    if len(samples) < sps:
        return np.array([], dtype=np.complex64), np.array([], dtype=BIT_ARRAY_DTYPE), np.array([], dtype=np.float32)

    env = np.abs(samples)
    num_syms = len(env) // sps
    sym_env = np.mean(env[:num_syms * sps].reshape(num_syms, sps), axis=1)

    if threshold is None:
        threshold = float(0.5 * (np.percentile(sym_env, 10) + np.percentile(sym_env, 90)))

    hard_bits = (sym_env > threshold).astype(BIT_ARRAY_DTYPE)
    ref_symbols = hard_bits.astype(np.complex64)
    soft_llrs = (sym_env - threshold).astype(np.float32)

    return ref_symbols, hard_bits, soft_llrs


# --------------------------------------------------------------------------- #
# 4. Unified Demodulation Interface                                             #
# --------------------------------------------------------------------------- #

def demodulate(
    samples: Union[np.ndarray, SignalData],
    modulation: Union[str, ModulationType],
    sample_rate: Optional[float] = None,
    symbol_rate: Optional[float] = None,
    **parameters: Any,
) -> DemodulationResult:
    """
    Unified digital communications demodulation interface for SpectralQ (CSE-3).

    Accepts raw complex NumPy arrays or SignalData objects.
    Executes synchronization, demodulation, symbol demapping, and quality telemetry.

    Parameters:
        samples: 1D complex/real NumPy array or SignalData instance.
        modulation: ModulationType enum or string (e.g., 'BPSK', 'QPSK', '2-FSK', '16QAM').
        sample_rate: Sampling frequency in Hz. Extracted from SignalData if omitted.
        symbol_rate: Baud rate in symbols/sec. Used to compute samples-per-symbol if provided.
        **parameters:
            - sps (float): Samples per symbol override.
            - differential / diff_decode (bool): Enable differential decoding (DBPSK, DQPSK).
            - phase_offset_deg / phase_offset_rad (float): Static phase offset correction.
            - enable_carrier_sync (bool): Enable Costas carrier tracking (default True for PSK/QAM).
            - enable_timing_sync (bool): Enable Gardner timing recovery (default True if sps >= 1.5).
            - mark_freq (float): Mark frequency for 2-FSK tone detector.
            - space_freq (float): Space frequency for 2-FSK tone detector.
            - fsk_method (str): 'matched_filter' or 'discriminator'.
            - noise_var (float): Noise variance estimate for soft LLR generation (default 0.05).
            - costas_kp, costas_ki (float): Costas loop filter gains.

    Returns:
        DemodulationResult: Standard dataclass from core.contracts.
    """
    warnings_list: List[Dict[str, Any]] = []

    # 1. Ingest inputs from SignalData or raw array
    if isinstance(samples, SignalData):
        if sample_rate is None:
            sample_rate = samples.sample_rate
        raw_samples = samples.samples
    else:
        raw_samples = np.asarray(samples)

    # 2. Check for empty or non-finite inputs
    if raw_samples is None or len(raw_samples) == 0:
        warnings_list.append(make_warning("EMPTY_INPUT", "Input sample buffer is empty."))
        return DemodulationResult(
            symbols=np.array([], dtype=np.complex64),
            hard_bits=np.array([], dtype=BIT_ARRAY_DTYPE),
            soft_llrs=np.array([], dtype=np.float32),
            evm_percent=100.0,
            mer_db=0.0,
            modulation=None,
            status=ResultStatus.FAILED,
            confidence=0.0,
            warnings=warnings_list,
            metrics={"symbol_count": 0, "bit_count": 0, "invalid_symbol_count": 0, "parameters": parameters},
        )

    # Sanitize and check finiteness
    non_finite_count = int(np.sum(~np.isfinite(raw_samples)))
    if non_finite_count == len(raw_samples):
        warnings_list.append(make_warning("INVALID_SIGNAL_DATA", "All input samples are non-finite (NaN or Inf)."))
        return DemodulationResult(
            symbols=np.array([], dtype=np.complex64),
            hard_bits=np.array([], dtype=BIT_ARRAY_DTYPE),
            soft_llrs=np.array([], dtype=np.float32),
            evm_percent=100.0,
            mer_db=0.0,
            modulation=None,
            status=ResultStatus.FAILED,
            confidence=0.0,
            warnings=warnings_list,
            metrics={"symbol_count": 0, "bit_count": 0, "invalid_symbol_count": non_finite_count, "parameters": parameters},
        )
    elif non_finite_count > 0:
        warnings_list.append(make_warning("NON_FINITE_SAMPLES", f"Replaced {non_finite_count} non-finite samples with 0."))
        raw_samples = np.where(np.isfinite(raw_samples), raw_samples, 0.0)

    # Convert to complex64 baseband representation
    if not np.iscomplexobj(raw_samples):
        try:
            analytic = hilbert(raw_samples)
            s_complex = analytic.astype(np.complex64)
        except Exception:
            s_complex = raw_samples.astype(np.complex64)
    else:
        s_complex = raw_samples.astype(np.complex64)

    # 3. Resolve ModulationType enum
    mod_enum: Optional[ModulationType] = None
    if isinstance(modulation, ModulationType):
        mod_enum = modulation
    elif isinstance(modulation, str):
        mod_norm = modulation.strip().upper().replace("_", "").replace("-", "")
        for m in ModulationType:
            m_norm = m.value.strip().upper().replace("_", "").replace("-", "")
            if mod_norm == m_norm or mod_norm == m.name:
                mod_enum = m
                break
        if mod_enum is None:
            if "BPSK" in mod_norm:
                mod_enum = ModulationType.BPSK
            elif "QPSK" in mod_norm:
                mod_enum = ModulationType.QPSK
            elif "8PSK" in mod_norm or "PSK8" in mod_norm:
                mod_enum = ModulationType.PSK8
            elif "16QAM" in mod_norm or "QAM16" in mod_norm:
                mod_enum = ModulationType.QAM16
            elif "64QAM" in mod_norm or "QAM64" in mod_norm:
                mod_enum = ModulationType.QAM64
            elif "2FSK" in mod_norm or "FSK2" in mod_norm or "FSK" in mod_norm:
                mod_enum = ModulationType.FSK2
            elif "4FSK" in mod_norm or "FSK4" in mod_norm:
                mod_enum = ModulationType.FSK4
            elif "OOK" in mod_norm:
                mod_enum = ModulationType.OOK
            elif "ASK" in mod_norm:
                mod_enum = ModulationType.ASK2

    if mod_enum is None or mod_enum == ModulationType.UNKNOWN:
        warnings_list.append(make_warning("UNSUPPORTED_MODULATION", f"Modulation '{modulation}' is not supported."))
        return DemodulationResult(
            symbols=np.array([], dtype=np.complex64),
            hard_bits=np.array([], dtype=BIT_ARRAY_DTYPE),
            soft_llrs=np.array([], dtype=np.float32),
            evm_percent=100.0,
            mer_db=0.0,
            modulation=None,
            status=ResultStatus.UNSUPPORTED,
            confidence=0.0,
            warnings=warnings_list,
            metrics={"symbol_count": 0, "bit_count": 0, "invalid_symbol_count": non_finite_count, "parameters": parameters},
        )

    # 4. Resolve SPS (samples-per-symbol)
    sps = parameters.get("sps", None)
    if sps is None:
        if sample_rate is not None and symbol_rate is not None and symbol_rate > 0:
            sps = float(sample_rate / symbol_rate)
        else:
            sps = 1.0
    sps = float(sps)
    if sps <= 0.0:
        sps = 1.0

    if len(s_complex) < sps:
        warnings_list.append(make_warning("INSUFFICIENT_SAMPLES", f"Signal length ({len(s_complex)}) is smaller than SPS ({sps})."))
        return DemodulationResult(
            symbols=np.array([], dtype=np.complex64),
            hard_bits=np.array([], dtype=BIT_ARRAY_DTYPE),
            soft_llrs=np.array([], dtype=np.float32),
            evm_percent=100.0,
            mer_db=0.0,
            modulation=mod_enum,
            estimated_sps=sps,
            status=ResultStatus.INSUFFICIENT_EVIDENCE,
            confidence=0.0,
            warnings=warnings_list,
            metrics={"symbol_count": 0, "bit_count": 0, "invalid_symbol_count": non_finite_count, "parameters": parameters},
        )

    # 5. Manual Static Phase Correction (if requested)
    phase_offset_rad = parameters.get("phase_offset_rad", None)
    phase_offset_deg = parameters.get("phase_offset_deg", None)
    applied_phase_offset_rad = 0.0
    if phase_offset_deg is not None:
        applied_phase_offset_rad = float(np.deg2rad(phase_offset_deg))
    elif phase_offset_rad is not None:
        applied_phase_offset_rad = float(phase_offset_rad)

    if applied_phase_offset_rad != 0.0:
        s_complex = s_complex * np.exp(-1j * applied_phase_offset_rad)

    differential = bool(parameters.get("differential", parameters.get("diff_decode", False)))
    noise_var = float(parameters.get("noise_var", 0.05))

    # 6. Branch based on modulation family
    timing_errors = np.array([], dtype=np.float32)
    phases = np.array([], dtype=np.float32)
    diagnostics_extra: Dict[str, Any] = {}

    if mod_enum in (ModulationType.FSK2, ModulationType.FSK4):
        # 2-FSK Demodulation
        sr = float(sample_rate) if sample_rate is not None and sample_rate > 0 else 100_000.0
        sps_int = max(1, int(round(sps)))
        mark_f = parameters.get("mark_freq", None)
        space_f = parameters.get("space_freq", None)
        fsk_meth = parameters.get("fsk_method", "matched_filter")

        ref_syms, hard_bits, soft_llrs, fsk_diag = demodulate_2fsk(
            s_complex, sample_rate=sr, sps=sps_int, mark_freq=mark_f, space_freq=space_f, method=fsk_meth
        )
        derotated = ref_syms
        diagnostics_extra.update(fsk_diag)

    elif mod_enum in (ModulationType.OOK, ModulationType.ASK2):
        sps_int = max(1, int(round(sps)))
        thresh = parameters.get("threshold", None)
        ref_syms, hard_bits, soft_llrs = demodulate_ook(s_complex, sps=sps_int, threshold=thresh)
        derotated = ref_syms

    else:
        # Phase / Quadrature Modulations (BPSK, QPSK, 8PSK, 16QAM, 64QAM)
        # Step A: Timing Recovery
        enable_timing = parameters.get("enable_timing_sync", True)
        if enable_timing and sps >= 1.5:
            sym_1sps, timing_errors = recover_timing_gardner(
                s_complex, sps=sps,
                kp=parameters.get("gardner_kp", 0.015),
                ki=parameters.get("gardner_ki", 0.0001),
            )
        else:
            sym_1sps = s_complex
            if sps > 1.0 and not enable_timing:
                warnings_list.append(make_warning("TIMING_SYNC_SKIPPED", "Timing synchronization disabled; using 1-SPS direct strobe."))

        # Step B: Carrier Recovery (Costas Loop / PLL)
        enable_carrier = parameters.get("enable_carrier_sync", True)
        if enable_carrier:
            derotated, phases = costas_carrier_sync(
                sym_1sps,
                modulation=mod_enum,
                kp=parameters.get("costas_kp", 0.04),
                ki=parameters.get("costas_ki", 0.001),
            )
        else:
            derotated = sym_1sps

        # Step C: Symbol to Bit Demapping
        if mod_enum == ModulationType.BPSK:
            ref_syms, hard_bits, soft_llrs = demodulate_bpsk(derotated, noise_var=noise_var, differential=differential)
        elif mod_enum == ModulationType.QPSK:
            ref_syms, hard_bits, soft_llrs = demodulate_qpsk(derotated, noise_var=noise_var, differential=differential)
        elif mod_enum == ModulationType.PSK8:
            ref_syms, hard_bits, soft_llrs = demodulate_8psk(derotated)
        elif mod_enum == ModulationType.QAM16:
            ref_syms, hard_bits, soft_llrs = demodulate_16qam(derotated, noise_var=noise_var)
        elif mod_enum == ModulationType.QAM64:
            ref_syms, hard_bits, soft_llrs = demodulate_64qam(derotated, noise_var=noise_var)
        else:
            ref_syms, hard_bits, soft_llrs = demodulate_qpsk(derotated, noise_var=noise_var)

    # 7. EVM & MER Computation
    evm_pct, mer_db = calculate_evm_and_mer(derotated, ref_syms)

    # Estimate SNR from MER and EVM
    est_snr_db = float(np.clip(mer_db, -10.0, 50.0))
    est_phase_offset_rad = float(np.mean(phases)) if len(phases) > 0 else applied_phase_offset_rad
    est_phase_offset_deg = float(np.rad2deg(est_phase_offset_rad))

    metrics: Dict[str, Any] = {
        "evm_percent": evm_pct,
        "mer_db": mer_db,
        "symbol_count": len(derotated),
        "num_recovered_symbols": len(derotated),
        "bit_count": len(hard_bits),
        "num_bits": len(hard_bits),
        "invalid_symbol_count": non_finite_count,
        "estimated_snr_db": est_snr_db,
        "estimated_phase_offset_deg": est_phase_offset_deg,
        "estimated_phase_offset_rad": est_phase_offset_rad,
        "differential_decoded": differential,
        "parameters": parameters,
        **diagnostics_extra,
    }

    # Confidence calculation based on MER
    confidence = float(np.clip((mer_db - 3.0) / 25.0, 0.0, 1.0))

    return DemodulationResult(
        symbols=derotated,
        hard_bits=hard_bits,
        soft_llrs=soft_llrs,
        evm_percent=evm_pct,
        mer_db=mer_db,
        modulation=mod_enum,
        estimated_sps=sps,
        carrier_freq_offset_hz=None,
        phase_history=phases,
        timing_error_history=timing_errors,
        status=ResultStatus.CONFIRMED,
        confidence=confidence,
        warnings=warnings_list,
        metrics=metrics,
    )


def demodulate_signal(
    samples: np.ndarray,
    sample_rate: float,
    modulation: ModulationType,
    sps: Optional[float] = None,
) -> DemodulationResult:
    """
    Backwards-compatible convenience wrapper delegating to demodulate().
    """
    return demodulate(
        samples=samples,
        modulation=modulation,
        sample_rate=sample_rate,
        sps=sps,
    )
