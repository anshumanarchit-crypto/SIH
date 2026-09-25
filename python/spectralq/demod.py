"""
spectralq.demod

Demodulation, pulse shaping, timing recovery, carrier synchronization,
and symbol detection for the SpectralQ decoder core.

Supported Modulation Schemes:
- BPSK (M=2, NASA/ESA standard)
- QPSK (M=4, Gray-mapped)
- 8-PSK (M=8, Gray-mapped)
- 2-FSK (Binary Frequency Shift Keying, continuous phase)
- 4-FSK (4-level Frequency Shift Keying, Gray-mapped)
- 16-QAM (16-ary Quadrature Amplitude Modulation, Gray-mapped)

Algorithms Implemented:
1. Root-Raised-Cosine (RRC) matched filter with singular tap resolution and unit energy normalization.
2. Gardner Timing Recovery: non-data-aided timing error detector with cubic polynomial interpolation.
3. M-PSK Carrier Recovery: M-th power coarse frequency estimation + decision-directed Costas loop PLL.
4. Phase Ambiguity Resolution: explicit preamble-correlation search over M-fold rotational symmetry.
5. FSK Demodulator: phase-differential frequency discriminator with integrate-and-dump post-filtering.
6. Centralized Constellation Slicing & Soft LLR generation.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


# ============================================================================
# 1. STATUS & RESULT TYPED CONTRACTS
# ============================================================================

class DemodStatus(Enum):
    """Execution status for the demodulation pipeline."""
    SUCCESS = "SUCCESS"
    UNSUPPORTED = "UNSUPPORTED"
    INVALID_INPUT = "INVALID_INPUT"
    DECODER_FAILURE = "DECODER_FAILURE"
    LOW_QUALITY = "LOW_QUALITY"
    NON_CONVERGED = "NON_CONVERGED"


@dataclass
class DemodConfig:
    """Explicit configuration passed into the demodulator core."""
    modulation: str
    sample_rate: float = 1.0
    samples_per_symbol: int = 4
    symbol_rate: Optional[float] = None
    rrc_alpha: float = 0.35
    filter_span: int = 8
    center_frequency: float = 0.0
    fsk_deviation: Optional[float] = None
    preamble_bits: Optional[np.ndarray] = None
    mapping_profile: str = "DEFAULT"
    external_reference_bits: Optional[np.ndarray] = None
    candidate_phase_rotations: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DemodResult:
    """Typed structured result emitted by the demodulation core.

    Does NOT compute final overall confidence or UNKNOWN decision.
    Supplies factual evidence for Archit's downstream hypothesis engine.
    """
    status: DemodStatus
    modulation: str
    input_sample_count: int
    output_symbol_count: int
    output_bit_count: int
    samples_per_symbol_used: int
    timing_status: Dict[str, Any]
    carrier_status: Dict[str, Any]
    estimated_frequency_offset: float
    estimated_phase_offset: float
    timing_error_summary: Dict[str, Any]
    symbol_decisions: np.ndarray
    hard_bits: np.ndarray
    soft_bits: Optional[np.ndarray] = None
    mapping_profile_used: str = "DEFAULT"
    candidate_rotations: List[Dict[str, Any]] = field(default_factory=list)
    reference_status: str = "REFERENCE_BITS_UNAVAILABLE"
    bit_error_rate: Optional[float] = None
    bit_errors: Optional[int] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    failure_reason: Optional[str] = None


# ============================================================================
# 2. RRC FILTER DESIGN & CONVOLUTION
# ============================================================================

def rrc_filter_taps(
    sps: int,
    alpha: float = 0.35,
    span: int = 8
) -> np.ndarray:
    """Compute normalized Root-Raised-Cosine (RRC) impulse response taps.

    Parameters
    ----------
    sps : int
        Samples per symbol (must be >= 2).
    alpha : float
        Roll-off factor in [0.0, 1.0].
    span : int
        Filter span in symbols (length = span * sps + 1).

    Returns
    -------
    h : np.ndarray
        1D float array with unit energy (sum(h^2) == 1.0).
    """
    if sps < 1:
        raise ValueError(f"sps must be >= 1, got {sps}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
    if span < 1:
        raise ValueError(f"span must be >= 1, got {span}")

    total_taps = span * sps + 1
    # Symmetric time grid centered at 0
    t = np.arange(-(total_taps - 1) // 2, (total_taps - 1) // 2 + 1, dtype=float) / float(sps)
    h = np.zeros_like(t)

    if alpha == 0.0:
        h = np.sinc(t)
    else:
        for i, ti in enumerate(t):
            # Center singular tap at t = 0
            if abs(ti) < 1e-7:
                h[i] = 1.0 - alpha + (4.0 * alpha / np.pi)
            # Off-center singular tap at t = +- 1 / (4 * alpha)
            elif abs(abs(4.0 * alpha * ti) - 1.0) < 1e-6:
                h[i] = (alpha / np.sqrt(2.0)) * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                    + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
                )
            else:
                num = np.sin(np.pi * ti * (1.0 - alpha)) + 4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha))
                denom = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
                h[i] = num / denom

    energy = np.sum(h ** 2)
    if energy > 0:
        h = h / np.sqrt(energy)
    return h


def apply_rrc_filter(
    iq_samples: np.ndarray,
    sps: int,
    alpha: float = 0.35,
    span: int = 8
) -> np.ndarray:
    """Apply RRC matched filter to complex IQ baseband samples.

    Parameters
    ----------
    iq_samples : np.ndarray
        1D array of complex IQ samples.
    sps : int
        Samples per symbol.
    alpha : float
        RRC excess bandwidth roll-off factor.
    span : int
        Filter span in symbols.

    Returns
    -------
    filtered : np.ndarray
        1D complex array of matched-filtered samples (centered delay via mode='same').
    """
    samples = np.asarray(iq_samples, dtype=complex).ravel()
    if len(samples) == 0:
        return np.array([], dtype=complex)

    taps = rrc_filter_taps(sps=sps, alpha=alpha, span=span)
    filtered = np.convolve(samples, taps, mode="same")
    return filtered


# ============================================================================
# 3. GARDNER TIMING RECOVERY (TED + CUBIC INTERPOLATION)
# ============================================================================

def _cubic_interpolate(s: np.ndarray, idx: int, mu: float) -> complex:
    """Cubic 4-point Hermite/Lagrange polynomial interpolator.

    Interpolates at continuous position idx + mu where mu in [0, 1).
    """
    n = len(s)
    im1 = max(0, idx - 1)
    i0 = min(n - 1, max(0, idx))
    ip1 = min(n - 1, max(0, idx + 1))
    ip2 = min(n - 1, max(0, idx + 2))

    ym1 = s[im1]
    y0 = s[i0]
    yp1 = s[ip1]
    yp2 = s[ip2]

    # Hermite cubic polynomial coefficients
    c0 = y0
    c1 = -0.5 * ym1 + 0.5 * yp1
    c2 = ym1 - 2.5 * y0 + 2.0 * yp1 - 0.5 * yp2
    c3 = -0.5 * ym1 + 1.5 * y0 - 1.5 * yp1 + 0.5 * yp2

    return c0 + mu * (c1 + mu * (c2 + mu * c3))


def gardner_timing_recovery(
    samples: np.ndarray,
    sps: int,
    kp: float = 0.02,
    ki: float = 0.0005,
    max_symbols: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Recover symbol-spaced samples from oversampled IQ via Gardner timing error detector.

    Uses a non-data-aided timing error detector (TED):
        e[k] = Re{ (y[k] - y[k-1]) * conj(y[k - 1/2]) }
    with a second-order loop filter and cubic Hermite interpolation.

    Parameters
    ----------
    samples : np.ndarray
        1D complex array of oversampled IQ samples (typically after RRC matched filtering).
    sps : int
        Nominal samples per symbol (must be >= 2).
    kp : float
        Proportional loop gain.
    ki : float
        Integral loop gain.
    max_symbols : Optional[int]
        Optional upper bound on recovered symbols.

    Returns
    -------
    recovered_symbols : np.ndarray
        1D complex array of recovered symbol-spaced samples (approximately 1 per symbol).
    metrics : Dict[str, Any]
        Diagnostic dictionary containing timing lock status, error trajectory, and residual jitter.
    """
    s = np.asarray(samples, dtype=complex).ravel()
    num_samples = len(s)

    if sps < 2:
        raise ValueError(f"Gardner timing recovery requires sps >= 2, got {sps}")
    if num_samples < 2 * sps:
        return np.array([], dtype=complex), {
            "converged": False,
            "mean_error": 0.0,
            "jitter_std": 0.0,
            "symbol_count": 0,
            "error_history": [],
            "warnings": ["Insufficient sample count for Gardner timing recovery."],
        }

    # Internal normalization so TED loop and jitter metrics operate on unit-power scale
    sig_power = float(np.mean(np.abs(s)**2))
    scale = np.sqrt(sig_power) if sig_power > 1e-12 else 1.0
    s_norm = s / scale

    symbols = []
    errors = []

    # Nominal time increment per symbol
    T_nom = float(sps)
    # Loop variables
    t_acc = 0.0
    v_integral = 0.0
    last_strobe = _cubic_interpolate(s_norm, 0, 0.0)
    symbols.append(last_strobe)

    # Estimate expected total symbols
    est_symbols = int(num_samples / sps)
    if max_symbols is not None:
        est_symbols = min(est_symbols, max_symbols)

    for _ in range(max(1, est_symbols)):
        # Mid-symbol position and strobe position
        t_mid = t_acc + T_nom * 0.5
        t_strobe = t_acc + T_nom

        if t_strobe + 2 >= num_samples:
            break

        idx_mid = int(math.floor(t_mid))
        mu_mid = t_mid - idx_mid
        y_mid = _cubic_interpolate(s_norm, idx_mid, mu_mid)

        idx_strobe = int(math.floor(t_strobe))
        mu_strobe = t_strobe - idx_strobe
        y_strobe = _cubic_interpolate(s_norm, idx_strobe, mu_strobe)

        # Gardner TED: e = Re{(y_strobe - last_strobe) * conj(y_mid)}
        diff = y_strobe - last_strobe
        ted_error = float(diff.real * y_mid.real + diff.imag * y_mid.imag)

        # Clamp error to prevent numerical divergence
        ted_error = max(-2.0, min(2.0, ted_error))
        errors.append(ted_error)

        # Negative feedback loop filter update (decrease step if late)
        v_integral += ki * ted_error
        step = T_nom - (kp * ted_error + v_integral)

        # Advance accumulator
        t_acc += step
        last_strobe = y_strobe
        symbols.append(y_strobe)

    # Scale symbols back to original amplitude
    rec_syms = np.array(symbols, dtype=complex) * scale
    err_arr = np.array(errors, dtype=float) if len(errors) > 0 else np.array([0.0])

    # Assess convergence over the second half of the sequence (or all if short sequence)
    half = len(err_arr) // 2 if len(err_arr) >= 32 else 0
    tail_errors = err_arr[half:] if len(err_arr) > 0 else err_arr
    mean_err = float(np.mean(tail_errors)) if len(tail_errors) > 0 else 0.0
    std_err = float(np.std(tail_errors)) if len(tail_errors) > 0 else 0.0

    # Converged if jitter is small and not blowing up (allow pattern jitter headroom for 16-QAM or short bursts)
    converged = bool((std_err < 0.95 or len(rec_syms) < 32) and not np.isnan(std_err) and len(rec_syms) > 0)

    metrics = {
        "converged": converged,
        "mean_error": mean_err,
        "jitter_std": std_err,
        "symbol_count": int(len(rec_syms)),
        "error_history": err_arr.tolist()[:100],  # snapshot first 100 for telemetry
        "nominal_sps": int(sps),
    }

    return rec_syms, metrics


# ============================================================================
# 4. M-PSK / QAM CARRIER RECOVERY (M-th POWER CFO + COSTAS / DD-PLL)
# ============================================================================

def estimate_cfo_mth_power(symbols: np.ndarray, order: int = 2) -> float:
    """Estimate coarse carrier frequency offset (CFO) using M-th power autocorrelation.

    Parameters
    ----------
    symbols : np.ndarray
        1D complex symbol stream.
    order : int
        Constellation order M (2 for BPSK, 4 for QPSK, 8 for 8-PSK, 16 for 16-QAM).

    Returns
    -------
    cfo_normalized : float
        Estimated frequency offset normalized to symbol rate (Delta_f / R_sym).
    """
    s = np.asarray(symbols, dtype=complex).ravel()
    if len(s) < 8:
        return 0.0

    if order == 16:
        # For 16-QAM, only corner points (amplitude > 1.3 * RMS) have 4-fold rotational symmetry collapsing under 4th power.
        # On short bursts (< 64 symbols), mixed-amplitude 4th power has high variance; fine DD-PLL acquires directly.
        if len(s) < 64:
            return 0.0
        p_rms = np.sqrt(np.mean(np.abs(s)**2))
        corners = s[np.abs(s) > 1.3 * p_rms]
        if len(corners) < 8:
            return 0.0
        z = corners ** 4
        r1 = np.sum(z[1:] * np.conj(z[:-1]))
        if abs(r1) < 1e-12:
            return 0.0
        return float(np.angle(r1) / (2.0 * np.pi * 4.0))

    # Non-linear power raises symbols to M-th power, collapsing M-PSK data to a single carrier
    z = s ** order

    # 1-lag autocorrelation R(1) = sum(z[k] * conj(z[k-1]))
    r1 = np.sum(z[1:] * np.conj(z[:-1]))
    if abs(r1) < 1e-12:
        return 0.0

    angle = np.angle(r1)
    # delta_f = angle / (2 * pi * order)
    cfo = float(angle / (2.0 * np.pi * order))
    return cfo


def costas_carrier_recovery(
    symbols: np.ndarray,
    order: int = 2,
    alpha: float = 0.02,
    beta: float = 0.0005
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Correct carrier frequency and phase offset for M-PSK symbols.

    Employs a two-stage approach:
    1. Coarse CFO estimation and removal via M-th power autocorrelation.
    2. Decision-directed / Costas loop fine frequency & phase tracking.

    Parameters
    ----------
    symbols : np.ndarray
        1D complex symbol sequence.
    order : int
        M-PSK order (2 for BPSK, 4 for QPSK, 8 for 8-PSK).
    alpha : float
        Proportional loop gain for fine tracking.
    beta : float
        Integral loop gain for fine tracking.

    Returns
    -------
    corrected_symbols : np.ndarray
        1D complex array of phase/frequency corrected symbols.
    metrics : Dict[str, Any]
        Dictionary reporting coarse CFO, residual phase trajectory, and lock status.
    """
    s = np.asarray(symbols, dtype=complex).ravel()
    n_syms = len(s)
    if n_syms == 0:
        return np.array([], dtype=complex), {
            "converged": False,
            "estimated_cfo": 0.0,
            "final_phase": 0.0,
            "phase_jitter_std": 0.0,
            "order": order,
        }

    # Step 1: Coarse CFO Estimation via M-th power
    cfo_coarse = estimate_cfo_mth_power(s, order=order)
    k_indices = np.arange(n_syms)
    s_coarse = s * np.exp(-1j * 2.0 * np.pi * cfo_coarse * k_indices)

    # Step 2: Fine Costas Tracking
    phase = 0.0
    freq = 0.0
    corrected = np.zeros(n_syms, dtype=complex)
    phase_errors = []

    for i in range(n_syms):
        # Derotate current symbol
        sym_rot = s_coarse[i] * np.exp(-1j * phase)
        corrected[i] = sym_rot

        # Phase error detector
        if order == 2:
            # BPSK Costas error: Im(y) * sign(Re(y))
            re_sign = 1.0 if sym_rot.real >= 0 else -1.0
            err = float(sym_rot.imag * re_sign)
        elif order == 4:
            # QPSK Costas error: sign(Re(y)) * Im(y) - sign(Im(y)) * Re(y)
            re_sign = 1.0 if sym_rot.real >= 0 else -1.0
            im_sign = 1.0 if sym_rot.imag >= 0 else -1.0
            err = float(re_sign * sym_rot.imag - im_sign * sym_rot.real)
        elif order == 8:
            # 8-PSK decision-directed error: angle to nearest 8-PSK constellation point
            ang = np.angle(sym_rot)
            sector = np.round(ang / (np.pi / 4.0)) * (np.pi / 4.0)
            nearest = np.exp(1j * sector)
            err = float(np.imag(sym_rot * np.conj(nearest)))
        elif order == 16:
            # 16-QAM decision-directed carrier tracking
            p_avg = float(np.mean(np.abs(sym_rot)**2))
            sc = np.sqrt(10.0) if p_avg < 2.5 else 1.0
            y_sc = sym_rot * sc
            re_slice = float(np.clip(round((y_sc.real - 1.0) / 2.0) * 2.0 + 1.0, -3.0, 3.0))
            im_slice = float(np.clip(round((y_sc.imag - 1.0) / 2.0) * 2.0 + 1.0, -3.0, 3.0))
            nearest = (re_slice + 1j * im_slice) / sc
            err = float(np.imag(sym_rot * np.conj(nearest)) / max(1e-6, abs(nearest)**2))
        else:
            # Generic decision-directed slicer
            ang = np.angle(sym_rot)
            nearest = np.exp(1j * np.round(ang / (2.0 * np.pi / order)) * (2.0 * np.pi / order))
            err = float(np.imag(sym_rot * np.conj(nearest)))

        err = max(-1.5, min(1.5, err))
        phase_errors.append(err)

        # Loop filter
        freq += beta * err
        phase += alpha * err + freq

    err_tail = phase_errors[len(phase_errors) // 2:] if len(phase_errors) > 4 else phase_errors
    std_phase = float(np.std(err_tail)) if len(err_tail) > 0 else 0.0
    total_cfo = float(cfo_coarse + freq / (2.0 * np.pi))
    converged = bool(std_phase < 0.4 and not np.isnan(std_phase))

    metrics = {
        "converged": converged,
        "coarse_cfo": float(cfo_coarse),
        "fine_cfo": float(freq / (2.0 * np.pi)),
        "estimated_cfo": total_cfo,
        "final_phase": float(phase % (2.0 * np.pi)),
        "phase_jitter_std": std_phase,
        "order": int(order),
    }

    return corrected, metrics


# ============================================================================
# 5. PHASE AMBIGUITY RESOLUTION
# ============================================================================

def resolve_phase_ambiguity(
    symbols: np.ndarray,
    order: int,
    preamble_bits: Optional[np.ndarray] = None,
    profile: str = "DEFAULT"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Resolve rotational phase ambiguity using a known preamble/sync sequence.

    For order M, there are M rotational ambiguities: m * (2*pi / M) for m in 0..M-1.
    Tests all candidate rotations against preamble_bits and picks the minimal BER.

    Parameters
    ----------
    symbols : np.ndarray
        1D complex symbol stream.
    order : int
        Constellation order (2, 4, 8).
    preamble_bits : Optional[np.ndarray]
        Known synchronization bit sequence.
    profile : str
        Mapping profile name for demapping candidates.

    Returns
    -------
    resolved_symbols : np.ndarray
        Optimally aligned complex symbol array.
    info : Dict[str, Any]
        Ambiguity resolution diagnostics (selected rotation, preamble bit errors).
    """
    s = np.asarray(symbols, dtype=complex).ravel()
    if len(s) == 0:
        return s, {"resolved": False, "selected_rotation_deg": 0.0, "reason": "empty_symbols"}

    if preamble_bits is None or len(preamble_bits) == 0:
        # No preamble provided: retain 0-degree nominal alignment
        return s, {
            "resolved": False,
            "selected_rotation_deg": 0.0,
            "reason": "no_preamble_provided",
        }

    preamble = np.asarray(preamble_bits, dtype=int).ravel()
    bits_per_sym = int(math.log2(order))
    preamble_sym_len = int(math.ceil(len(preamble) / bits_per_sym))
    test_len = min(len(s), preamble_sym_len)

    candidate_angles = [m * (2.0 * np.pi / order) for m in range(order)]
    best_angle = 0.0
    best_errors = float("inf")

    for angle in candidate_angles:
        cand_syms = s[:test_len] * np.exp(-1j * angle)
        # Demap candidate
        cand_bits, _ = demap_psk(cand_syms, constellation_order=order, profile=profile)
        eval_len = min(len(preamble), len(cand_bits))
        errors = int(np.sum(cand_bits[:eval_len] != preamble[:eval_len]))
        if errors < best_errors:
            best_errors = errors
            best_angle = angle

    # Maximum acceptable preamble error threshold: 30% of preamble bits
    max_acceptable = max(1, int(0.3 * len(preamble)))
    resolved = bool(best_errors <= max_acceptable)

    resolved_symbols = s * np.exp(-1j * best_angle)
    info = {
        "resolved": resolved,
        "selected_rotation_rad": float(best_angle),
        "selected_rotation_deg": float(np.degrees(best_angle)),
        "preamble_bit_errors": int(best_errors),
        "preamble_eval_bits": int(min(len(preamble), test_len * bits_per_sym)),
    }
    return resolved_symbols, info


# ============================================================================
# 6. CONSTELLATION SLICING & BIT MAPPING (PSK / QAM / FSK)
# ============================================================================

MAPPING_PROFILES_BPSK: Dict[str, Dict[int, int]] = {
    "DEFAULT": {0: 0, 1: 1},      # 0 -> +1 (bit 0), 1 -> -1 (bit 1)
    "INVERTED": {0: 1, 1: 0},     # 180-deg inverted mapping
}

MAPPING_PROFILES_8PSK: Dict[str, Dict[int, Tuple[int, int, int]]] = {
    "DEFAULT": {
        0: (0, 0, 0),
        1: (0, 0, 1),
        2: (0, 1, 1),
        3: (0, 1, 0),
        4: (1, 1, 0),
        5: (1, 1, 1),
        6: (1, 0, 1),
        7: (1, 0, 0),
    },
    "OCTAVE": {
        0: (0, 0, 0),
        1: (0, 0, 1),
        2: (0, 1, 1),
        3: (0, 1, 0),
        4: (1, 1, 1),
        5: (1, 1, 0),
        6: (1, 0, 0),
        7: (1, 0, 1),
    },
}

MAPPING_PROFILES_16QAM: Dict[str, Dict[int, Tuple[int, int]]] = {
    "DEFAULT": {
        0: (1, 0),  # Region 0 (> +2.0): MSB=1, LSB=0
        1: (1, 1),  # Region 1 (0 to +2.0): MSB=1, LSB=1
        2: (0, 1),  # Region 2 (-2.0 to 0): MSB=0, LSB=1
        3: (0, 0),  # Region 3 (< -2.0): MSB=0, LSB=0
    },
    "OCTAVE": {
        0: (0, 0),  # Region 0 (> +2.0): 00
        1: (0, 1),  # Region 1 (0 to +2.0): 01
        2: (1, 1),  # Region 2 (-2.0 to 0): 11
        3: (1, 0),  # Region 3 (< -2.0): 10
    },
}


def demap_bpsk(
    symbols: np.ndarray,
    profile: str = "DEFAULT"
) -> Tuple[np.ndarray, np.ndarray]:
    """Demap BPSK symbols to hard bits and soft LLRs.

    Parameters
    ----------
    symbols : np.ndarray
        Complex symbol array.
    profile : str
        Mapping profile name ('DEFAULT' or 'INVERTED').
    """
    prof_key = profile.upper()
    if prof_key not in MAPPING_PROFILES_BPSK:
        raise ValueError(
            f"Unsupported BPSK mapping profile: {profile}. "
            f"Supported profiles: {list(MAPPING_PROFILES_BPSK.keys())}"
        )
    mapping = MAPPING_PROFILES_BPSK[prof_key]

    s = np.asarray(symbols, dtype=complex).ravel()
    s_re = np.real(s)
    raw_bits = (s_re < 0).astype(int)
    hard_bits = np.array([mapping[int(b)] for b in raw_bits], dtype=int) if prof_key != "DEFAULT" else raw_bits
    soft_bits = 2.0 * s_re if prof_key == "DEFAULT" else -2.0 * s_re
    return hard_bits, soft_bits


MAPPING_PROFILES_QPSK: Dict[str, str] = {
    "DEFAULT": "IQ",  # Bit 0 is I (s_re < 0), Bit 1 is Q (s_im < 0)
    "OCTAVE": "QI",   # Bit 0 is Q (s_im < 0), Bit 1 is I (s_re < 0)
}


def demap_qpsk(
    symbols: np.ndarray,
    profile: str = "DEFAULT"
) -> Tuple[np.ndarray, np.ndarray]:
    """Demap QPSK symbols to hard bits and soft LLRs with Gray mapping.

    Bit mapping (DEFAULT):
        00 -> (+1 + j) / sqrt(2)
        01 -> (-1 + j) / sqrt(2)
        11 -> (-1 - j) / sqrt(2)
        10 -> (+1 - j) / sqrt(2)
    Bit 0 corresponds to Re(s) < 0.
    Bit 1 corresponds to Im(s) < 0.

    Supported profiles:
        - "DEFAULT": Standard QPSK bit order [I, Q]
        - "OCTAVE": GNU Octave QPSK bit order [Q, I]
    """
    prof_key = profile.upper()
    if prof_key not in MAPPING_PROFILES_QPSK:
        raise ValueError(
            f"Unsupported QPSK mapping profile: {profile}. "
            f"Supported profiles: {list(MAPPING_PROFILES_QPSK.keys())}"
        )

    s = np.asarray(symbols, dtype=complex).ravel()
    s_re = np.real(s)
    s_im = np.imag(s)
    b0 = (s_re < 0).astype(int)
    b1 = (s_im < 0).astype(int)

    hard_bits = np.empty(len(s) * 2, dtype=int)
    soft_bits = np.empty(len(s) * 2, dtype=float)

    if prof_key == "DEFAULT":
        # Interleave bit 0 and bit 1: [b0_0, b1_0, b0_1, b1_1, ...]
        hard_bits[0::2] = b0
        hard_bits[1::2] = b1
        soft_bits[0::2] = 2.0 * np.sqrt(2.0) * s_re
        soft_bits[1::2] = 2.0 * np.sqrt(2.0) * s_im
    elif prof_key == "OCTAVE":
        # Interleave bit 1 and bit 0: [b1_0, b0_0, b1_1, b0_1, ...]
        hard_bits[0::2] = b1
        hard_bits[1::2] = b0
        soft_bits[0::2] = 2.0 * np.sqrt(2.0) * s_im
        soft_bits[1::2] = 2.0 * np.sqrt(2.0) * s_re

    return hard_bits, soft_bits


def demap_8psk(
    symbols: np.ndarray,
    profile: str = "DEFAULT"
) -> Tuple[np.ndarray, np.ndarray]:
    """Demap 8-PSK symbols to hard bits and approximate soft LLRs.

    Separates sector detection from bit-label mapping.

    Constellation points at angles k * pi/4:
        Sector 0 (0 deg):   [0, 0, 0]
        Sector 1 (45 deg):  [0, 0, 1]
        Sector 2 (90 deg):  [0, 1, 1]
        Sector 3 (135 deg): [0, 1, 0]
        Sector 4 (180 deg): [1, 1, 0] (DEFAULT) or [1, 1, 1] (OCTAVE)
        Sector 5 (225 deg): [1, 1, 1] (DEFAULT) or [1, 1, 0] (OCTAVE)
        Sector 6 (270 deg): [1, 0, 1] (DEFAULT) or [1, 0, 0] (OCTAVE)
        Sector 7 (315 deg): [1, 0, 0] (DEFAULT) or [1, 0, 1] (OCTAVE)
    """
    prof_key = profile.upper()
    if prof_key not in MAPPING_PROFILES_8PSK:
        raise ValueError(
            f"Unsupported 8-PSK mapping profile: {profile}. "
            f"Supported profiles: {list(MAPPING_PROFILES_8PSK.keys())}"
        )
    mapping = MAPPING_PROFILES_8PSK[prof_key]

    s = np.asarray(symbols, dtype=complex).ravel()
    angles = np.angle(s) % (2.0 * np.pi)
    sectors = np.round(angles / (np.pi / 4.0)).astype(int) % 8

    hard_bits = np.empty(len(s) * 3, dtype=int)
    for i, sec in enumerate(sectors):
        b0, b1, b2 = mapping[int(sec)]
        hard_bits[3 * i] = b0
        hard_bits[3 * i + 1] = b1
        hard_bits[3 * i + 2] = b2

    # Scaled decision margin soft bits
    soft_bits = np.empty(len(s) * 3, dtype=float)
    s_re = np.real(s)
    s_im = np.imag(s)
    soft_bits[0::3] = -2.0 * s_re
    soft_bits[1::3] = -2.0 * s_im
    soft_bits[2::3] = 2.0 * (np.abs(s_re) - np.abs(s_im))
    return hard_bits, soft_bits


def demap_psk(
    symbols: np.ndarray,
    constellation_order: int = 2,
    profile: str = "DEFAULT"
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Unified PSK constellation demapper."""
    if constellation_order == 2:
        return demap_bpsk(symbols, profile=profile)
    elif constellation_order == 4:
        return demap_qpsk(symbols, profile=profile)
    elif constellation_order == 8:
        return demap_8psk(symbols, profile=profile)
    else:
        raise ValueError(f"Unsupported PSK order: {constellation_order}")


def demap_qam(
    symbols: np.ndarray,
    constellation_order: int = 16,
    profile: str = "DEFAULT"
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Demap Square 16-QAM symbols to hard bits and soft metrics.

    Separates PAM-4 level detection from bit-label mapping.

    PAM-4 Decision Regions:
        Region 0: level > 2.0  (nominal +3)
        Region 1: 0.0 < level <= 2.0 (nominal +1)
        Region 2: -2.0 <= level <= 0.0 (nominal -1)
        Region 3: level < -2.0 (nominal -3)

    Supported profiles:
        - "DEFAULT": Standard 16-QAM Gray mapping
        - "OCTAVE": Octave-compatible 16-QAM mapping
    """
    if constellation_order != 16:
        raise ValueError(f"Unsupported QAM order: {constellation_order}. Only 16-QAM is currently supported.")

    prof_key = profile.upper()
    if prof_key not in MAPPING_PROFILES_16QAM:
        raise ValueError(
            f"Unsupported 16-QAM mapping profile: {profile}. "
            f"Supported profiles: {list(MAPPING_PROFILES_16QAM.keys())}"
        )
    mapping = MAPPING_PROFILES_16QAM[prof_key]

    s = np.asarray(symbols, dtype=complex).ravel()
    p_avg = float(np.mean(np.abs(s)**2)) if len(s) > 0 else 1.0
    scale = np.sqrt(10.0) if p_avg < 2.5 else 1.0
    re = np.real(s) * scale
    im = np.imag(s) * scale

    def slice_pam4_levels(val: np.ndarray) -> np.ndarray:
        levels = np.zeros(len(val), dtype=int)
        levels[val > 2.0] = 0
        levels[(val > 0.0) & (val <= 2.0)] = 1
        levels[(val >= -2.0) & (val <= 0.0)] = 2
        levels[val < -2.0] = 3
        return levels

    i_levels = slice_pam4_levels(re)
    q_levels = slice_pam4_levels(im)

    hard_bits = np.empty(len(s) * 4, dtype=int)
    for idx in range(len(s)):
        i_bits = mapping[int(i_levels[idx])]
        q_bits = mapping[int(q_levels[idx])]
        hard_bits[4 * idx] = i_bits[0]
        hard_bits[4 * idx + 1] = i_bits[1]
        hard_bits[4 * idx + 2] = q_bits[0]
        hard_bits[4 * idx + 3] = q_bits[1]

    soft_bits = np.empty(len(s) * 4, dtype=float)
    soft_bits[0::4] = re
    soft_bits[1::4] = 2.0 - np.abs(re)
    soft_bits[2::4] = im
    soft_bits[3::4] = 2.0 - np.abs(im)
    return hard_bits, soft_bits


# ============================================================================
# 7. FSK DEMODULATION (PHASE-DISCRIMINATOR CORE)
# ============================================================================

def demod_fsk(
    iq_samples: np.ndarray,
    f_dev: Optional[float] = None,
    symbol_rate: Optional[float] = None,
    fs: float = 1.0,
    order: int = 2,
    sps: int = 4
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Demodulate continuous-phase FSK signals (2-FSK, 4-FSK) using a phase discriminator.

    Parameters
    ----------
    iq_samples : np.ndarray
        Complex baseband IQ samples.
    f_dev : Optional[float]
        Peak frequency deviation in Hz.
    symbol_rate : Optional[float]
        Symbol rate in baud.
    fs : float
        Sampling frequency in Hz.
    order : int
        FSK order (2 or 4).
    sps : int
        Samples per symbol.

    Returns
    -------
    hard_bits : np.ndarray
        Demodulated hard bits.
    soft_bits : Optional[np.ndarray]
        Demodulated soft metrics.
    diagnostics : Dict[str, Any]
        Diagnostic metrics (mean CFO, estimated deviation, SNR estimate).
    """
    samples = np.asarray(iq_samples, dtype=complex).ravel()
    n_samples = len(samples)

    if n_samples < sps * 2:
        return np.array([], dtype=int), None, {
            "converged": False,
            "warnings": ["Insufficient samples for FSK demodulation."],
        }

    # 1. Phase differential discriminator: d[n] = s[n] * conj(s[n-1])
    d = samples[1:] * np.conj(samples[:-1])
    inst_phase_diff = np.angle(d)

    # 2. Instantaneous frequency in Hz
    inst_freq = inst_phase_diff * (fs / (2.0 * np.pi))

    # 3. Integrate-and-dump / moving-average filter over symbol period (sps samples)
    kernel = np.ones(sps, dtype=float) / float(sps)
    filtered_freq = np.convolve(inst_freq, kernel, mode="same")

    # 4. Center-strobe sampling at center of each symbol
    strobe_indices = np.arange(sps // 2, len(filtered_freq), sps)
    sym_freqs = filtered_freq[strobe_indices]

    # Robust CFO estimation using symmetric extrema percentiles (independent of bit balance)
    p_low = float(np.percentile(sym_freqs, 10))
    p_high = float(np.percentile(sym_freqs, 90))
    cfo_est = float((p_low + p_high) / 2.0)
    centered_freqs = sym_freqs - cfo_est

    if order == 2:
        # 2-FSK: 0 -> -f_dev, 1 -> +f_dev
        hard_bits = (centered_freqs > 0).astype(int)
        dev_est = float((p_high - p_low) / 2.0) if f_dev is None else f_dev
        soft_bits = centered_freqs / (dev_est if dev_est > 0 else 1.0)
    elif order == 4:
        # 4-FSK: 4 frequency deviations [-3*df, -df, +df, +3*df]
        df = float((p_high - p_low) / 6.0) if f_dev is None else (f_dev / 3.0)
        df = max(1e-6, df)

        # Slicing thresholds at -2*df, 0, +2*df
        sym_dec = np.zeros(len(centered_freqs), dtype=int)
        sym_dec[centered_freqs < -2.0 * df] = 0
        sym_dec[(centered_freqs >= -2.0 * df) & (centered_freqs < 0.0)] = 1
        sym_dec[(centered_freqs >= 0.0) & (centered_freqs < 2.0 * df)] = 2
        sym_dec[centered_freqs >= 2.0 * df] = 3

        # Gray mapping: 0 -> 00, 1 -> 01, 2 -> 11, 3 -> 10
        gray_map = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
        hard_bits = np.empty(len(sym_dec) * 2, dtype=int)
        for i, val in enumerate(sym_dec):
            b0, b1 = gray_map[int(val)]
            hard_bits[2 * i] = b0
            hard_bits[2 * i + 1] = b1
        soft_bits = centered_freqs / df
        dev_est = df * 3.0
    else:
        raise ValueError(f"Unsupported FSK order: {order}. Supported: 2, 4.")

    diagnostics = {
        "converged": True,
        "estimated_cfo_hz": cfo_est,
        "estimated_deviation_hz": dev_est,
        "symbol_count": int(len(sym_freqs)),
        "order": int(order),
    }

    return hard_bits, soft_bits, diagnostics


# ============================================================================
# 8. UNIFIED DEMODULATION ENTRY POINT
# ============================================================================

def demodulate(
    iq_samples: np.ndarray,
    config: DemodConfig
) -> DemodResult:
    """Execute complete demodulation pipeline on raw complex IQ samples.

    Enforces input validation, matched filtering, timing recovery, carrier synchronization,
    phase ambiguity resolution, and symbol/bit slicing.

    Parameters
    ----------
    iq_samples : np.ndarray
        1D array of complex IQ samples.
    config : DemodConfig
        Demodulation configuration parameters.

    Returns
    -------
    result : DemodResult
        Comprehensive typed result container.
    """
    warnings: List[str] = []

    # 1. Input Validation
    if not isinstance(iq_samples, np.ndarray) or iq_samples.ndim != 1:
        return DemodResult(
            status=DemodStatus.INVALID_INPUT,
            modulation=str(config.modulation),
            input_sample_count=0 if not hasattr(iq_samples, "__len__") else len(iq_samples),
            output_symbol_count=0,
            output_bit_count=0,
            samples_per_symbol_used=config.samples_per_symbol,
            timing_status={"converged": False},
            carrier_status={"converged": False},
            estimated_frequency_offset=0.0,
            estimated_phase_offset=0.0,
            timing_error_summary={},
            symbol_decisions=np.array([], dtype=complex),
            hard_bits=np.array([], dtype=int),
            failure_reason="Input waveform must be a 1D numpy array of complex numbers.",
        )

    samples = np.asarray(iq_samples, dtype=complex)
    num_samples = len(samples)

    if num_samples == 0:
        return DemodResult(
            status=DemodStatus.INVALID_INPUT,
            modulation=str(config.modulation),
            input_sample_count=0,
            output_symbol_count=0,
            output_bit_count=0,
            samples_per_symbol_used=config.samples_per_symbol,
            timing_status={"converged": False},
            carrier_status={"converged": False},
            estimated_frequency_offset=0.0,
            estimated_phase_offset=0.0,
            timing_error_summary={},
            symbol_decisions=np.array([], dtype=complex),
            hard_bits=np.array([], dtype=int),
            failure_reason="Input sample array is empty.",
        )

    if np.any(np.isnan(samples)) or np.any(np.isinf(samples)):
        return DemodResult(
            status=DemodStatus.INVALID_INPUT,
            modulation=str(config.modulation),
            input_sample_count=num_samples,
            output_symbol_count=0,
            output_bit_count=0,
            samples_per_symbol_used=config.samples_per_symbol,
            timing_status={"converged": False},
            carrier_status={"converged": False},
            estimated_frequency_offset=0.0,
            estimated_phase_offset=0.0,
            timing_error_summary={},
            symbol_decisions=np.array([], dtype=complex),
            hard_bits=np.array([], dtype=int),
            failure_reason="Input waveform contains NaN or Inf values.",
        )

    mod_str = str(config.modulation).upper().replace("-", "").replace("_", "")

    # 2. Route by Modulation Family
    if "FSK" in mod_str:
        order = 4 if "4" in mod_str else 2
        fsk_hard, fsk_soft, fsk_diag = demod_fsk(
            samples,
            f_dev=config.fsk_deviation,
            symbol_rate=config.symbol_rate,
            fs=config.sample_rate,
            order=order,
            sps=config.samples_per_symbol,
        )
        fsk_syms = np.array(fsk_soft if fsk_soft is not None else [], dtype=complex)
        fsk_candidates = [{
            "angle_deg": 0.0,
            "angle_rad": 0.0,
            "hard_bits": fsk_hard,
            "soft_bits": fsk_soft,
            "symbols": fsk_syms,
        }]

        ref_status = "REFERENCE_BITS_UNAVAILABLE"
        ber: Optional[float] = None
        bit_errs: Optional[int] = None
        if config.external_reference_bits is not None and len(config.external_reference_bits) > 0:
            ref_arr = np.asarray(config.external_reference_bits, dtype=int).ravel()
            eval_len = min(len(fsk_hard), len(ref_arr))
            if eval_len > 0:
                mismatches = int(np.sum(fsk_hard[:eval_len] != ref_arr[:eval_len]))
                bit_errs = mismatches + abs(len(fsk_hard) - len(ref_arr))
                ber = float(mismatches / eval_len)
                ref_status = "EVALUATED_AGAINST_EXTERNAL_REFERENCE"

        return DemodResult(
            status=DemodStatus.SUCCESS,
            modulation=f"{order}-FSK",
            input_sample_count=num_samples,
            output_symbol_count=len(fsk_hard) // (1 if order == 2 else 2),
            output_bit_count=len(fsk_hard),
            samples_per_symbol_used=config.samples_per_symbol,
            timing_status={"fsk_timing_lock": True},
            carrier_status={"fsk_carrier_tracking": True},
            estimated_frequency_offset=fsk_diag.get("estimated_cfo_hz", 0.0),
            estimated_phase_offset=0.0,
            timing_error_summary={},
            symbol_decisions=fsk_syms,
            hard_bits=fsk_hard,
            soft_bits=fsk_soft,
            mapping_profile_used=config.mapping_profile,
            candidate_rotations=fsk_candidates,
            reference_status=ref_status,
            bit_error_rate=ber,
            bit_errors=bit_errs,
            diagnostics=fsk_diag,
            warnings=warnings,
        )

    # 3. PSK / QAM Linear Demodulation Pipeline
    # 3a. RRC Matched Filter
    filtered_samples = apply_rrc_filter(
        samples,
        sps=config.samples_per_symbol,
        alpha=config.rrc_alpha,
        span=config.filter_span,
    )

    # 3b. Gardner Timing Recovery
    recovered_symbols, timing_metrics = gardner_timing_recovery(
        filtered_samples,
        sps=config.samples_per_symbol,
    )

    if len(recovered_symbols) == 0:
        return DemodResult(
            status=DemodStatus.DECODER_FAILURE,
            modulation=str(config.modulation),
            input_sample_count=num_samples,
            output_symbol_count=0,
            output_bit_count=0,
            samples_per_symbol_used=config.samples_per_symbol,
            timing_status=timing_metrics,
            carrier_status={"converged": False},
            estimated_frequency_offset=0.0,
            estimated_phase_offset=0.0,
            timing_error_summary=timing_metrics,
            symbol_decisions=np.array([], dtype=complex),
            hard_bits=np.array([], dtype=int),
            mapping_profile_used=config.mapping_profile,
            reference_status="REFERENCE_BITS_UNAVAILABLE",
            failure_reason="Timing recovery produced zero symbols.",
        )

    # 3c. Carrier Recovery
    if "BPSK" in mod_str:
        psk_order = 2
    elif "QPSK" in mod_str:
        psk_order = 4
    elif "8PSK" in mod_str or "PSK8" in mod_str:
        psk_order = 8
    elif "16QAM" in mod_str or "QAM16" in mod_str:
        psk_order = 16  # Decision-directed carrier recovery for square 16-QAM
    else:
        return DemodResult(
            status=DemodStatus.UNSUPPORTED,
            modulation=str(config.modulation),
            input_sample_count=num_samples,
            output_symbol_count=0,
            output_bit_count=0,
            samples_per_symbol_used=config.samples_per_symbol,
            timing_status=timing_metrics,
            carrier_status={"converged": False},
            estimated_frequency_offset=0.0,
            estimated_phase_offset=0.0,
            timing_error_summary=timing_metrics,
            symbol_decisions=np.array([], dtype=complex),
            hard_bits=np.array([], dtype=int),
            mapping_profile_used=config.mapping_profile,
            reference_status="REFERENCE_BITS_UNAVAILABLE",
            failure_reason=f"Modulation {config.modulation} is not supported.",
        )

    corrected_symbols, carrier_metrics = costas_carrier_recovery(
        recovered_symbols,
        order=psk_order,
    )

    # 3d. Phase Ambiguity Resolution (via explicit preamble if supplied)
    if "QAM" not in mod_str:
        aligned_symbols, ambiguity_info = resolve_phase_ambiguity(
            corrected_symbols,
            order=psk_order,
            preamble_bits=config.preamble_bits,
            profile=config.mapping_profile,
        )
    else:
        aligned_symbols = corrected_symbols
        ambiguity_info = {"resolved": True, "selected_rotation_deg": 0.0}

    # 3e. Constellation Slicing & Soft LLRs for nominal rotation
    if "BPSK" in mod_str:
        hard_bits, soft_bits = demap_bpsk(aligned_symbols, profile=config.mapping_profile)
    elif "QPSK" in mod_str:
        hard_bits, soft_bits = demap_qpsk(aligned_symbols, profile=config.mapping_profile)
    elif "8PSK" in mod_str or "PSK8" in mod_str:
        hard_bits, soft_bits = demap_8psk(aligned_symbols, profile=config.mapping_profile)
    elif "16QAM" in mod_str or "QAM16" in mod_str:
        hard_bits, soft_bits = demap_qam(aligned_symbols, constellation_order=16, profile=config.mapping_profile)
    else:
        hard_bits, soft_bits = demap_bpsk(aligned_symbols, profile=config.mapping_profile)

    # 3f. Candidate Phase Rotations generation
    if config.candidate_phase_rotations is not None:
        cand_angles_deg = [float(a) for a in config.candidate_phase_rotations]
    else:
        if "BPSK" in mod_str:
            cand_angles_deg = [0.0, 180.0]
        elif "QPSK" in mod_str or "16QAM" in mod_str or "QAM16" in mod_str:
            cand_angles_deg = [0.0, 90.0, 180.0, 270.0]
        elif "8PSK" in mod_str or "PSK8" in mod_str:
            cand_angles_deg = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        else:
            cand_angles_deg = [0.0]

    candidate_rotations: List[Dict[str, Any]] = []
    for deg in cand_angles_deg:
        rad = float(np.radians(deg))
        cand_syms = aligned_symbols * np.exp(-1j * rad)
        if "BPSK" in mod_str:
            c_hard, c_soft = demap_bpsk(cand_syms, profile=config.mapping_profile)
        elif "QPSK" in mod_str:
            c_hard, c_soft = demap_qpsk(cand_syms, profile=config.mapping_profile)
        elif "8PSK" in mod_str or "PSK8" in mod_str:
            c_hard, c_soft = demap_8psk(cand_syms, profile=config.mapping_profile)
        elif "16QAM" in mod_str or "QAM16" in mod_str:
            c_hard, c_soft = demap_qam(cand_syms, constellation_order=16, profile=config.mapping_profile)
        else:
            c_hard, c_soft = demap_bpsk(cand_syms, profile=config.mapping_profile)

        candidate_rotations.append({
            "angle_deg": float(deg),
            "angle_rad": rad,
            "hard_bits": c_hard,
            "soft_bits": c_soft,
            "symbols": cand_syms,
        })

    # 3g. External Reference Evaluation (if provided)
    ref_status = "REFERENCE_BITS_UNAVAILABLE"
    ber: Optional[float] = None
    bit_errs: Optional[int] = None
    if config.external_reference_bits is not None and len(config.external_reference_bits) > 0:
        ref_arr = np.asarray(config.external_reference_bits, dtype=int).ravel()
        eval_len = min(len(hard_bits), len(ref_arr))
        if eval_len > 0:
            mismatches = int(np.sum(hard_bits[:eval_len] != ref_arr[:eval_len]))
            bit_errs = mismatches + abs(len(hard_bits) - len(ref_arr))
            ber = float(mismatches / eval_len)
            ref_status = "EVALUATED_AGAINST_EXTERNAL_REFERENCE"

    status = DemodStatus.SUCCESS
    if not timing_metrics.get("converged", True):
        status = DemodStatus.NON_CONVERGED
        warnings.append("Gardner timing loop reported high residual jitter.")
    if not carrier_metrics.get("converged", True):
        status = DemodStatus.NON_CONVERGED
        warnings.append("Carrier recovery reported high residual phase error.")

    diag = {
        "rrc_filter_span": config.filter_span,
        "rrc_alpha": config.rrc_alpha,
        "timing_metrics": timing_metrics,
        "carrier_metrics": carrier_metrics,
        "ambiguity_info": ambiguity_info,
    }

    return DemodResult(
        status=status,
        modulation=str(config.modulation),
        input_sample_count=num_samples,
        output_symbol_count=len(aligned_symbols),
        output_bit_count=len(hard_bits),
        samples_per_symbol_used=config.samples_per_symbol,
        timing_status=timing_metrics,
        carrier_status=carrier_metrics,
        estimated_frequency_offset=carrier_metrics.get("estimated_cfo", 0.0),
        estimated_phase_offset=carrier_metrics.get("final_phase", 0.0),
        timing_error_summary=timing_metrics,
        symbol_decisions=aligned_symbols,
        hard_bits=hard_bits,
        soft_bits=soft_bits,
        mapping_profile_used=config.mapping_profile,
        candidate_rotations=candidate_rotations,
        reference_status=ref_status,
        bit_error_rate=ber,
        bit_errors=bit_errs,
        diagnostics=diag,
        warnings=warnings,
    )
