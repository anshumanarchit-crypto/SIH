"""
Signal Normalization and Preprocessing Module for SpectralQ.

Implements deterministic DSP algorithms:
- DC offset removal
- Blind Gram-Schmidt IQ imbalance correction
- Unit power (RMS) normalization
- Coarse carrier frequency offset (CFO) estimation & correction (M-th power & spectral methods)
- Root-Raised-Cosine (RRC) matched filtering
- Bandpass/lowpass noise rejection filtering
- Automatic Gain Control (AGC)
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, fftconvolve

logger = logging.getLogger("spectralq.preprocessing")


def remove_dc_offset(samples: np.ndarray) -> np.ndarray:
    """
    Remove DC bias (zero-frequency offset) by subtracting the mean.
    Preserves input dtype.
    """
    if len(samples) == 0:
        return samples
    mean_val = np.mean(samples)
    return samples - mean_val


def correct_iq_imbalance(samples: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Blind IQ imbalance correction using Gram-Schmidt orthogonalization.

    Compensates for:
    - Amplitude imbalance between I and Q branches
    - Phase quadrature errors (non-90-degree LO phase split)

    Returns:
        (corrected_samples, metrics_dict)
    """
    if len(samples) == 0:
        return samples, {"gain_imbalance_db": 0.0, "phase_error_deg": 0.0}

    i = np.real(samples)
    q = np.imag(samples)

    # 1. Zero-mean the components
    i_zm = i - np.mean(i)
    q_zm = q - np.mean(q)

    # 2. Estimate power and cross-correlation
    p_i = np.mean(i_zm ** 2)
    p_q = np.mean(q_zm ** 2)

    if p_i < 1e-12 or p_q < 1e-12:
        return samples, {"gain_imbalance_db": 0.0, "phase_error_deg": 0.0}

    # If power ratio is extreme (>13 dB), signal is 1D (BPSK / Real ASK); do not amplify orthogonal noise
    ratio = p_q / p_i
    if ratio < 0.05 or ratio > 20.0:
        return samples, {"gain_imbalance_db": float(10.0 * np.log10(max(ratio, 1e-6))), "phase_error_deg": 0.0}

    # Amplitude imbalance estimate
    gain_imbalance_db = 10.0 * np.log10(ratio)

    # Cross-correlation coefficient (phase error indicator)
    p_iq = np.mean(i_zm * q_zm)
    sin_phi = p_iq / np.sqrt(p_i * p_q)
    sin_phi = np.clip(sin_phi, -0.999, 0.999)
    phase_error_deg = float(np.arcsin(sin_phi) * 180.0 / np.pi)

    # 3. Gram-Schmidt Orthogonalization
    i_norm = i_zm / np.sqrt(p_i)
    q_temp = q_zm - (p_iq / p_i) * i_zm
    p_q_temp = np.mean(q_temp ** 2)
    if p_q_temp < 1e-12:
        p_q_temp = 1e-12
    q_norm = q_temp / np.sqrt(p_q_temp)

    corrected = (i_norm + 1j * q_norm).astype(np.complex64)

    metrics = {
        "gain_imbalance_db": float(gain_imbalance_db),
        "phase_error_deg": phase_error_deg,
        "iq_correlation": float(p_iq / np.sqrt(p_i * p_q)),
    }
    return corrected, metrics


def normalize_power(samples: np.ndarray, target_power: float = 1.0) -> np.ndarray:
    """
    Scale signal to have average power (RMS squared) equal to target_power.
    """
    if len(samples) == 0:
        return samples
    current_power = np.mean(np.abs(samples) ** 2)
    if current_power < 1e-14:
        return samples
    scale_factor = np.sqrt(target_power / current_power)
    return samples * scale_factor


def design_rrc_filter(
    sps: int,
    beta: float = 0.35,
    span: int = 10,
) -> np.ndarray:
    """
    Design Root-Raised-Cosine (RRC) pulse shaping / matched filter.

    Args:
        sps: Samples per symbol (must be >= 1).
        beta: Roll-off factor (0.0 to 1.0).
        span: Filter length in symbols (total taps = span * sps + 1).

    Returns:
        Unit-energy 1D float array of filter coefficients.
    """
    if sps < 1:
        raise ValueError(f"sps must be >= 1, got {sps}")
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0.0, 1.0], got {beta}")

    num_taps = span * sps + 1
    t = np.arange(-num_taps // 2 + 1, num_taps // 2 + 1) / float(sps)
    h = np.zeros(num_taps, dtype=np.float64)

    for idx, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[idx] = 1.0 - beta + (4.0 * beta / np.pi)
        elif beta > 0 and np.isclose(abs(ti), 1.0 / (4.0 * beta)):
            val = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
            h[idx] = val
        else:
            numerator = np.sin(np.pi * ti * (1.0 - beta)) + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
            denominator = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[idx] = numerator / denominator

    # Normalize filter to unit energy
    energy = np.sqrt(np.sum(h ** 2))
    if energy > 0:
        h = h / energy
    return h.astype(np.float32)


def apply_rrc_filter(
    samples: np.ndarray,
    sps: int = 8,
    beta: float = 0.35,
    span: int = 10,
) -> np.ndarray:
    """
    Apply RRC matched filter to samples via FFT convolution.
    """
    if len(samples) == 0:
        return samples
    h = design_rrc_filter(sps=sps, beta=beta, span=span)
    filtered = fftconvolve(samples, h, mode="same")
    return filtered.astype(samples.dtype if np.iscomplexobj(samples) else np.complex64)


def estimate_and_correct_cfo(
    samples: np.ndarray,
    sample_rate: float,
    m_order: int = 4,
    search_range_hz: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Estimate and correct Coarse Carrier Frequency Offset (CFO).

    Uses M-th power non-linearity:
      - m_order=1 for CW / Unmodulated carrier / AM
      - m_order=2 for BPSK / MSK
      - m_order=4 for QPSK / QAM

    Returns:
        (corrected_samples, estimated_cfo_hz)
    """
    if len(samples) == 0 or sample_rate <= 0:
        return samples, 0.0

    n_samples = len(samples)
    # Use up to 32768 samples for high FFT resolution
    sub_len = min(n_samples, 32768)
    sub_sig = samples[:sub_len]

    # Raise signal to M-th power to collapse modulation phase
    sig_m = sub_sig ** m_order

    # Zero-padding for fine frequency interpolation
    fft_size = max(65536, 1 << int(np.ceil(np.log2(sub_len * 4))))
    spectrum = np.fft.fftshift(np.fft.fft(sig_m, n=fft_size))
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate))

    if search_range_hz is not None:
        mask = np.abs(freqs) <= search_range_hz
        if np.any(mask):
            freqs = freqs[mask]
            spectrum = spectrum[mask]

    peak_idx = np.argmax(np.abs(spectrum))
    estimated_m_freq = freqs[peak_idx]

    # The actual carrier offset is the peak frequency divided by M
    cfo_hz = float(estimated_m_freq / m_order)

    # Correct CFO
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    correction_phasor = np.exp(-1j * 2.0 * np.pi * cfo_hz * t).astype(np.complex64)
    corrected = (samples * correction_phasor).astype(np.complex64)

    return corrected, cfo_hz


def apply_bandpass_filter(
    samples: np.ndarray,
    sample_rate: float,
    low_cutoff_hz: float,
    high_cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter.
    """
    if len(samples) == 0:
        return samples

    nyquist = 0.5 * sample_rate
    low = max(1e-3, low_cutoff_hz / nyquist)
    high = min(0.999, high_cutoff_hz / nyquist)

    if low >= high:
        return samples

    b, a = butter(order, [low, high], btype="bandpass")

    if np.iscomplexobj(samples):
        i_filt = filtfilt(b, a, np.real(samples))
        q_filt = filtfilt(b, a, np.imag(samples))
        return (i_filt + 1j * q_filt).astype(np.complex64)
    else:
        return filtfilt(b, a, samples).astype(np.float32)


def automatic_gain_control(
    samples: np.ndarray,
    target_level: float = 1.0,
    mu: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    First-order feedback Automatic Gain Control (AGC).

    Args:
        samples: Input complex/real samples.
        target_level: Desired output envelope level.
        mu: Adaptation step size (rate of gain adjustment).

    Returns:
        (leveled_samples, gain_history)
    """
    n = len(samples)
    if n == 0:
        return samples, np.array([])

    out = np.empty(n, dtype=np.complex64 if np.iscomplexobj(samples) else np.float32)
    gains = np.empty(n, dtype=np.float32)

    gain = 1.0
    for idx in range(n):
        s = samples[idx]
        y = s * gain
        env = abs(y)
        err = target_level - env
        gain = max(1e-4, gain + mu * err)
        out[idx] = y
        gains[idx] = gain

    return out, gains
