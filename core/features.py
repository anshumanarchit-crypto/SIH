"""
Feature Extraction Module for SpectralQ.

Implements mathematically rigorous feature extractors:
- Higher-Order Cumulants (C20, C21, C40, C41, C42, C63, C80)
- Instantaneous Spectral & Temporal Statistics (gamma_max, sigma_ap, sigma_dp, sigma_aa, sigma_af)
- Higher-order Moments, Kurtosis & Skewness
- Non-Data-Aided SNR Estimation (M2M4 and Spectral Floor)
- Baud / Symbol Rate & Samples-Per-Symbol (SPS) Estimators
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import welch

logger = logging.getLogger("spectralq.features")


from core.contracts import (
    SpectralFeatures,
    ResultStatus,
    make_warning,
    IQ_CONVENTION,
)


def compute_higher_order_cumulants(samples: np.ndarray) -> Dict[str, complex]:
    """
    Compute sample higher-order moments and cumulants for zero-mean normalized signal:
    C20, C21, C40, C41, C42, C63, C80.
    """
    if len(samples) < 16:
        return {
            "c20": 0.0 + 0j, "c21": 0.0, "c40": 0.0 + 0j,
            "c41": 0.0 + 0j, "c42": 0.0, "c63": 0.0, "c80": 0.0 + 0j,
        }

    # Ensure zero-mean
    s = samples - np.mean(samples)
    p = np.mean(np.abs(s) ** 2)
    if p > 1e-12:
        s = s / np.sqrt(p)

    s_conj = np.conj(s)
    s_sq = s ** 2
    s_abs_sq = np.abs(s) ** 2

    # Moments: M_pq = E[s^(p-q) * (s*)^q]
    m20 = np.mean(s_sq)
    m21 = np.mean(s_abs_sq)  # real, equals 1.0 for unit variance
    m40 = np.mean(s ** 4)
    m41 = np.mean((s ** 3) * s_conj)
    m42 = np.mean(s_abs_sq ** 2)
    m63 = np.mean(s_abs_sq ** 3)
    m80 = np.mean(s ** 8)

    # Cumulants
    c20 = m20
    c21 = float(np.real(m21))
    c40 = m40 - 3.0 * (m20 ** 2)
    c41 = m41 - 3.0 * m20 * m21
    c42 = float(np.real(m42 - np.abs(m20) ** 2 - 2.0 * (m21 ** 2)))
    c63 = float(np.real(m63 - 9.0 * c42 * c21 - 6.0 * (c21 ** 3)))
    c80 = m80 - 35.0 * (m40 ** 2)

    return {
        "c20": c20,
        "c21": c21,
        "c40": c40,
        "c41": c41,
        "c42": c42,
        "c63": c63,
        "c80": c80,
    }


def compute_instantaneous_statistics(
    samples: np.ndarray,
    sample_rate: float,
) -> Dict[str, float]:
    """
    Extract instantaneous spectral & temporal statistics:
      - gamma_max: Max spectral power density of normalized centered instantaneous amplitude.
      - sigma_ap: Standard deviation of direct centered instantaneous phase.
      - sigma_dp: Standard deviation of nonlinear (detrended) phase.
      - sigma_aa: Standard deviation of absolute normalized centered instantaneous amplitude.
      - sigma_af: Standard deviation of normalized centered instantaneous frequency.
      - kurtosis & skewness of envelope and phase.
    """
    n = len(samples)
    if n < 32:
        return {
            "gamma_max": 0.0, "sigma_ap": 0.0, "sigma_dp": 0.0,
            "sigma_aa": 0.0, "sigma_af": 0.0,
            "kurtosis_amp": 0.0, "skewness_amp": 0.0, "kurtosis_phase": 0.0,
            "spectral_flatness": 1.0,
        }

    # 1. Instantaneous amplitude
    a = np.abs(samples)
    mu_a = np.mean(a)
    if mu_a < 1e-12:
        mu_a = 1e-12
    a_cn = (a / mu_a) - 1.0  # normalized centered amplitude

    # gamma_max: max normalized power spectral density of a_cn
    fft_a_cn = np.fft.fft(a_cn)
    gamma_max = float(np.max(np.abs(fft_a_cn) ** 2) / float(n))

    # sigma_aa
    sigma_aa = float(np.std(np.abs(a_cn)))

    # Amplitude skewness and kurtosis
    std_a = np.std(a)
    if std_a > 1e-9:
        skewness_amp = float(np.mean(((a - mu_a) / std_a) ** 3))
        kurtosis_amp = float(np.mean(((a - mu_a) / std_a) ** 4)) - 3.0  # excess kurtosis
    else:
        skewness_amp = 0.0
        kurtosis_amp = 0.0

    # 2. Instantaneous phase
    # Filter out samples with very low amplitude to avoid phase singularity noise
    amp_thresh = 0.1 * mu_a
    valid_mask = a > amp_thresh

    if np.sum(valid_mask) > 16:
        valid_samples = samples[valid_mask]
        phase_raw = np.unwrap(np.angle(valid_samples))
        # Centered phase
        phase_centered = phase_raw - np.mean(phase_raw)
        sigma_ap = float(np.std(phase_centered))

        # Nonlinear phase (detrend linear phase slope / CFO)
        t_indices = np.arange(len(phase_raw))
        if len(t_indices) > 1:
            p_fit = np.polyfit(t_indices, phase_raw, 1)
            phase_linear = np.polyval(p_fit, t_indices)
            phase_nl = phase_raw - phase_linear
            sigma_dp = float(np.std(phase_nl))
        else:
            sigma_dp = sigma_ap

        std_phase = np.std(phase_centered)
        if std_phase > 1e-6:
            kurtosis_phase = float(np.mean((phase_centered / std_phase) ** 4)) - 3.0
        else:
            kurtosis_phase = 0.0
    else:
        sigma_ap = 0.0
        sigma_dp = 0.0
        kurtosis_phase = 0.0

    # 3. Instantaneous frequency
    phase_all = np.unwrap(np.angle(samples))
    diff_phase = np.diff(phase_all)
    inst_freq = diff_phase * (sample_rate / (2.0 * np.pi))
    mu_f = np.mean(inst_freq)
    std_f = np.std(inst_freq)
    # Normalized frequency std (divided by sample rate)
    sigma_af = float(std_f / sample_rate) if sample_rate > 0 else 0.0

    # 4. Spectral Flatness (Wiener entropy)
    freqs_w, psd_w = welch(samples, fs=sample_rate, nperseg=min(1024, n))
    psd_w = psd_w + 1e-15
    geom_mean = np.exp(np.mean(np.log(psd_w)))
    arith_mean = np.mean(psd_w)
    spectral_flatness = float(geom_mean / arith_mean) if arith_mean > 0 else 1.0

    return {
        "gamma_max": gamma_max,
        "sigma_ap": sigma_ap,
        "sigma_dp": sigma_dp,
        "sigma_aa": sigma_aa,
        "sigma_af": sigma_af,
        "kurtosis_amp": kurtosis_amp,
        "skewness_amp": skewness_amp,
        "kurtosis_phase": kurtosis_phase,
        "spectral_flatness": spectral_flatness,
    }


def estimate_snr_m2m4(samples: np.ndarray, assumed_ka: float = 1.0, sample_rate: float = 100_000.0) -> float:
    """
    Robust Hybrid SNR estimator combining M2M4 moments with Welch Spectral in-band/noise-floor ratio.

    Args:
        samples: Baseband complex samples.
        assumed_ka: Kurtosis factor of constellation.
        sample_rate: Sampling rate in Hz.

    Returns:
        Estimated SNR in dB.
    """
    if len(samples) < 32:
        return 0.0

    s = samples - np.mean(samples)
    n = len(s)

    # 1. Welch spectral noise floor & in-band peak power
    freqs, psd = welch(s, fs=sample_rate, nperseg=min(1024, n))
    psd_sorted = np.sort(psd)
    noise_floor = float(np.median(psd_sorted[: max(1, len(psd_sorted) // 4)]))
    signal_peak = float(np.percentile(psd, 90))

    spectral_snr_db = 0.0
    if noise_floor > 1e-15 and signal_peak > noise_floor:
        spectral_snr_db = float(10.0 * np.log10((signal_peak - noise_floor) / noise_floor))

    # 2. Moment-based M2M4
    m2 = np.mean(np.abs(s) ** 2)
    m4 = np.mean(np.abs(s) ** 4)

    if m2 < 1e-12:
        return -20.0

    disc = 2.0 * (m2 ** 2) - m4
    if disc > 0:
        s_power = np.sqrt(disc)
        n_power = max(m2 - s_power, 1e-12)
        m2m4_snr_db = float(10.0 * np.log10(max(s_power / n_power, 1e-3)))
    else:
        m2m4_snr_db = -10.0

    # Pick the most defensible SNR (spectral SNR is robust to pulse shaping & burst padding)
    final_snr = max(m2m4_snr_db, spectral_snr_db)
    return float(np.clip(final_snr, -20.0, 50.0))


def estimate_symbol_rate(
    samples: np.ndarray,
    sample_rate: float,
    min_sps: float = 2.0,
    max_sps: float = 64.0,
) -> Tuple[float, float]:
    """
    Estimate Symbol (Baud) Rate using wave-difference non-linearity: |s[n] - s[n-1]|^2.
    Isolates cyclostationary symbol clock lines and rejects burst boxcar DC leakage.

    Returns:
        (estimated_baud_rate_hz, estimated_samples_per_symbol)
    """
    n = len(samples)
    if n < 64 or sample_rate <= 0:
        return 1000.0, 8.0

    # 1. Wave-difference non-linearity: transitions produce periodic clock spikes
    diff_s = np.abs(np.diff(samples)) ** 2
    diff_zm = diff_s - np.mean(diff_s)

    # 2. High-resolution FFT
    n_fft = max(8192, 1 << int(np.ceil(np.log2(min(len(diff_zm) * 4, 32768)))))
    fft_y = np.abs(np.fft.rfft(diff_zm, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    min_freq = sample_rate / max_sps
    max_freq = sample_rate / min_sps
    mask = (freqs >= min_freq) & (freqs <= max_freq)

    if not np.any(mask):
        return sample_rate / 8.0, 8.0

    masked_freqs = freqs[mask]
    masked_spec = fft_y[mask]

    peak_idx = np.argmax(masked_spec)
    est_baud = float(masked_freqs[peak_idx])

    if est_baud <= 0:
        est_baud = sample_rate / 8.0

    est_sps = float(sample_rate / est_baud)
    return est_baud, est_sps


def extract_all_features(
    samples: np.ndarray,
    sample_rate: float,
    center_freq_offset_hz: float = 0.0,
) -> SpectralFeatures:
    """
    Orchestrate full feature extraction pipeline on signal samples.
    """
    hoc = compute_higher_order_cumulants(samples)
    inst_stats = compute_instantaneous_statistics(samples, sample_rate)
    snr_db = estimate_snr_m2m4(samples, sample_rate=sample_rate)
    baud_rate, sps = estimate_symbol_rate(samples, sample_rate)

    raw_metrics = {
        "c20_mag": float(np.abs(hoc["c20"])),
        "c21": float(hoc["c21"]),
        "c40_mag": float(np.abs(hoc["c40"])),
        "c41_mag": float(np.abs(hoc["c41"])),
        "c42": float(hoc["c42"]),
        "c63": float(hoc["c63"]),
        "c80_mag": float(np.abs(hoc["c80"])),
        **inst_stats,
        "snr_db": snr_db,
        "baud_rate": baud_rate,
        "sps": sps,
        "cfo_hz": center_freq_offset_hz,
    }

    return SpectralFeatures(
        c20=hoc["c20"],
        c21=hoc["c21"],
        c40=hoc["c40"],
        c41=hoc["c41"],
        c42=hoc["c42"],
        c63=hoc["c63"],
        c80=hoc["c80"],
        gamma_max=inst_stats["gamma_max"],
        sigma_ap=inst_stats["sigma_ap"],
        sigma_dp=inst_stats["sigma_dp"],
        sigma_aa=inst_stats["sigma_aa"],
        sigma_af=inst_stats["sigma_af"],
        kurtosis_amp=inst_stats["kurtosis_amp"],
        skewness_amp=inst_stats["skewness_amp"],
        kurtosis_phase=inst_stats["kurtosis_phase"],
        snr_db=snr_db,
        estimated_baud_rate=baud_rate,
        estimated_sps=sps,
        spectral_flatness=inst_stats["spectral_flatness"],
        carrier_freq_offset_hz=center_freq_offset_hz,
        raw_metrics=raw_metrics,
    )
