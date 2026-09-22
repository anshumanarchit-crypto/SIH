"""
SpectralQ Signal Visualization Engine (CSE-2)
=============================================
Engineering-grade, reusable visualization functions for RF signal processing,
SDR baseband analysis, spectral analytics, constellation diagrams, and more.

Design Principles
-----------------
- All plot functions accept raw NumPy arrays + explicit metadata parameters
  so they can be used independently of the pipeline.
- High-level wrappers (``plot_from_signal``) accept ``SignalData`` objects
  (from ``core.contracts``) for pipeline integration.
- NEVER import Streamlit here; this module is backend-only and reusable by CSE-5.
- All functions return a ``matplotlib.figure.Figure`` object. The caller is
  responsible for saving or displaying it.
- Functions that compute transform data (FFT, spectrogram) also return the
  underlying arrays so callers can inspect or re-plot without recomputation.
- Every function handles degenerate inputs (empty array, very short, NaN/Inf)
  gracefully by rendering a clearly-labelled "insufficient data" placeholder.
- DC removal before visualization is opt-in (``remove_dc=True``), never automatic.
- Frequency axis units are clearly labelled; when ``sample_rate`` is None the
  axis shows *normalized* frequency ∈ [−0.5, 0.5] (cycles/sample) with an
  explicit note on the axis label.

API Summary
-----------
Core plot functions (arrays):
  plot_time_domain(samples, sample_rate, ...)     → Figure
  plot_fft(samples, sample_rate, ...)             → (Figure, freqs, magnitude_db)
  plot_spectrogram(samples, sample_rate, ...)     → (Figure, f, t, Sxx_db)
  plot_constellation(symbols, ...)                → Figure
  plot_amplitude(samples, sample_rate, ...)       → Figure
  plot_phase(samples, sample_rate, ...)           → Figure

Legacy / bonus functions (kept for pipeline compatibility):
  plot_psd(samples, sample_rate, ...)             → Figure
  plot_eye_diagram(samples, sps, ...)             → Figure
  plot_sync_correlation(corr_curve, ...)          → Figure
  plot_feature_radar(features, ...)               → Figure

SignalData-aware entry points (CSE-1 integration):
  plot_from_signal(sig, plot_type, **kwargs)      → Figure or tuple

Contracts
---------
- ``SignalData`` is imported from ``core.contracts`` (CSE-0), never redefined.
- ``ResultStatus`` and ``validate_confidence`` are imported from ``core.contracts``
  for constellation classification overlay. No ad-hoc status strings.
- IQ convention: complex sample = I + jQ (I = np.real, Q = np.imag).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless and web servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
from scipy import signal as scipy_signal

from core.contracts import (
    IQ_CONVENTION,
    ResultStatus,
    SignalData,
    validate_confidence,
)

logger = logging.getLogger("spectralq.visualization")

# --------------------------------------------------------------------------- #
# Theme                                                                         #
# --------------------------------------------------------------------------- #

DARK_THEME: Dict[str, str] = {
    "bg_figure":   "#0B0F19",
    "bg_axes":     "#111827",
    "grid":        "#1F2937",
    "text":        "#E5E7EB",
    "muted":       "#9CA3AF",
    "i_channel":   "#00F0FF",   # Cyber Cyan  — I component
    "q_channel":   "#FF007F",   # Neon Magenta — Q component
    "envelope":    "#FBBF24",   # Amber Gold   — magnitude/envelope
    "phase_line":  "#A78BFA",   # Soft Violet  — phase trajectory
    "ideal_pts":   "#10B981",   # Emerald       — ideal constellation
    "threshold":   "#EF4444",   # Neon Red      — thresholds
    "sync_peak":   "#8B5CF6",   # Bright Violet — correlation peaks
    "cmap":        "viridis",
}


def apply_dark_theme(fig: plt.Figure, ax_list=None) -> None:
    """Apply the SpectralQ dark-SDR-lab styling to *fig* and all axes."""
    fig.patch.set_facecolor(DARK_THEME["bg_figure"])
    if ax_list is None:
        ax_list = fig.axes
    elif not isinstance(ax_list, (list, tuple, np.ndarray)):
        ax_list = [ax_list]
    for ax in ax_list:
        ax.set_facecolor(DARK_THEME["bg_axes"])
        ax.tick_params(colors=DARK_THEME["text"], labelsize=9)
        ax.xaxis.label.set_color(DARK_THEME["text"])
        ax.yaxis.label.set_color(DARK_THEME["text"])
        ax.title.set_color(DARK_THEME["text"])
        for spine in ax.spines.values():
            spine.set_color(DARK_THEME["grid"])
            spine.set_linewidth(1.0)
        ax.grid(True, linestyle="--", alpha=0.35, color=DARK_THEME["grid"])


# --------------------------------------------------------------------------- #
# Internal Utilities                                                            #
# --------------------------------------------------------------------------- #

_MIN_SAMPLES = 2  # Minimum number of samples required for any meaningful plot


def _placeholder_fig(message: str, figsize: Tuple[float, float] = (8, 4)) -> plt.Figure:
    """Return a styled figure with a centred message (used for degenerate inputs)."""
    fig, ax = plt.subplots(figsize=figsize)
    apply_dark_theme(fig, ax)
    ax.text(
        0.5, 0.5, message,
        color=DARK_THEME["muted"], ha="center", va="center",
        transform=ax.transAxes, fontsize=11, wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def _sanitize(samples: np.ndarray, context: str = "") -> np.ndarray:
    """
    Replace NaN and Inf with zero and emit a structured warning.

    This guarantees plot functions never crash due to invalid values.  The
    caller should validate data quality upstream; this is a last-resort safety net.
    """
    if not np.all(np.isfinite(samples)):
        n_bad = int(np.sum(~np.isfinite(samples)))
        logger.warning(
            "[visualization%s] %d non-finite values (NaN/Inf) replaced with 0.",
            f" {context}" if context else "",
            n_bad,
        )
        samples = np.where(np.isfinite(samples), samples, 0.0)
    return samples


def _downsample(arr: np.ndarray, max_pts: int) -> np.ndarray:
    """Uniformly downsample ``arr`` to at most ``max_pts`` points (deterministic)."""
    n = len(arr)
    if n <= max_pts:
        return arr
    idx = np.linspace(0, n - 1, max_pts, dtype=int)
    return arr[idx]


def _freq_axis(n_fft: int, sample_rate: Optional[float], is_complex: bool) -> Tuple[np.ndarray, str]:
    """
    Return (freq_array, x_label) for an FFT of transform size ``n_fft``.

    Args:
        n_fft: The FFT/transform size (NOT the number of output bins).
               For complex FFT: n_fft bins. For real FFT: n_fft//2+1 bins.
    When ``sample_rate`` is None, returns normalized frequency in
    [−0.5, 0.5] (cycles/sample) for complex, [0, 0.5] for real, with an
    explicit note in the label.
    """
    if sample_rate is not None:
        if is_complex:
            freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / sample_rate))
            label = "Frequency (Hz)"
        else:
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
            label = "Frequency (Hz)"
    else:
        if is_complex:
            freqs = np.fft.fftshift(np.fft.fftfreq(n_fft))
            label = "Normalized Frequency (cycles/sample) [sample_rate unknown]"
        else:
            freqs = np.fft.rfftfreq(n_fft)
            label = "Normalized Frequency (cycles/sample) [sample_rate unknown]"
    return freqs, label


def _remove_dc(samples: np.ndarray) -> np.ndarray:
    """Subtract the complex/real mean (DC removal)."""
    return samples - np.mean(samples)


# --------------------------------------------------------------------------- #
# 1. Time-Domain Waveform                                                       #
# --------------------------------------------------------------------------- #

def plot_time_domain(
    samples: np.ndarray,
    sample_rate: Optional[float] = None,
    max_samples: int = 2000,
    remove_dc: bool = False,
    title: str = "Baseband Time-Domain Waveform",
) -> plt.Figure:
    """
    Plot the I component, Q component, and instantaneous magnitude of a signal.

    Parameters
    ----------
    samples:
        Complex (I+jQ) or real 1-D NumPy array.
    sample_rate:
        Sampling rate in Hz. If None, the x-axis shows sample index.
    max_samples:
        Maximum number of samples rendered (downsampled if longer).
    remove_dc:
        If True, subtract the complex mean before plotting.
    title:
        Plot title string.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if samples is None or len(samples) < _MIN_SAMPLES:
        return _placeholder_fig(f"Insufficient data for time-domain plot (n={len(samples) if samples is not None else 0})")

    s = np.asarray(samples)
    s = _sanitize(s, "time_domain")
    if remove_dc:
        s = _remove_dc(s)
    s = _downsample(s, max_samples)
    n = len(s)

    # Build time axis
    if sample_rate is not None and sample_rate > 0:
        t = np.arange(n) / sample_rate * 1e6  # microseconds
        x_label = "Time (\u00b5s)"
    else:
        t = np.arange(n)
        x_label = "Sample Index"

    is_complex = np.iscomplexobj(s)

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax_iq, ax_mag = axes
    apply_dark_theme(fig, axes)

    if is_complex:
        ax_iq.plot(t, np.real(s), label="In-Phase (I)", color=DARK_THEME["i_channel"], lw=1.3, alpha=0.9)
        ax_iq.plot(t, np.imag(s), label="Quadrature (Q)", color=DARK_THEME["q_channel"], lw=1.3, alpha=0.8)
    else:
        ax_iq.plot(t, s, label="Real Amplitude", color=DARK_THEME["i_channel"], lw=1.3)

    ax_iq.set_ylabel("Amplitude (V)", fontsize=10)
    ax_iq.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax_iq.legend(
        loc="upper right",
        facecolor=DARK_THEME["bg_axes"], edgecolor=DARK_THEME["grid"],
        labelcolor=DARK_THEME["text"], fontsize=9,
    )

    # Instantaneous magnitude (envelope)
    env = np.abs(s)
    ax_mag.plot(t, env, color=DARK_THEME["envelope"], lw=1.2, label="|s(t)|")
    ax_mag.set_xlabel(x_label, fontsize=10)
    ax_mag.set_ylabel("|s(t)|", fontsize=10)
    ax_mag.legend(
        loc="upper right",
        facecolor=DARK_THEME["bg_axes"], edgecolor=DARK_THEME["grid"],
        labelcolor=DARK_THEME["text"], fontsize=9,
    )

    dc_note = "  [DC removed]" if remove_dc else ""
    fig.suptitle(
        f"N={len(samples):,} samples{dc_note}",
        color=DARK_THEME["muted"], fontsize=8, y=0.02,
    )
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 2. FFT / Power Spectrum                                                       #
# --------------------------------------------------------------------------- #

def plot_fft(
    samples: np.ndarray,
    sample_rate: Optional[float] = None,
    center_freq: float = 0.0,
    n_fft: Optional[int] = None,
    window: str = "hann",
    remove_dc: bool = False,
    db_range: float = 80.0,
    title: str = "FFT Magnitude Spectrum",
) -> Tuple[plt.Figure, np.ndarray, np.ndarray]:
    """
    Compute and plot the FFT magnitude spectrum.

    Parameters
    ----------
    samples:
        Complex (I+jQ) or real 1-D NumPy array.
    sample_rate:
        Sampling rate in Hz. If None, normalized frequency is shown and a
        warning note is added to the axis label.
    center_freq:
        RF center frequency in Hz added to the frequency axis (offset).
        Only used when ``sample_rate`` is not None.
    n_fft:
        FFT size. Defaults to the next power-of-two >= len(samples).
    window:
        Window function name (any scipy.signal.get_window argument).
    remove_dc:
        If True, subtract the complex mean before computing the FFT.
    db_range:
        Dynamic range of the displayed spectrum in dB (clips floor at peak - db_range).
    title:
        Plot title string.

    Returns
    -------
    fig : matplotlib.figure.Figure
    freqs : np.ndarray
        Frequency array (Hz if sample_rate known, cycles/sample otherwise).
    magnitude_db : np.ndarray
        FFT magnitude in dB, same shape as ``freqs``.
    """
    if samples is None or len(samples) < _MIN_SAMPLES:
        fig = _placeholder_fig(f"Insufficient data for FFT (n={len(samples) if samples is not None else 0})")
        return fig, np.array([]), np.array([])

    s = np.asarray(samples, dtype=np.complex128 if np.iscomplexobj(samples) else np.float64)
    s = _sanitize(s, "fft")
    if remove_dc:
        s = _remove_dc(s)

    is_complex = np.iscomplexobj(s)
    n = len(s)
    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(n)))
    n_fft = max(n_fft, n)  # Never zero-pad to less than signal length

    # Apply window
    try:
        win = scipy_signal.get_window(window, n)
    except Exception:
        win = np.ones(n)
        logger.warning("Unknown window '%s'; using rectangular.", window)
    win_rms = np.sqrt(np.mean(win ** 2))
    if win_rms < 1e-12:
        win_rms = 1.0

    s_windowed = s * win

    if is_complex:
        spectrum = np.fft.fftshift(np.fft.fft(s_windowed, n=n_fft))
        n_freq_bins = n_fft
    else:
        spectrum = np.fft.rfft(s_windowed, n=n_fft)
        n_freq_bins = n_fft // 2 + 1

    magnitude = np.abs(spectrum) / (n * win_rms)
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-20))

    # Pass n_fft (transform size) — _freq_axis calls rfftfreq(n_fft) → n_fft//2+1 bins
    # for real, fftfreq(n_fft) → n_fft bins for complex. Both match spectrum shape.
    freqs, freq_label = _freq_axis(n_fft, sample_rate, is_complex)

    # When center_freq is non-zero and sample_rate is known, offset the axis
    if center_freq != 0.0 and sample_rate is not None:
        freqs = freqs + center_freq
        freq_label = "Frequency (Hz)"

    # Clip dynamic range
    peak_db = float(np.max(magnitude_db))
    floor_db = peak_db - db_range
    magnitude_db_clipped = np.maximum(magnitude_db, floor_db)

    # Peak annotation
    peak_idx = int(np.argmax(magnitude_db))
    peak_freq = float(freqs[peak_idx])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    apply_dark_theme(fig, ax)

    ax.plot(freqs, magnitude_db_clipped, color=DARK_THEME["i_channel"], lw=1.3, label="FFT |X(f)| (dB)")
    ax.axvline(peak_freq, color=DARK_THEME["sync_peak"], linestyle="--", lw=1.2,
               label=f"Peak: {peak_freq:.4g} Hz ({peak_db:.1f} dB)")

    noise_floor_est = float(np.percentile(magnitude_db_clipped, 25))
    ax.axhline(noise_floor_est, color=DARK_THEME["muted"], linestyle=":", lw=1.1,
               label=f"Noise floor ~{noise_floor_est:.1f} dB")

    ax.fill_between(freqs, noise_floor_est, magnitude_db_clipped,
                    where=(magnitude_db_clipped >= noise_floor_est),
                    color=DARK_THEME["i_channel"], alpha=0.12)

    ax.set_xlabel(freq_label, fontsize=10)
    ax.set_ylabel("Magnitude (dB)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_ylim(floor_db - 5, peak_db + 5)

    win_label = f"Window: {window}  |  N_FFT: {n_fft}"
    if remove_dc:
        win_label += "  |  DC removed"
    ax.legend(
        loc="upper right",
        facecolor=DARK_THEME["bg_axes"], edgecolor=DARK_THEME["grid"],
        labelcolor=DARK_THEME["text"], fontsize=9,
        title=win_label, title_fontsize=7,
    )

    plt.tight_layout()
    return fig, freqs, magnitude_db


# --------------------------------------------------------------------------- #
# 3. Spectrogram / Waterfall                                                    #
# --------------------------------------------------------------------------- #

def plot_spectrogram(
    samples: np.ndarray,
    sample_rate: Optional[float] = None,
    center_freq: float = 0.0,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    window: str = "hann",
    db_range: float = 50.0,
    remove_dc: bool = False,
    title: str = "Waterfall Spectrogram (STFT)",
) -> Tuple[plt.Figure, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute and plot an STFT waterfall spectrogram.

    Parameters
    ----------
    samples:
        Complex (I+jQ) or real 1-D NumPy array.
    sample_rate:
        Sampling rate in Hz. If None, time axis shows segment index and
        frequency axis shows normalized frequency.
    center_freq:
        RF center frequency offset in Hz (added to frequency axis when
        ``sample_rate`` is known).
    nperseg:
        Length of each FFT segment in samples.
    noverlap:
        Number of samples overlapping between segments. Defaults to
        ``nperseg // 2``.
    window:
        Window function name.
    db_range:
        Colourmap dynamic range in dB.
    remove_dc:
        If True, subtract the complex mean before computing the spectrogram.
    title:
        Plot title string.

    Returns
    -------
    fig : matplotlib.figure.Figure
    f : np.ndarray
        Frequency array (Hz or cycles/sample).
    t : np.ndarray
        Time array (seconds or segment index).
    Sxx_db : np.ndarray
        Spectrogram power in dB, shape (len(f), len(t)).
    """
    if samples is None or len(samples) < _MIN_SAMPLES:
        fig = _placeholder_fig("Insufficient data for spectrogram")
        return fig, np.array([]), np.array([]), np.array([[]])

    s = np.asarray(samples)
    s = _sanitize(s, "spectrogram")
    if remove_dc:
        s = _remove_dc(s)

    is_complex = np.iscomplexobj(s)
    n = len(s)

    # Adapt nperseg/noverlap to short signals
    effective_nperseg = min(nperseg, n)
    if noverlap is None:
        effective_noverlap = effective_nperseg // 2
    else:
        effective_noverlap = min(noverlap, effective_nperseg - 1)

    try:
        f, t, Sxx = scipy_signal.spectrogram(
            s,
            fs=sample_rate if sample_rate is not None else 1.0,
            window=window,
            nperseg=effective_nperseg,
            noverlap=effective_noverlap,
            return_onesided=not is_complex,
            scaling="density",
            mode="psd",
        )
    except Exception as exc:
        logger.error("spectrogram computation failed: %s", exc)
        fig = _placeholder_fig(f"Spectrogram computation failed: {exc}")
        return fig, np.array([]), np.array([]), np.array([[]])

    if is_complex:
        f = np.fft.fftshift(f)
        Sxx = np.fft.fftshift(Sxx, axes=0)

    Sxx_db = 10.0 * np.log10(np.maximum(Sxx, 1e-20))

    # Frequency axis
    if sample_rate is not None:
        f_plot = f + center_freq
        f_label = "Frequency (Hz)"
        t_plot = t
        t_label = "Time (s)"
    else:
        f_plot = f  # normalized [0,0.5] or [-0.5,0.5]
        f_label = "Normalized Frequency [sample_rate unknown]"
        t_plot = np.arange(Sxx_db.shape[1])
        t_label = "Segment Index"

    vmax = float(np.max(Sxx_db))
    vmin = max(vmax - db_range, float(np.min(Sxx_db)))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    apply_dark_theme(fig, ax)

    im = ax.pcolormesh(
        t_plot, f_plot, Sxx_db,
        shading="gouraud", cmap=DARK_THEME["cmap"],
        vmin=vmin, vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.tick_params(colors=DARK_THEME["text"], labelsize=8)
    cbar.set_label("Power (dB)", color=DARK_THEME["text"], fontsize=9)
    for spine in cbar.ax.spines.values():
        spine.set_color(DARK_THEME["grid"])

    ax.set_xlabel(t_label, fontsize=10)
    ax.set_ylabel(f_label, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    params_text = f"nperseg={effective_nperseg}  noverlap={effective_noverlap}  window={window}"
    ax.text(
        0.01, 0.98, params_text,
        transform=ax.transAxes, color=DARK_THEME["muted"],
        fontsize=7, va="top", ha="left",
    )

    plt.tight_layout()
    return fig, f_plot, t_plot, Sxx_db


# --------------------------------------------------------------------------- #
# 4. I/Q Constellation                                                          #
# --------------------------------------------------------------------------- #

def plot_constellation(
    symbols: np.ndarray,
    mod_type: Optional[str] = None,
    classification_status: Optional[ResultStatus] = None,
    confidence: Optional[float] = None,
    max_points: int = 3000,
    show_ideal: bool = True,
    title: str = "IQ Constellation Diagram",
) -> plt.Figure:
    """
    Plot an I/Q constellation scatter diagram.

    Parameters
    ----------
    symbols:
        Complex array of received symbols (I + jQ per IQ_CONVENTION).
    mod_type:
        Optional modulation type string (e.g. ``'BPSK'``, ``'QPSK'``,
        ``'16QAM'``). When provided and ``show_ideal=True``, ideal reference
        constellation points are overlaid.
    classification_status:
        Optional ``ResultStatus`` from ``core.contracts``.  If provided, the
        status and confidence are annotated on the plot.  Must be a
        ``ResultStatus`` enum member, not an ad-hoc string.
    confidence:
        Optional confidence float in [0.0, 1.0] from ``core.contracts``.
        Validated with ``validate_confidence``.
    max_points:
        Maximum number of symbols plotted (deterministic downsampling).
    show_ideal:
        If True, overlay ideal constellation reference points (not fabricated
        — only drawn when ``mod_type`` is recognized).
    title:
        Plot title string.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if symbols is None or len(symbols) < 1:
        return _placeholder_fig("Insufficient symbols for constellation")

    # Validate contracts inputs
    if classification_status is not None and not isinstance(classification_status, ResultStatus):
        raise TypeError(
            f"classification_status must be a ResultStatus (from core.contracts), "
            f"got {type(classification_status).__name__!r}. "
            "Do not pass raw strings."
        )
    validated_conf = validate_confidence(confidence) if confidence is not None else None

    syms = np.asarray(symbols, dtype=np.complex128)
    syms = _sanitize(syms, "constellation")

    # Deterministic downsampling: use evenly spaced indices
    n_orig = len(syms)
    if n_orig > max_points:
        idx = np.linspace(0, n_orig - 1, max_points, dtype=int)
        syms = syms[idx]

    i_vals = np.real(syms)
    q_vals = np.imag(syms)

    fig, ax = plt.subplots(figsize=(6, 6))
    apply_dark_theme(fig, ax)

    # Reference cross-hairs and unit circle
    ax.axhline(0, color=DARK_THEME["grid"], lw=1.0, linestyle="-", zorder=1)
    ax.axvline(0, color=DARK_THEME["grid"], lw=1.0, linestyle="-", zorder=1)
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color=DARK_THEME["grid"], linestyle=":", lw=0.8, zorder=1)

    # Received symbol scatter
    ax.scatter(
        i_vals, q_vals,
        c=DARK_THEME["i_channel"], alpha=0.45, s=14, edgecolors="none",
        label=f"Received ({len(syms):,} symbols)", zorder=3,
    )

    # Ideal reference points
    if show_ideal and mod_type is not None:
        ideal_pts = _get_ideal_constellation(mod_type)
        if ideal_pts is not None:
            ax.scatter(
                np.real(ideal_pts), np.imag(ideal_pts),
                c=DARK_THEME["ideal_pts"], marker="+", s=150, lw=2.5,
                label="Ideal alphabet", zorder=5,
            )

    ax.set_xlabel("In-Phase (I)", fontsize=10)
    ax.set_ylabel("Quadrature (Q)", fontsize=10)

    mod_str = f" [{mod_type}]" if mod_type else ""
    ax.set_title(f"{title}{mod_str}", fontsize=12, fontweight="bold", pad=8)

    # Symmetrical limits with a small margin
    lim = float(max(1.8, np.percentile(np.abs(syms), 99.0) * 1.3))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    ax.legend(
        loc="upper right",
        facecolor=DARK_THEME["bg_axes"], edgecolor=DARK_THEME["grid"],
        labelcolor=DARK_THEME["text"], fontsize=8,
    )

    # Classification annotation
    if classification_status is not None:
        conf_str = f"  conf={validated_conf:.2f}" if validated_conf is not None else ""
        ann_text = f"Status: {classification_status.value}{conf_str}"
        ann_color = DARK_THEME["ideal_pts"] if classification_status == ResultStatus.CONFIRMED else DARK_THEME["muted"]
        ax.text(
            0.03, 0.97, ann_text,
            transform=ax.transAxes, color=ann_color,
            fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc=DARK_THEME["bg_axes"],
                      ec=ann_color, lw=1.0),
        )

    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 5. Instantaneous Amplitude                                                    #
# --------------------------------------------------------------------------- #

def plot_amplitude(
    samples: np.ndarray,
    sample_rate: Optional[float] = None,
    max_samples: int = 2000,
    remove_dc: bool = False,
    title: str = "Instantaneous Amplitude |s(t)|",
) -> plt.Figure:
    """
    Plot the instantaneous amplitude (envelope) of a signal.

    Parameters
    ----------
    samples:
        Complex or real 1-D NumPy array.
    sample_rate:
        Sampling rate in Hz. If None, x-axis shows sample index.
    max_samples:
        Maximum number of samples rendered.
    remove_dc:
        If True, remove DC offset before computing amplitude.
    title:
        Plot title string.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if samples is None or len(samples) < _MIN_SAMPLES:
        return _placeholder_fig("Insufficient data for amplitude plot")

    s = np.asarray(samples)
    s = _sanitize(s, "amplitude")
    if remove_dc:
        s = _remove_dc(s)
    s = _downsample(s, max_samples)
    n = len(s)

    amp = np.abs(s)

    if sample_rate is not None and sample_rate > 0:
        t = np.arange(n) / sample_rate * 1e6
        x_label = "Time (\u00b5s)"
    else:
        t = np.arange(n)
        x_label = "Sample Index"

    amp_db = 20.0 * np.log10(np.maximum(amp, 1e-12))

    fig, (ax_lin, ax_db) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    apply_dark_theme(fig, [ax_lin, ax_db])

    # Linear amplitude
    ax_lin.plot(t, amp, color=DARK_THEME["envelope"], lw=1.3, label="|s(t)|")
    ax_lin.fill_between(t, 0, amp, color=DARK_THEME["envelope"], alpha=0.15)
    ax_lin.set_ylabel("Amplitude", fontsize=10)
    ax_lin.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax_lin.legend(
        loc="upper right",
        facecolor=DARK_THEME["bg_axes"], edgecolor=DARK_THEME["grid"],
        labelcolor=DARK_THEME["text"], fontsize=9,
    )

    # dB amplitude
    ax_db.plot(t, amp_db, color=DARK_THEME["q_channel"], lw=1.2, label="|s(t)| (dB)")
    ax_db.set_xlabel(x_label, fontsize=10)
    ax_db.set_ylabel("Amplitude (dB)", fontsize=10)
    ax_db.legend(
        loc="upper right",
        facecolor=DARK_THEME["bg_axes"], edgecolor=DARK_THEME["grid"],
        labelcolor=DARK_THEME["text"], fontsize=9,
    )

    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 6. Instantaneous Phase Trajectory                                             #
# --------------------------------------------------------------------------- #

def plot_phase(
    samples: np.ndarray,
    sample_rate: Optional[float] = None,
    max_samples: int = 2000,
    unwrap: bool = True,
    remove_dc: bool = False,
    title: str = "Instantaneous Phase Trajectory",
) -> plt.Figure:
    """
    Plot the instantaneous phase of a complex signal.

    Parameters
    ----------
    samples:
        Complex (I+jQ) 1-D NumPy array.  For real arrays, the Hilbert
        analytic signal is used to compute instantaneous phase.
    sample_rate:
        Sampling rate in Hz. If None, x-axis shows sample index.
    max_samples:
        Maximum number of samples rendered.
    unwrap:
        If True, apply ``np.unwrap`` to remove 2π discontinuities.
    remove_dc:
        If True, remove DC offset before computing phase.
    title:
        Plot title string.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if samples is None or len(samples) < _MIN_SAMPLES:
        return _placeholder_fig("Insufficient data for phase plot")

    s = np.asarray(samples)
    s = _sanitize(s, "phase")
    if remove_dc:
        s = _remove_dc(s)
    s = _downsample(s, max_samples)
    n = len(s)

    if not np.iscomplexobj(s):
        # Compute analytic signal to get instantaneous phase for real input
        from scipy.signal import hilbert as scipy_hilbert
        s = scipy_hilbert(s.astype(np.float64)).astype(np.complex128)

    phase_rad = np.angle(s)
    if unwrap:
        phase_rad = np.unwrap(phase_rad)

    phase_deg = np.degrees(phase_rad)

    if sample_rate is not None and sample_rate > 0:
        t = np.arange(n) / sample_rate * 1e6
        x_label = "Time (\u00b5s)"
    else:
        t = np.arange(n)
        x_label = "Sample Index"

    fig, ax = plt.subplots(figsize=(10, 4))
    apply_dark_theme(fig, ax)

    ax.plot(t, phase_deg, color=DARK_THEME["phase_line"], lw=1.2, label="Phase (deg)")
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("Phase (degrees)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    unwrap_note = " (unwrapped)" if unwrap else " (wrapped, -\u03c0 to \u03c0)"
    ax.legend(
        loc="upper right",
        facecolor=DARK_THEME["bg_axes"], edgecolor=DARK_THEME["grid"],
        labelcolor=DARK_THEME["text"], fontsize=9,
        title=f"Phase{unwrap_note}", title_fontsize=7,
    )

    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 7. Welch PSD (legacy / pipeline compatibility alias)                          #
# --------------------------------------------------------------------------- #

def plot_psd(
    samples: np.ndarray,
    sample_rate: Optional[float] = None,
    center_freq: float = 0.0,
    nperseg: int = 1024,
    title: str = "Power Spectral Density (Welch)",
) -> plt.Figure:
    """
    Compute and plot Welch Power Spectral Density.

    This function is retained for backwards compatibility and pipeline
    integration. New code should prefer ``plot_fft``.

    Parameters
    ----------
    samples:
        Complex or real 1-D NumPy array.
    sample_rate:
        Sampling rate in Hz. If None, normalized frequency is shown.
    center_freq:
        RF center frequency offset in Hz.
    nperseg:
        Welch segment length in samples.
    title:
        Plot title string.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if samples is None or len(samples) < _MIN_SAMPLES:
        return _placeholder_fig("Insufficient data for PSD")

    s = np.asarray(samples)
    s = _sanitize(s, "psd")
    is_complex = np.iscomplexobj(s)
    effective_nperseg = min(len(s), nperseg)

    fs = sample_rate if sample_rate is not None else 1.0

    f, pxx = scipy_signal.welch(
        s, fs=fs,
        window="hann",
        nperseg=effective_nperseg,
        return_onesided=not is_complex,
        scaling="density",
    )
    if is_complex:
        f = np.fft.fftshift(f)
        pxx = np.fft.fftshift(pxx)

    pxx_db = 10.0 * np.log10(np.maximum(pxx, 1e-20))

    if sample_rate is not None:
        freq_axis = f + center_freq
        freq_label = "Frequency (Hz)" if center_freq == 0 else f"Frequency (Hz)  [CF={center_freq/1e6:.3f} MHz]"
    else:
        freq_axis = f
        freq_label = "Normalized Frequency [sample_rate unknown]"

    noise_floor_est = float(np.percentile(pxx_db, 25))
    peak_power = float(np.max(pxx_db))
    peak_freq = float(freq_axis[np.argmax(pxx_db)])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    apply_dark_theme(fig, ax)

    ax.plot(freq_axis, pxx_db, color=DARK_THEME["i_channel"], lw=1.3, label="PSD (dB/Hz)")
    ax.axhline(noise_floor_est, color=DARK_THEME["muted"], linestyle=":", lw=1.2,
               label=f"Est. noise floor: {noise_floor_est:.1f} dB/Hz")
    ax.axvline(peak_freq, color=DARK_THEME["sync_peak"], linestyle="--", lw=1.2,
               label=f"Peak: {peak_freq:.4g} Hz ({peak_power:.1f} dB)")
    ax.fill_between(
        freq_axis, noise_floor_est, pxx_db,
        where=(pxx_db >= noise_floor_est),
        color=DARK_THEME["i_channel"], alpha=0.15,
    )

    ax.set_xlabel(freq_label, fontsize=10)
    ax.set_ylabel("Power Spectral Density (dB/Hz)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.legend(
        loc="upper right",
        facecolor=DARK_THEME["bg_axes"], edgecolor=DARK_THEME["grid"],
        labelcolor=DARK_THEME["text"], fontsize=9,
    )
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 8. Eye Diagram (bonus — used by pipeline)                                     #
# --------------------------------------------------------------------------- #

def plot_eye_diagram(
    samples: np.ndarray,
    sps: int,
    num_traces: int = 50,
    title: str = "Eye Diagram",
) -> plt.Figure:
    """
    Plot a folded eye diagram over 2 symbol periods.

    Parameters
    ----------
    samples:
        Complex or real 1-D NumPy array of pulse-shaped baseband samples.
    sps:
        Samples per symbol.
    num_traces:
        Maximum number of overlaid traces.
    title:
        Plot title string.

    Returns
    -------
    matplotlib.figure.Figure
    """
    trace_len = 2 * sps
    if sps <= 0 or trace_len <= 0 or len(samples) < trace_len * 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        apply_dark_theme(fig, ax)
        ax.text(
            0.5, 0.5, "Insufficient samples or invalid SPS for eye diagram",
            color=DARK_THEME["muted"], ha="center", va="center",
        )
        return fig

    fig, (ax_i, ax_q) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    apply_dark_theme(fig, [ax_i, ax_q])

    t = np.linspace(-1.0, 1.0, trace_len)
    total_traces = min(num_traces, (len(samples) - trace_len) // sps)

    for i in range(total_traces):
        idx_start = i * sps
        chunk = samples[idx_start : idx_start + trace_len]
        if len(chunk) == trace_len:
            ax_i.plot(t, np.real(chunk), color=DARK_THEME["i_channel"], alpha=0.25, lw=1.0)
            if np.iscomplexobj(samples):
                ax_q.plot(t, np.imag(chunk), color=DARK_THEME["q_channel"], alpha=0.25, lw=1.0)

    ax_i.axvline(0, color=DARK_THEME["envelope"], linestyle="--", lw=1.2, label="Strobe")
    ax_q.axvline(0, color=DARK_THEME["envelope"], linestyle="--", lw=1.2, label="Strobe")

    ax_i.set_xlabel("Symbol Period (T)", fontsize=10)
    ax_i.set_ylabel("In-Phase Amplitude", fontsize=10)
    ax_i.set_title("I Eye Opening", fontsize=11, fontweight="bold")
    ax_i.legend(
        loc="upper right",
        facecolor=DARK_THEME["bg_axes"], edgecolor=DARK_THEME["grid"],
        labelcolor=DARK_THEME["text"], fontsize=8,
    )

    ax_q.set_xlabel("Symbol Period (T)", fontsize=10)
    ax_q.set_ylabel("Quadrature Amplitude", fontsize=10)
    ax_q.set_title("Q Eye Opening", fontsize=11, fontweight="bold")
    ax_q.legend(
        loc="upper right",
        facecolor=DARK_THEME["bg_axes"], edgecolor=DARK_THEME["grid"],
        labelcolor=DARK_THEME["text"], fontsize=8,
    )

    plt.suptitle(title, color=DARK_THEME["text"], fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 9. Sync Correlation (bonus — used by pipeline)                                #
# --------------------------------------------------------------------------- #

def plot_sync_correlation(
    corr_curve: np.ndarray,
    peak_idx: int,
    threshold: float = 0.85,
    title: str = "Preamble Sync Cross-Correlation",
) -> plt.Figure:
    """
    Plot a normalized cross-correlation curve with sync threshold annotation.

    Parameters
    ----------
    corr_curve:
        1-D array of normalized correlation values.
    peak_idx:
        Index of the detected synchronization peak.
    threshold:
        Detection threshold to annotate on the plot.
    title:
        Plot title string.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if corr_curve is None or len(corr_curve) == 0:
        return _placeholder_fig("No correlation data")

    corr_curve = np.asarray(corr_curve)
    corr_curve = _sanitize(corr_curve, "sync_corr")

    fig, ax = plt.subplots(figsize=(10, 3.8))
    apply_dark_theme(fig, ax)

    indices = np.arange(len(corr_curve))
    ax.plot(indices, corr_curve, color=DARK_THEME["i_channel"], lw=1.2, label="|R_xy[n]|")
    ax.axhline(threshold, color=DARK_THEME["threshold"], linestyle="--", lw=1.2,
               label=f"Threshold ({threshold:.2f})")

    if 0 <= peak_idx < len(corr_curve):
        peak_val = float(corr_curve[peak_idx])
        ax.axvline(peak_idx, color=DARK_THEME["sync_peak"], linestyle="-", lw=1.5)
        ax.scatter([peak_idx], [peak_val], color=DARK_THEME["sync_peak"], s=80, zorder=5,
                   label=f"Peak @ bit {peak_idx} ({peak_val:.3f})")
        offset = max(1, len(corr_curve) // 20)
        ax.annotate(
            f"Sync @ {peak_idx}\n{peak_val:.3f}",
            xy=(peak_idx, peak_val),
            xytext=(peak_idx + offset, peak_val * 0.85),
            arrowprops=dict(facecolor=DARK_THEME["sync_peak"], shrink=0.08, width=1, headwidth=6),
            color=DARK_THEME["text"], fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc=DARK_THEME["bg_axes"],
                      ec=DARK_THEME["sync_peak"], lw=1),
        )

    ax.set_xlabel("Bit Offset", fontsize=10)
    ax.set_ylabel("Correlation", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(
        loc="upper right",
        facecolor=DARK_THEME["bg_axes"], edgecolor=DARK_THEME["grid"],
        labelcolor=DARK_THEME["text"], fontsize=9,
    )
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 10. Feature Radar (bonus — used by pipeline)                                  #
# --------------------------------------------------------------------------- #

def plot_feature_radar(
    features: Dict[str, float],
    title: str = "Signal Feature Profile",
) -> plt.Figure:
    """
    Plot a horizontal bar chart of normalized signal features.

    Parameters
    ----------
    features:
        Dictionary of feature name → float value.
    title:
        Plot title string.

    Returns
    -------
    matplotlib.figure.Figure
    """
    keys = ["c20_norm", "c40_norm", "c42_norm", "gamma_max", "sigma_dp", "sigma_aa", "spec_flatness"]
    labels = ["|C20| Norm", "|C40| Norm", "C42 Norm", "gamma_max", "sigma_dp", "sigma_aa", "Spectral Flatness"]

    values, display_labels = [], []
    for k, lbl in zip(keys, labels):
        if k in features:
            values.append(float(min(max(features[k], 0.0), 5.0)))
            display_labels.append(lbl)

    if not values:
        return _placeholder_fig("No feature data to display")

    fig, ax = plt.subplots(figsize=(8, max(3, len(values) * 0.6 + 1)))
    apply_dark_theme(fig, ax)

    y_pos = np.arange(len(display_labels))
    bars = ax.barh(y_pos, values, color=DARK_THEME["i_channel"], alpha=0.8,
                   edgecolor=DARK_THEME["q_channel"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Metric Magnitude", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}", va="center", ha="left",
            color=DARK_THEME["text"], fontsize=8,
        )

    ax.set_xlim(0, max(values, default=1.0) * 1.25 + 0.2)
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 11. SignalData-Aware Entry Point (CSE-1 integration)                          #
# --------------------------------------------------------------------------- #

def plot_from_signal(
    sig: SignalData,
    plot_type: str,
    **kwargs: Any,
) -> Any:
    """
    Unified entry point that accepts a ``SignalData`` object (from CSE-1)
    and dispatches to the appropriate visualization function.

    Parameters
    ----------
    sig:
        A ``SignalData`` instance produced by ``core.io.load_wav`` or
        ``core.io.load_iq``.
    plot_type:
        One of: ``'time'``, ``'fft'``, ``'spectrogram'``, ``'constellation'``,
        ``'amplitude'``, ``'phase'``, ``'psd'``.
    **kwargs:
        Additional keyword arguments forwarded to the underlying plot function.
        ``sample_rate`` and ``center_freq`` are automatically extracted from
        the ``SignalData`` object but can be overridden via kwargs.

    Returns
    -------
    Figure or (Figure, *arrays) depending on plot_type.

    Raises
    ------
    ValueError
        If ``plot_type`` is not recognized.
    TypeError
        If ``sig`` is not a ``SignalData`` instance.
    """
    if not isinstance(sig, SignalData):
        raise TypeError(
            f"sig must be a SignalData (from core.contracts), got {type(sig).__name__!r}."
        )

    # Inject metadata from SignalData unless caller overrides
    kwargs.setdefault("sample_rate", sig.sample_rate)
    kwargs.setdefault("center_freq", sig.center_freq)

    pt = plot_type.lower().strip()
    if pt == "time":
        # time_domain does not accept center_freq
        kwargs.pop("center_freq", None)
        return plot_time_domain(sig.samples, **kwargs)
    elif pt == "fft":
        return plot_fft(sig.samples, **kwargs)
    elif pt in ("spectrogram", "waterfall"):
        return plot_spectrogram(sig.samples, **kwargs)
    elif pt in ("constellation", "iq"):
        # constellation does not use center_freq or sample_rate
        kwargs.pop("center_freq", None)
        kwargs.pop("sample_rate", None)
        return plot_constellation(sig.samples, **kwargs)
    elif pt == "amplitude":
        kwargs.pop("center_freq", None)
        return plot_amplitude(sig.samples, **kwargs)
    elif pt == "phase":
        kwargs.pop("center_freq", None)
        return plot_phase(sig.samples, **kwargs)
    elif pt == "psd":
        return plot_psd(sig.samples, **kwargs)
    else:
        raise ValueError(
            f"Unknown plot_type {plot_type!r}. "
            "Supported: 'time', 'fft', 'spectrogram', 'constellation', "
            "'amplitude', 'phase', 'psd'."
        )


# --------------------------------------------------------------------------- #
# Internal: Ideal Constellation Reference Points                                #
# --------------------------------------------------------------------------- #

def _get_ideal_constellation(mod_type: Optional[str]) -> Optional[np.ndarray]:
    """
    Return ideal symbol coordinates for known modulation types.

    Points are normalized to unit average symbol energy.
    Returns None if ``mod_type`` is unknown (never fabricates).
    """
    if not mod_type:
        return None
    mt = str(mod_type).upper().strip()

    if "BPSK" in mt:
        return np.array([-1.0, 1.0], dtype=np.complex128)

    # Check 64QAM BEFORE 16QAM/QPSK to avoid substring false matches
    if "64QAM" in mt or "64-QAM" in mt:
        grid = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
        pts = np.array([r + 1j * c for r in grid for c in grid], dtype=np.complex128)
        return pts / np.sqrt(np.mean(np.abs(pts) ** 2))

    if "16QAM" in mt or "16-QAM" in mt:
        grid = np.array([-3.0, -1.0, 1.0, 3.0])
        pts = np.array([r + 1j * c for r in grid for c in grid], dtype=np.complex128)
        return pts / np.sqrt(np.mean(np.abs(pts) ** 2))

    if "QPSK" in mt or "4PSK" in mt or "4QAM" in mt:
        return np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex128) / np.sqrt(2)

    if "8PSK" in mt:
        angles = np.arange(8) * (2 * np.pi / 8)
        return np.exp(1j * angles).astype(np.complex128)

    if "OOK" in mt or "2-ASK" in mt or "2ASK" in mt:
        return np.array([0.0 + 0j, 1.0 + 0j], dtype=np.complex128)

    if "2FSK" in mt or "2-FSK" in mt or "BFSK" in mt:
        # FSK is not an amplitude/phase modulation; no standard IQ point constellation
        return None

    if "4FSK" in mt or "4-FSK" in mt:
        return None

    return None
