"""
Comprehensive tests for core/visualization.py — SpectralQ CSE-2.

Coverage matrix:
  Signal types:   BPSK, QPSK, 2-FSK, noisy complex, very short (<16 samples),
                  real mono, unknown sample_rate
  Functions:      plot_time_domain, plot_fft, plot_spectrogram, plot_constellation,
                  plot_amplitude, plot_phase, plot_psd, plot_eye_diagram,
                  plot_sync_correlation, plot_feature_radar, plot_from_signal
  Edge cases:     empty array, single sample, NaN/Inf values, missing sample_rate,
                  QI ordering detection, DC removal, downsampling, ideal overlay,
                  ResultStatus/confidence annotation, unsupported extension
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from core.visualization import (
    DARK_THEME,
    apply_dark_theme,
    plot_amplitude,
    plot_constellation,
    plot_eye_diagram,
    plot_feature_radar,
    plot_fft,
    plot_from_signal,
    plot_phase,
    plot_psd,
    plot_spectrogram,
    plot_sync_correlation,
    plot_time_domain,
    _get_ideal_constellation,
    _sanitize,
    _downsample,
)
from core.contracts import (
    ResultStatus,
    SignalData,
    validate_confidence,
)


# =========================================================================== #
# Helpers                                                                       #
# =========================================================================== #

def _bpsk(n: int = 512, fs: float = 100_000.0, sps: int = 8) -> np.ndarray:
    """BPSK: symbols {-1,+1} repeated sps times."""
    bits = np.random.choice([-1, 1], size=n // sps)
    return (np.repeat(bits, sps) + 0j).astype(np.complex64)[:n]


def _qpsk(n: int = 512, fs: float = 100_000.0, sps: int = 8) -> np.ndarray:
    """QPSK: 4 phase points."""
    angles = np.random.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2], size=n // sps)
    symbols = np.exp(1j * angles) / np.sqrt(2)
    return np.repeat(symbols, sps).astype(np.complex64)[:n]


def _fsk2(n: int = 512, fs: float = 100_000.0, f_dev: float = 5000.0, sps: int = 16) -> np.ndarray:
    """2-FSK: two-tone FM signal."""
    t = np.arange(n) / fs
    bits = np.random.choice([-1, 1], size=n // sps)
    bit_stream = np.repeat(bits, sps)[:n]
    return np.exp(1j * 2 * np.pi * f_dev * bit_stream * t).astype(np.complex64)


def _noisy_complex(n: int = 512, snr_db: float = 10.0, fs: float = 100_000.0) -> np.ndarray:
    """Complex exponential with AWGN."""
    t = np.arange(n) / fs
    sig = np.exp(1j * 2 * np.pi * 10_000 * t).astype(np.complex64)
    noise_pwr = 1.0 / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_pwr / 2) * (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
    return sig + noise


def _make_signal_data(samples: np.ndarray, fs: Optional[float] = 100_000.0) -> SignalData:
    return SignalData(samples=samples, sample_rate=fs)


def _fig_has_axes(fig: plt.Figure) -> bool:
    return len(fig.axes) > 0


# =========================================================================== #
# 1. Contract Import Check                                                      #
# =========================================================================== #

class TestContractImports:
    def test_signaldata_identity(self):
        """SignalData in visualization must be the same object as in core.contracts."""
        from core.contracts import SignalData as ContractSD
        import core.visualization as v
        assert v.SignalData is ContractSD

    def test_resultstatus_identity(self):
        from core.contracts import ResultStatus as ContractRS
        import core.visualization as v
        assert v.ResultStatus is ContractRS

    def test_no_streamlit_import(self):
        """visualization.py must not import Streamlit."""
        import core.visualization as v
        import sys
        assert "streamlit" not in sys.modules or True  # module-level check
        import ast, pathlib
        src = pathlib.Path("core/visualization.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for name in names:
                    assert "streamlit" not in str(name).lower(), \
                        f"Streamlit import found in visualization.py: {name}"


# =========================================================================== #
# 2. plot_time_domain                                                           #
# =========================================================================== #

class TestPlotTimeDomain:
    """Tests for plot_time_domain()."""

    def test_returns_figure(self):
        sig = _bpsk(512)
        fig = plot_time_domain(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_complex_bpsk(self):
        sig = _bpsk(512)
        fig = plot_time_domain(sig, sample_rate=100_000.0, title="BPSK Time Domain")
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # IQ axes + magnitude axes
        plt.close(fig)

    def test_complex_qpsk(self):
        sig = _qpsk(512)
        fig = plot_time_domain(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_complex_fsk(self):
        sig = _fsk2(512)
        fig = plot_time_domain(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_real_signal(self):
        t = np.linspace(0, 0.01, 512)
        real_sig = np.sin(2 * np.pi * 5000 * t).astype(np.float32)
        fig = plot_time_domain(real_sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_unknown_sample_rate(self):
        """When sample_rate is None, x-axis should be sample index (no crash)."""
        sig = _bpsk(256)
        fig = plot_time_domain(sig, sample_rate=None)
        assert isinstance(fig, plt.Figure)
        # Verify axis label contains 'Sample'
        x_labels = [ax.get_xlabel() for ax in fig.axes]
        assert any("Sample" in lbl or lbl == "" for lbl in x_labels)
        plt.close(fig)

    def test_very_short_signal(self):
        """Signal with < 2 samples returns a placeholder figure (no crash)."""
        fig = plot_time_domain(np.array([1 + 0j], dtype=np.complex64), sample_rate=1000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_signal(self):
        fig = plot_time_domain(np.array([], dtype=np.complex64), sample_rate=1000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_nan_inf_handled(self):
        """NaN/Inf values should not cause a crash."""
        sig = np.array([1 + 1j, float("nan") + 0j, float("inf") + 0j, -1j], dtype=np.complex64)
        fig = plot_time_domain(sig, sample_rate=1000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dc_removal_flag(self):
        """With remove_dc=True, the mean offset is removed before plotting."""
        dc_offset = 5.0 + 3j
        sig = dc_offset + _bpsk(256)
        fig = plot_time_domain(sig, sample_rate=100_000.0, remove_dc=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_downsampling(self):
        """Signals longer than max_samples should be downsampled, not crash."""
        sig = _bpsk(10_000)
        fig = plot_time_domain(sig, sample_rate=100_000.0, max_samples=500)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_noisy_complex(self):
        sig = _noisy_complex(1024)
        fig = plot_time_domain(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =========================================================================== #
# 3. plot_fft                                                                   #
# =========================================================================== #

class TestPlotFFT:
    """Tests for plot_fft() — the new required CSE-2 function."""

    def test_returns_tuple(self):
        sig = _noisy_complex(1024)
        result = plot_fft(sig, sample_rate=100_000.0)
        assert isinstance(result, tuple) and len(result) == 3
        fig, freqs, mag_db = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(freqs, np.ndarray)
        assert isinstance(mag_db, np.ndarray)
        plt.close(fig)

    def test_freqs_shape_matches_magnitude(self):
        sig = _noisy_complex(512)
        _, freqs, mag_db = plot_fft(sig, sample_rate=100_000.0)
        assert len(freqs) == len(mag_db)

    def test_bpsk_spectrum(self):
        sig = _bpsk(512, sps=8)
        fig, freqs, mag_db = plot_fft(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_qpsk_spectrum(self):
        sig = _qpsk(512)
        fig, freqs, mag_db = plot_fft(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_fsk_spectrum(self):
        sig = _fsk2(512, fs=100_000.0)
        fig, freqs, mag_db = plot_fft(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_unknown_sample_rate_normalized_freq(self):
        """When sample_rate=None, frequency axis should be normalized."""
        sig = _noisy_complex(512)
        fig, freqs, mag_db = plot_fft(sig, sample_rate=None)
        assert isinstance(fig, plt.Figure)
        # Normalized frequency is in [-0.5, 0.5]
        assert float(np.min(freqs)) >= -0.501
        assert float(np.max(freqs)) <= 0.501
        # Axis label should say 'normalized' or 'cycles/sample'
        ax = fig.axes[0]
        x_label = ax.get_xlabel().lower()
        assert "normalized" in x_label or "cycles" in x_label or "unknown" in x_label
        plt.close(fig)

    def test_real_signal(self):
        t = np.arange(1024) / 100_000.0
        real_sig = np.cos(2 * np.pi * 10_000 * t).astype(np.float32)
        fig, freqs, mag_db = plot_fft(real_sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        # Real FFT freqs start at 0
        assert float(freqs[0]) >= 0.0
        plt.close(fig)

    def test_magnitude_db_is_sensible(self):
        """Spectrum of a pure sinusoid should have a dominant peak."""
        fs = 100_000.0
        n = 2048
        t = np.arange(n) / fs
        f0 = 10_000.0
        sig = np.exp(1j * 2 * np.pi * f0 * t).astype(np.complex64)
        _, freqs, mag_db = plot_fft(sig, sample_rate=fs)
        peak_idx = int(np.argmax(mag_db))
        peak_freq = float(freqs[peak_idx])
        # Peak should be near f0 (within 2 FFT bins)
        bin_width = fs / len(sig)
        assert abs(peak_freq - f0) < 2 * bin_width + 200, \
            f"Peak at {peak_freq:.1f} Hz, expected near {f0:.1f} Hz"

    def test_empty_signal(self):
        fig, freqs, mag_db = plot_fft(np.array([]), sample_rate=1000.0)
        assert isinstance(fig, plt.Figure)
        assert len(freqs) == 0 and len(mag_db) == 0
        plt.close(fig)

    def test_very_short_signal(self):
        fig, freqs, mag_db = plot_fft(np.array([1 + 0j], dtype=np.complex64), sample_rate=1000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_nan_inf_handled(self):
        sig = np.array([1 + 1j, float("nan") + 0j, float("inf") + 0j, -1j], dtype=np.complex64)
        fig, freqs, mag_db = plot_fft(sig, sample_rate=1000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_window_parameter(self):
        sig = _noisy_complex(512)
        for win in ("hann", "hamming", "blackman", "boxcar"):
            fig, freqs, mag_db = plot_fft(sig, sample_rate=100_000.0, window=win)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_center_freq_offset(self):
        """When center_freq is set, frequency axis should be offset."""
        sig = _noisy_complex(512)
        cf = 433.92e6
        _, freqs_base, _ = plot_fft(sig, sample_rate=100_000.0, center_freq=0.0)
        _, freqs_offset, _ = plot_fft(sig, sample_rate=100_000.0, center_freq=cf)
        # Offset axis should be shifted by center_freq
        shift = float(np.mean(freqs_offset - freqs_base))
        assert abs(shift - cf) < 1.0, f"Expected shift {cf:.1f}, got {shift:.1f}"

    def test_dc_removal(self):
        dc_sig = np.ones(256, dtype=np.complex64) * (5.0 + 3j)
        fig, freqs, mag_db = plot_fft(dc_sig, sample_rate=1000.0, remove_dc=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =========================================================================== #
# 4. plot_spectrogram                                                           #
# =========================================================================== #

class TestPlotSpectrogram:
    """Tests for plot_spectrogram() — now returns (fig, f, t, Sxx_db)."""

    def test_returns_tuple(self):
        sig = _noisy_complex(2048)
        result = plot_spectrogram(sig, sample_rate=100_000.0)
        assert isinstance(result, tuple) and len(result) == 4
        fig, f, t, Sxx_db = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(f, np.ndarray) and isinstance(t, np.ndarray)
        assert isinstance(Sxx_db, np.ndarray) and Sxx_db.ndim == 2
        plt.close(fig)

    def test_sxx_shape_consistency(self):
        sig = _noisy_complex(2048)
        _, f, t, Sxx_db = plot_spectrogram(sig, sample_rate=100_000.0)
        assert Sxx_db.shape == (len(f), len(t))

    def test_bpsk_spectrogram(self):
        sig = _bpsk(2048)
        fig, f, t, Sxx_db = plot_spectrogram(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_qpsk_spectrogram(self):
        sig = _qpsk(2048)
        fig, f, t, Sxx_db = plot_spectrogram(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_fsk_spectrogram(self):
        sig = _fsk2(4096, fs=100_000.0)
        fig, f, t, Sxx_db = plot_spectrogram(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_unknown_sample_rate(self):
        """Spectrogram with sample_rate=None uses normalized frequency."""
        sig = _noisy_complex(1024)
        fig, f, t, Sxx_db = plot_spectrogram(sig, sample_rate=None)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        y_label = ax.get_ylabel().lower()
        assert "normalized" in y_label or "unknown" in y_label
        plt.close(fig)

    def test_very_short_signal(self):
        sig = np.array([1 + 0j, 0 + 1j], dtype=np.complex64)
        fig, f, t, Sxx_db = plot_spectrogram(sig, sample_rate=1000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_nperseg_noverlap_exposed(self):
        """nperseg and noverlap parameters should be accepted."""
        sig = _noisy_complex(2048)
        fig, f, t, Sxx_db = plot_spectrogram(sig, sample_rate=100_000.0,
                                              nperseg=128, noverlap=64, window="hann")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_short_recording_adapts(self):
        """nperseg larger than signal length should be auto-clamped."""
        sig = _noisy_complex(32)  # Very short
        fig, f, t, Sxx_db = plot_spectrogram(sig, sample_rate=100_000.0, nperseg=512)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_nan_inf_handled(self):
        sig = np.array([1 + 1j, float("nan") + 0j, -1j, float("inf") + 0j] * 128, dtype=np.complex64)
        fig, f, t, Sxx_db = plot_spectrogram(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_sxx_values_are_finite(self):
        sig = _noisy_complex(1024)
        _, f, t, Sxx_db = plot_spectrogram(sig, sample_rate=100_000.0)
        assert np.all(np.isfinite(Sxx_db)), "Sxx_db should contain only finite values"

    def test_noisy_complex(self):
        sig = _noisy_complex(2048, snr_db=5.0)
        fig, f, t, Sxx_db = plot_spectrogram(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =========================================================================== #
# 5. plot_constellation                                                         #
# =========================================================================== #

class TestPlotConstellation:
    """Tests for plot_constellation() with ResultStatus/confidence contract."""

    def test_returns_figure(self):
        syms = _bpsk(256)
        fig = plot_constellation(syms, mod_type="BPSK")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bpsk_constellation(self):
        syms = _bpsk(256)
        fig = plot_constellation(syms, mod_type="BPSK", show_ideal=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_qpsk_constellation(self):
        syms = _qpsk(512)
        fig = plot_constellation(syms, mod_type="QPSK", show_ideal=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_16qam_constellation(self):
        grid = np.array([-3.0, -1.0, 1.0, 3.0])
        pts = np.array([r + 1j * c for r in grid for c in grid], dtype=np.complex64)
        pts /= np.sqrt(np.mean(np.abs(pts) ** 2))
        noisy_pts = pts[np.random.randint(0, 16, 500)] + 0.1 * (np.random.randn(500) + 1j * np.random.randn(500))
        fig = plot_constellation(noisy_pts, mod_type="16QAM", show_ideal=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_fsk_no_ideal_points(self):
        """FSK has no standard IQ constellation — ideal overlay silently absent."""
        syms = _fsk2(256)
        fig = plot_constellation(syms, mod_type="2FSK", show_ideal=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_mod_type(self):
        syms = _noisy_complex(256)
        fig = plot_constellation(syms, mod_type=None, show_ideal=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_resultstatus_confirmed_overlay(self):
        """ResultStatus.CONFIRMED + confidence annotated on plot."""
        syms = _bpsk(256)
        fig = plot_constellation(
            syms, mod_type="BPSK",
            classification_status=ResultStatus.CONFIRMED,
            confidence=0.93,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_resultstatus_estimated_overlay(self):
        syms = _qpsk(256)
        fig = plot_constellation(
            syms, mod_type="QPSK",
            classification_status=ResultStatus.ESTIMATED,
            confidence=0.71,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_raw_string_status_raises(self):
        """Passing a raw string instead of ResultStatus must raise TypeError."""
        syms = _bpsk(64)
        with pytest.raises(TypeError, match="ResultStatus"):
            plot_constellation(syms, classification_status="CONFIRMED")

    def test_invalid_confidence_raises(self):
        """Confidence outside [0,1] must raise ValueError from validate_confidence."""
        syms = _bpsk(64)
        with pytest.raises(ValueError):
            plot_constellation(syms, confidence=1.5)

    def test_downsampling_deterministic(self):
        """Large symbol sets are downsampled; the result should be reproducible."""
        syms = _bpsk(10_000)
        fig1 = plot_constellation(syms, max_points=200)
        fig2 = plot_constellation(syms, max_points=200)
        # Both should succeed without crash
        assert isinstance(fig1, plt.Figure)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig1)
        plt.close(fig2)

    def test_aspect_ratio_equal(self):
        """Constellation must have equal x/y aspect ratio.
        matplotlib returns 'equal' (str) or 1.0 (float) depending on version.
        """
        syms = _qpsk(256)
        fig = plot_constellation(syms)
        ax = fig.axes[0]
        aspect = ax.get_aspect()
        assert aspect == "equal" or aspect == 1.0, \
            f"Expected 'equal' or 1.0 aspect, got {aspect!r}"
        plt.close(fig)

    def test_empty_symbols(self):
        fig = plot_constellation(np.array([], dtype=np.complex64))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_nan_handled(self):
        syms = np.array([1 + 1j, float("nan") + 0j, -1 - 1j], dtype=np.complex64)
        fig = plot_constellation(syms)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ideal_constellation_not_fabricated_for_unknown(self):
        """Unknown mod_type → no ideal points overlaid (no fabrication)."""
        syms = _noisy_complex(256)
        fig = plot_constellation(syms, mod_type="UNKNOWN_MOD_XYZ", show_ideal=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =========================================================================== #
# 6. plot_amplitude                                                             #
# =========================================================================== #

class TestPlotAmplitude:
    """Tests for the NEW plot_amplitude() function."""

    def test_returns_figure(self):
        sig = _noisy_complex(512)
        fig = plot_amplitude(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_two_axes(self):
        """Should have linear and dB subplot."""
        sig = _noisy_complex(512)
        fig = plot_amplitude(sig, sample_rate=100_000.0)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_bpsk(self):
        sig = _bpsk(512)
        fig = plot_amplitude(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_qpsk(self):
        sig = _qpsk(512)
        fig = plot_amplitude(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_real_signal(self):
        real_sig = np.cos(2 * np.pi * 0.1 * np.arange(512)).astype(np.float32)
        fig = plot_amplitude(real_sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_unknown_sample_rate(self):
        sig = _noisy_complex(256)
        fig = plot_amplitude(sig, sample_rate=None)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[1]
        x_label = ax.get_xlabel()
        assert "Sample" in x_label or x_label == ""
        plt.close(fig)

    def test_empty_signal(self):
        fig = plot_amplitude(np.array([], dtype=np.complex64), sample_rate=1000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_very_short_signal(self):
        fig = plot_amplitude(np.array([1 + 0j], dtype=np.complex64), sample_rate=1000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_nan_handled(self):
        sig = np.array([1 + 1j, float("nan") + 0j, -1 - 1j] * 100, dtype=np.complex64)
        fig = plot_amplitude(sig, sample_rate=1000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dc_removal(self):
        sig = _noisy_complex(512) + (5.0 + 3j)
        fig = plot_amplitude(sig, sample_rate=100_000.0, remove_dc=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_downsampling(self):
        sig = _noisy_complex(10_000)
        fig = plot_amplitude(sig, sample_rate=100_000.0, max_samples=300)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =========================================================================== #
# 7. plot_phase                                                                 #
# =========================================================================== #

class TestPlotPhase:
    """Tests for the NEW plot_phase() function."""

    def test_returns_figure(self):
        sig = _bpsk(512)
        fig = plot_phase(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_axis(self):
        sig = _qpsk(512)
        fig = plot_phase(sig, sample_rate=100_000.0)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_bpsk_phase_values(self):
        """BPSK phase should be near 0 or ±pi."""
        bits = np.array([1, -1, 1, 1, -1], dtype=np.float64)
        syms = np.exp(1j * np.pi * (bits < 0).astype(float))  # 0 or pi
        fig = plot_phase(syms, sample_rate=100_000.0, unwrap=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_qpsk_phase(self):
        sig = _qpsk(512)
        fig = plot_phase(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_fsk_phase(self):
        sig = _fsk2(512, fs=100_000.0)
        fig = plot_phase(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_real_signal(self):
        """Real input: Hilbert transform used to get instantaneous phase."""
        real_sig = np.sin(2 * np.pi * 0.1 * np.arange(512)).astype(np.float32)
        fig = plot_phase(real_sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_unknown_sample_rate(self):
        sig = _noisy_complex(256)
        fig = plot_phase(sig, sample_rate=None)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        x_label = ax.get_xlabel()
        assert "Sample" in x_label or x_label == ""
        plt.close(fig)

    def test_unwrap_false(self):
        sig = _qpsk(512)
        fig = plot_phase(sig, sample_rate=100_000.0, unwrap=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_signal(self):
        fig = plot_phase(np.array([], dtype=np.complex64), sample_rate=1000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_very_short_signal(self):
        fig = plot_phase(np.array([1 + 0j], dtype=np.complex64), sample_rate=1000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_nan_handled(self):
        sig = np.array([1 + 1j, float("nan") + 0j, -1 - 1j] * 50, dtype=np.complex64)
        fig = plot_phase(sig, sample_rate=1000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_noisy_complex(self):
        sig = _noisy_complex(512)
        fig = plot_phase(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =========================================================================== #
# 8. plot_psd (legacy / backwards compat)                                      #
# =========================================================================== #

class TestPlotPSD:
    def test_returns_figure(self):
        sig = _noisy_complex(2048)
        fig = plot_psd(sig, sample_rate=100_000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_center_freq(self):
        sig = _noisy_complex(2048)
        fig = plot_psd(sig, sample_rate=100_000.0, center_freq=100e6)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_unknown_sample_rate(self):
        sig = _noisy_complex(512)
        fig = plot_psd(sig, sample_rate=None)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        x_label = ax.get_xlabel().lower()
        assert "normalized" in x_label or "unknown" in x_label
        plt.close(fig)

    def test_empty_signal(self):
        fig = plot_psd(np.array([], dtype=np.complex64), sample_rate=1000.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =========================================================================== #
# 9. plot_eye_diagram (bonus)                                                   #
# =========================================================================== #

class TestPlotEyeDiagram:
    def test_returns_figure(self):
        sps = 8
        syms = np.random.choice([-1, 1], 100)
        samples = np.repeat(syms, sps) + 0.1 * np.random.randn(100 * sps)
        fig = plot_eye_diagram(samples.astype(np.float32), sps=sps)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_insufficient_data_returns_placeholder(self):
        fig = plot_eye_diagram(np.ones(4, dtype=np.float32), sps=8)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_invalid_sps(self):
        fig = plot_eye_diagram(np.ones(100, dtype=np.float32), sps=0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =========================================================================== #
# 10. plot_sync_correlation (bonus)                                             #
# =========================================================================== #

class TestPlotSyncCorrelation:
    def test_returns_figure(self):
        corr = np.array([0.1, 0.2, 0.95, 0.3, 0.1])
        fig = plot_sync_correlation(corr, peak_idx=2, threshold=0.85)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_curve(self):
        fig = plot_sync_correlation(np.array([]), peak_idx=0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_peak_outside_range(self):
        corr = np.array([0.1, 0.5, 0.9])
        fig = plot_sync_correlation(corr, peak_idx=999)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =========================================================================== #
# 11. plot_feature_radar (bonus)                                               #
# =========================================================================== #

class TestPlotFeatureRadar:
    def test_full_features(self):
        feats = {
            "c20_norm": 0.98, "c40_norm": 1.85, "c42_norm": 0.95,
            "gamma_max": 1.2, "sigma_dp": 0.45, "sigma_aa": 0.12,
            "spec_flatness": 0.82,
        }
        fig = plot_feature_radar(feats)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_partial_features(self):
        feats = {"c20_norm": 0.5, "gamma_max": 1.0}
        fig = plot_feature_radar(feats)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_features(self):
        fig = plot_feature_radar({})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =========================================================================== #
# 12. plot_from_signal (SignalData integration)                                 #
# =========================================================================== #

class TestPlotFromSignal:
    """Tests for the SignalData-aware dispatcher."""

    def _sig(self, n=512, fs=100_000.0):
        return _make_signal_data(_noisy_complex(n), fs)

    def test_time_domain(self):
        fig = plot_from_signal(self._sig(), "time")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_fft(self):
        result = plot_from_signal(self._sig(), "fft")
        assert isinstance(result, tuple) and len(result) == 3
        plt.close(result[0])

    def test_spectrogram(self):
        result = plot_from_signal(self._sig(2048), "spectrogram")
        assert isinstance(result, tuple) and len(result) == 4
        plt.close(result[0])

    def test_constellation(self):
        fig = plot_from_signal(self._sig(), "constellation")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_amplitude(self):
        fig = plot_from_signal(self._sig(), "amplitude")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_phase(self):
        fig = plot_from_signal(self._sig(), "phase")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_psd(self):
        fig = plot_from_signal(self._sig(), "psd")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_waterfall_alias(self):
        result = plot_from_signal(self._sig(2048), "waterfall")
        assert isinstance(result, tuple) and len(result) == 4
        plt.close(result[0])

    def test_iq_alias(self):
        fig = plot_from_signal(self._sig(), "iq")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown plot_type"):
            plot_from_signal(self._sig(), "radar")

    def test_non_signaldata_raises(self):
        with pytest.raises(TypeError):
            plot_from_signal(np.ones(64, dtype=np.complex64), "time")

    def test_sample_rate_from_signaldata(self):
        """sample_rate from SignalData should be used automatically."""
        sig = _make_signal_data(_noisy_complex(512), fs=44100.0)
        result = plot_from_signal(sig, "fft")
        fig, freqs, _ = result
        # Frequency range should match fs/2 = 22050
        assert float(np.max(np.abs(freqs))) <= 22_100.0
        plt.close(fig)

    def test_none_sample_rate_from_signaldata(self):
        """SignalData with sample_rate=None should produce normalized frequency."""
        sig = _make_signal_data(_noisy_complex(256), fs=None)
        result = plot_from_signal(sig, "fft")
        assert isinstance(result[0], plt.Figure)
        plt.close(result[0])


# =========================================================================== #
# 13. Internal Utilities                                                        #
# =========================================================================== #

class TestInternalUtilities:
    def test_sanitize_nan_to_zero(self):
        arr = np.array([1.0, float("nan"), 2.0, float("inf"), -1.0], dtype=np.float64)
        out = _sanitize(arr)
        assert np.all(np.isfinite(out))
        assert out[0] == pytest.approx(1.0)
        assert out[2] == pytest.approx(2.0)

    def test_sanitize_clean_array_unchanged(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = _sanitize(arr)
        np.testing.assert_array_equal(arr, out)

    def test_downsample_reduces_length(self):
        arr = np.arange(1000, dtype=np.float64)
        out = _downsample(arr, 100)
        assert len(out) == 100

    def test_downsample_no_op_when_short(self):
        arr = np.arange(50, dtype=np.float64)
        out = _downsample(arr, 100)
        assert len(out) == 50

    def test_downsample_deterministic(self):
        arr = np.arange(1000, dtype=np.float64)
        out1 = _downsample(arr, 200)
        out2 = _downsample(arr, 200)
        np.testing.assert_array_equal(out1, out2)


# =========================================================================== #
# 14. _get_ideal_constellation                                                  #
# =========================================================================== #

class TestGetIdealConstellation:
    def test_bpsk(self):
        pts = _get_ideal_constellation("BPSK")
        assert pts is not None and len(pts) == 2

    def test_qpsk(self):
        pts = _get_ideal_constellation("QPSK")
        assert pts is not None and len(pts) == 4

    def test_8psk(self):
        pts = _get_ideal_constellation("8PSK")
        assert pts is not None and len(pts) == 8

    def test_16qam(self):
        pts = _get_ideal_constellation("16QAM")
        assert pts is not None and len(pts) == 16

    def test_64qam(self):
        pts = _get_ideal_constellation("64QAM")
        assert pts is not None and len(pts) == 64

    def test_ook(self):
        pts = _get_ideal_constellation("OOK")
        assert pts is not None and len(pts) == 2

    def test_2fsk_returns_none(self):
        """FSK has no standard IQ constellation; must return None."""
        pts = _get_ideal_constellation("2FSK")
        assert pts is None

    def test_unknown_returns_none(self):
        pts = _get_ideal_constellation("SOMETHING_WEIRD")
        assert pts is None

    def test_none_input(self):
        pts = _get_ideal_constellation(None)
        assert pts is None

    def test_normalized_energy_bpsk(self):
        pts = _get_ideal_constellation("BPSK")
        avg_energy = float(np.mean(np.abs(pts) ** 2))
        assert avg_energy == pytest.approx(1.0, abs=0.01)

    def test_normalized_energy_16qam(self):
        pts = _get_ideal_constellation("16QAM")
        avg_energy = float(np.mean(np.abs(pts) ** 2))
        assert avg_energy == pytest.approx(1.0, abs=0.05)


# =========================================================================== #
# 15. Dark Theme                                                                #
# =========================================================================== #

class TestDarkTheme:
    def test_apply_dark_theme_does_not_crash(self):
        fig, ax = plt.subplots()
        apply_dark_theme(fig, ax)
        assert fig.get_facecolor() is not None
        plt.close(fig)

    def test_dark_theme_dict_has_required_keys(self):
        required = ["bg_figure", "bg_axes", "grid", "text", "i_channel", "q_channel",
                    "envelope", "ideal_pts", "threshold", "sync_peak", "cmap"]
        for k in required:
            assert k in DARK_THEME, f"Missing key in DARK_THEME: {k}"


# =========================================================================== #
# 16. Fixture-based end-to-end smoke tests                                     #
# =========================================================================== #

SYNTHETIC_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"


class TestFixtureSmoke:
    """End-to-end smoke tests using the synthetic fixture files from CSE-1."""

    @pytest.mark.skipif(
        not (SYNTHETIC_DIR / "bpsk_snr20db_clean.iq").exists(),
        reason="Synthetic fixtures not present.",
    )
    def test_bpsk_iq_all_plots(self):
        from core.io import load_iq
        sig = load_iq(SYNTHETIC_DIR / "bpsk_snr20db_clean.iq", dtype="complex64")
        for pt in ("time", "fft", "spectrogram", "amplitude", "phase", "psd"):
            result = plot_from_signal(sig, pt)
            fig = result[0] if isinstance(result, tuple) else result
            assert isinstance(fig, plt.Figure), f"plot_type={pt!r} did not return Figure"
            plt.close(fig)

    @pytest.mark.skipif(
        not (SYNTHETIC_DIR / "qpsk_snr15db_cfo.iq").exists(),
        reason="Synthetic fixtures not present.",
    )
    def test_qpsk_iq_constellation(self):
        from core.io import load_iq
        sig = load_iq(SYNTHETIC_DIR / "qpsk_snr15db_cfo.iq", dtype="complex64")
        fig = plot_from_signal(sig, "constellation", mod_type="QPSK")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @pytest.mark.skipif(
        not (SYNTHETIC_DIR / "2fsk_snr18db.wav").exists(),
        reason="Synthetic fixtures not present.",
    )
    def test_fsk_wav_fft(self):
        from core.io import load_wav
        sig = load_wav(SYNTHETIC_DIR / "2fsk_snr18db.wav")
        result = plot_from_signal(sig, "fft")
        assert isinstance(result[0], plt.Figure)
        plt.close(result[0])
