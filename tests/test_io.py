"""
Unit tests for core/io.py — SpectralQ CSE-1: Robust Signal File Ingestion.

Coverage:
  - load_wav / load_wav_file:  mono, stereo, int16, int32, float32, QI order
  - load_iq / load_iq_file:   int8, uint8, int16, float32, float64, QI layout
  - load_signal:              extension dispatch
  - validate_signal:          valid / invalid / partial signals
  - normalize_signal:         DC removal, amplitude norm, gain, clipping detection
  - infer_basic_metadata:     structural completeness
  - save_iq_file / save_wav_file: roundtrip and companion JSON
  - Error paths: missing file, empty file, bad dtype, unsupported extension,
                 odd element count, missing sample_rate warning, NaN/Inf values
"""

import json
import struct
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from scipy.io import wavfile

# Public API under test
from core.io import (
    infer_basic_metadata,
    load_iq,
    load_iq_file,
    load_signal,
    load_wav,
    load_wav_file,
    normalize_signal,
    save_iq_file,
    save_wav_file,
    validate_signal,
)
from core.contracts import SignalData, ResultStatus, IQ_CONVENTION, BIT_ORDERING


# =========================================================================== #
# Helpers                                                                       #
# =========================================================================== #

def _write_raw_iq(path: Path, i: np.ndarray, q: np.ndarray, elem_dtype: np.dtype) -> None:
    """Write interleaved IQ pairs to a raw binary file."""
    n = min(len(i), len(q))
    interleaved = np.empty(n * 2, dtype=elem_dtype)
    interleaved[0::2] = i[:n].astype(elem_dtype)
    interleaved[1::2] = q[:n].astype(elem_dtype)
    interleaved.tofile(str(path))


def _write_raw_qi(path: Path, i: np.ndarray, q: np.ndarray, elem_dtype: np.dtype) -> None:
    """Write interleaved QI pairs (Q first) to a raw binary file."""
    n = min(len(i), len(q))
    interleaved = np.empty(n * 2, dtype=elem_dtype)
    interleaved[0::2] = q[:n].astype(elem_dtype)
    interleaved[1::2] = i[:n].astype(elem_dtype)
    interleaved.tofile(str(path))


def _make_complex_signal(n: int = 512, freq_hz: float = 1000.0, fs: float = 44100.0) -> np.ndarray:
    """Return a complex exponential test signal."""
    t = np.arange(n) / fs
    return np.exp(1j * 2 * np.pi * freq_hz * t).astype(np.complex64)


def _make_real_signal(n: int = 512, freq_hz: float = 1000.0, fs: float = 44100.0) -> np.ndarray:
    """Return a real cosine test signal."""
    t = np.arange(n) / fs
    return np.cos(2 * np.pi * freq_hz * t).astype(np.float32)


# =========================================================================== #
# 1. SignalData contract (imported from contracts — identity check)            #
# =========================================================================== #

class TestSignalDataContract:
    def test_imports_from_contracts(self):
        """SignalData must come from core.contracts, not be redefined."""
        from core.contracts import SignalData as ContractSignalData
        import core.io as io_mod
        assert io_mod.SignalData is ContractSignalData

    def test_basic_creation_complex(self):
        samples = np.array([1+2j, 3+4j], dtype=np.complex64)
        sig = SignalData(samples=samples, sample_rate=10000.0)
        assert sig.num_samples == 2
        assert sig.is_complex is True
        assert sig.duration == pytest.approx(2 / 10000.0)

    def test_basic_creation_real(self):
        samples = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sig = SignalData(samples=samples, sample_rate=1000.0, is_complex=False)
        assert sig.num_samples == 3
        assert sig.is_complex is False

    def test_none_sample_rate_allowed(self):
        """sample_rate=None is legal per CSE-1 spec."""
        samples = np.zeros(100, dtype=np.complex64)
        sig = SignalData(samples=samples, sample_rate=None)
        assert sig.sample_rate is None
        assert sig.duration is None

    def test_invalid_sample_rate_raises(self):
        samples = np.zeros(10, dtype=np.complex64)
        with pytest.raises(ValueError):
            SignalData(samples=samples, sample_rate=0.0)
        with pytest.raises(ValueError):
            SignalData(samples=samples, sample_rate=-1.0)

    def test_iq_convention_constant(self):
        assert "I + jQ" in IQ_CONVENTION
        assert "real" in IQ_CONVENTION.lower()

    def test_bit_ordering_constant(self):
        assert BIT_ORDERING == "MSB_FIRST"


# =========================================================================== #
# 2. WAV Loading                                                               #
# =========================================================================== #

class TestLoadWav:
    """Tests for load_wav() (new CSE-1 API)."""

    # ---- Mono tests --------------------------------------------------------

    def test_mono_wav_int16_to_analytic(self, tmp_path):
        """Mono int16 WAV → Hilbert analytic complex signal."""
        n = 1024
        fs = 44100
        t = np.arange(n) / fs
        real_data = np.cos(2 * np.pi * 1000 * t)
        # Scale to int16 range
        data_int16 = (real_data * 30000).astype(np.int16)
        p = tmp_path / "mono_int16.wav"
        wavfile.write(str(p), fs, data_int16)

        sig = load_wav(p, to_analytic=True)
        assert sig.sample_rate == float(fs)
        assert sig.is_complex is True
        assert sig.num_samples == n
        # Analytic envelope should be approximately equal to the normalized amplitude
        # int16 quantization creates ~1% amplitude error; exclude edge effects (Hilbert boundary)
        env = np.abs(sig.samples[100:-100])
        peak_norm = float(np.max(np.abs(data_int16)) / 32768.0)
        assert np.allclose(env, peak_norm, atol=0.12), f"Mean envelope: {env.mean():.4f}, expected ≈{peak_norm:.4f}"

    def test_mono_wav_float32_no_analytic(self, tmp_path):
        """Mono float32 WAV with to_analytic=False → real signal."""
        data = np.sin(2 * np.pi * 100 * np.arange(256) / 8000).astype(np.float32)
        p = tmp_path / "mono_float.wav"
        wavfile.write(str(p), 8000, data)

        sig = load_wav(p, to_analytic=False)
        assert sig.is_complex is False
        assert sig.num_samples == 256
        assert sig.sample_rate == 8000.0

    def test_mono_wav_int32(self, tmp_path):
        """Mono int32 WAV is supported."""
        data = (np.sin(2 * np.pi * 440 * np.arange(256) / 44100) * 2**30).astype(np.int32)
        p = tmp_path / "mono_int32.wav"
        wavfile.write(str(p), 44100, data)

        sig = load_wav(p, to_analytic=False)
        assert sig.num_samples == 256
        assert sig.sample_rate == 44100.0
        # Values should be normalized close to [-1,1]
        assert float(np.max(np.abs(sig.samples))) <= 1.5

    def test_mono_wav_uint8(self, tmp_path):
        """Mono uint8 WAV is supported and re-centered."""
        data = (np.sin(2 * np.pi * 440 * np.arange(256) / 8000) * 120 + 128).astype(np.uint8)
        p = tmp_path / "mono_uint8.wav"
        wavfile.write(str(p), 8000, data)

        sig = load_wav(p, to_analytic=False)
        assert sig.num_samples == 256
        # Mean should be near zero after re-centering
        assert abs(float(np.mean(sig.samples))) < 0.15

    # ---- Stereo tests ------------------------------------------------------

    def test_stereo_iq_interpretation(self, tmp_path):
        """Stereo WAV: Ch0=I, Ch1=Q → complex baseband."""
        n = 512
        fs = 44100
        t = np.arange(n) / fs
        i_comp = np.cos(2 * np.pi * 1000 * t).astype(np.float32)
        q_comp = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        stereo = np.column_stack((i_comp, q_comp))
        p = tmp_path / "stereo_iq.wav"
        wavfile.write(str(p), fs, stereo)

        sig = load_wav(p, iq_interpretation=True)
        assert sig.is_complex is True
        assert sig.num_samples == n
        assert sig.sample_rate == float(fs)
        # Verify I/Q structure: envelope should be near 1.0
        env = np.abs(sig.samples)
        assert np.allclose(env, 1.0, atol=0.01), f"Mean envelope {env.mean():.4f}"

    def test_stereo_no_iq_interpretation(self, tmp_path):
        """Stereo WAV with iq_interpretation=False → real 2D array."""
        stereo = np.random.randn(256, 2).astype(np.float32)
        p = tmp_path / "stereo_raw.wav"
        wavfile.write(str(p), 44100, stereo)

        sig = load_wav(p, iq_interpretation=False)
        assert sig.is_complex is False
        assert sig.samples.shape == (256, 2)

    def test_stereo_roundtrip(self, tmp_path):
        """Save complex → stereo WAV → load back and verify IQ fidelity."""
        original = _make_complex_signal(n=512)
        sig_in = SignalData(samples=original, sample_rate=44100.0)
        p = tmp_path / "roundtrip_iq.wav"
        save_wav_file(sig_in, p)

        sig_out = load_wav(p)
        assert sig_out.is_complex is True
        assert sig_out.num_samples == 512
        # Normalised correlation
        corr = np.abs(np.vdot(sig_out.samples, original))
        corr /= np.linalg.norm(sig_out.samples) * np.linalg.norm(original)
        assert corr > 0.998, f"Correlation too low: {corr:.6f}"

    # ---- Error paths -------------------------------------------------------

    def test_missing_wav_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_wav(tmp_path / "does_not_exist.wav")

    def test_empty_wav_raises(self, tmp_path):
        p = tmp_path / "empty.wav"
        p.write_bytes(b"")
        with pytest.raises(ValueError, match="too small"):
            load_wav(p)

    def test_corrupt_wav_raises(self, tmp_path):
        p = tmp_path / "corrupt.wav"
        p.write_bytes(b"RIFF" + b"\x00" * 50)  # Minimal corrupt RIFF
        with pytest.raises(ValueError):
            load_wav(p)

    # ---- Backwards-compat alias -------------------------------------------

    def test_load_wav_file_alias(self, tmp_path):
        """load_wav_file() is a compatible alias for load_wav()."""
        data = np.sin(2 * np.pi * 440 * np.arange(256) / 44100).astype(np.float32)
        p = tmp_path / "alias.wav"
        wavfile.write(str(p), 44100, data)
        sig1 = load_wav(p, to_analytic=False)
        sig2 = load_wav_file(p, to_analytic=False)
        np.testing.assert_array_equal(sig1.samples, sig2.samples)
        assert sig1.sample_rate == sig2.sample_rate


# =========================================================================== #
# 3. Raw IQ Loading                                                            #
# =========================================================================== #

class TestLoadIQ:
    """Tests for load_iq() (new CSE-1 API)."""

    # ---- float32 (complex64 on disk) ---------------------------------------

    def test_float32_iq_basic(self, tmp_path):
        """float32 IQ interleaved → complex64 output."""
        n = 256
        i = np.cos(2 * np.pi * 0.1 * np.arange(n)).astype(np.float32)
        q = np.sin(2 * np.pi * 0.1 * np.arange(n)).astype(np.float32)
        p = tmp_path / "test.iq"
        _write_raw_iq(p, i, q, np.float32)

        sig = load_iq(p, dtype="float32", sample_rate=100000.0)
        assert sig.num_samples == n
        assert sig.sample_rate == 100000.0
        assert sig.is_complex is True
        np.testing.assert_allclose(np.real(sig.samples), i, atol=1e-6)
        np.testing.assert_allclose(np.imag(sig.samples), q, atol=1e-6)

    def test_complex64_alias(self, tmp_path):
        """'complex64' dtype alias works like 'float32'."""
        i = np.ones(64, dtype=np.float32) * 0.5
        q = np.ones(64, dtype=np.float32) * -0.5
        p = tmp_path / "c64.iq"
        _write_raw_iq(p, i, q, np.float32)
        sig = load_iq(p, dtype="complex64", sample_rate=1e6)
        assert sig.num_samples == 64

    # ---- int16 -------------------------------------------------------------

    def test_int16_iq(self, tmp_path):
        """int16 IQ is correctly normalized to [-1, 1]."""
        n = 128
        i_raw = (np.cos(2 * np.pi * 0.05 * np.arange(n)) * 30000).astype(np.int16)
        q_raw = (np.sin(2 * np.pi * 0.05 * np.arange(n)) * 30000).astype(np.int16)
        p = tmp_path / "test_i16.raw"
        _write_raw_iq(p, i_raw, q_raw, np.int16)

        sig = load_iq(p, dtype="int16", sample_rate=50000.0)
        assert sig.num_samples == n
        expected_i = i_raw.astype(np.float32) / 32768.0
        expected_q = q_raw.astype(np.float32) / 32768.0
        np.testing.assert_allclose(np.real(sig.samples), expected_i, atol=1e-5)
        np.testing.assert_allclose(np.imag(sig.samples), expected_q, atol=1e-5)

    def test_int16_roundtrip(self, tmp_path):
        """save_iq_file (int16) → load_iq (int16) roundtrip within 1% tolerance."""
        original = _make_complex_signal(n=256)
        sig = SignalData(samples=original, sample_rate=100_000.0)
        p = tmp_path / "roundtrip.raw"
        save_iq_file(sig, p, data_type="int16", write_metadata_json=True)

        loaded = load_iq(p, dtype="int16")   # sample_rate from companion JSON
        assert loaded.num_samples == 256
        np.testing.assert_allclose(
            np.abs(loaded.samples), np.abs(original), rtol=0.01, atol=0.01
        )

    # ---- int8, uint8 -------------------------------------------------------

    def test_int8_iq(self, tmp_path):
        """int8 IQ is normalized to [-1, 1]."""
        n = 64
        i_raw = (np.cos(2 * np.pi * 0.1 * np.arange(n)) * 100).astype(np.int8)
        q_raw = (np.sin(2 * np.pi * 0.1 * np.arange(n)) * 100).astype(np.int8)
        p = tmp_path / "test_i8.raw"
        _write_raw_iq(p, i_raw, q_raw, np.int8)

        sig = load_iq(p, dtype="int8", sample_rate=1e6)
        assert sig.num_samples == n
        expected_i = i_raw.astype(np.float32) / 128.0
        np.testing.assert_allclose(np.real(sig.samples), expected_i, atol=1e-5)

    def test_uint8_iq(self, tmp_path):
        """uint8 IQ is centered at 127.5 and normalized."""
        n = 64
        i_raw = (np.cos(2 * np.pi * 0.1 * np.arange(n)) * 100 + 128).astype(np.uint8)
        q_raw = (np.sin(2 * np.pi * 0.1 * np.arange(n)) * 100 + 128).astype(np.uint8)
        p = tmp_path / "test_u8.raw"
        _write_raw_iq(p, i_raw, q_raw, np.uint8)

        sig = load_iq(p, dtype="uint8", sample_rate=1e6)
        assert sig.num_samples == n
        # Values should be in approximately [-1, 1]
        assert float(np.max(np.abs(sig.samples))) <= 1.1

    # ---- float64 / complex128 ----------------------------------------------

    def test_float64_iq(self, tmp_path):
        """float64 IQ is supported and returns complex128-precision."""
        n = 32
        i = np.cos(2 * np.pi * 0.1 * np.arange(n)).astype(np.float64)
        q = np.sin(2 * np.pi * 0.1 * np.arange(n)).astype(np.float64)
        p = tmp_path / "test_f64.iq"
        _write_raw_iq(p, i, q, np.float64)

        sig = load_iq(p, dtype="float64", sample_rate=1e6)
        assert sig.num_samples == n

    # ---- QI ordering -------------------------------------------------------

    def test_qi_ordering(self, tmp_path):
        """QI ordering correctly swaps I and Q channels."""
        n = 128
        i_true = np.ones(n, dtype=np.float32) * 0.7
        q_true = np.ones(n, dtype=np.float32) * -0.3
        p = tmp_path / "qi.iq"
        _write_raw_qi(p, i_true, q_true, np.float32)

        sig_qi = load_iq(p, dtype="float32", iq_format="QI", sample_rate=1e6)
        # After QI → IQ reordering: I should be i_true, Q should be q_true
        np.testing.assert_allclose(np.real(sig_qi.samples), i_true, atol=1e-6)
        np.testing.assert_allclose(np.imag(sig_qi.samples), q_true, atol=1e-6)

    def test_iq_vs_qi_differ(self, tmp_path):
        """Loading same file with IQ vs QI ordering gives different results (non-symmetric signal)."""
        n = 64
        i_vals = np.linspace(0.1, 0.9, n, dtype=np.float32)
        q_vals = np.linspace(-0.9, -0.1, n, dtype=np.float32)
        p = tmp_path / "asymmetric.iq"
        _write_raw_iq(p, i_vals, q_vals, np.float32)

        sig_iq = load_iq(p, dtype="float32", iq_format="IQ", sample_rate=1e6)
        sig_qi = load_iq(p, dtype="float32", iq_format="QI", sample_rate=1e6)

        # IQ interprets i_vals as I and q_vals as Q
        np.testing.assert_allclose(np.real(sig_iq.samples), i_vals, atol=1e-6)
        # QI interprets i_vals (even elements) as Q and q_vals (odd elements) as I
        np.testing.assert_allclose(np.real(sig_qi.samples), q_vals, atol=1e-6)

    # ---- Sample rate resolution --------------------------------------------

    def test_missing_sample_rate_yields_none_with_warning(self, tmp_path):
        """Raw IQ without sample_rate and no companion JSON → None + warning."""
        n = 64
        i = np.zeros(n, dtype=np.float32)
        q = np.zeros(n, dtype=np.float32)
        p = tmp_path / "orphan.iq"
        _write_raw_iq(p, i, q, np.float32)

        sig = load_iq(p, dtype="float32", sample_rate=None)
        assert sig.sample_rate is None
        # Must emit a SAMPLE_RATE_UNKNOWN warning
        codes = [w["code"] for w in sig.warnings]
        assert "SAMPLE_RATE_UNKNOWN" in codes, f"Expected SAMPLE_RATE_UNKNOWN, got: {codes}"

    def test_sample_rate_from_companion_json(self, tmp_path):
        """When sample_rate is None, reads from companion .json."""
        n = 64
        i = np.ones(n, dtype=np.float32) * 0.1
        q = np.ones(n, dtype=np.float32) * 0.1
        p = tmp_path / "with_meta.iq"
        _write_raw_iq(p, i, q, np.float32)

        meta = {"sample_rate": 250000.0, "center_freq": 433.92e6}
        json_p = tmp_path / "with_meta.json"
        json_p.write_text(json.dumps(meta))

        sig = load_iq(p, dtype="float32", sample_rate=None)
        assert sig.sample_rate == pytest.approx(250000.0)
        assert len([w for w in sig.warnings if w["code"] == "SAMPLE_RATE_UNKNOWN"]) == 0

    def test_explicit_sample_rate_overrides_json(self, tmp_path):
        """Explicit sample_rate overrides any companion JSON value."""
        n = 32
        i = np.zeros(n, dtype=np.float32)
        q = np.zeros(n, dtype=np.float32)
        p = tmp_path / "override.iq"
        _write_raw_iq(p, i, q, np.float32)

        json_p = tmp_path / "override.json"
        json_p.write_text(json.dumps({"sample_rate": 99999.0}))

        sig = load_iq(p, dtype="float32", sample_rate=500000.0)
        assert sig.sample_rate == 500000.0

    # ---- Odd element count -------------------------------------------------

    def test_odd_element_count_warning(self, tmp_path):
        """File with odd number of float32 elements → warning + last element discarded."""
        # Write 101 float32 elements (should yield 50 complex samples)
        raw = np.arange(101, dtype=np.float32)
        p = tmp_path / "odd.iq"
        raw.tofile(str(p))

        sig = load_iq(p, dtype="float32", sample_rate=1e6)
        assert sig.num_samples == 50
        codes = [w["code"] for w in sig.warnings]
        assert "ODD_IQ_ELEMENT_COUNT" in codes

    # ---- Offset / max_samples ----------------------------------------------

    def test_offset_and_max_samples(self, tmp_path):
        """offset_samples and max_samples slice the file correctly."""
        n = 256
        i = np.arange(n, dtype=np.float32)
        q = np.zeros(n, dtype=np.float32)
        p = tmp_path / "slice.iq"
        _write_raw_iq(p, i, q, np.float32)

        sig = load_iq(p, dtype="float32", sample_rate=1e6, offset_samples=10, max_samples=50)
        assert sig.num_samples == 50
        np.testing.assert_allclose(np.real(sig.samples), i[10:60], atol=1e-6)

    # ---- Error paths -------------------------------------------------------

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_iq(tmp_path / "ghost.iq", dtype="float32", sample_rate=1e6)

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.iq"
        p.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            load_iq(p, dtype="float32", sample_rate=1e6)

    def test_unsupported_dtype_raises(self, tmp_path):
        p = tmp_path / "data.iq"
        p.write_bytes(b"\x00" * 64)
        with pytest.raises(ValueError, match="Unsupported dtype"):
            load_iq(p, dtype="float16", sample_rate=1e6)

    def test_unsupported_iq_format_raises(self, tmp_path):
        p = tmp_path / "data.iq"
        p.write_bytes(b"\x00" * 64)
        with pytest.raises(ValueError, match="iq_format"):
            load_iq(p, dtype="float32", iq_format="XI", sample_rate=1e6)

    def test_offset_exceeds_file_raises(self, tmp_path):
        n = 32
        i = np.zeros(n, dtype=np.float32)
        q = np.zeros(n, dtype=np.float32)
        p = tmp_path / "small.iq"
        _write_raw_iq(p, i, q, np.float32)
        with pytest.raises(ValueError, match="offset"):
            load_iq(p, dtype="float32", sample_rate=1e6, offset_samples=9999)

    def test_negative_sample_rate_raises(self, tmp_path):
        n = 32
        i = np.zeros(n, dtype=np.float32)
        q = np.zeros(n, dtype=np.float32)
        p = tmp_path / "neg_sr.iq"
        _write_raw_iq(p, i, q, np.float32)
        with pytest.raises(ValueError):
            load_iq(p, dtype="float32", sample_rate=-5000.0)

    # ---- Backwards-compat alias -------------------------------------------

    def test_load_iq_file_alias(self, tmp_path):
        """load_iq_file() is a compatible alias."""
        n = 64
        i = np.cos(np.linspace(0, 2 * np.pi, n)).astype(np.float32)
        q = np.sin(np.linspace(0, 2 * np.pi, n)).astype(np.float32)
        p = tmp_path / "alias_iq.iq"
        _write_raw_iq(p, i, q, np.float32)

        sig1 = load_iq(p, dtype="float32", sample_rate=1e6)
        sig2 = load_iq_file(p, data_type="complex64", sample_rate=1e6)
        np.testing.assert_array_almost_equal(sig1.samples, sig2.samples)


# =========================================================================== #
# 4. load_signal dispatcher                                                    #
# =========================================================================== #

class TestLoadSignal:
    def test_dispatches_wav(self, tmp_path):
        data = np.zeros(64, dtype=np.float32)
        p = tmp_path / "test.wav"
        wavfile.write(str(p), 8000, data)
        sig = load_signal(p, to_analytic=False)
        assert sig.source_format == "wav"

    def test_dispatches_iq(self, tmp_path):
        i = np.ones(64, dtype=np.float32) * 0.3
        q = np.ones(64, dtype=np.float32) * 0.4
        p = tmp_path / "test.iq"
        _write_raw_iq(p, i, q, np.float32)
        sig = load_signal(p, sample_rate=1e6)
        assert sig.is_complex is True

    def test_dispatches_raw(self, tmp_path):
        i = np.zeros(64, dtype=np.float32)
        q = np.zeros(64, dtype=np.float32)
        p = tmp_path / "test.raw"
        _write_raw_iq(p, i, q, np.float32)
        sig = load_signal(p, sample_rate=1e6)
        assert sig.num_samples == 64

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "signal.mp3"
        p.write_bytes(b"\x00" * 64)
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_signal(p, sample_rate=1e6)


# =========================================================================== #
# 5. validate_signal                                                           #
# =========================================================================== #

class TestValidateSignal:
    def test_valid_signal_passes(self):
        sig = SignalData(
            samples=_make_complex_signal(256),
            sample_rate=44100.0,
            source_path="test.iq",
        )
        report = validate_signal(sig)
        assert report["valid"] is True
        assert len(report["issues"]) == 0

    def test_empty_samples_invalid(self):
        sig = SignalData(samples=np.array([], dtype=np.complex64), sample_rate=44100.0)
        report = validate_signal(sig)
        assert report["valid"] is False
        assert any("empty" in i.lower() for i in report["issues"])

    def test_nan_samples_invalid(self):
        samples = np.array([1+1j, float("nan")+0j, -1-1j], dtype=np.complex64)
        sig = SignalData(samples=samples, sample_rate=44100.0)
        report = validate_signal(sig)
        assert report["valid"] is False
        assert any("non-finite" in i for i in report["issues"])

    def test_none_sample_rate_produces_warning(self):
        sig = SignalData(samples=_make_complex_signal(64), sample_rate=None)
        report = validate_signal(sig)
        # valid=True (not a hard error) but warning emitted
        codes = [w["code"] for w in report["warnings"]]
        assert "SAMPLE_RATE_UNKNOWN" in codes

    def test_very_short_signal_warning(self):
        sig = SignalData(samples=np.ones(4, dtype=np.complex64), sample_rate=1000.0)
        report = validate_signal(sig)
        codes = [w["code"] for w in report["warnings"]]
        assert "VERY_SHORT_SIGNAL" in codes

    def test_is_complex_flag_mismatch_warning(self):
        # Declare complex but pass real array
        samples = np.ones(64, dtype=np.float32)
        sig = SignalData(samples=samples, sample_rate=1000.0, is_complex=True)
        # Note: SignalData.__post_init__ casts float to complex64 if is_complex=True,
        # so this actually converts — the warning won't fire. Test the opposite:
        # declare real but pass complex array
        samples_c = np.ones(64, dtype=np.complex64)
        sig2 = SignalData(samples=samples_c, sample_rate=1000.0, is_complex=False)
        report = validate_signal(sig2)
        codes = [w["code"] for w in report["warnings"]]
        assert "IS_COMPLEX_FLAG_MISMATCH" in codes


# =========================================================================== #
# 6. normalize_signal                                                          #
# =========================================================================== #

class TestNormalizeSignal:
    def _make_sig(self, samples: np.ndarray, sr: float = 44100.0) -> SignalData:
        return SignalData(samples=samples, sample_rate=sr, is_complex=np.iscomplexobj(samples))

    def test_dc_removal(self):
        """DC offset is removed correctly."""
        dc = 3.0 + 2j
        samples = dc + np.zeros(128, dtype=np.complex64)
        sig = self._make_sig(samples)
        norm = normalize_signal(sig, remove_dc=True, normalize_amplitude=False)
        mean_i = float(np.mean(np.real(norm.samples)))
        mean_q = float(np.mean(np.imag(norm.samples)))
        assert abs(mean_i) < 1e-5, f"I mean after DC removal: {mean_i}"
        assert abs(mean_q) < 1e-5, f"Q mean after DC removal: {mean_q}"

    def test_amplitude_normalization(self):
        """After normalization, RMS power = 1.0."""
        samples = _make_complex_signal(256) * 5.0
        sig = self._make_sig(samples)
        norm = normalize_signal(sig, remove_dc=False, normalize_amplitude=True)
        rms = float(np.sqrt(np.mean(np.abs(norm.samples) ** 2)))
        assert rms == pytest.approx(1.0, abs=1e-5)

    def test_gain_scaling(self):
        """Gain factor is applied after normalization."""
        samples = _make_complex_signal(128)
        sig = self._make_sig(samples)
        norm = normalize_signal(sig, remove_dc=False, normalize_amplitude=True, gain=2.0)
        rms = float(np.sqrt(np.mean(np.abs(norm.samples) ** 2)))
        assert rms == pytest.approx(2.0, abs=0.01)

    def test_original_not_modified(self):
        """normalize_signal must not alter the input SignalData."""
        samples = _make_complex_signal(64) * 10.0
        sig = self._make_sig(samples)
        orig_mean = float(np.abs(np.mean(sig.samples)))
        _ = normalize_signal(sig, remove_dc=True, normalize_amplitude=True)
        # Original should be unchanged
        assert float(np.abs(np.mean(sig.samples))) == pytest.approx(orig_mean, rel=1e-5)

    def test_clipping_detection(self):
        """Clipping detection warning is emitted when samples exceed threshold."""
        # Create a signal with spike
        samples = np.zeros(128, dtype=np.complex64)
        samples[10] = 5.0 + 5j  # large spike
        sig = self._make_sig(samples)
        norm = normalize_signal(sig, remove_dc=False, normalize_amplitude=False,
                                gain=1.0, clip_threshold=1.0)
        codes = [w["code"] for w in norm.warnings]
        assert "CLIPPING_DETECTED" in codes

    def test_no_clipping_warning_when_within_threshold(self):
        """No clipping warning when all samples are within threshold."""
        samples = _make_complex_signal(128)  # envelope ≈ 1.0
        sig = self._make_sig(samples)
        norm = normalize_signal(sig, remove_dc=False, normalize_amplitude=False,
                                clip_threshold=2.0)
        codes = [w["code"] for w in norm.warnings]
        assert "CLIPPING_DETECTED" not in codes

    def test_zero_power_signal_warning(self):
        """Zero-power signal emits ZERO_POWER_SIGNAL warning and skips normalization."""
        samples = np.zeros(64, dtype=np.complex64)
        sig = self._make_sig(samples)
        norm = normalize_signal(sig, remove_dc=False, normalize_amplitude=True)
        codes = [w["code"] for w in norm.warnings]
        assert "ZERO_POWER_SIGNAL" in codes

    def test_nan_samples_raises(self):
        """NaN in samples causes ValueError before processing."""
        samples = np.array([1+1j, float("nan")+0j], dtype=np.complex64)
        sig = self._make_sig(samples)
        with pytest.raises(ValueError, match="invalid values"):
            normalize_signal(sig)


# =========================================================================== #
# 7. infer_basic_metadata                                                      #
# =========================================================================== #

class TestInferBasicMetadata:
    def test_all_keys_present(self):
        sig = SignalData(samples=_make_complex_signal(512), sample_rate=44100.0)
        meta = infer_basic_metadata(sig)
        required_keys = {
            "num_samples", "duration_sec", "sample_rate_hz", "center_freq_hz",
            "is_complex", "dtype", "rms_power_dbfs", "peak_magnitude",
            "dynamic_range_db", "dc_offset_i", "dc_offset_q", "estimated_bandwidth_hz",
        }
        assert required_keys.issubset(set(meta.keys()))

    def test_values_correct(self):
        sig = SignalData(samples=_make_complex_signal(512), sample_rate=44100.0)
        meta = infer_basic_metadata(sig)
        assert meta["num_samples"] == 512
        assert meta["sample_rate_hz"] == 44100.0
        assert meta["estimated_bandwidth_hz"] == pytest.approx(22050.0)
        assert meta["rms_power_dbfs"] == pytest.approx(0.0, abs=0.5)  # complex exp → RMS ≈ 1.0

    def test_none_sample_rate(self):
        sig = SignalData(samples=np.zeros(64, dtype=np.complex64), sample_rate=None)
        meta = infer_basic_metadata(sig)
        assert meta["duration_sec"] is None
        assert meta["estimated_bandwidth_hz"] is None

    def test_empty_signal(self):
        sig = SignalData(samples=np.array([], dtype=np.complex64), sample_rate=None)
        meta = infer_basic_metadata(sig)
        assert meta["num_samples"] == 0
        assert meta["rms_power_dbfs"] == -120.0


# =========================================================================== #
# 8. Serialization round-trips                                                 #
# =========================================================================== #

class TestSerialization:
    def test_iq_roundtrip_complex64(self, tmp_path):
        original = _make_complex_signal(n=1000, freq_hz=100.0, fs=100_000.0)
        sig = SignalData(samples=original, sample_rate=100_000.0, center_freq=433.92e6)
        p = tmp_path / "rt_c64.iq"
        save_iq_file(sig, p, data_type="complex64")

        loaded = load_iq(p, dtype="complex64")  # sample_rate from companion JSON
        assert loaded.sample_rate == 100_000.0
        assert loaded.center_freq == 433.92e6
        np.testing.assert_allclose(loaded.samples, original, rtol=1e-5, atol=1e-5)

    def test_iq_roundtrip_int16(self, tmp_path):
        original = _make_complex_signal(n=500)
        sig = SignalData(samples=original, sample_rate=50_000.0)
        p = tmp_path / "rt_i16.raw"
        save_iq_file(sig, p, data_type="int16")

        loaded = load_iq(p, dtype="int16")
        np.testing.assert_allclose(loaded.samples, original, rtol=0.01, atol=0.01)

    def test_companion_json_written(self, tmp_path):
        """save_iq_file writes a companion .json with sample_rate."""
        sig = SignalData(
            samples=np.zeros(32, dtype=np.complex64),
            sample_rate=250_000.0,
            center_freq=915e6,
        )
        p = tmp_path / "meta_test.iq"
        save_iq_file(sig, p, write_metadata_json=True)
        json_p = p.with_suffix(".json")
        assert json_p.exists()
        meta = json.loads(json_p.read_text())
        assert meta["sample_rate"] == 250_000.0
        assert meta["center_freq"] == 915e6

    def test_wav_stereo_roundtrip(self, tmp_path):
        original = _make_complex_signal(n=882, fs=44100.0)
        sig = SignalData(samples=original, sample_rate=44100.0)
        p = tmp_path / "iq_wav.wav"
        save_wav_file(sig, p)

        loaded = load_wav(p)
        assert loaded.sample_rate == 44100.0
        assert loaded.is_complex is True
        corr = np.abs(np.vdot(loaded.samples, original))
        corr /= np.linalg.norm(loaded.samples) * np.linalg.norm(original)
        assert corr > 0.998

    def test_save_wav_without_sample_rate_raises(self, tmp_path):
        sig = SignalData(samples=np.zeros(64, dtype=np.complex64), sample_rate=None)
        with pytest.raises(ValueError, match="sample_rate"):
            save_wav_file(sig, tmp_path / "no_sr.wav")


# =========================================================================== #
# 9. Demo Fixture Smoke Test                                                   #
# =========================================================================== #

SYNTHETIC_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"

class TestDemoFixtures:
    """Smoke-test the pre-generated synthetic fixture files."""

    @pytest.mark.skipif(
        not (SYNTHETIC_DIR / "bpsk_snr20db_clean.iq").exists(),
        reason="Synthetic fixtures not generated yet.",
    )
    def test_load_bpsk_iq_fixture(self):
        p = SYNTHETIC_DIR / "bpsk_snr20db_clean.iq"
        sig = load_iq(p, dtype="complex64")  # reads sample_rate from companion JSON
        assert sig.num_samples > 0
        assert sig.sample_rate is not None
        assert sig.is_complex is True
        report = validate_signal(sig)
        assert report["valid"] is True

    @pytest.mark.skipif(
        not (SYNTHETIC_DIR / "bpsk_snr20db_clean.wav").exists(),
        reason="Synthetic fixtures not generated yet.",
    )
    def test_load_bpsk_wav_fixture(self):
        p = SYNTHETIC_DIR / "bpsk_snr20db_clean.wav"
        sig = load_wav(p)
        assert sig.num_samples > 0
        assert sig.is_complex is True
        report = validate_signal(sig)
        assert report["valid"] is True

    @pytest.mark.skipif(
        not (SYNTHETIC_DIR / "16qam_snr22db.iq").exists(),
        reason="Synthetic fixtures not generated yet.",
    )
    def test_load_16qam_iq_fixture(self):
        p = SYNTHETIC_DIR / "16qam_snr22db.iq"
        sig = load_iq(p, dtype="complex64")
        assert sig.num_samples > 0
        report = validate_signal(sig)
        assert report["valid"] is True
