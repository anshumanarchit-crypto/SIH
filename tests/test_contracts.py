"""
Unit tests for SpectralQ Shared Contracts (CSE-0).

Verifies:
- Dataclass field presence and types.
- Enum definitions and completeness (ResultStatus, ModulationType).
- IQ convention and bit ordering constants.
- Confidence semantics (0.0 != None, [0.0, 1.0] bounds checking).
- Structured warning creation and formatting.
- Type identity across all core modules (single source of truth).
"""

import pytest
import numpy as np
from dataclasses import is_dataclass, fields

from core.contracts import (
    IQ_CONVENTION,
    BIT_ORDERING,
    BIT_ORDERING_CONVENTION,
    BIT_ARRAY_DTYPE,
    ResultStatus,
    validate_confidence,
    make_warning,
    SignalData,
    ModulationType,
    SpectralFeatures,
    ModulationResult,
    DemodulationResult,
    CorrelationResult,
    SyncDetection,
    PacketHeader,
    DecodedFrame,
    PreprocessingResult,
    PipelineConfig,
    PipelineResult,
)


# -----------------------------------------------------------------------------
# 1. IQ CONVENTION & BIT ORDERING CONVENTION
# -----------------------------------------------------------------------------
def test_iq_convention_pinned():
    """Verify IQ convention constant is defined, non-empty, and explicitly documents I+jQ."""
    assert isinstance(IQ_CONVENTION, str)
    assert len(IQ_CONVENTION) > 20
    assert "I + jQ" in IQ_CONVENTION
    assert "real" in IQ_CONVENTION.lower()
    assert "imag" in IQ_CONVENTION.lower()


def test_bit_ordering_convention_pinned():
    """Verify bit ordering convention is MSB-first and dtype is uint8."""
    assert BIT_ORDERING == "MSB_FIRST"
    assert isinstance(BIT_ORDERING_CONVENTION, str)
    assert "MSB-first" in BIT_ORDERING_CONVENTION
    assert BIT_ARRAY_DTYPE == np.uint8


# -----------------------------------------------------------------------------
# 2. RESULT STATUS ENUM
# -----------------------------------------------------------------------------
def test_result_status_enum():
    """Verify ResultStatus enum contains all required standardized statuses."""
    expected_members = {
        "CONFIRMED": "CONFIRMED",
        "ESTIMATED": "ESTIMATED",
        "UNSUPPORTED": "UNSUPPORTED",
        "INSUFFICIENT_EVIDENCE": "INSUFFICIENT_EVIDENCE",
        "FAILED": "FAILED",
    }
    for name, value in expected_members.items():
        assert hasattr(ResultStatus, name)
        assert getattr(ResultStatus, name).value == value
        assert isinstance(getattr(ResultStatus, name), ResultStatus)


# -----------------------------------------------------------------------------
# 3. CONFIDENCE SEMANTICS
# -----------------------------------------------------------------------------
def test_confidence_semantics_valid():
    """Verify valid confidence values [0.0, 1.0] and None handling."""
    assert validate_confidence(0.0) == 0.0
    assert validate_confidence(0.5) == 0.5
    assert validate_confidence(1.0) == 1.0
    assert validate_confidence(None) is None
    # Verify 0.0 is distinct from None
    assert validate_confidence(0.0) is not None
    assert validate_confidence(0.0) != validate_confidence(None)


def test_confidence_semantics_invalid():
    """Verify invalid confidence values raise ValueError."""
    with pytest.raises(ValueError):
        validate_confidence(-0.01)

    with pytest.raises(ValueError):
        validate_confidence(1.01)

    with pytest.raises(ValueError):
        validate_confidence("high")  # Non-float string


# -----------------------------------------------------------------------------
# 4. STRUCTURED WARNING HELPER
# -----------------------------------------------------------------------------
def test_make_warning_format():
    """Verify make_warning returns standardized dict with uppercase code."""
    w1 = make_warning("low_snr", "Signal SNR is below 3 dB threshold.")
    assert w1 == {
        "code": "LOW_SNR",
        "message": "Signal SNR is below 3 dB threshold.",
        "details": {},
    }

    w2 = make_warning("cfo_high", "Significant carrier offset detected.", {"cfo_hz": 4500.0})
    assert w2["code"] == "CFO_HIGH"
    assert w2["message"] == "Significant carrier offset detected."
    assert w2["details"]["cfo_hz"] == 4500.0


# -----------------------------------------------------------------------------
# 5. SIGNALDATA DATACLASS
# -----------------------------------------------------------------------------
def test_signal_data_contract():
    """Verify SignalData dataclass fields, defaults, properties, and validation."""
    assert is_dataclass(SignalData)
    field_names = {f.name for f in fields(SignalData)}
    required_fields = {
        "samples", "sample_rate", "source_path", "source_format",
        "dtype", "num_samples", "duration", "center_freq",
        "is_complex", "metadata", "warnings"
    }
    assert required_fields.issubset(field_names)

    # Valid initialization
    samples = np.array([1+2j, 3+4j, -1-1j], dtype=np.complex64)
    sig = SignalData(
        samples=samples,
        sample_rate=1000.0,
        source_path="test.iq",
        source_format="complex64",
    )
    assert sig.num_samples == 3
    assert sig.duration == pytest.approx(0.003)
    assert sig.duration_sec == pytest.approx(0.003)
    assert "complex" in sig.dtype
    assert sig.is_complex is True
    assert isinstance(sig.metadata, dict)
    assert isinstance(sig.warnings, list)
    assert sig.power_db > -100.0
    assert sig.peak_magnitude > 0.0

    # Test copy
    sig_copy = sig.copy()
    assert sig_copy.num_samples == sig.num_samples
    assert np.allclose(sig_copy.samples, sig.samples)


def test_signal_data_optional_sample_rate():
    """Verify SignalData supports None sample_rate when unspecified."""
    samples = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sig = SignalData(samples=samples, sample_rate=None, is_complex=False)
    assert sig.sample_rate is None
    assert sig.duration is None
    assert sig.num_samples == 3


def test_signal_data_invalid_sample_rate():
    """Verify non-positive sample_rate raises ValueError."""
    samples = np.array([1+1j, 2+2j], dtype=np.complex64)
    with pytest.raises(ValueError):
        SignalData(samples=samples, sample_rate=0.0)
    with pytest.raises(ValueError):
        SignalData(samples=samples, sample_rate=-50.0)


# -----------------------------------------------------------------------------
# 6. DEMODULATION RESULT & CORRELATION RESULT
# -----------------------------------------------------------------------------
def test_demodulation_result_contract():
    """Verify DemodulationResult dataclass fields and status default."""
    assert is_dataclass(DemodulationResult)
    field_names = {f.name for f in fields(DemodulationResult)}
    assert {"symbols", "hard_bits", "status", "confidence", "evm_percent", "mer_db"}.issubset(field_names)

    demod = DemodulationResult(
        symbols=np.array([1+1j, -1-1j], dtype=np.complex64),
        hard_bits=np.array([1, 0, 0, 1], dtype=np.uint8),
        evm_percent=4.5,
        mer_db=26.9,
        modulation=ModulationType.QPSK,
        status=ResultStatus.CONFIRMED,
        confidence=0.98,
    )
    assert demod.status == ResultStatus.CONFIRMED
    assert demod.confidence == 0.98
    assert demod.hard_bits.dtype == np.uint8


def test_correlation_result_contract():
    """Verify CorrelationResult (SyncDetection) dataclass fields."""
    assert is_dataclass(CorrelationResult)
    assert SyncDetection is CorrelationResult
    field_names = {f.name for f in fields(CorrelationResult)}
    assert {"found", "sync_name", "bit_index", "correlation_score", "status", "confidence"}.issubset(field_names)

    corr = CorrelationResult(
        found=True,
        sync_name="CCSDS_32",
        bit_index=64,
        correlation_score=0.99,
        is_inverted=False,
        phase_rotation_deg=0.0,
        sync_word_len=32,
        confidence=0.99,
    )
    assert corr.found is True
    assert corr.status == ResultStatus.CONFIRMED


# -----------------------------------------------------------------------------
# 7. PIPELINE RESULT CONTRACT
# -----------------------------------------------------------------------------
def test_pipeline_result_contract():
    """Verify PipelineResult dataclass fields and helper properties."""
    assert is_dataclass(PipelineResult)
    field_names = {f.name for f in fields(PipelineResult)}
    expected = {
        "signal_data", "preprocessed_samples", "iq_metrics", "cfo_hz",
        "features", "modulation", "demodulation", "deinterleaved_bits",
        "fec_bits", "sync_detection", "decoded_frame", "status", "confidence"
    }
    assert expected.issubset(field_names)


# -----------------------------------------------------------------------------
# 8. SINGLE SOURCE OF TRUTH (IDENTITY) CHECK
# -----------------------------------------------------------------------------
def test_single_source_of_truth_imports():
    """Verify all downstream core modules import types from core.contracts without redefinition."""
    import core.io as io_mod
    import core.modulation as mod_mod
    import core.demodulation as demod_mod
    import core.correlation as corr_mod
    import core.pipeline as pipe_mod

    assert io_mod.SignalData is SignalData
    assert mod_mod.ModulationType is ModulationType
    assert mod_mod.ModulationResult is ModulationResult
    assert demod_mod.DemodulationResult is DemodulationResult
    assert corr_mod.SyncDetection is SyncDetection
    assert corr_mod.CorrelationResult is CorrelationResult
    assert corr_mod.DecodedFrame is DecodedFrame
    assert pipe_mod.PipelineConfig is PipelineConfig
    assert pipe_mod.PipelineResult is PipelineResult
    assert pipe_mod.ResultStatus is ResultStatus
