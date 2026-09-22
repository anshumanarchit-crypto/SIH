"""
SpectralQ — Autonomous Signal Intelligence & Demodulation Platform.
From raw waveform to readable bits.
"""

__version__ = "1.0.0"

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
    FECResult,
)

__all__ = [
    "IQ_CONVENTION",
    "BIT_ORDERING",
    "BIT_ORDERING_CONVENTION",
    "BIT_ARRAY_DTYPE",
    "ResultStatus",
    "validate_confidence",
    "make_warning",
    "SignalData",
    "ModulationType",
    "SpectralFeatures",
    "ModulationResult",
    "DemodulationResult",
    "CorrelationResult",
    "SyncDetection",
    "PacketHeader",
    "DecodedFrame",
    "PreprocessingResult",
    "PipelineConfig",
    "PipelineResult",
    "FECResult",
]

