"""
SpectralQ Shared Contracts (CSE-0)
==================================
The single source of truth for interface contracts, data models, enum definitions,
and conventions across all SpectralQ modules (CSE-1 through CSE-5).

Every downstream module (io, preprocessing, features, modulation, demodulation,
deinterleave, fec, correlation, pipeline) MUST import these types from `core.contracts`
and MUST NOT redefine them.

Conventions Pinned:
- IQ_CONVENTION: complex sample = I + jQ (I = real, Q = imag).
- BIT_ORDERING: MSB-first (Big-Endian bit ordering).
- ResultStatus: Universal output status enum for all modules.
- Confidence Semantics: float in [0.0, 1.0] or None (where None != 0.0).
- Standard Structured Warnings: make_warning(code, message, details).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np


# -----------------------------------------------------------------------------
# 1. IQ CONVENTION
# -----------------------------------------------------------------------------
IQ_CONVENTION = (
    "complex sample = I + jQ, where I is the real part (In-Phase) and "
    "Q is the imaginary part (Quadrature). Positive frequencies represent "
    "counter-clockwise rotation in the complex plane (exp(+j*2*pi*f*t)). "
    "All DSP modules must strictly follow this convention."
)


# -----------------------------------------------------------------------------
# 2. BIT ORDERING CONVENTION
# -----------------------------------------------------------------------------
BIT_ORDERING = "MSB_FIRST"  # Big-Endian bit ordering

BIT_ORDERING_CONVENTION = (
    "Bit ordering convention: MSB-first (Big-Endian). "
    "For symbol-to-bit demapping and packet framing, bit 0 represents the Most Significant Bit (MSB). "
    "Bit arrays are numpy uint8 arrays containing binary values (0 or 1), where index 0 represents "
    "the first bit in time (earliest transmitted bit). When packing bits into bytes via np.packbits, "
    "bitorder='big' is strictly used. This conforms to standard aerospace, IEEE, CCSDS, and telecom protocols."
)

BIT_ARRAY_DTYPE = np.uint8


# -----------------------------------------------------------------------------
# 3. UNIVERSAL RESULT STATUS ENUM
# -----------------------------------------------------------------------------
class ResultStatus(str, Enum):
    """
    Standard status enum used by EVERY module's output contract.
    No module may invent its own status strings.
    """
    CONFIRMED = "CONFIRMED"
    ESTIMATED = "ESTIMATED"
    UNSUPPORTED = "UNSUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    FAILED = "FAILED"


# -----------------------------------------------------------------------------
# 4. CONFIDENCE SEMANTICS & VALIDATION
# -----------------------------------------------------------------------------
def validate_confidence(confidence: Optional[float]) -> Optional[float]:
    """
    Validate that confidence is None or a float within [0.0, 1.0].
    
    Confidence Semantics:
    - float in [0.0, 1.0]: 0.0 means 0% certainty, 1.0 means 100% certainty.
    - None: explicitly indicates 'not computed / not applicable'.
    - IMPORTANT: 0.0 != None.
    
    Raises:
        ValueError: If confidence is not None and outside [0.0, 1.0].
    """
    if confidence is None:
        return None
    try:
        conf_val = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Confidence must be a float or None, got: {confidence}") from exc

    if not (0.0 <= conf_val <= 1.0):
        raise ValueError(f"Confidence must be within [0.0, 1.0] or None, got: {conf_val}")
    return conf_val


# -----------------------------------------------------------------------------
# 5. STRUCTURED WARNING FORMAT
# -----------------------------------------------------------------------------
def make_warning(
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a standard structured warning dictionary for programmatic filtering and telemetry.
    
    Args:
        code: Machine-readable uppercase warning code (e.g., 'LOW_SNR', 'IQ_IMBALANCE_HIGH').
        message: Human-readable diagnostic description.
        details: Optional dictionary containing numerical or contextual metrics.
        
    Returns:
        Structured warning dictionary with 'code', 'message', and 'details' keys.
    """
    return {
        "code": str(code).upper().strip(),
        "message": str(message),
        "details": details if details is not None else {},
    }


# -----------------------------------------------------------------------------
# 6. CORE DATACLASSES
# -----------------------------------------------------------------------------

@dataclass
class SignalData:
    """
    Standardized internal representation of a digital signal.
    
    Attributes:
        samples: Baseband complex or real samples array.
        sample_rate: Sampling frequency in Hz (float or None if unknown).
        source_path: Path to source file or description (default "").
        source_format: File/signal format string (e.g., "complex64", "wav", "int16").
        dtype: Data type string representation.
        num_samples: Total number of samples in the buffer.
        duration: Signal duration in seconds (float or None if sample_rate is None).
        center_freq: Center RF frequency in Hz (default 0.0).
        is_complex: True if complex I/Q baseband, False if real IF.
        metadata: Arbitrary companion metadata key-value pairs.
        warnings: List of structured warning dictionaries.
    """
    samples: np.ndarray
    sample_rate: Optional[float] = None
    source_path: str = ""
    source_format: str = ""
    dtype: str = ""
    num_samples: int = 0
    duration: Optional[float] = None
    center_freq: float = 0.0
    is_complex: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.samples = np.asarray(self.samples)
        if self.is_complex and not np.iscomplexobj(self.samples):
            self.samples = self.samples.astype(np.complex64)
            
        self.num_samples = len(self.samples)
        if not self.dtype:
            self.dtype = str(self.samples.dtype)
            
        if self.sample_rate is not None:
            if self.sample_rate <= 0:
                raise ValueError(f"Sample rate must be positive and non-zero, got: {self.sample_rate}")
            if self.duration is None:
                self.duration = float(self.num_samples / self.sample_rate)
        else:
            self.duration = None

    @property
    def duration_sec(self) -> Optional[float]:
        """Signal duration in seconds (alias for duration)."""
        return self.duration

    @property
    def power_db(self) -> float:
        """Average signal power in decibels (relative to 1V RMS)."""
        if len(self.samples) == 0:
            return -120.0
        p = float(np.mean(np.abs(self.samples) ** 2))
        return float(10.0 * np.log10(max(p, 1e-12)))

    @property
    def peak_magnitude(self) -> float:
        """Peak instantaneous magnitude."""
        return float(np.max(np.abs(self.samples))) if len(self.samples) > 0 else 0.0

    def copy(self) -> SignalData:
        """Return a deep copy of the signal data."""
        return SignalData(
            samples=self.samples.copy(),
            sample_rate=self.sample_rate,
            source_path=self.source_path,
            source_format=self.source_format,
            dtype=self.dtype,
            num_samples=self.num_samples,
            duration=self.duration,
            center_freq=self.center_freq,
            is_complex=self.is_complex,
            metadata=self.metadata.copy(),
            warnings=[w.copy() for w in self.warnings],
        )


class ModulationType(str, Enum):
    """Supported digital and analog modulation schemes."""
    BPSK = "BPSK"
    QPSK = "QPSK"
    PSK8 = "8PSK"
    QAM16 = "16QAM"
    QAM64 = "64QAM"
    FSK2 = "2-FSK"
    FSK4 = "4-FSK"
    OOK = "OOK"
    ASK2 = "2-ASK"
    AM_DSB = "AM-DSB"
    FM = "FM"
    UNKNOWN = "Unknown / Insufficient Evidence"


@dataclass
class SpectralFeatures:
    """Comprehensive container for extracted signal features."""
    # Higher-Order Cumulants
    c20: complex
    c21: float
    c40: complex
    c41: complex
    c42: float
    c63: float
    c80: complex

    # Instantaneous Statistics
    gamma_max: float
    sigma_ap: float
    sigma_dp: float
    sigma_aa: float
    sigma_af: float
    kurtosis_amp: float
    skewness_amp: float
    kurtosis_phase: float

    # Signal Quality & Timing
    snr_db: float
    estimated_baud_rate: float
    estimated_sps: float
    spectral_flatness: float
    carrier_freq_offset_hz: float

    # Contract telemetry
    status: ResultStatus = ResultStatus.ESTIMATED
    confidence: Optional[float] = None
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    raw_metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.confidence is not None:
            self.confidence = validate_confidence(self.confidence)


@dataclass
class ModulationResult:
    """Output contract for modulation classification."""
    modulation: ModulationType
    confidence: Optional[float]
    probabilities: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    classification_method: str = "HYBRID_CUMULANTS_RF"
    rejection_reason: str = ""
    features: Optional[SpectralFeatures] = None
    is_unknown: bool = False
    status: ResultStatus = ResultStatus.CONFIRMED
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.confidence is not None:
            self.confidence = validate_confidence(self.confidence)
        if self.modulation == ModulationType.UNKNOWN:
            self.is_unknown = True
            if self.status == ResultStatus.CONFIRMED:
                self.status = ResultStatus.INSUFFICIENT_EVIDENCE


@dataclass
class DemodulationResult:
    """Output contract for demodulation and symbol synchronization."""
    symbols: np.ndarray
    hard_bits: np.ndarray
    soft_llrs: Optional[np.ndarray] = None
    evm_percent: float = 0.0
    mer_db: float = 0.0
    modulation: Optional[ModulationType] = None
    estimated_sps: Optional[float] = None
    carrier_freq_offset_hz: Optional[float] = None
    timing_offset_samples: int = 0
    phase_history: np.ndarray = field(default_factory=lambda: np.array([]))
    timing_error_history: np.ndarray = field(default_factory=lambda: np.array([]))
    status: ResultStatus = ResultStatus.CONFIRMED
    confidence: Optional[float] = None
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.symbols = np.asarray(self.symbols)
        self.hard_bits = np.asarray(self.hard_bits, dtype=BIT_ARRAY_DTYPE) if len(self.hard_bits) > 0 else np.array([], dtype=BIT_ARRAY_DTYPE)
        if self.confidence is not None:
            self.confidence = validate_confidence(self.confidence)

    @property
    def bits(self) -> np.ndarray:
        """Alias for hard_bits."""
        return self.hard_bits

    @property
    def diagnostics(self) -> Dict[str, Any]:
        """Alias for metrics / diagnostics."""
        return self.metrics

    @property
    def parameters(self) -> Dict[str, Any]:
        """Extract parameters dictionary from metrics."""
        return self.metrics.get("parameters", {})


@dataclass
class FECResult:
    """Output contract for forward error correction decoding."""
    decoded_bits: np.ndarray
    fec_scheme: str
    status: ResultStatus = ResultStatus.CONFIRMED
    corrected_errors: int = 0
    input_bit_count: int = 0
    output_bit_count: int = 0
    bit_error_rate: Optional[float] = None
    confidence: Optional[float] = None
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.decoded_bits = np.asarray(self.decoded_bits, dtype=BIT_ARRAY_DTYPE) if len(self.decoded_bits) > 0 else np.array([], dtype=BIT_ARRAY_DTYPE)
        if self.output_bit_count == 0 and len(self.decoded_bits) > 0:
            self.output_bit_count = len(self.decoded_bits)
        if self.confidence is not None:
            self.confidence = validate_confidence(self.confidence)

    @property
    def bits(self) -> np.ndarray:
        """Alias for decoded_bits."""
        return self.decoded_bits



@dataclass
class CorrelationResult:
    """Output contract for preamble/sync word correlation search."""
    found: bool
    sync_name: str
    bit_index: int
    correlation_score: float
    is_inverted: bool
    phase_rotation_deg: float = 0.0
    sync_word_len: int = 0
    correlation_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    status: ResultStatus = ResultStatus.CONFIRMED
    confidence: Optional[float] = None
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.confidence is not None:
            self.confidence = validate_confidence(self.confidence)
        if not self.found and self.status == ResultStatus.CONFIRMED:
            self.status = ResultStatus.INSUFFICIENT_EVIDENCE


# Alias SyncDetection to CorrelationResult for seamless backwards-compatibility
SyncDetection = CorrelationResult


@dataclass
class PacketHeader:
    """Structured packet header metadata."""
    version: int = 1
    packet_type: int = 0
    sequence_num: int = 0
    payload_len: int = 0
    crc_type: str = "CRC-16-CCITT"
    received_crc: int = 0
    computed_crc: int = 0
    crc_valid: bool = False


@dataclass
class DecodedFrame:
    """Structured container for decoded packet frames."""
    sync_name: str
    sync_index: int
    is_inverted: bool
    version: int
    packet_type: int
    seq_num: int
    payload_length: int
    raw_payload_bytes: bytes
    payload_text: str
    hex_dump: str
    crc_valid: bool
    crc_type: str = "CRC-16-CCITT"
    crc_expected: int = 0
    crc_actual: int = 0
    phase_rotation_deg: float = 0.0
    status: ResultStatus = ResultStatus.CONFIRMED
    confidence: Optional[float] = None
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.confidence is not None:
            self.confidence = validate_confidence(self.confidence)


@dataclass
class PreprocessingResult:
    """Output contract for signal preprocessing and conditioning."""
    samples: np.ndarray
    dc_offset: complex = 0.0 + 0j
    iq_imbalance: Dict[str, float] = field(default_factory=dict)
    estimated_cfo_hz: float = 0.0
    rrc_applied: bool = False
    status: ResultStatus = ResultStatus.CONFIRMED
    warnings: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Configuration options for SpectralQ pipeline."""
    # Preprocessing
    enable_dc_removal: bool = True
    enable_iq_balance: bool = True
    enable_iq_correction: bool = True  # alias
    enable_cfo_correction: bool = True
    enable_power_norm: bool = True
    enable_rrc_filter: bool = True
    enable_rrc_filtering: bool = True  # alias
    enable_costas_sync: bool = True
    rrc_beta: float = 0.35
    rrc_alpha: float = 0.35  # alias
    rrc_span: int = 8

    # Manual Overrides (None = automatic estimation)
    manual_modulation: Optional[ModulationType] = None
    manual_sps: Optional[float] = None
    manual_cfo_hz: Optional[float] = None

    # De-interleaving
    deinterleave_scheme: str = "none"  # 'none', 'block', 'convolutional'
    block_rows: int = 8
    block_cols: int = 8
    conv_branches: int = 4
    conv_delay_step: int = 2

    # FEC
    fec_scheme: str = "auto"  # 'auto', 'viterbi_hard', 'viterbi_soft', 'hamming', 'rs', 'none'
    rs_n: int = 30
    rs_k: int = 20

    # Synchronization & Framing
    sync_pattern: Optional[str] = None  # None searches all standard preambles
    sync_threshold: float = 0.75
    crc_type: str = "crc16"


@dataclass
class PipelineResult:
    """End-to-end execution telemetry and reconstructed data."""
    signal_data: SignalData
    preprocessed_samples: np.ndarray
    iq_metrics: Dict[str, float]
    cfo_hz: float
    features: SpectralFeatures
    modulation: ModulationResult
    demodulation: DemodulationResult
    deinterleaved_bits: np.ndarray
    fec_bits: np.ndarray
    sync_detection: CorrelationResult
    decoded_frame: Optional[DecodedFrame]
    correlation: Optional[CorrelationResult] = None
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)
    total_time_ms: float = 0.0
    status_summary: str = "SUCCESS"
    status: ResultStatus = ResultStatus.CONFIRMED
    confidence: Optional[float] = None
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.correlation is None and self.sync_detection is not None:
            self.correlation = self.sync_detection
        if self.confidence is not None:
            self.confidence = validate_confidence(self.confidence)

    # Telemetry Helper Properties
    @property
    def estimated_snr_db(self) -> float:
        return self.features.snr_db if self.features else 0.0

    @property
    def estimated_baud_rate(self) -> float:
        return self.features.estimated_baud_rate if self.features else 0.0

    @property
    def sync_found(self) -> bool:
        return self.sync_detection.found if self.sync_detection else False

    @property
    def sync_word_name(self) -> str:
        return self.sync_detection.sync_name if self.sync_detection else "NONE"

    @property
    def sync_bit_index(self) -> int:
        return self.sync_detection.bit_index if self.sync_detection else -1

    @property
    def sync_metrics(self) -> np.ndarray:
        return self.sync_detection.correlation_curve if self.sync_detection else np.array([])

    @property
    def packet_header(self) -> Optional[PacketHeader]:
        if not self.decoded_frame:
            return None
        return PacketHeader(
            version=self.decoded_frame.version,
            packet_type=self.decoded_frame.packet_type,
            sequence_num=self.decoded_frame.seq_num,
            payload_len=self.decoded_frame.payload_length,
            crc_type="CRC-16-CCITT",
            received_crc=0,
            computed_crc=0,
            crc_valid=self.decoded_frame.crc_valid,
        )

    @property
    def crc_valid(self) -> bool:
        return self.decoded_frame.crc_valid if self.decoded_frame else False

    @property
    def payload_bytes(self) -> bytes:
        return self.decoded_frame.raw_payload_bytes if self.decoded_frame else b""

    @property
    def mod_classification_result(self) -> Optional[ModulationResult]:
        return self.modulation
