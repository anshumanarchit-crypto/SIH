"""
spectralq.decoder_api

Stable decoder facade and typed interface contract for the SpectralQ decoder core.

ARCHITECTURAL CONTRACT:
INPUT:
    Complex IQ waveform (np.ndarray of complex64/complex128) + explicit DecoderConfig

PIPELINE:
    waveform
       ↓
    demodulation (demod.py)
       ↓
    transmitted coded/interleaved bits (TX BITS)
       ↓
    de-interleaving (interleave.py)
       ↓
    FEC decoding (fec.py)
       ↓
    recovered source bits (SOURCE BITS)
       ↓
    bit-stream intelligence (bitintel.py)
       ↓
    DecoderResult (structured typed metrics and evidence)

BOUNDARY RULE:
- Do NOT calculate Archit's final confidence score here.
- Do NOT calculate the final UNKNOWN decision here.
- Those belong strictly to the downstream hypothesis/evidence aggregation layer.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List, Optional
import numpy as np


class DecoderStatus(Enum):
    """Execution status for the decoder pipeline."""
    SUCCESS = auto()
    UNSUPPORTED = auto()
    INVALID_INPUT = auto()
    DECODER_FAILURE = auto()
    LOW_QUALITY = auto()
    NON_CONVERGED = auto()


class ModulationType(Enum):
    """Supported modulation schemes."""
    BPSK = "BPSK"
    QPSK = "QPSK"
    PSK8 = "8-PSK"
    QAM16 = "16-QAM"
    QAM64 = "64-QAM"
    FSK2 = "2-FSK"
    FSK4 = "4-FSK"
    UNKNOWN = "UNKNOWN"


class FECType(Enum):
    """Supported Forward Error Correction schemes."""
    CONVOLUTIONAL_K7 = "CONV_K7_171_133"
    REED_SOLOMON_255_223 = "RS_255_223"
    LDPC = "LDPC"
    CONCATENATED_RS_CONV = "CONCAT_RS_CONV"
    NONE = "NONE"


class InterleaverType(Enum):
    """Supported interleaving algorithms."""
    BLOCK = "BLOCK"
    DIAGONAL = "DIAGONAL"
    PSEUDORANDOM = "PSEUDORANDOM"
    CONVOLUTIONAL = "CONVOLUTIONAL"
    NONE = "NONE"


@dataclass
class DecoderConfig:
    """Explicit configuration passed into the decoder facade."""
    modulation: ModulationType
    fec_type: FECType
    interleaver_type: InterleaverType
    sample_rate: float
    center_frequency: float = 0.0
    samples_per_symbol: int = 4
    rrc_alpha: float = 0.35
    mapping_profile: str = "DEFAULT"
    candidate_phase_evaluation: bool = True
    external_reference_bits: Optional[np.ndarray] = None
    interleaver_params: Dict[str, Any] = field(default_factory=dict)
    fec_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecoderResult:
    """Typed structured result emitted by the decoder core facade.

    Does NOT compute final overall confidence or UNKNOWN decision.
    Supplies structured evidence for Archit's hypothesis engine.
    """
    status: DecoderStatus
    modulation: ModulationType
    fec_type: FECType
    interleaver_type: InterleaverType
    bit_count: int
    symbols: Optional[np.ndarray] = None
    hard_bits: Optional[np.ndarray] = None
    soft_bits: Optional[np.ndarray] = None
    recovered_source_bits: Optional[np.ndarray] = None
    timing_status: Dict[str, Any] = field(default_factory=dict)
    carrier_status: Dict[str, Any] = field(default_factory=dict)
    decoder_success: bool = False
    convergence: bool = False
    iterations: int = 0
    syndrome_status: Optional[bool] = None
    crc_info: Dict[str, Any] = field(default_factory=dict)
    selected_rotation_deg: float = 0.0
    re_encode_errors: Optional[int] = None
    candidate_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class DecoderPipeline:
    """Facade orchestrating the complete decoder core pipeline."""

    def __init__(self):
        pass

    def decode(self, iq_waveform: np.ndarray, config: DecoderConfig) -> DecoderResult:
        """Execute the decoding pipeline on the provided waveform.

        Orchestrates demodulation, de-interleaving, and FEC decoding into a
        coherent pipeline execution.
        """
        if not isinstance(iq_waveform, np.ndarray) or iq_waveform.ndim != 1:
            return DecoderResult(
                status=DecoderStatus.INVALID_INPUT,
                modulation=config.modulation,
                fec_type=config.fec_type,
                interleaver_type=config.interleaver_type,
                bit_count=0,
                warnings=["Input waveform must be a 1D numpy array of complex samples."]
            )

        if len(iq_waveform) == 0:
            return DecoderResult(
                status=DecoderStatus.INVALID_INPUT,
                modulation=config.modulation,
                fec_type=config.fec_type,
                interleaver_type=config.interleaver_type,
                bit_count=0,
                warnings=["Input waveform is empty."]
            )

        from spectralq.demod import demodulate, DemodConfig, DemodStatus
        from spectralq.encode_chain import decode_chain, encode_chain, EncodeConfig

        # Map DecoderConfig to DemodConfig
        mod_val = config.modulation.value if hasattr(config.modulation, "value") else str(config.modulation)
        fsk_dev = config.metadata.get("fsk_deviation")
        preamble = config.metadata.get("preamble_bits")
        demod_cfg = DemodConfig(
            modulation=mod_val,
            sample_rate=config.sample_rate,
            samples_per_symbol=config.samples_per_symbol,
            rrc_alpha=config.rrc_alpha,
            center_frequency=config.center_frequency,
            fsk_deviation=fsk_dev,
            preamble_bits=preamble,
            mapping_profile=config.mapping_profile,
            external_reference_bits=config.external_reference_bits,
            metadata=config.metadata,
        )

        demod_res = demodulate(iq_waveform, demod_cfg)

        status_map = {
            DemodStatus.SUCCESS: DecoderStatus.SUCCESS,
            DemodStatus.UNSUPPORTED: DecoderStatus.UNSUPPORTED,
            DemodStatus.INVALID_INPUT: DecoderStatus.INVALID_INPUT,
            DemodStatus.DECODER_FAILURE: DecoderStatus.DECODER_FAILURE,
            DemodStatus.LOW_QUALITY: DecoderStatus.LOW_QUALITY,
            DemodStatus.NON_CONVERGED: DecoderStatus.NON_CONVERGED,
        }
        dec_status = status_map.get(demod_res.status, DecoderStatus.DECODER_FAILURE)

        if demod_res.status not in (DemodStatus.SUCCESS, DemodStatus.NON_CONVERGED):
            return DecoderResult(
                status=dec_status,
                modulation=config.modulation,
                fec_type=config.fec_type,
                interleaver_type=config.interleaver_type,
                bit_count=0,
                symbols=demod_res.symbol_decisions,
                hard_bits=demod_res.hard_bits,
                soft_bits=demod_res.soft_bits,
                recovered_source_bits=None,
                timing_status=demod_res.timing_status,
                carrier_status=demod_res.carrier_status,
                decoder_success=False,
                diagnostics={"demod": demod_res.diagnostics},
                warnings=demod_res.warnings,
            )

        # FEC decoding stage if FEC is configured
        fec_val = config.fec_type.value if hasattr(config.fec_type, "value") else str(config.fec_type)
        intl_val = config.interleaver_type.value if hasattr(config.interleaver_type, "value") else str(config.interleaver_type)

        enc_cfg = EncodeConfig(
            case_id=config.metadata.get("case_id", "GENERIC"),
            modulation=mod_val,
            fec_type=fec_val,
            interleaver_type=intl_val,
            fec_params=config.fec_params,
            interleaver_params=config.interleaver_params,
        )

        # Candidate phase evaluation using re-encode consistency
        chosen_cand_deg = 0.0
        chosen_re_encode_errors: Optional[int] = None
        chosen_bits = demod_res.hard_bits
        chosen_syms = demod_res.symbol_decisions
        chosen_soft = demod_res.soft_bits
        chosen_dec_res = None
        candidate_eval_records: List[Dict[str, Any]] = []

        if config.candidate_phase_evaluation and config.fec_type != FECType.NONE and len(demod_res.candidate_rotations) > 1:
            best_re_err = float("inf")
            for cand in demod_res.candidate_rotations:
                deg = float(cand["angle_deg"])
                c_bits = cand["hard_bits"]
                c_dec = decode_chain(
                    received_bits=c_bits,
                    config=enc_cfg,
                    interleaver_meta=config.metadata.get("interleaver_metadata"),
                )
                re_err = None
                if c_dec.decoder_success and len(c_dec.recovered_source_bits) > 0:
                    try:
                        re_enc = encode_chain(
                            source_bits=c_dec.recovered_source_bits,
                            config=enc_cfg,
                        )
                        eval_len = min(len(c_bits), len(re_enc.tx_bits))
                        re_err = int(np.sum(c_bits[:eval_len] != re_enc.tx_bits[:eval_len])) + abs(len(c_bits) - len(re_enc.tx_bits))
                    except Exception:
                        re_err = None

                candidate_eval_records.append({
                    "angle_deg": deg,
                    "decoder_success": c_dec.decoder_success,
                    "re_encode_errors": re_err,
                })

                if re_err is not None and re_err < best_re_err:
                    best_re_err = re_err
                    chosen_cand_deg = deg
                    chosen_re_encode_errors = re_err
                    chosen_bits = c_bits
                    chosen_syms = cand["symbols"]
                    chosen_soft = cand["soft_bits"]
                    chosen_dec_res = c_dec
                    if re_err == 0:
                        # Exact re-encode match found!
                        break

        if chosen_dec_res is None:
            chosen_dec_res = decode_chain(
                received_bits=demod_res.hard_bits,
                config=enc_cfg,
                interleaver_meta=config.metadata.get("interleaver_metadata"),
            )

        final_status = DecoderStatus.SUCCESS if chosen_dec_res.decoder_success else DecoderStatus.DECODER_FAILURE
        combined_warnings = demod_res.warnings + chosen_dec_res.warnings
        final_ber = demod_res.bit_error_rate
        final_errors = demod_res.bit_errors
        if config.external_reference_bits is not None and len(config.external_reference_bits) > 0 and chosen_bits is not None:
            ref_arr = np.asarray(config.external_reference_bits, dtype=int).ravel()
            eval_len = min(len(chosen_bits), len(ref_arr))
            if eval_len > 0:
                mismatches = int(np.sum(chosen_bits[:eval_len] != ref_arr[:eval_len]))
                final_errors = mismatches + abs(len(chosen_bits) - len(ref_arr))
                final_ber = float(mismatches / eval_len)

        combined_diag = {
            "demod": demod_res.diagnostics,
            "fec": chosen_dec_res.fec_metrics,
            "candidate_evaluations": candidate_eval_records,
            "mapping_profile_used": demod_res.mapping_profile_used,
            "reference_status": demod_res.reference_status,
            "bit_error_rate": final_ber,
            "bit_errors": final_errors,
            "reference_bit_count": len(config.external_reference_bits) if config.external_reference_bits is not None else None,
        }

        return DecoderResult(
            status=final_status,
            modulation=config.modulation,
            fec_type=config.fec_type,
            interleaver_type=config.interleaver_type,
            bit_count=len(chosen_dec_res.recovered_source_bits),
            symbols=chosen_syms,
            hard_bits=chosen_bits,
            soft_bits=chosen_soft,
            recovered_source_bits=chosen_dec_res.recovered_source_bits,
            timing_status=demod_res.timing_status,
            carrier_status=demod_res.carrier_status,
            decoder_success=chosen_dec_res.decoder_success,
            selected_rotation_deg=chosen_cand_deg,
            re_encode_errors=chosen_re_encode_errors,
            candidate_evaluations=candidate_eval_records,
            diagnostics=combined_diag,
            warnings=combined_warnings,
        )
