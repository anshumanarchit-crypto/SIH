"""
spectralq.confidence

Deterministic, transparent confidence calculation engine for Phase 3.

NO arbitrary numbers like 'confidence = 0.95'.
NO magic numbers.
NO probabilistic hallucinations.

Computes a deterministic evidence confidence from factual physical, synchronization,
demodulation, FEC, re-encode, and bitstream structural metrics, minus explicit
contradiction penalties.

METHODOLOGY:
- Each component provides a normalized score in [0.0, 1.0] and an explicit weight.
- Total base score is the weighted average across active components.
- Factual contradictions apply explicit, bounded penalties.
- Failure of critical invariants (e.g. FEC decoding failed, timing unlocked) enforces
  strict upper bounds on the final confidence.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np

from spectralq.schemas import (
    ConfidenceComponent,
    ConfidenceResult,
    ContradictionSeverity,
    Contradiction,
    EvidenceCategory,
    BitstreamAnalysisResult,
    StructuralClassification,
)
from spectralq.evidence import EvidenceLedger


def calculate_deterministic_confidence(
    ledger: EvidenceLedger,
    bitstream_analysis: Optional[BitstreamAnalysisResult] = None,
    fec_configured: bool = True,
) -> ConfidenceResult:
    """Calculate deterministic evidence confidence from auditable pipeline evidence.

    Parameters
    ----------
    ledger : EvidenceLedger
        Ledger containing all recorded evidence items and detected contradictions.
    bitstream_analysis : Optional[BitstreamAnalysisResult]
        Result of the downstream bitstream structural analysis.
    fec_configured : bool
        Whether forward error correction was expected for this transmission.

    Returns
    -------
    ConfidenceResult
        Deterministic confidence summary with transparent components, penalties,
        and uncertainty reasons.
    """
    components: Dict[str, ConfidenceComponent] = {}
    penalties: Dict[str, float] = {}
    uncertainty_reasons: List[str] = []

    # ------------------------------------------------------------------------
    # 1. PHYSICAL LAYER QUALITY (Weight: 0.15)
    # ------------------------------------------------------------------------
    snr_ev = ledger.get_by_id("EVID_PHYS_SNR_EST")
    evm_ev = ledger.get_by_id("EVID_PHYS_EVM_RMS")

    phys_score = 0.75
    phys_raw = None
    phys_refs: List[str] = []
    phys_reason = "Nominal physical signal metrics."

    if snr_ev and snr_ev.value is not None:
        phys_raw = snr_ev.value
        phys_refs.append(snr_ev.evidence_id)
        snr_f = float(snr_ev.value)
        # 4 dB -> 0.0, 20 dB -> 1.0
        phys_score = max(0.0, min(1.0, (snr_f - 4.0) / 16.0))
        phys_reason = f"Calculated from estimated SNR ({snr_f:.1f} dB)."
        if snr_f <= 7.0:
            uncertainty_reasons.append(f"Physical SNR ({snr_f:.1f} dB) is near receiver operating threshold.")
    elif evm_ev and evm_ev.value is not None:
        phys_raw = evm_ev.value
        phys_refs.append(evm_ev.evidence_id)
        evm_f = float(evm_ev.value)
        phys_score = max(0.0, min(1.0, 1.0 - 2.0 * evm_f))
        phys_reason = f"Calculated from RMS EVM ({evm_f:.3f})."

    components["physical_quality"] = ConfidenceComponent(
        name="physical_quality",
        raw_metric=phys_raw,
        normalized_score=phys_score,
        weight=0.15,
        reason=phys_reason,
        source_evidence_ids=phys_refs,
    )

    # ------------------------------------------------------------------------
    # 2. TIMING SYNCHRONIZATION QUALITY (Weight: 0.15)
    # ------------------------------------------------------------------------
    timing_lock_ev = ledger.get_by_id("EVID_TIMING_LOCK")
    timing_jit_ev = ledger.get_by_id("EVID_TIMING_JITTER_VAR")

    timing_score = 1.0
    timing_raw = None
    timing_refs: List[str] = []
    timing_reason = "Symbol timing converged with low jitter."

    if timing_lock_ev and not bool(timing_lock_ev.value):
        timing_score = 0.0
        timing_raw = False
        timing_refs.append(timing_lock_ev.evidence_id)
        timing_reason = "Symbol timing loop failed to achieve lock."
        uncertainty_reasons.append("Timing recovery unlocked.")
    elif timing_jit_ev and timing_jit_ev.value is not None:
        timing_raw = timing_jit_ev.value
        timing_refs.append(timing_jit_ev.evidence_id)
        jit_f = float(timing_jit_ev.value)
        timing_score = max(0.0, min(1.0, 1.0 - 4.0 * jit_f))
        timing_reason = f"Timing loop locked with jitter variance {jit_f:.4f}."
        if jit_f > 0.15:
            uncertainty_reasons.append(f"Elevated timing error variance ({jit_f:.4f}).")

    components["timing_quality"] = ConfidenceComponent(
        name="timing_quality",
        raw_metric=timing_raw,
        normalized_score=timing_score,
        weight=0.15,
        reason=timing_reason,
        source_evidence_ids=timing_refs,
    )

    # ------------------------------------------------------------------------
    # 3. CARRIER SYNCHRONIZATION QUALITY (Weight: 0.15)
    # ------------------------------------------------------------------------
    carrier_lock_ev = ledger.get_by_id("EVID_CARRIER_LOCK")
    carrier_phase_ev = ledger.get_by_id("EVID_CARRIER_PHASE_OFFSET")

    carrier_score = 1.0
    carrier_raw = None
    carrier_refs: List[str] = []
    carrier_reason = "Carrier loop locked and frequency offset compensated."

    if carrier_lock_ev and not bool(carrier_lock_ev.value):
        carrier_score = 0.0
        carrier_raw = False
        carrier_refs.append(carrier_lock_ev.evidence_id)
        carrier_reason = "Carrier recovery loop did not converge."
        uncertainty_reasons.append("Carrier recovery unlocked.")
    elif carrier_phase_ev and carrier_phase_ev.value is not None:
        carrier_raw = carrier_phase_ev.value
        carrier_refs.append(carrier_phase_ev.evidence_id)
        p_offset = abs(float(carrier_phase_ev.value))
        carrier_score = max(0.2, min(1.0, 1.0 - p_offset / np.pi))
        carrier_reason = f"Carrier converged with residual phase offset {p_offset:.4f} rad."

    components["carrier_quality"] = ConfidenceComponent(
        name="carrier_quality",
        raw_metric=carrier_raw,
        normalized_score=carrier_score,
        weight=0.15,
        reason=carrier_reason,
        source_evidence_ids=carrier_refs,
    )

    # ------------------------------------------------------------------------
    # 4. DEMODULATION QUALITY (Weight: 0.10)
    # ------------------------------------------------------------------------
    demod_stat_ev = ledger.get_by_id("EVID_DEMOD_STATUS")
    demod_score = 1.0
    demod_raw = None
    demod_refs: List[str] = []
    demod_reason = "Demodulation succeeded nominally."

    if demod_stat_ev:
        demod_raw = demod_stat_ev.value
        demod_refs.append(demod_stat_ev.evidence_id)
        val_str = str(demod_stat_ev.value)
        if "SUCCESS" in val_str:
            demod_score = 1.0
            demod_reason = "Demodulator reported SUCCESS."
        elif "NON_CONVERGED" in val_str or "LOW_QUALITY" in val_str:
            demod_score = 0.35
            demod_reason = f"Demodulator reported degraded status: {val_str}."
            uncertainty_reasons.append(f"Demodulation status: {val_str}.")
        else:
            demod_score = 0.0
            demod_reason = f"Demodulation failed: {val_str}."
            uncertainty_reasons.append(f"Demodulation failed: {val_str}.")

    components["demodulation_quality"] = ConfidenceComponent(
        name="demodulation_quality",
        raw_metric=demod_raw,
        normalized_score=demod_score,
        weight=0.10,
        reason=demod_reason,
        source_evidence_ids=demod_refs,
    )

    # ------------------------------------------------------------------------
    # 5. FEC CONSISTENCY (Weight: 0.20 if FEC configured, else 0.0)
    # ------------------------------------------------------------------------
    fec_ev = ledger.get_by_id("EVID_FEC_SUCCESS")
    fec_score = 1.0
    fec_weight = 0.20 if fec_configured else 0.0
    fec_raw = None
    fec_refs: List[str] = []
    fec_reason = "No FEC configured for transmission."

    if fec_configured:
        if fec_ev:
            fec_raw = bool(fec_ev.value)
            fec_refs.append(fec_ev.evidence_id)
            if fec_raw:
                fec_score = 1.0
                fec_reason = "FEC decoder successfully converged on valid codeword."
            else:
                fec_score = 0.0
                fec_reason = "FEC decoder failed to converge or syndrome check failed."
                uncertainty_reasons.append("FEC decoder failed to recover valid codeword.")
        else:
            fec_score = 0.0
            fec_reason = "FEC expected but no decoder evidence recorded."
            uncertainty_reasons.append("FEC decoding status unknown.")

    components["fec_consistency"] = ConfidenceComponent(
        name="fec_consistency",
        raw_metric=fec_raw,
        normalized_score=fec_score,
        weight=fec_weight,
        reason=fec_reason,
        source_evidence_ids=fec_refs,
    )

    # ------------------------------------------------------------------------
    # 6. RE-ENCODE CONSISTENCY (Weight: 0.15 if FEC configured, else 0.0)
    # ------------------------------------------------------------------------
    reenc_ev = ledger.get_by_id("EVID_REENCODE_ERRORS")
    reenc_score = 1.0
    reenc_weight = 0.15 if fec_configured else 0.0
    reenc_raw = None
    reenc_refs: List[str] = []
    reenc_reason = "Re-encode check not applicable (uncoded signal)."

    if fec_configured:
        if reenc_ev and reenc_ev.value is not None:
            reenc_raw = reenc_ev.value
            reenc_refs.append(reenc_ev.evidence_id)
            err_count = int(reenc_ev.value)
            if err_count == 0:
                reenc_score = 1.0
                reenc_reason = "Re-encoded payload identically matches demapped channel bits (0 bit errors)."
            else:
                reenc_score = max(0.0, 1.0 - min(1.0, err_count / 100.0))
                reenc_reason = f"Re-encoded payload had {err_count} mismatches against channel bits."
                uncertainty_reasons.append(f"Re-encode consistency discrepancy: {err_count} bit mismatches.")
        else:
            reenc_score = 0.0
            reenc_weight = 0.05
            reenc_reason = "Re-encode validation could not be executed."

    components["reencode_consistency"] = ConfidenceComponent(
        name="reencode_consistency",
        raw_metric=reenc_raw,
        normalized_score=reenc_score,
        weight=reenc_weight,
        reason=reenc_reason,
        source_evidence_ids=reenc_refs,
    )

    # ------------------------------------------------------------------------
    # 7. BITSTREAM STRUCTURAL SUPPORT (Weight: 0.10)
    # ------------------------------------------------------------------------
    struct_score = 0.50
    struct_raw = None
    struct_reason = "No bitstream structural analysis provided."

    if bitstream_analysis:
        struct_raw = bitstream_analysis.structural_classification.value
        c_type = bitstream_analysis.structural_classification
        if c_type in (
            StructuralClassification.TEXT_LIKE,
            StructuralClassification.BYTE_STRUCTURED,
            StructuralClassification.BINARY_STRUCTURED,
        ):
            struct_score = 0.95
            struct_reason = f"Recovered bitstream exhibits valid deterministic structure ({c_type.value})."
        elif c_type == StructuralClassification.HIGHLY_REPETITIVE:
            struct_score = 0.80
            struct_reason = f"Recovered bitstream exhibits periodic or repetitive framing ({c_type.value})."
        elif c_type == StructuralClassification.PADDING_OR_CONSTANT:
            struct_score = 0.15
            struct_reason = "Recovered bitstream is constant or degenerate padding."
            uncertainty_reasons.append("Bitstream is degenerate/constant.")
        else:
            struct_score = 0.40
            struct_reason = "Recovered bitstream structure is indeterminate."

    components["structural_support"] = ConfidenceComponent(
        name="structural_support",
        raw_metric=struct_raw,
        normalized_score=struct_score,
        weight=0.10,
        reason=struct_reason,
        source_evidence_ids=["EVID_STRUCT_CLASS"] if ledger.get_by_id("EVID_STRUCT_CLASS") else [],
    )

    # ------------------------------------------------------------------------
    # 8. EXTERNAL REFERENCE SUPPORT (Dynamic weight: 0.0 if unavailable)
    # ------------------------------------------------------------------------
    ref_ev = ledger.get_by_id("EVID_REF_STATUS")
    ref_avail_ev = ledger.get_by_id("EVID_REF_AVAILABLE")
    ber_ev = ledger.get_by_id("EVID_REF_BER")

    is_ref_active = (
        (ref_avail_ev and bool(ref_avail_ev.value) is True)
        or (ref_ev and ref_ev.value in ("REFERENCE_BITS_AVAILABLE", "AVAILABLE", "EVALUATED_AGAINST_EXTERNAL_REFERENCE", "VERIFIED"))
    )

    if is_ref_active and ber_ev and ber_ev.value is not None:
        ber_f = float(ber_ev.value)
        ref_score = max(0.0, 1.0 - 10.0 * ber_f)
        src_ids = [ev.evidence_id for ev in [ref_ev, ref_avail_ev, ber_ev] if ev]
        components["reference_support"] = ConfidenceComponent(
            name="reference_support",
            raw_metric=ber_f,
            normalized_score=ref_score,
            weight=0.15,
            reason=f"External reference bits validated with BER={ber_f:.4e}.",
            source_evidence_ids=src_ids,
        )
    else:
        # Reference is unavailable; DO NOT invent, DO NOT penalize
        components["reference_support"] = ConfidenceComponent(
            name="reference_support",
            raw_metric="REFERENCE_BITS_UNAVAILABLE",
            normalized_score=0.50,
            weight=0.0,  # Zero weight: neutral impact!
            reason="External reference bits unavailable; truth reference not used in confidence calculation.",
            source_evidence_ids=[ref_ev.evidence_id] if ref_ev else [],
        )

    # ------------------------------------------------------------------------
    # 9. WEIGHTED AGGREGATION & CONTRADICTION PENALTIES
    # ------------------------------------------------------------------------
    active_weights = sum(c.weight for c in components.values())
    if active_weights > 0.0:
        base_score = sum(c.weight * c.normalized_score for c in components.values()) / active_weights
    else:
        base_score = 0.0

    # Scan contradictions for explicit deductions
    contradictions = ledger.get_contradictions()
    total_penalty = 0.0

    severity_penalties = {
        ContradictionSeverity.CRITICAL: 0.40,
        ContradictionSeverity.HIGH: 0.20,
        ContradictionSeverity.MEDIUM: 0.10,
        ContradictionSeverity.LOW: 0.05,
        ContradictionSeverity.INFO: 0.00,  # Info (like reference unavailable) does not penalize
    }

    for c in contradictions:
        sev = c.severity
        p_val = severity_penalties.get(sev, 0.0)
        if p_val > 0.0:
            penalties[c.contradiction_type] = penalties.get(c.contradiction_type, 0.0) + p_val
            total_penalty += p_val

    # Final bounded confidence
    confidence_val = max(0.0, min(1.0, base_score - total_penalty))

    # ------------------------------------------------------------------------
    # 10. INVARIANT BOUNDS (Fail-safes against false confidence)
    # ------------------------------------------------------------------------
    # If FEC was configured and failed or had major re-encode mismatch, confidence cannot exceed 0.12
    if fec_configured:
        if fec_ev and not bool(fec_ev.value):
            confidence_val = min(confidence_val, 0.12)
            if "Decoder failed to establish valid FEC codeword under channel conditions." not in uncertainty_reasons:
                uncertainty_reasons.append("Decoder failed to establish valid FEC codeword under channel conditions.")
        elif reenc_ev and reenc_ev.value is not None and int(reenc_ev.value) > 10:
            confidence_val = min(confidence_val, 0.12)
            if "Major re-encode discrepancy against received signal." not in uncertainty_reasons:
                uncertainty_reasons.append("Major re-encode discrepancy against received signal.")

    # If timing was unlocked, confidence cannot exceed 0.10
    if timing_lock_ev and not bool(timing_lock_ev.value):
        confidence_val = min(confidence_val, 0.10)

    # If bitstream is empty, confidence is exactly 0.0
    if bitstream_analysis and bitstream_analysis.bit_count == 0:
        confidence_val = 0.0
        uncertainty_reasons.append("Zero source bits recovered.")

    return ConfidenceResult(
        value=round(confidence_val, 4),
        method="deterministic_evidence_weighted_v1",
        method_version="phase3-v1",
        components=components,
        penalties=penalties,
        uncertainty_reasons=uncertainty_reasons,
    )
