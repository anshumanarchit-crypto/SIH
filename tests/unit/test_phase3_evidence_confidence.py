"""
tests/unit/test_phase3_evidence_confidence.py

Unit tests for Phase 3 Evidence Ledger, Contradiction Detection,
Hypothesis Evaluation, and Deterministic Confidence.
"""

from __future__ import annotations
import numpy as np
import pytest

from spectralq.schemas import (
    EvidenceCategory,
    EvidenceReliability,
    EvidenceItem,
    Contradiction,
    ContradictionSeverity,
    HypothesisStatus,
    StructuralClassification,
    BitstreamAnalysisResult,
)
from spectralq.evidence import EvidenceLedger
from spectralq.hypothesis import build_candidate_hypotheses
from spectralq.confidence import calculate_deterministic_confidence
from spectralq.decoder_api import DecoderResult, DecoderStatus, ModulationType, FECType, InterleaverType
from spectralq.bitintel import generate_decoder_evidence


def test_evidence_ledger_storage_and_query():
    """Verify evidence ledger stores and indexes items by category and ID."""
    ledger = EvidenceLedger()
    item1 = EvidenceItem(
        evidence_id="EVID_TEST_1",
        category=EvidenceCategory.TIMING,
        metric="timing_lock",
        value=True,
        unit="boolean",
        source="test",
        reliability=EvidenceReliability.HIGH,
        interpretation="Timing loop locked.",
    )
    item2 = EvidenceItem(
        evidence_id="EVID_TEST_2",
        category=EvidenceCategory.FEC,
        metric="decoder_success",
        value=True,
        unit="boolean",
        source="test",
        reliability=EvidenceReliability.HIGH,
        interpretation="FEC converged.",
    )
    ledger.add_item(item1)
    ledger.add_item(item2)

    assert len(ledger.get_items()) == 2
    assert ledger.get_by_id("EVID_TEST_1") == item1
    assert ledger.get_by_id("EVID_NONEXISTENT") is None

    timing_items = ledger.get_by_category(EvidenceCategory.TIMING)
    assert len(timing_items) == 1
    assert timing_items[0].evidence_id == "EVID_TEST_1"


def test_contradiction_detection_scenarios():
    """Verify automatic detection of timing, carrier, SNR, and FEC contradictions."""
    ledger = EvidenceLedger()

    # 1. Unlocked timing
    ledger.add_item(EvidenceItem(
        evidence_id="EVID_TIMING_LOCK",
        category=EvidenceCategory.TIMING,
        metric="timing_lock",
        value=False,
        unit="boolean",
        source="test",
        reliability=EvidenceReliability.HIGH,
        interpretation="Timing failed.",
    ))
    # 2. Near-threshold SNR
    ledger.add_item(EvidenceItem(
        evidence_id="EVID_PHYS_SNR_EST",
        category=EvidenceCategory.PHYSICAL_SIGNAL,
        metric="estimated_snr_db",
        value=5.5,
        unit="dB",
        source="test",
        reliability=EvidenceReliability.MEDIUM,
        interpretation="Low SNR.",
    ))
    # 3. Failed FEC
    ledger.add_item(EvidenceItem(
        evidence_id="EVID_FEC_SUCCESS",
        category=EvidenceCategory.FEC,
        metric="decoder_success",
        value=False,
        unit="boolean",
        source="test",
        reliability=EvidenceReliability.HIGH,
        interpretation="FEC failed.",
    ))
    # 4. Missing reference
    ledger.add_item(EvidenceItem(
        evidence_id="EVID_REF_STATUS",
        category=EvidenceCategory.REFERENCE_AVAILABILITY,
        metric="reference_status",
        value="REFERENCE_BITS_UNAVAILABLE",
        unit="status",
        source="test",
        reliability=EvidenceReliability.HIGH,
        interpretation="Reference unavailable.",
    ))

    contradictions = ledger.detect_contradictions()
    contra_types = {c.contradiction_type for c in contradictions}

    assert "TIMING_UNLOCKED" in contra_types
    assert "NEAR_THRESHOLD_SNR" in contra_types
    assert "FEC_DECODE_FAILED" in contra_types
    assert "REFERENCE_UNAVAILABLE" in contra_types

    # Verify severity assignment
    timing_contra = next(c for c in contradictions if c.contradiction_type == "TIMING_UNLOCKED")
    assert timing_contra.severity == ContradictionSeverity.CRITICAL

    ref_contra = next(c for c in contradictions if c.contradiction_type == "REFERENCE_UNAVAILABLE")
    assert ref_contra.severity == ContradictionSeverity.INFO


def test_candidate_hypothesis_evaluation_ranking():
    """Verify deterministic ranking of competing candidate hypotheses."""
    candidate_records = [
        {"angle_deg": 0.0, "decoder_success": False, "re_encode_errors": 120},
        {"angle_deg": 90.0, "decoder_success": False, "re_encode_errors": 115},
        {"angle_deg": 180.0, "decoder_success": True, "re_encode_errors": 0},
        {"angle_deg": 270.0, "decoder_success": False, "re_encode_errors": 118},
    ]

    hypotheses, selected = build_candidate_hypotheses(
        candidate_records=candidate_records,
        mapping_profile="DEFAULT",
        fec_type_str="CONV_K7_171_133",
        selected_angle_deg=180.0,
    )

    assert selected is not None
    assert selected.hypothesis_id == "HYP_ROT_180"
    assert selected.re_encode_errors == 0
    assert selected.status == HypothesisStatus.CONFIRMED_BY_MULTIPLE_EVIDENCE
    assert selected.rank == 1

    # Rejected candidates should have rank > 1
    rejected = [h for h in hypotheses if h.status == HypothesisStatus.REJECTED]
    assert len(rejected) == 3


def test_candidate_hypothesis_ambiguity_detection():
    """Verify that equal-scoring candidates are flagged as AMBIGUOUS."""
    candidate_records = [
        {"angle_deg": 0.0, "decoder_success": True, "re_encode_errors": 5},
        {"angle_deg": 180.0, "decoder_success": True, "re_encode_errors": 5},
    ]

    hypotheses, selected = build_candidate_hypotheses(
        candidate_records=candidate_records,
        mapping_profile="DEFAULT",
        fec_type_str="CONV_K7_171_133",
    )

    assert len(hypotheses) == 2
    assert hypotheses[0].status == HypothesisStatus.AMBIGUOUS
    assert hypotheses[1].status == HypothesisStatus.AMBIGUOUS


def test_deterministic_confidence_reproducibility():
    """Verify that confidence calculation produces exact identical results across repeated calls."""
    ledger = EvidenceLedger()
    ledger.add_item(EvidenceItem(
        evidence_id="EVID_PHYS_SNR_EST",
        category=EvidenceCategory.PHYSICAL_SIGNAL,
        metric="estimated_snr_db",
        value=15.0,
        unit="dB",
        source="test",
        reliability=EvidenceReliability.MEDIUM,
        interpretation="Good SNR.",
    ))
    ledger.add_item(EvidenceItem(
        evidence_id="EVID_TIMING_LOCK",
        category=EvidenceCategory.TIMING,
        metric="timing_lock",
        value=True,
        unit="boolean",
        source="test",
        reliability=EvidenceReliability.HIGH,
        interpretation="Timing locked.",
    ))
    ledger.add_item(EvidenceItem(
        evidence_id="EVID_CARRIER_LOCK",
        category=EvidenceCategory.CARRIER,
        metric="carrier_lock",
        value=True,
        unit="boolean",
        source="test",
        reliability=EvidenceReliability.HIGH,
        interpretation="Carrier locked.",
    ))
    ledger.add_item(EvidenceItem(
        evidence_id="EVID_FEC_SUCCESS",
        category=EvidenceCategory.FEC,
        metric="decoder_success",
        value=True,
        unit="boolean",
        source="test",
        reliability=EvidenceReliability.HIGH,
        interpretation="FEC pass.",
    ))
    ledger.add_item(EvidenceItem(
        evidence_id="EVID_REENCODE_ERRORS",
        category=EvidenceCategory.REENCODE,
        metric="re_encode_errors",
        value=0,
        unit="bits",
        source="test",
        reliability=EvidenceReliability.HIGH,
        interpretation="Re-encode match.",
    ))

    c1 = calculate_deterministic_confidence(ledger, fec_configured=True)
    c2 = calculate_deterministic_confidence(ledger, fec_configured=True)

    assert c1.value == c2.value
    assert c1.to_dict() == c2.to_dict()
    assert c1.value is not None
    assert c1.value > 0.85


def test_confidence_invariant_fec_failure():
    """Verify invariant: FEC failure strictly caps confidence to <= 0.12."""
    ledger = EvidenceLedger()
    ledger.add_item(EvidenceItem(
        evidence_id="EVID_FEC_SUCCESS",
        category=EvidenceCategory.FEC,
        metric="decoder_success",
        value=False,
        unit="boolean",
        source="test",
        reliability=EvidenceReliability.HIGH,
        interpretation="FEC failed.",
    ))
    ledger.detect_contradictions()

    res = calculate_deterministic_confidence(ledger, fec_configured=True)
    assert res.value is not None
    assert res.value <= 0.12
    assert any("fec" in r.lower() for r in res.uncertainty_reasons)


def test_confidence_missing_reference_is_neutral():
    """Verify that missing reference does NOT penalize confidence or fabricate BER."""
    ledger = EvidenceLedger()
    ledger.add_item(EvidenceItem(
        evidence_id="EVID_REF_STATUS",
        category=EvidenceCategory.REFERENCE_AVAILABILITY,
        metric="reference_status",
        value="REFERENCE_BITS_UNAVAILABLE",
        unit="status",
        source="test",
        reliability=EvidenceReliability.HIGH,
        interpretation="No truth reference available.",
    ))
    ledger.detect_contradictions()

    res = calculate_deterministic_confidence(ledger, fec_configured=False)
    # The reference component weight should be 0.0 (neutral)
    ref_comp = res.components["reference_support"]
    assert ref_comp.weight == 0.0
    assert "REFERENCE_UNAVAILABLE" not in res.penalties


def test_generate_decoder_evidence_facade_integration():
    """Verify end-to-end evidence generation from a simulated DecoderResult."""
    bits = np.tile([1, 0, 0, 1, 0, 1, 1, 0], 32).astype(np.uint8)  # 256 bits
    dec_res = DecoderResult(
        status=DecoderStatus.SUCCESS,
        modulation=ModulationType.BPSK,
        fec_type=FECType.CONVOLUTIONAL_K7,
        interleaver_type=InterleaverType.BLOCK,
        bit_count=len(bits),
        recovered_source_bits=bits,
        hard_bits=bits,
        decoder_success=True,
        selected_rotation_deg=180.0,
        re_encode_errors=0,
        candidate_evaluations=[
            {"angle_deg": 0.0, "decoder_success": False, "re_encode_errors": 100},
            {"angle_deg": 180.0, "decoder_success": True, "re_encode_errors": 0},
        ],
        timing_status={"timing_lock": True, "timing_error_variance": 0.001},
        carrier_status={"carrier_lock": True, "residual_phase_offset_rad": 0.02},
        diagnostics={"demod": {"estimated_snr_db": 16.0, "evm_rms": 0.08}},
    )

    handoff = generate_decoder_evidence(dec_res, case_id="SIMULATED_TEST")
    assert handoff.case_id == "SIMULATED_TEST"
    assert handoff.overall_status == "CONFIRMED_BY_MULTIPLE_EVIDENCE"
    assert handoff.selected_hypothesis is not None
    assert handoff.selected_hypothesis.hypothesis_id == "HYP_ROT_180"
    assert handoff.confidence is not None
    assert handoff.confidence.value is not None and handoff.confidence.value > 0.85
    assert handoff.bitstream_analysis is not None
    assert handoff.bitstream_analysis.bit_count == 256

    # Verify JSON serialization
    json_str = handoff.to_json()
    assert isinstance(json_str, str)
    assert "spectralq-decoder-evidence-v1" in json_str
