"""
tests/golden/test_phase3_official_handoff.py

Golden validation suite for Phase 3:
- End-to-end evidence generation across official Sinchana captures (G1 through G7)
- Zero Phase 2.6 regressions
- Verification of G7 near-threshold insufficient evidence reporting
- Verification of G1/G5 uncoded reference unavailability preservation
- Machine-readable handoff JSON serialization and determinism
- Production code truth-isolation audit
"""

from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
import pytest

from spectralq.demod import demodulate, DemodConfig
from spectralq.decoder_api import (
    DecoderConfig,
    DecoderPipeline,
    ModulationType,
    FECType,
    InterleaverType,
    DecoderStatus,
    DecoderResult,
)
from spectralq.bitintel import generate_decoder_evidence

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL_DIR = _REPO_ROOT / "data" / "official" / "sinchana" / "golden"
_GOLDEN_DIR = _REPO_ROOT / "data" / "golden"


def _load_cf32(filename: str) -> np.ndarray:
    path = _OFFICIAL_DIR / filename
    assert path.exists(), f"Capture file missing: {path}"
    raw = np.fromfile(path, dtype=np.float32)
    return raw[0::2] + 1j * raw[1::2]


def _load_meta(filename: str) -> dict:
    path = _OFFICIAL_DIR / filename
    assert path.exists(), f"Metadata file missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase3_official_g2_evidence_package():
    """Verify official G2 BPSK generates complete evidence package with 0 errors and CONFIRMED status."""
    iq = _load_cf32("G2_BPSK_conv_block.cf32")
    meta = _load_meta("G2_BPSK_conv_block.truth.json")

    dec_cfg = DecoderConfig(
        modulation=ModulationType.BPSK,
        fec_type=FECType.CONVOLUTIONAL_K7,
        interleaver_type=InterleaverType.BLOCK,
        sample_rate=float(meta["fs_hz"]),
        samples_per_symbol=int(meta["sps"]),
        rrc_alpha=float(meta["rolloff"]),
        candidate_phase_evaluation=True,
        interleaver_params={"rows": 16, "cols": 34},
        metadata={"case_id": "G2"},
    )
    pipeline = DecoderPipeline()
    res = pipeline.decode(iq, dec_cfg)

    # Decode integrity
    assert res.decoder_success is True
    assert res.selected_rotation_deg == 180.0
    assert res.re_encode_errors == 0

    # Phase 3 Evidence Handoff
    handoff = generate_decoder_evidence(res, case_id="G2")
    assert handoff.overall_status == "CONFIRMED_BY_MULTIPLE_EVIDENCE"
    assert handoff.selected_hypothesis is not None
    assert handoff.selected_hypothesis.parameters["angle_deg"] == 180.0
    assert handoff.confidence is not None
    assert handoff.confidence.value is not None and handoff.confidence.value > 0.85
    assert handoff.bitstream_analysis is not None
    assert handoff.bitstream_analysis.bit_count == 266
    assert handoff.bitstream_analysis.remainder_bits == 2

    # JSON exportability
    json_doc = json.loads(handoff.to_json())
    assert json_doc["case_id"] == "G2"
    assert json_doc["decoder"]["success"] is True


def test_phase3_official_g3_evidence_package():
    """Verify official G3 8-PSK generates complete evidence package with OCTAVE mapping profile."""
    iq = _load_cf32("G3_8PSK_RS_diagonal.cf32")
    meta = _load_meta("G3_8PSK_RS_diagonal.truth.json")

    dec_cfg = DecoderConfig(
        modulation=ModulationType.PSK8,
        fec_type=FECType.REED_SOLOMON_255_223,
        interleaver_type=InterleaverType.DIAGONAL,
        sample_rate=float(meta["fs_hz"]),
        samples_per_symbol=int(meta["sps"]),
        rrc_alpha=float(meta["rolloff"]),
        mapping_profile="OCTAVE",
        candidate_phase_evaluation=True,
        interleaver_params={"rows": 40, "cols": 51},
        metadata={"case_id": "G3"},
    )
    pipeline = DecoderPipeline()
    res = pipeline.decode(iq, dec_cfg)

    assert res.decoder_success is True
    assert res.re_encode_errors == 0

    handoff = generate_decoder_evidence(res, case_id="G3")
    assert handoff.overall_status == "CONFIRMED_BY_MULTIPLE_EVIDENCE"
    assert handoff.confidence is not None and handoff.confidence.value is not None and handoff.confidence.value > 0.85
    assert handoff.bitstream_analysis is not None
    assert handoff.bitstream_analysis.bit_count == 1784


def test_phase3_official_g4_evidence_package():
    """Verify official G4 16-QAM LDPC generates complete evidence package."""
    iq = _load_cf32("G4_16QAM_LDPC_pseudorandom.cf32")
    meta = _load_meta("G4_16QAM_LDPC_pseudorandom.truth.json")

    dec_cfg = DecoderConfig(
        modulation=ModulationType.QAM16,
        fec_type=FECType.LDPC,
        interleaver_type=InterleaverType.PSEUDORANDOM,
        sample_rate=float(meta["fs_hz"]),
        samples_per_symbol=int(meta["sps"]),
        rrc_alpha=float(meta["rolloff"]),
        mapping_profile="OCTAVE",
        candidate_phase_evaluation=True,
        interleaver_params={"seed": 42},
        metadata={"case_id": "G4"},
    )
    pipeline = DecoderPipeline()
    res = pipeline.decode(iq, dec_cfg)

    assert res.decoder_success is True
    assert res.re_encode_errors == 0

    handoff = generate_decoder_evidence(res, case_id="G4")
    assert handoff.overall_status == "CONFIRMED_BY_MULTIPLE_EVIDENCE"
    assert handoff.confidence is not None and handoff.confidence.value is not None and handoff.confidence.value > 0.85
    assert handoff.bitstream_analysis is not None
    assert handoff.bitstream_analysis.bit_count == 50


def test_phase3_official_g5_coded_evidence_package():
    """Verify official G5 2-FSK RS+Conv regression anchor generates complete evidence package."""
    iq = _load_cf32("G5_2FSK_RS_Conv_interleaved.cf32")
    meta = _load_meta("G5_2FSK_RS_Conv_interleaved.truth.json")

    dec_cfg = DecoderConfig(
        modulation=ModulationType.FSK2,
        fec_type=FECType.CONCATENATED_RS_CONV,
        interleaver_type=InterleaverType.CONVOLUTIONAL,
        sample_rate=float(meta["fs_hz"]),
        samples_per_symbol=int(meta["sps"]),
        interleaver_params={"branches": 6},
        metadata={
            "case_id": "G5",
            "fsk_deviation": float(meta["fsk_deviation_hz"]),
            "interleaver_metadata": {"branches": 6},
        },
    )
    pipeline = DecoderPipeline()
    res = pipeline.decode(iq, dec_cfg)

    assert res.decoder_success is True

    handoff = generate_decoder_evidence(res, case_id="G5_coded")
    assert handoff.overall_status in ("CONFIRMED_BY_MULTIPLE_EVIDENCE", "SUPPORTED")
    assert handoff.confidence is not None and handoff.confidence.value is not None and handoff.confidence.value > 0.85
    assert handoff.bitstream_analysis is not None
    assert handoff.bitstream_analysis.bit_count == 1784


def test_phase3_official_g6_evidence_package():
    """Verify official G6 BPSK Conv+interleaved generates complete evidence package."""
    iq = _load_cf32("G6_BPSK_conv_interleaved.cf32")
    meta = _load_meta("G6_BPSK_conv_interleaved.truth.json")

    dec_cfg = DecoderConfig(
        modulation=ModulationType.BPSK,
        fec_type=FECType.CONVOLUTIONAL_K7,
        interleaver_type=InterleaverType.CONVOLUTIONAL,
        sample_rate=float(meta["fs_hz"]),
        samples_per_symbol=int(meta["sps"]),
        rrc_alpha=float(meta["rolloff"]),
        candidate_phase_evaluation=True,
        interleaver_params={"branches": 4},
        metadata={"case_id": "G6", "interleaver_metadata": {"branches": 4}},
    )
    pipeline = DecoderPipeline()
    res = pipeline.decode(iq, dec_cfg)

    assert res.decoder_success is True
    assert res.selected_rotation_deg == 180.0
    assert res.re_encode_errors == 0

    handoff = generate_decoder_evidence(res, case_id="G6")
    assert handoff.overall_status == "CONFIRMED_BY_MULTIPLE_EVIDENCE"
    assert handoff.confidence is not None and handoff.confidence.value is not None and handoff.confidence.value > 0.85
    assert handoff.bitstream_analysis is not None
    assert handoff.bitstream_analysis.bit_count == 256


def test_phase3_official_g1_uncoded_missing_reference():
    """Verify G1 uncoded truthfully reports REFERENCE_BITS_UNAVAILABLE without fabricating BER."""
    iq = _load_cf32("G1_QPSK_uncoded.cf32")
    meta = _load_meta("G1_QPSK_uncoded.truth.json")

    demod_cfg = DemodConfig(
        modulation="QPSK",
        sample_rate=float(meta["fs_hz"]),
        samples_per_symbol=int(meta["sps"]),
        rrc_alpha=float(meta["rolloff"]),
    )
    demod_res = demodulate(iq, demod_cfg)

    dec_res = DecoderResult(
        status=DecoderStatus.SUCCESS,
        modulation=ModulationType.QPSK,
        fec_type=FECType.NONE,
        interleaver_type=InterleaverType.NONE,
        bit_count=len(demod_res.hard_bits),
        hard_bits=demod_res.hard_bits,
        symbols=demod_res.symbol_decisions,
        decoder_success=True,
        diagnostics={
            "demod": demod_res.diagnostics,
            "reference_status": "REFERENCE_BITS_UNAVAILABLE",
        },
    )

    handoff = generate_decoder_evidence(dec_res, case_id="G1")
    assert handoff.overall_status == "REFERENCE_BITS_UNAVAILABLE"
    assert handoff.reference["status"] == "REFERENCE_BITS_UNAVAILABLE"
    assert handoff.reference["ber"] is None
    assert handoff.reference["bit_errors"] is None
    assert handoff.bitstream_analysis is not None
    assert handoff.bitstream_analysis.bit_count > 0


def test_phase3_official_g7_near_threshold_insufficient_evidence():
    """Verify G7 near-threshold capture produces informative evidence package without forced success."""
    iq = _load_cf32("G7_QPSK_conv_near_threshold.cf32")
    meta = _load_meta("G7_QPSK_conv_near_threshold.truth.json")

    dec_cfg = DecoderConfig(
        modulation=ModulationType.QPSK,
        fec_type=FECType.CONVOLUTIONAL_K7,
        interleaver_type=InterleaverType.NONE,
        sample_rate=float(meta["fs_hz"]),
        samples_per_symbol=int(meta["sps"]),
        rrc_alpha=float(meta["rolloff"]),
        candidate_phase_evaluation=True,
        metadata={"case_id": "G7"},
    )
    pipeline = DecoderPipeline()
    res = pipeline.decode(iq, dec_cfg)

    # G7 must NOT be falsely reported as successful
    handoff = generate_decoder_evidence(res, case_id="G7", snr_est_db=6.0)

    assert handoff.overall_status == "FAILED_TO_DECODE_NEAR_THRESHOLD_STRESS_CASE"
    assert handoff.decoder["success"] is False
    assert handoff.confidence is not None and handoff.confidence.value is not None
    assert handoff.confidence.value <= 0.15

    # Check that near-threshold contradiction was recorded
    contra_types = {c.contradiction_type for c in handoff.contradictions}
    assert "NEAR_THRESHOLD_SNR" in contra_types
    assert ("REENCODE_DISCREPANCY" in contra_types or "FEC_DECODE_FAILED" in contra_types)

    # Uncertainty reason explanation
    assert any("near-threshold" in r.lower() for r in handoff.uncertainty_reasons)


def test_phase3_handoff_determinism():
    """Verify that calling generate_decoder_evidence twice yields bit-exact identical JSON."""
    iq = _load_cf32("G2_BPSK_conv_block.cf32")
    meta = _load_meta("G2_BPSK_conv_block.truth.json")

    dec_cfg = DecoderConfig(
        modulation=ModulationType.BPSK,
        fec_type=FECType.CONVOLUTIONAL_K7,
        interleaver_type=InterleaverType.BLOCK,
        sample_rate=float(meta["fs_hz"]),
        samples_per_symbol=int(meta["sps"]),
        rrc_alpha=float(meta["rolloff"]),
        candidate_phase_evaluation=True,
        interleaver_params={"rows": 16, "cols": 34},
        metadata={"case_id": "G2"},
    )
    pipeline = DecoderPipeline()
    res = pipeline.decode(iq, dec_cfg)

    h1 = generate_decoder_evidence(res, case_id="G2")
    h2 = generate_decoder_evidence(res, case_id="G2")

    json1 = h1.to_json()
    json2 = h2.to_json()

    assert json1 == json2


def test_production_code_truth_isolation_audit():
    """Audit python/spectralq/ to guarantee zero imports or references to truth files."""
    prod_dir = _REPO_ROOT / "python" / "spectralq"
    forbidden_terms = [
        "truth.json",
        "official_validation_results.json",
        "official_manifest.json",
        "sinchana_zip",
        "SIH-main",
        "core.demodulation",
        "core.pipeline",
        "golden/G",
    ]

    for py_file in prod_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in content, f"Forbidden term '{term}' found in production file {py_file.name}"
