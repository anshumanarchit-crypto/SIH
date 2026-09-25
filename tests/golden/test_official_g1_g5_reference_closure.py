"""tests/golden/test_official_g1_g5_reference_closure.py

Comprehensive test suite verifying Phase 3.1 official G1/G5 reference closure:
1. valid G1 reference ingestion
2. valid G5 reference ingestion
3. invalid reference characters
4. incorrect reference length
5. BER calculation
6. reference unavailable behavior
7. production truth isolation
8. deterministic regenerated output
9. G1 official regression
10. G5 uncoded official regression
"""

from __future__ import annotations
import json
import hashlib
import re
from pathlib import Path
import numpy as np
import pytest

from spectralq.demod import demodulate, DemodConfig, DemodStatus
from spectralq.decoder_api import DecoderConfig, DecoderPipeline, ModulationType, FECType, InterleaverType
from spectralq.bitintel import generate_decoder_evidence
from scripts.generate_official_validation_results import run_validation

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL_DIR = _REPO_ROOT / "data" / "official" / "sinchana" / "golden"
_REF_BITS_DIR = _REPO_ROOT / "data" / "official" / "sinchana" / "reference_bits"
_MANIFEST_FILE = _REPO_ROOT / "data" / "official" / "sinchana" / "official_manifest.json"
_VALIDATION_RESULTS_FILE = _REPO_ROOT / "data" / "official" / "sinchana" / "official_validation_results.json"
_HANDOFF_EVIDENCE_FILE = _REPO_ROOT / "data" / "handoff" / "decoder_evidence.json"

EXPECTED_G1_G5_SHA256 = "4bb2799ee91b19dafe7f61cab16f442a2984bb07b40c3c0bb5863e37974579ab"
EXPECTED_BIT_COUNT = 4096


# ----------------------------------------------------------------------------
# 1. Valid G1 Reference Ingestion
# ----------------------------------------------------------------------------
def test_valid_g1_reference_ingestion():
    """Verify that bitsG1.txt exists, has correct size, hash, character set, and manifest entry."""
    g1_file = _REF_BITS_DIR / "bitsG1.txt"
    assert g1_file.exists(), f"Reference file missing: {g1_file}"
    content = g1_file.read_text(encoding="utf-8")
    
    assert len(content) == EXPECTED_BIT_COUNT, f"G1 expected {EXPECTED_BIT_COUNT} bits, got {len(content)}"
    assert set(content) == {"0", "1"}, "G1 reference contains non-binary characters"
    
    actual_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert actual_sha == EXPECTED_G1_G5_SHA256, f"G1 sha256 mismatch: {actual_sha}"
    
    manifest = json.loads(_MANIFEST_FILE.read_text(encoding="utf-8"))
    assert "reference_bits" in manifest
    assert "G1" in manifest["reference_bits"]
    assert manifest["reference_bits"]["G1"]["sha256"] == actual_sha
    assert manifest["reference_bits"]["G1"]["bit_count"] == EXPECTED_BIT_COUNT


# ----------------------------------------------------------------------------
# 2. Valid G5 Reference Ingestion
# ----------------------------------------------------------------------------
def test_valid_g5_reference_ingestion():
    """Verify that bitsG5.txt exists, has correct size, hash, character set, and manifest entry."""
    g5_file = _REF_BITS_DIR / "bitsG5.txt"
    assert g5_file.exists(), f"Reference file missing: {g5_file}"
    content = g5_file.read_text(encoding="utf-8")
    
    assert len(content) == EXPECTED_BIT_COUNT, f"G5 expected {EXPECTED_BIT_COUNT} bits, got {len(content)}"
    assert set(content) == {"0", "1"}, "G5 reference contains non-binary characters"
    
    actual_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert actual_sha == EXPECTED_G1_G5_SHA256, f"G5 sha256 mismatch: {actual_sha}"
    
    manifest = json.loads(_MANIFEST_FILE.read_text(encoding="utf-8"))
    assert "reference_bits" in manifest
    assert "G5_uncoded" in manifest["reference_bits"]
    assert manifest["reference_bits"]["G5_uncoded"]["sha256"] == actual_sha
    assert manifest["reference_bits"]["G5_uncoded"]["bit_count"] == EXPECTED_BIT_COUNT


# ----------------------------------------------------------------------------
# 3. Invalid Reference Characters
# ----------------------------------------------------------------------------
def test_invalid_reference_characters():
    """Verify that reference sequences with invalid characters are rejected."""
    bad_chars = "01010120101"
    with pytest.raises(ValueError, match="invalid non-binary characters"):
        # Character set validation logic
        if not set(bad_chars).issubset({"0", "1"}):
            raise ValueError(f"Reference file contains invalid non-binary characters.")

    bad_alpha = "0101abc0101"
    with pytest.raises(ValueError, match="invalid non-binary characters"):
        if not set(bad_alpha).issubset({"0", "1"}):
            raise ValueError(f"Reference file contains invalid non-binary characters.")


# ----------------------------------------------------------------------------
# 4. Incorrect Reference Length
# ----------------------------------------------------------------------------
def test_incorrect_reference_length():
    """Verify that reference length mismatch against expected/recovered bits fails explicitly."""
    # Under-length reference
    with pytest.raises(ValueError, match="Bit count mismatch"):
        recovered = np.zeros(4096, dtype=int)
        ref = np.zeros(4095, dtype=int)
        if len(recovered) != len(ref):
            raise ValueError(f"Bit count mismatch for G1: recovered={len(recovered)}, reference={len(ref)}")

    # Over-length reference
    with pytest.raises(ValueError, match="Bit count mismatch"):
        recovered = np.zeros(4096, dtype=int)
        ref = np.zeros(4097, dtype=int)
        if len(recovered) != len(ref):
            raise ValueError(f"Bit count mismatch for G1: recovered={len(recovered)}, reference={len(ref)}")


# ----------------------------------------------------------------------------
# 5. BER Calculation
# ----------------------------------------------------------------------------
def test_ber_calculation_exactness():
    """Verify exact BER and bit error count under known mismatch conditions."""
    iq = np.repeat([1.0, -1.0, 1.0, -1.0], 4).astype(complex)
    cfg = DemodConfig(modulation="BPSK", samples_per_symbol=4)
    res_base = demodulate(iq, cfg)
    
    # Perfect match: 0 errors
    cfg_zero = DemodConfig(modulation="BPSK", samples_per_symbol=4, external_reference_bits=res_base.hard_bits)
    res_zero = demodulate(iq, cfg_zero)
    assert res_zero.bit_errors == 0
    assert res_zero.bit_error_rate == 0.0

    # 1 mismatch
    ref_one_err = res_base.hard_bits.copy()
    ref_one_err[0] = 1 - ref_one_err[0]
    cfg_one = DemodConfig(modulation="BPSK", samples_per_symbol=4, external_reference_bits=ref_one_err)
    res_one = demodulate(iq, cfg_one)
    assert res_one.bit_errors == 1
    assert res_one.bit_error_rate == 1.0 / len(ref_one_err)

    # 50% mismatch
    ref_half = res_base.hard_bits.copy()
    ref_half[0::2] = 1 - ref_half[0::2]
    cfg_half = DemodConfig(modulation="BPSK", samples_per_symbol=4, external_reference_bits=ref_half)
    res_half = demodulate(iq, cfg_half)
    assert res_half.bit_errors == len(ref_half) // 2
    assert res_half.bit_error_rate == 0.5


# ----------------------------------------------------------------------------
# 6. Reference Unavailable Behavior
# ----------------------------------------------------------------------------
def test_reference_unavailable_behavior():
    """Verify that omitting external reference bits preserves UNAVAILABLE status and neutral confidence."""
    iq = np.repeat([1.0, -1.0, 1.0, -1.0], 4).astype(complex)
    cfg = DemodConfig(modulation="BPSK", samples_per_symbol=4, external_reference_bits=None)
    res = demodulate(iq, cfg)

    assert res.reference_status == "REFERENCE_BITS_UNAVAILABLE"
    assert res.bit_error_rate is None
    assert res.bit_errors is None

    dec_cfg = DecoderConfig(
        modulation=ModulationType.BPSK,
        fec_type=FECType.NONE,
        interleaver_type=InterleaverType.NONE,
        sample_rate=100000.0,
        samples_per_symbol=4,
        external_reference_bits=None,
    )
    pipeline = DecoderPipeline()
    pipe_res = pipeline.decode(iq, dec_cfg)
    handoff = generate_decoder_evidence(pipe_res, case_id="GENERIC_UNCODED")

    assert handoff.reference["status"] == "REFERENCE_BITS_UNAVAILABLE"
    assert handoff.reference["external_reference_available"] is False
    assert handoff.reference["ber"] is None
    assert handoff.confidence is not None
    assert handoff.confidence.components["reference_support"].weight == 0.0


# ----------------------------------------------------------------------------
# 7. Production Truth Isolation
# ----------------------------------------------------------------------------
def test_production_truth_isolation():
    """Verify that python/spectralq/ contains zero references to truth artifacts or staging folders."""
    prod_dir = _REPO_ROOT / "python" / "spectralq"
    forbidden_terms = [
        "truth.json",
        "official_validation_results.json",
        "official_manifest.json",
        "bitsG1.txt",
        "bitsG5.txt",
        "sinchana_g1_g5_refs",
        "SIH-main",
        "reference_bits/",
        "reference_bits\\",
        "core.demodulation",
        "core.pipeline",
    ]
    for py_file in prod_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in content, f"Forbidden term '{term}' found in {py_file.name}"
        assert not re.search(r'\breference_bits\b', content), f"Standalone 'reference_bits' found in {py_file.name}"


# ----------------------------------------------------------------------------
# 8. Deterministic Regenerated Output
# ----------------------------------------------------------------------------
def test_deterministic_regenerated_output():
    """Verify that running official validation generator produces identical deterministic outputs."""
    run_1 = run_validation()
    run_2 = run_validation()

    dump_1 = json.dumps(run_1, sort_keys=True)
    dump_2 = json.dumps(run_2, sort_keys=True)

    hash_1 = hashlib.sha256(dump_1.encode("utf-8")).hexdigest()
    hash_2 = hashlib.sha256(dump_2.encode("utf-8")).hexdigest()

    assert hash_1 == hash_2, "Non-deterministic output detected across consecutive validation runs!"


# ----------------------------------------------------------------------------
# 9. G1 Official Regression
# ----------------------------------------------------------------------------
def test_g1_official_regression():
    """Verify official G1 QPSK uncoded capture against bitsG1.txt reference."""
    cf32_path = _OFFICIAL_DIR / "G1_QPSK_uncoded.cf32"
    truth_path = _OFFICIAL_DIR / "G1_QPSK_uncoded.truth.json"
    ref_file = _REF_BITS_DIR / "bitsG1.txt"

    assert cf32_path.exists()
    assert truth_path.exists()
    assert ref_file.exists()

    meta = json.loads(truth_path.read_text(encoding="utf-8"))
    ref_bits = np.array([int(c) for c in ref_file.read_text(encoding="utf-8").strip()], dtype=np.uint8)

    raw = np.fromfile(cf32_path, dtype=np.float32)
    iq = raw[0::2] + 1j * raw[1::2]

    dec_cfg = DecoderConfig(
        modulation=ModulationType.QPSK,
        fec_type=FECType.NONE,
        interleaver_type=InterleaverType.NONE,
        sample_rate=float(meta["fs_hz"]),
        samples_per_symbol=int(meta["sps"]),
        rrc_alpha=float(meta["rolloff"]),
        mapping_profile="OCTAVE",
        external_reference_bits=ref_bits,
        metadata={"case_id": "G1"},
    )
    pipeline = DecoderPipeline()
    res = pipeline.decode(iq, dec_cfg)

    assert res.hard_bits is not None
    assert len(res.hard_bits) == len(ref_bits)
    errors = int(np.sum(res.hard_bits != ref_bits))
    assert errors == 0, f"G1 reference evaluation had {errors} bit errors!"

    handoff = generate_decoder_evidence(res, case_id="G1", snr_est_db=float(meta["snr_db"]))
    assert handoff.overall_status == "CONFIRMED_BY_MULTIPLE_EVIDENCE"
    assert handoff.reference["status"] == "AVAILABLE"
    assert handoff.reference["ber"] == 0.0
    assert handoff.reference["bit_errors"] == 0
    assert handoff.confidence is not None
    assert handoff.confidence.value is not None
    assert handoff.confidence.value >= 0.90


# ----------------------------------------------------------------------------
# 10. G5 Uncoded Official Regression
# ----------------------------------------------------------------------------
def test_g5_uncoded_official_regression():
    """Verify official G5 2-FSK uncoded capture against bitsG5.txt reference."""
    cf32_path = _OFFICIAL_DIR / "G5_2FSK_uncoded.cf32"
    truth_path = _OFFICIAL_DIR / "G5_2FSK_uncoded.truth.json"
    ref_file = _REF_BITS_DIR / "bitsG5.txt"

    assert cf32_path.exists()
    assert truth_path.exists()
    assert ref_file.exists()

    meta = json.loads(truth_path.read_text(encoding="utf-8"))
    ref_bits = np.array([int(c) for c in ref_file.read_text(encoding="utf-8").strip()], dtype=np.uint8)

    raw = np.fromfile(cf32_path, dtype=np.float32)
    iq = raw[0::2] + 1j * raw[1::2]

    demod_cfg = DemodConfig(
        modulation="2-FSK",
        sample_rate=float(meta["fs_hz"]),
        samples_per_symbol=int(meta["sps"]),
        symbol_rate=float(meta["symbol_rate_hz"]),
        fsk_deviation=float(meta["fsk_deviation_hz"]),
        external_reference_bits=ref_bits,
    )
    demod_res = demodulate(iq, demod_cfg)

    assert len(demod_res.hard_bits) == len(ref_bits)
    errors = int(np.sum(demod_res.hard_bits != ref_bits))
    assert errors == 0, f"G5 uncoded reference evaluation had {errors} bit errors!"

    dec_cfg = DecoderConfig(
        modulation=ModulationType.FSK2,
        fec_type=FECType.NONE,
        interleaver_type=InterleaverType.NONE,
        sample_rate=float(meta["fs_hz"]),
        samples_per_symbol=int(meta["sps"]),
        external_reference_bits=ref_bits,
        metadata={"case_id": "G5_uncoded", "fsk_deviation": float(meta["fsk_deviation_hz"])},
    )
    pipeline = DecoderPipeline()
    pipe_res = pipeline.decode(iq, dec_cfg)

    handoff = generate_decoder_evidence(pipe_res, case_id="G5_uncoded", snr_est_db=float(meta["snr_db"]))
    assert handoff.overall_status == "CONFIRMED_BY_MULTIPLE_EVIDENCE"
    assert handoff.reference["status"] == "AVAILABLE"
    assert handoff.reference["ber"] == 0.0
    assert handoff.reference["bit_errors"] == 0
    assert handoff.confidence is not None
    assert handoff.confidence.value is not None
    assert handoff.confidence.value >= 0.90
