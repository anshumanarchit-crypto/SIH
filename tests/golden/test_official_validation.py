"""tests/golden/test_official_validation.py

Test suite validating Sinchana's official Octave captures and ensuring production isolation.
"""

from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
import pytest

from spectralq.demod import demodulate, DemodConfig
from spectralq.encode_chain import decode_chain, EncodeConfig
from spectralq.decoder_api import DecoderConfig, DecoderPipeline, ModulationType, FECType, InterleaverType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL_DIR = _REPO_ROOT / "data" / "official" / "sinchana" / "golden"
_MANIFEST_FILE = _REPO_ROOT / "data" / "official" / "sinchana" / "official_manifest.json"
_HANDOFF_DIR = _REPO_ROOT / "data" / "handoff"
_GOLDEN_DIR = _REPO_ROOT / "data" / "golden"


def test_official_manifest_integrity():
    """Verify that all official captures and truth metadata match official_manifest.json."""
    assert _MANIFEST_FILE.exists(), "Official manifest missing"
    manifest = json.loads(_MANIFEST_FILE.read_text(encoding="utf-8"))
    captures = manifest["captures"]
    assert len(captures) == 8, f"Expected 8 captures, got {len(captures)}"

    for base_name, entry in captures.items():
        cf32_path = _OFFICIAL_DIR / entry["cf32_file"]
        truth_path = _OFFICIAL_DIR / entry["truth_file"]

        assert cf32_path.exists(), f"Capture missing: {cf32_path}"
        assert truth_path.exists(), f"Truth metadata missing: {truth_path}"

        actual_cf32_hash = hashlib.sha256(cf32_path.read_bytes()).hexdigest()
        actual_truth_hash = hashlib.sha256(truth_path.read_bytes()).hexdigest()

        assert actual_cf32_hash == entry["cf32_sha256"], f"Hash mismatch for {entry['cf32_file']}"
        assert actual_truth_hash == entry["truth_sha256"], f"Hash mismatch for {entry['truth_file']}"


def test_official_cf32_byte_counts():
    """Verify that every .cf32 file has exactly num_samples * 8 bytes."""
    for cf32_path in _OFFICIAL_DIR.glob("*.cf32"):
        truth_path = cf32_path.with_suffix(".truth.json")
        meta = json.loads(truth_path.read_text(encoding="utf-8"))
        num_samples = meta["num_samples"]
        expected_bytes = num_samples * 8
        actual_bytes = cf32_path.stat().st_size
        assert actual_bytes == expected_bytes, f"{cf32_path.name}: expected {expected_bytes} bytes, got {actual_bytes}"


@pytest.mark.parametrize("case_id,truth_name", [
    ("G2", "G2_BPSK_conv_block.truth.json"),
    ("G3", "G3_8PSK_RS_diagonal.truth.json"),
    ("G4", "G4_16QAM_LDPC_pseudorandom.truth.json"),
    ("G5", "G5_2FSK_RS_Conv_interleaved.truth.json"),
    ("G6", "G6_BPSK_conv_interleaved.truth.json"),
    ("G7", "G7_QPSK_conv_near_threshold.truth.json"),
])
def test_official_tx_hash_match(case_id: str, truth_name: str):
    """Verify that Sinchana's truth bit_sha256 matches our local handoff bitstreams."""
    truth_path = _OFFICIAL_DIR / truth_name
    truth_meta = json.loads(truth_path.read_text(encoding="utf-8"))
    expected_hash = truth_meta["bit_sha256"]

    handoff_file = _HANDOFF_DIR / f"bits{case_id}.txt"
    assert handoff_file.exists(), f"Handoff file missing for {case_id}"
    text_content = handoff_file.read_text(encoding="utf-8").strip()
    our_hash = hashlib.sha256(text_content.encode("utf-8")).hexdigest()

    assert our_hash == expected_hash, f"TX bit hash mismatch for {case_id}: local={our_hash}, truth={expected_hash}"


def test_official_g5_coded_end_to_end():
    """Verify that official G5 2-FSK capture demodulates and decodes end-to-end with 0 BER."""
    cf32_path = _OFFICIAL_DIR / "G5_2FSK_RS_Conv_interleaved.cf32"
    truth_path = _OFFICIAL_DIR / "G5_2FSK_RS_Conv_interleaved.truth.json"
    meta = json.loads(truth_path.read_text(encoding="utf-8"))

    raw = np.fromfile(cf32_path, dtype=np.float32)
    iq = raw[0::2] + 1j * raw[1::2]

    demod_cfg = DemodConfig(
        modulation="2-FSK",
        samples_per_symbol=int(meta["sps"]),
        sample_rate=float(meta["fs_hz"]),
        symbol_rate=float(meta["symbol_rate_hz"]),
        fsk_deviation=float(meta["fsk_deviation_hz"]),
    )
    demod_res = demodulate(iq, demod_cfg)

    # 1. Demodulation matches TX bits with 0 errors
    tx_file = _HANDOFF_DIR / "bitsG5.txt"
    tx_bits = np.array([int(c) for c in tx_file.read_text(encoding="utf-8").strip()], dtype=np.uint8)
    assert len(demod_res.hard_bits) == len(tx_bits)
    demod_err = int(np.sum(demod_res.hard_bits != tx_bits))
    assert demod_err == 0, f"G5 official demodulation bit errors: {demod_err}"

    # 2. Decode chain recovers 1784 source bits with 0 errors
    enc_cfg = EncodeConfig(
        fec_type=FECType.CONCATENATED_RS_CONV,
        interleaver_type=InterleaverType.CONVOLUTIONAL,
        interleaver_params={"branches": 6},
    )
    dec_res = decode_chain(demod_res.hard_bits, enc_cfg)
    assert dec_res.decoder_success is True

    source_file = _GOLDEN_DIR / "G5" / "source_bits.txt"
    source_bits = np.array([int(c) for c in source_file.read_text(encoding="utf-8").strip()], dtype=np.uint8)
    assert len(dec_res.recovered_source_bits) == len(source_bits)
    src_err = int(np.sum(dec_res.recovered_source_bits != source_bits))
    assert src_err == 0, f"G5 official source bit errors: {src_err}"


def test_official_g2_coded_end_to_end():
    """Verify that official G2 BPSK capture resolves 180-deg phase ambiguity and decodes to 0 source bit errors."""
    cf32_path = _OFFICIAL_DIR / "G2_BPSK_conv_block.cf32"
    truth_path = _OFFICIAL_DIR / "G2_BPSK_conv_block.truth.json"
    meta = json.loads(truth_path.read_text(encoding="utf-8"))

    raw = np.fromfile(cf32_path, dtype=np.float32)
    iq = raw[0::2] + 1j * raw[1::2]

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

    assert res.decoder_success is True
    assert res.selected_rotation_deg == 180.0
    assert res.re_encode_errors == 0

    source_file = _GOLDEN_DIR / "G2" / "source_bits.txt"
    source_bits = np.array([int(c) for c in source_file.read_text(encoding="utf-8").strip()], dtype=np.uint8)
    assert res.recovered_source_bits is not None
    assert len(res.recovered_source_bits) >= len(source_bits)
    assert int(np.sum(res.recovered_source_bits[:len(source_bits)] != source_bits)) == 0


def test_official_g3_coded_end_to_end():
    """Verify that official G3 8-PSK capture with OCTAVE mapping profile decodes to 0 source bit errors."""
    cf32_path = _OFFICIAL_DIR / "G3_8PSK_RS_diagonal.cf32"
    truth_path = _OFFICIAL_DIR / "G3_8PSK_RS_diagonal.truth.json"
    meta = json.loads(truth_path.read_text(encoding="utf-8"))

    raw = np.fromfile(cf32_path, dtype=np.float32)
    iq = raw[0::2] + 1j * raw[1::2]

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

    source_file = _GOLDEN_DIR / "G3" / "source_bits.txt"
    source_bits = np.array([int(c) for c in source_file.read_text(encoding="utf-8").strip()], dtype=np.uint8)
    assert res.recovered_source_bits is not None
    assert len(res.recovered_source_bits) == len(source_bits)
    assert int(np.sum(res.recovered_source_bits != source_bits)) == 0


def test_official_g4_coded_end_to_end():
    """Verify that official G4 16-QAM capture with OCTAVE mapping profile and LDPC decodes to 0 source bit errors."""
    cf32_path = _OFFICIAL_DIR / "G4_16QAM_LDPC_pseudorandom.cf32"
    truth_path = _OFFICIAL_DIR / "G4_16QAM_LDPC_pseudorandom.truth.json"
    meta = json.loads(truth_path.read_text(encoding="utf-8"))

    raw = np.fromfile(cf32_path, dtype=np.float32)
    iq = raw[0::2] + 1j * raw[1::2]

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

    source_file = _GOLDEN_DIR / "G4" / "source_bits.txt"
    source_bits = np.array([int(c) for c in source_file.read_text(encoding="utf-8").strip()], dtype=np.uint8)
    assert res.recovered_source_bits is not None
    assert len(res.recovered_source_bits) == len(source_bits)
    assert int(np.sum(res.recovered_source_bits != source_bits)) == 0


def test_official_g6_coded_end_to_end():
    """Verify that official G6 BPSK capture resolves 180-deg phase ambiguity and decodes to 0 source bit errors."""
    cf32_path = _OFFICIAL_DIR / "G6_BPSK_conv_interleaved.cf32"
    truth_path = _OFFICIAL_DIR / "G6_BPSK_conv_interleaved.truth.json"
    meta = json.loads(truth_path.read_text(encoding="utf-8"))

    raw = np.fromfile(cf32_path, dtype=np.float32)
    iq = raw[0::2] + 1j * raw[1::2]

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

    source_file = _GOLDEN_DIR / "G6" / "source_bits.txt"
    source_bits = np.array([int(c) for c in source_file.read_text(encoding="utf-8").strip()], dtype=np.uint8)
    assert res.recovered_source_bits is not None
    assert len(res.recovered_source_bits) == len(source_bits)
    assert int(np.sum(res.recovered_source_bits != source_bits)) == 0


def test_official_g1_g5_uncoded_reference_status():
    """Verify that G1 and G5 uncoded report reference-evaluated status with zero BER."""
    validation_results_path = _OFFICIAL_DIR.parent / "official_validation_results.json"
    assert validation_results_path.exists()
    results = json.loads(validation_results_path.read_text(encoding="utf-8"))

    for case_id in ("G1", "G5_uncoded"):
        case = results["cases"][case_id]
        assert case["overall_status"] == "DEMODULATED_AND_EVALUATED_AGAINST_REFERENCE"
        assert case["first_failed_stage"] == "NONE_STAGE_PASSED"
        assert case["demod_result"]["demod_ber"] == 0.0
        assert case["demod_result"]["demod_bit_errors"] == 0
        assert case["bit_reference_validation"]["reference_status"] == "AVAILABLE"
        assert case["bit_reference_validation"]["external_reference_available"] is True
        assert case["bit_reference_validation"]["reference_bit_count"] == 4096
        p3 = case["phase3_evidence"]
        assert p3["overall_status"] == "CONFIRMED_BY_MULTIPLE_EVIDENCE"
        assert p3["reference"]["status"] == "AVAILABLE"
        assert p3["reference"]["ber"] == 0.0
        assert p3["reference"]["bit_errors"] == 0
        assert p3["confidence"]["value"] >= 0.90


def test_official_g7_diagnostic_preservation():
    """Verify that G7 near-threshold capture is diagnosed accurately without forcing success."""
    validation_results_path = _OFFICIAL_DIR.parent / "official_validation_results.json"
    assert validation_results_path.exists()
    results = json.loads(validation_results_path.read_text(encoding="utf-8"))

    g7 = results["cases"]["G7"]
    assert g7["overall_status"] == "FAILED_TO_DECODE_NEAR_THRESHOLD_STRESS_CASE"
    assert g7["decoder_result"]["source_exact_match"] is False
    assert g7["decoder_result"]["source_bit_errors"] > 0
    assert g7["first_failed_stage"] == "PHYSICAL_LAYER_NEAR_THRESHOLD_6DB_STRESS_LIMIT"


def test_production_truth_isolation():
    """Verify that production code in python/spectralq/ has no imports or references to truth artifacts."""
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
    import re
    for py_file in prod_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in content, f"Forbidden term '{term}' referenced in production file {py_file.name}"
        assert not re.search(r'\breference_bits\b', content), f"Standalone 'reference_bits' referenced in {py_file.name}"

