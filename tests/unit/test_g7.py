"""
tests/unit/test_g7.py

Comprehensive test suite for Golden Case G7 extension:
1. NONE interleaver identity mapping
2. G7 convolutional round trip (K=7, polynomials 0o171, 0o133, terminated)
3. G7 encode_chain round trip
4. G7 length validation (source=256, coded=524, tx=524)
5. G7 deterministic generation (seed=202607)
6. G7 metadata schema and consistency
7. G7 TX file binary-format validation
8. G7 SHA-256 digest correctness
9. G2-G6 regression verification (hashes and lengths unchanged)
10. G7 golden round-trip bit-exact recovery (BER = 0.0 digital baseband)
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

from spectralq.fec import ConvolutionalCodec
from spectralq.interleave import (
    identity_interleave,
    identity_deinterleave,
    none_interleave,
    none_deinterleave,
)
from spectralq.encode_chain import (
    EncodeConfig,
    encode_chain,
    decode_chain,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------
# 1. NONE Interleaver Identity
# ----------------------------------------------------------------------------
def test_none_interleaver_identity():
    """Verify NONE interleaver is strict identity mapping with zero padding."""
    rng = np.random.default_rng(202607)
    bits = rng.integers(0, 2, size=524)

    interleaved, meta = identity_interleave(bits)
    assert np.array_equal(interleaved, bits), "interleave(input) must equal input"
    assert meta["interleaver_type"] == "NONE"
    assert meta["original_length"] == 524
    assert meta["padded_length"] == 524
    assert meta["padding_length"] == 0

    recovered = identity_deinterleave(interleaved, meta)
    assert np.array_equal(recovered, bits), "deinterleave(interleave(input)) must equal input"

    # Test aliases
    int_alias, meta_alias = none_interleave(bits)
    assert np.array_equal(int_alias, bits)
    rec_alias = none_deinterleave(int_alias, meta_alias)
    assert np.array_equal(rec_alias, bits)


# ----------------------------------------------------------------------------
# 2. G7 Convolutional Round Trip
# ----------------------------------------------------------------------------
def test_g7_convolutional_round_trip():
    """Verify ConvolutionalCodec(K=7, (0o171, 0o133)) encodes and decodes 256 bits."""
    codec = ConvolutionalCodec(constraint_length=7, polynomials=(0o171, 0o133))
    rng = np.random.default_rng(202607)
    source_bits = rng.integers(0, 2, size=256)

    coded_bits = codec.encode(source_bits)
    assert len(coded_bits) == 524, "Expected (256 + 6) * 2 = 524 coded bits"

    recovered_bits, metrics = codec.decode(coded_bits)
    assert metrics["decoder_success"] is True
    assert metrics["tail_bits_trimmed"] == 6
    assert len(recovered_bits) == 256
    assert np.array_equal(source_bits, recovered_bits), "Viterbi decode must recover exact source bits"


# ----------------------------------------------------------------------------
# 3. G7 Encode Chain Round Trip
# ----------------------------------------------------------------------------
def test_g7_encode_chain_round_trip():
    """Verify encode_chain -> decode_chain round trip for G7 configuration."""
    config = EncodeConfig(
        case_id="G7",
        modulation="QPSK",
        fec_type="CONVOLUTIONAL",
        interleaver_type="NONE",
        fec_params={"constraint_length": 7, "polynomials": (0o171, 0o133)},
        interleaver_params={},
    )
    rng = np.random.default_rng(202607)
    source_bits = rng.integers(0, 2, size=256)

    enc_res = encode_chain(source_bits, config)
    assert enc_res.source_bit_length == 256
    assert enc_res.tx_bit_length == 524
    assert len(enc_res.tx_bits) == 524
    assert np.array_equal(enc_res.fec_bits, enc_res.tx_bits)

    dec_res = decode_chain(enc_res.tx_bits, config, enc_res.interleaver_metadata)
    assert dec_res.decoder_success is True
    assert len(dec_res.recovered_source_bits) == 256
    assert np.array_equal(source_bits, dec_res.recovered_source_bits)


# ----------------------------------------------------------------------------
# 4. G7 Length Validation
# ----------------------------------------------------------------------------
def test_g7_lengths():
    """Verify G7 source and TX bit lengths on disk and in pipeline."""
    source_file = _REPO_ROOT / "data" / "golden" / "G7" / "source_bits.txt"
    handoff_file = _REPO_ROOT / "data" / "handoff" / "bitsG7.txt"

    source_content = source_file.read_text(encoding="utf-8").strip()
    tx_content = handoff_file.read_text(encoding="utf-8").strip()

    assert len(source_content) == 256, f"G7 source bit length must be 256, got {len(source_content)}"
    assert len(tx_content) == 524, f"G7 TX bit length must be 524, got {len(tx_content)}"


# ----------------------------------------------------------------------------
# 5. G7 Deterministic Generation
# ----------------------------------------------------------------------------
def test_g7_deterministic_generation():
    """Verify that seed 202607 deterministically yields the exact same bit sequence."""
    rng1 = np.random.default_rng(202607)
    src1 = rng1.integers(0, 2, size=256)

    rng2 = np.random.default_rng(202607)
    src2 = rng2.integers(0, 2, size=256)

    assert np.array_equal(src1, src2), "RNG with seed 202607 must be strictly deterministic"

    source_file = _REPO_ROOT / "data" / "golden" / "G7" / "source_bits.txt"
    disk_src = np.array([int(c) for c in source_file.read_text(encoding="utf-8").strip()], dtype=int)
    assert np.array_equal(src1, disk_src), "Source file on disk must match deterministic generation"


# ----------------------------------------------------------------------------
# 6. G7 Metadata Consistency
# ----------------------------------------------------------------------------
def test_g7_metadata_consistency():
    """Verify G7 metadata.json contains all mandatory keys and correct values."""
    meta_path = _REPO_ROOT / "data" / "golden" / "G7" / "metadata.json"
    assert meta_path.exists(), "data/golden/G7/metadata.json must exist"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert meta["case_id"] == "G7"
    assert meta["modulation"] == "QPSK"
    assert meta["fec_type"] == "CONVOLUTIONAL"
    assert meta["constraint_length"] == 7
    assert meta["generators"] == ["0171", "0133"]
    assert meta["termination"] == "terminated"
    assert meta["tail_bits"] == 6
    assert meta["interleaver_type"] == "NONE"
    assert meta["source_bit_length"] == 256
    assert meta["tx_bit_length"] == 524
    assert meta["source_seed"] == 202607
    assert meta["channel_snr_db"] is None, "channel_snr_db must be null (not fabricated)"
    assert meta["channel_snr_status"] == "TO_BE_SET_BY_SINCHANA_FOR_NEAR_THRESHOLD_WAVEFORM"
    assert "QPSK + convolutional near-threshold case" in meta["notes"]
    assert "implementation_choices" in meta
    assert meta["implementation_choices"]["source_bits"] == 256
    assert meta["implementation_choices"]["interleaver"] == "NONE"


# ----------------------------------------------------------------------------
# 7. G7 TX File Binary Format
# ----------------------------------------------------------------------------
def test_g7_tx_file_binary_format():
    """Verify bitsG7.txt contains strictly '0' and '1' characters and a trailing newline."""
    handoff_file = _REPO_ROOT / "data" / "handoff" / "bitsG7.txt"
    assert handoff_file.exists(), "data/handoff/bitsG7.txt must exist"

    raw_bytes = handoff_file.read_bytes()
    # Exactly 524 bits + 1 newline = 525 bytes (or 526 if CRLF on Windows)
    text = raw_bytes.decode("utf-8")
    assert text.endswith("\n"), "File must end with a newline"

    stripped = text.strip()
    assert len(stripped) == 524, f"Expected 524 characters, got {len(stripped)}"
    assert all(c in ("0", "1") for c in stripped), "File must contain only '0' and '1' characters"
    assert " " not in stripped, "No whitespace permitted"
    assert "#" not in stripped, "No comments permitted"


# ----------------------------------------------------------------------------
# 8. G7 SHA-256 Correctness
# ----------------------------------------------------------------------------
def test_g7_sha256_correctness():
    """Verify SHA-256 digests in metadata match actual file contents."""
    meta_path = _REPO_ROOT / "data" / "golden" / "G7" / "metadata.json"
    source_file = _REPO_ROOT / "data" / "golden" / "G7" / "source_bits.txt"
    handoff_file = _REPO_ROOT / "data" / "handoff" / "bitsG7.txt"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    source_str = source_file.read_text(encoding="utf-8").strip()
    tx_str = handoff_file.read_text(encoding="utf-8").strip()

    expected_src_hash = hashlib.sha256(source_str.encode("utf-8")).hexdigest()
    expected_tx_hash = hashlib.sha256(tx_str.encode("utf-8")).hexdigest()

    assert meta["source_bits_sha256"] == expected_src_hash
    assert meta["tx_bits_sha256"] == expected_tx_hash


# ----------------------------------------------------------------------------
# 9. G2-G6 Baseline Regression
# ----------------------------------------------------------------------------
def test_g2_g6_baseline_regression():
    """Verify that all G2-G6 golden source bits, TX bits, and metadata hashes are unchanged."""
    expected_hashes = {
        "G2": {
            "source_bits": "239b930d40753d2ccbbaae580b9b0c5ff8c4f3ac41d3bc570508e11b171cadf9",
            "tx_bits": "ca3b4f2c71b8f915a1cec76b761700ffe0027f1619a1c1849b274cf3756a9c3d",
            "metadata": "276d7b1129510e8a6f5570a83c1f16fbd7ba8ca8eaa89371a30a0b33d0db34d4",
            "source_len": 256,
            "tx_len": 544,
        },
        "G3": {
            "source_bits": "097345e71df389a08c4d800586ec7a8a1410b962bc345cf229efdb0561432030",
            "tx_bits": "5aca8e44e28d245a0951d46bcb05df93f8a5d13415cf49f3fac522a6a42f9282",
            "metadata": "b4d015aee81227574cacf1f573b70fd142e4d2160984df0da98ba728018e8368",
            "source_len": 1784,
            "tx_len": 2040,
        },
        "G4": {
            "source_bits": "7f2df79dc2f96bf806788f835e568f8b40d0bd5158da36e8aa6fc3e55ea9dcc5",
            "tx_bits": "10ed3d495d82ed7f1179105aec6b6439a787df58040edbdd72d3957a19481ee4",
            "metadata": "3a3593a7d7941867012dc373c036d4acd5a63706b052f9db4adffb53bf98b37a",
            "source_len": 50,
            "tx_len": 96,
        },
        "G5": {
            "source_bits": "202d78ff3393aa3ddf81d56aac4be830bd3edd9369195dd63c4ad72b71cd1d7c",
            "tx_bits": "2859b39c5c91c643782375e8e78600240eca93eb68c105de7901686c5b12333b",
            "metadata": "531432c7221a5809bf66f0cf0b4331affa0014bce703b36e18fa72cc3d3f33fc",
            "source_len": 1784,
            "tx_len": 4152,
        },
        "G6": {
            "source_bits": "b7502b740377e15ad0f946a1ce4450b0c854b50a1b592657afd9e374d0bc0ad0",
            "tx_bits": "3fbe252e8014a83361b2cbc1419a15d6b8761f57df4cfcca0b5dc811ea0c7f52",
            "metadata": "2acca4d4a22b664425af3d4212d40363129f761ab4840a77533df28e824e9874",
            "source_len": 256,
            "tx_len": 548,
        },
    }

    for case_id, exp in expected_hashes.items():
        s_file = _REPO_ROOT / "data" / "golden" / case_id / "source_bits.txt"
        t_file = _REPO_ROOT / "data" / "handoff" / f"bits{case_id}.txt"
        m_file = _REPO_ROOT / "data" / "golden" / case_id / "metadata.json"

        assert hashlib.sha256(s_file.read_bytes()).hexdigest() == exp["source_bits"]
        assert hashlib.sha256(t_file.read_bytes()).hexdigest() == exp["tx_bits"]
        assert hashlib.sha256(m_file.read_bytes()).hexdigest() == exp["metadata"]

        s_str = s_file.read_text(encoding="utf-8").strip()
        t_str = t_file.read_text(encoding="utf-8").strip()
        assert len(s_str) == exp["source_len"]
        assert len(t_str) == exp["tx_len"]


# ----------------------------------------------------------------------------
# 10. G7 Golden Round-Trip
# ----------------------------------------------------------------------------
def test_g7_golden_roundtrip_zero_error():
    """Verify G7 handoff bits decode to exact source bits with zero error under digital baseband."""
    source_file = _REPO_ROOT / "data" / "golden" / "G7" / "source_bits.txt"
    handoff_file = _REPO_ROOT / "data" / "handoff" / "bitsG7.txt"
    meta_file = _REPO_ROOT / "data" / "golden" / "G7" / "metadata.json"

    source_bits = np.array([int(c) for c in source_file.read_text(encoding="utf-8").strip()], dtype=int)
    tx_bits = np.array([int(c) for c in handoff_file.read_text(encoding="utf-8").strip()], dtype=int)
    meta = json.loads(meta_file.read_text(encoding="utf-8"))

    config = EncodeConfig(
        case_id="G7",
        modulation=meta["modulation"],
        fec_type=meta["fec_type"],
        interleaver_type=meta["interleaver_type"],
        fec_params=meta["fec_parameters"],
        interleaver_params=meta["interleaver_parameters"],
    )

    dec_res = decode_chain(
        received_bits=tx_bits,
        config=config,
        interleaver_meta=meta.get("interleaver_metadata"),
    )

    assert dec_res.decoder_success is True
    recovered = dec_res.recovered_source_bits
    assert len(recovered) == len(source_bits)
    assert np.array_equal(source_bits, recovered)
    bit_errors = int(np.sum(source_bits != recovered))
    assert bit_errors == 0
