"""
tests/unit/test_bitstream_intelligence.py

Comprehensive unit tests for Phase 3 bitstream intelligence and structural analysis.
"""

from __future__ import annotations
import math
import numpy as np
import pytest

from spectralq.schemas import (
    BitstreamAnalysisResult,
    StructuralClassification,
)
from spectralq.bitstream import analyze_bitstream
from spectralq.bitintel import (
    mine_sync_preamble,
    detect_frame_candidates,
    scan_crc_candidates,
    map_header_payload,
    cross_burst_consistency,
)


def test_bitstream_empty():
    """Verify analysis of empty bitstream returns safe degenerate result."""
    empty = np.array([], dtype=np.uint8)
    res = analyze_bitstream(empty)
    assert res.bit_count == 0
    assert res.byte_count == 0
    assert res.is_byte_aligned is False
    assert res.remainder_bits == 0
    assert res.empirical_entropy == 0.0
    assert res.total_runs == 0
    assert res.structural_classification == StructuralClassification.PADDING_OR_CONSTANT
    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["bit_count"] == 0


def test_bitstream_single_bit_and_short():
    """Verify analysis of short bitstreams (< 32 bits) without crash."""
    short_bits = np.array([1, 0, 1, 1], dtype=np.uint8)
    res = analyze_bitstream(short_bits)
    assert res.bit_count == 4
    assert res.byte_count == 0
    assert res.is_byte_aligned is False
    assert res.remainder_bits == 4
    assert res.structural_classification == StructuralClassification.UNKNOWN_STRUCTURE
    assert any("threshold" in j.lower() for j in res.classification_justification)


def test_bitstream_byte_alignment():
    """Verify byte alignment and remainder bits calculation."""
    bits_16 = np.ones(16, dtype=np.uint8)
    res_16 = analyze_bitstream(bits_16)
    assert res_16.bit_count == 16
    assert res_16.byte_count == 2
    assert res_16.is_byte_aligned is True
    assert res_16.remainder_bits == 0

    bits_19 = np.ones(19, dtype=np.uint8)
    res_19 = analyze_bitstream(bits_19)
    assert res_19.bit_count == 19
    assert res_19.byte_count == 2
    assert res_19.is_byte_aligned is False
    assert res_19.remainder_bits == 3


def test_bitstream_entropy_and_density():
    """Verify bit density and Shannon entropy calculations."""
    # Balanced 50/50 alternating bits
    alt_bits = np.tile([0, 1], 100).astype(np.uint8)
    res = analyze_bitstream(alt_bits)
    assert res.bit_count == 200
    assert res.zero_count == 100
    assert res.one_count == 100
    assert res.one_density == 0.50
    assert pytest.approx(res.empirical_entropy, 0.001) == 1.0

    # Imbalanced 75/25 bits
    imb_bits = np.array([1, 1, 1, 0] * 50, dtype=np.uint8)
    res_imb = analyze_bitstream(imb_bits)
    assert res_imb.one_density == 0.75
    expected_h = -(0.25 * math.log2(0.25) + 0.75 * math.log2(0.75))
    assert pytest.approx(res_imb.empirical_entropy, 0.001) == expected_h


def test_bitstream_run_length_structure():
    """Verify run-length metrics: max runs, mean run length, total runs."""
    # Pattern: 10 ones, 5 zeros, 20 ones, 2 zeros
    bits = np.concatenate([
        np.ones(10, dtype=np.uint8),
        np.zeros(5, dtype=np.uint8),
        np.ones(20, dtype=np.uint8),
        np.zeros(2, dtype=np.uint8),
    ])
    res = analyze_bitstream(bits)
    assert res.max_one_run == 20
    assert res.max_zero_run == 5
    assert res.total_runs == 4
    assert res.mean_run_length == (10 + 5 + 20 + 2) / 4.0
    assert res.run_length_distribution[10] == 1
    assert res.run_length_distribution[20] == 1


def test_bitstream_text_like_classification():
    """Verify detection of printable text-like ASCII payloads."""
    text = "SpectralQ Phase 3 Bitstream Intelligence Engine Validation Test!"
    text_bytes = text.encode("ascii")
    raw_arr = np.frombuffer(text_bytes, dtype=np.uint8)
    bits = np.unpackbits(raw_arr)

    res = analyze_bitstream(bits)
    assert res.is_byte_aligned is True
    assert res.aligned_byte_count == len(text_bytes)
    assert res.printable_ascii_ratio >= 0.95
    assert res.ascii_control_ratio <= 0.05
    assert res.structural_classification == StructuralClassification.TEXT_LIKE


def test_bitstream_binary_structured_classification():
    """Verify detection of balanced, high-entropy binary payloads."""
    # Deterministic pseudo-random LFSR sequence
    state = 0xACE1
    bits_list = []
    for _ in range(512):
        bit = (state ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1
        state = (state >> 1) | (bit << 15)
        bits_list.append(bit)
    bits = np.array(bits_list, dtype=np.uint8)

    res = analyze_bitstream(bits)
    assert res.bit_count == 512
    assert 0.45 <= res.one_density <= 0.55
    assert res.empirical_entropy >= 0.95
    assert res.structural_classification == StructuralClassification.BINARY_STRUCTURED


def test_bitstream_highly_repetitive_classification():
    """Verify detection of long runs and periodic repetition."""
    # Long run of 80 ones followed by some bits
    bits = np.concatenate([np.ones(80, dtype=np.uint8), np.tile([0, 1], 20).astype(np.uint8)])
    res = analyze_bitstream(bits)
    assert res.max_one_run == 80
    assert res.structural_classification == StructuralClassification.HIGHLY_REPETITIVE


def test_bitstream_padding_or_constant():
    """Verify extreme imbalance triggers PADDING_OR_CONSTANT."""
    all_zeros = np.zeros(256, dtype=np.uint8)
    res_z = analyze_bitstream(all_zeros)
    assert res_z.structural_classification == StructuralClassification.PADDING_OR_CONSTANT

    all_ones = np.ones(256, dtype=np.uint8)
    res_o = analyze_bitstream(all_ones)
    assert res_o.structural_classification == StructuralClassification.PADDING_OR_CONSTANT


def test_bitstream_analysis_determinism():
    """Verify that bitstream analysis produces identical results across runs."""
    rng = np.random.RandomState(42)
    sample_bits = rng.randint(0, 2, size=500).astype(np.uint8)

    res1 = analyze_bitstream(sample_bits)
    res2 = analyze_bitstream(sample_bits)

    assert res1.to_dict() == res2.to_dict()


def test_legacy_bitintel_functions():
    """Verify standalone preamble, frame, CRC, and header utilities."""
    # Preamble mining with known CCSDS sync marker
    ccsds_marker = np.array([0,0,0,1,1,0,1,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,0,1], dtype=np.uint8)
    frame = np.concatenate([ccsds_marker, np.zeros(64, dtype=np.uint8), ccsds_marker, np.zeros(64, dtype=np.uint8)])
    preambles = mine_sync_preamble(frame)
    assert len(preambles) > 0
    assert preambles[0]["pattern_name"] == "CCSDS_32"
    assert preambles[0]["occurrences"] == 2

    # Frame candidate carving
    frames = detect_frame_candidates(frame, sync_pattern=ccsds_marker)
    assert len(frames) == 1
    assert frames[0]["detected_frame_length_bits"] == len(ccsds_marker) + 64

    # Header and payload mapping
    hdr_pay = map_header_payload(frame, header_length_bits=32)
    assert len(hdr_pay["header"]) == 32
    assert len(hdr_pay["payload"]) == len(frame) - 32

    # Cross-burst consistency
    bursts = [{"bit_count": 96}, {"bit_count": 96}, {"bit_count": 96}]
    cb = cross_burst_consistency(bursts)
    assert cb["burst_count"] == 3
    assert cb["length_stability"] == 1.0
