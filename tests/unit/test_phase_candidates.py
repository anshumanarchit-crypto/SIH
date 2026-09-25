"""tests/unit/test_phase_candidates.py

Unit tests for general candidate phase ambiguity generation and re-encode evaluation.
"""

from __future__ import annotations
import numpy as np
import pytest

from spectralq.demod import (
    demodulate,
    DemodConfig,
    resolve_phase_ambiguity,
    demap_bpsk,
    demap_qpsk,
)
from spectralq.decoder_api import (
    DecoderConfig,
    DecoderPipeline,
    ModulationType,
    FECType,
    InterleaverType,
)
from spectralq.encode_chain import encode_chain, EncodeConfig


def test_bpsk_candidate_rotations_generation():
    """Verify BPSK produces candidate rotations for 0 deg and 180 deg."""
    # BPSK symbols: +1, -1, +1, +1
    syms = np.array([1.0, -1.0, 1.0, 1.0], dtype=complex)
    cfg = DemodConfig(modulation="BPSK")
    res = demodulate(syms, cfg)

    assert len(res.candidate_rotations) == 2
    angles = [c["angle_deg"] for c in res.candidate_rotations]
    assert angles == [0.0, 180.0]

    cand_0 = res.candidate_rotations[0]
    cand_180 = res.candidate_rotations[1]

    # At 0 deg, cand_0 matches nominal hard_bits
    assert np.array_equal(cand_0["hard_bits"], res.hard_bits)
    # At 180 deg, antipodal inversion inverts all bits
    assert np.array_equal(cand_180["hard_bits"], 1 - cand_0["hard_bits"])


def test_qpsk_candidate_rotations_generation():
    """Verify QPSK produces candidate rotations for all four rotations (0, 90, 180, 270 deg)."""
    syms = np.array([1.0 + 1j, -1.0 + 1j, -1.0 - 1j, 1.0 - 1j], dtype=complex) / np.sqrt(2.0)
    cfg = DemodConfig(modulation="QPSK")
    res = demodulate(syms, cfg)

    assert len(res.candidate_rotations) == 4
    angles = [c["angle_deg"] for c in res.candidate_rotations]
    assert angles == [0.0, 90.0, 180.0, 270.0]


def test_arbitrary_phase_rotation_candidates():
    """Verify explicitly configured custom candidate rotations."""
    syms = np.array([1.0, -1.0], dtype=complex)
    custom_angles = [0.0, 45.0, 180.0]
    cfg = DemodConfig(modulation="BPSK", candidate_phase_rotations=custom_angles)
    res = demodulate(syms, cfg)

    assert len(res.candidate_rotations) == 3
    angles = [c["angle_deg"] for c in res.candidate_rotations]
    assert angles == custom_angles


def test_candidate_re_encode_consistency():
    """Verify that re-encode consistency cleanly distinguishes 0-deg from 180-deg BPSK."""
    # Create valid encoded BPSK payload with Convolutional K=7 code
    rng = np.random.default_rng(2026)
    source_bits = rng.integers(0, 2, size=64, dtype=np.uint8)

    enc_cfg = EncodeConfig(
        modulation="BPSK",
        fec_type=FECType.CONVOLUTIONAL_K7.value,
        interleaver_type=InterleaverType.BLOCK.value,
        interleaver_params={"rows": 8, "cols": 18},
    )
    enc_res = encode_chain(source_bits, enc_cfg)
    tx_bits = enc_res.tx_bits

    # Modulate to BPSK symbols with 180 degree rotation (antipodal inversion)
    # Bit 0 -> +1, Bit 1 -> -1; inverted: 0 -> -1, 1 -> +1
    inverted_symbols = np.where(tx_bits == 0, -1.0, 1.0).astype(complex)
    sps = 4
    inverted_waveform = np.repeat(inverted_symbols, sps)

    dec_cfg = DecoderConfig(
        modulation=ModulationType.BPSK,
        fec_type=FECType.CONVOLUTIONAL_K7,
        interleaver_type=InterleaverType.BLOCK,
        sample_rate=1.0,
        samples_per_symbol=sps,
        candidate_phase_evaluation=True,
        interleaver_params={"rows": 8, "cols": 18},
    )

    pipeline = DecoderPipeline()
    dec_res = pipeline.decode(inverted_waveform, dec_cfg)

    assert dec_res.decoder_success is True
    # The decoder should automatically select 180 deg rotation because it yields 0 re-encode errors
    assert dec_res.selected_rotation_deg == 180.0
    assert dec_res.re_encode_errors == 0
    assert dec_res.recovered_source_bits is not None
    assert np.array_equal(dec_res.recovered_source_bits[:len(source_bits)], source_bits)
