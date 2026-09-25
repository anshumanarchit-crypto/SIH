"""
tests/integration/test_encode_chain.py

End-to-end integration tests for the SpectralQ encode and decode chain orchestrators.
Validates exact bit-for-bit round-trip recovery across all Golden configurations:
- G2: BPSK + Conv K=7 + Block Interleaver (256 source bits)
- G3: 8-PSK + RS(255, 223) + Diagonal Interleaver (1,784 source bits)
- G4: 16-QAM + LDPC (Gallager 96.3.963) + Pseudo-Random Interleaver (50 source bits)
- G5: 2-FSK + Concatenated RS+Conv + Convolutional Interleaver (1,784 source bits)
- G6: BPSK + Conv K=7 + Convolutional Interleaver (256 source bits)
- G7: QPSK + Conv K=7 + No Interleaver (NONE) (256 source bits)
"""

import hashlib
import numpy as np
import pytest

from spectralq.encode_chain import (
    EncodeConfig,
    encode_chain,
    decode_chain,
)


def compute_sha256(bits: np.ndarray) -> str:
    """Compute SHA-256 digest of bitstream string."""
    bit_str = "".join(str(b) for b in bits)
    return hashlib.sha256(bit_str.encode("utf-8")).hexdigest()


def test_round_trip_g2():
    """Verify G2: Conv K=7 + Block Interleaver on frozen length 256 bits."""
    config = EncodeConfig(
        case_id="G2",
        modulation="BPSK",
        fec_type="CONV_K7",
        interleaver_type="BLOCK",
        interleaver_params={"rows": 16, "cols": 34},  # capacity 544 >= 524 coded bits
    )
    rng = np.random.default_rng(2002)
    source_bits = rng.integers(0, 2, size=256)

    enc_res = encode_chain(source_bits, config)
    assert enc_res.source_bit_length == 256
    # 256 info -> 524 coded bits. Block capacity 16*34 = 544.
    assert enc_res.tx_bit_length == 544

    dec_res = decode_chain(enc_res.tx_bits, config, enc_res.interleaver_metadata)
    assert dec_res.decoder_success is True
    assert np.array_equal(source_bits, dec_res.recovered_source_bits)


def test_round_trip_g3():
    """Verify G3: RS(255, 223) + Diagonal Interleaver on frozen length 1,784 bits."""
    config = EncodeConfig(
        case_id="G3",
        modulation="8-PSK",
        fec_type="RS_255_223",
        interleaver_type="DIAGONAL",
        interleaver_params={"num_rows": 40, "num_cols": 51},  # capacity 2040 == 2040 coded bits
    )
    rng = np.random.default_rng(3003)
    source_bits = rng.integers(0, 2, size=1784)

    enc_res = encode_chain(source_bits, config)
    assert enc_res.source_bit_length == 1784
    assert enc_res.tx_bit_length == 2040

    dec_res = decode_chain(enc_res.tx_bits, config, enc_res.interleaver_metadata)
    assert dec_res.decoder_success is True
    assert np.array_equal(source_bits, dec_res.recovered_source_bits)


def test_round_trip_g4():
    """Verify G4: LDPC (Gallager 96.3.963) + PRNG Interleaver on frozen length 50 bits."""
    config = EncodeConfig(
        case_id="G4",
        modulation="16-QAM",
        fec_type="LDPC",
        interleaver_type="PSEUDORANDOM",
        interleaver_params={"seed": 42},
    )
    rng = np.random.default_rng(4004)
    source_bits = rng.integers(0, 2, size=50)

    enc_res = encode_chain(source_bits, config)
    assert enc_res.source_bit_length == 50
    assert enc_res.tx_bit_length == 96

    dec_res = decode_chain(enc_res.tx_bits, config, enc_res.interleaver_metadata)
    assert dec_res.decoder_success is True
    assert np.array_equal(source_bits, dec_res.recovered_source_bits)


def test_round_trip_g5():
    """Verify G5: Concatenated RS+Conv + Convolutional Interleaver on frozen length 1,784 bits."""
    config = EncodeConfig(
        case_id="G5",
        modulation="2-FSK",
        fec_type="CONCAT_RS_CONV",
        interleaver_type="CONVOLUTIONAL",
        interleaver_params={"num_branches": 6, "delay_step": 2},
    )
    rng = np.random.default_rng(5005)
    source_bits = rng.integers(0, 2, size=1784)

    enc_res = encode_chain(source_bits, config)
    assert enc_res.source_bit_length == 1784
    # RS(255, 223) -> 2040 bits -> Conv K=7 with tail -> 4092 bits
    # Convolutional interleaver with B=6, M=2 flushes (6-1)*2*6 = 60 bits
    # Total TX bits = 4092 + 60 = 4152 bits
    assert enc_res.tx_bit_length == 4092 + 60

    dec_res = decode_chain(enc_res.tx_bits, config, enc_res.interleaver_metadata)
    assert dec_res.decoder_success is True
    assert np.array_equal(source_bits, dec_res.recovered_source_bits)


def test_round_trip_g6():
    """Verify G6: Conv K=7 + Convolutional Interleaver on frozen length 256 bits."""
    config = EncodeConfig(
        case_id="G6",
        modulation="BPSK",
        fec_type="CONV_K7",
        interleaver_type="CONVOLUTIONAL",
        interleaver_params={"num_branches": 4, "delay_step": 2},
    )
    rng = np.random.default_rng(6006)
    source_bits = rng.integers(0, 2, size=256)

    enc_res = encode_chain(source_bits, config)
    assert enc_res.source_bit_length == 256
    # 256 info -> 524 coded bits -> Forney (B=4, M=2) flushes (4-1)*2*4 = 24 bits -> 548 bits
    assert enc_res.tx_bit_length == 524 + 24

    dec_res = decode_chain(enc_res.tx_bits, config, enc_res.interleaver_metadata)
    assert dec_res.decoder_success is True
    assert np.array_equal(source_bits, dec_res.recovered_source_bits)


def test_double_generation_determinism():
    """Verify that double execution produces identical bits and SHA-256 digests."""
    config = EncodeConfig(
        case_id="G2",
        modulation="BPSK",
        fec_type="CONV_K7",
        interleaver_type="BLOCK",
        interleaver_params={"rows": 16, "cols": 34},
    )
    rng1 = np.random.default_rng(777)
    source1 = rng1.integers(0, 2, size=256)
    enc1 = encode_chain(source1, config)
    hash_src1 = compute_sha256(source1)
    hash_tx1 = compute_sha256(enc1.tx_bits)

    rng2 = np.random.default_rng(777)
    source2 = rng2.integers(0, 2, size=256)
    enc2 = encode_chain(source2, config)
    hash_src2 = compute_sha256(source2)
    hash_tx2 = compute_sha256(enc2.tx_bits)

    assert hash_src1 == hash_src2
    assert hash_tx1 == hash_tx2
    assert np.array_equal(enc1.tx_bits, enc2.tx_bits)


def test_round_trip_g7():
    """Verify G7: QPSK + Conv K=7 + No Interleaver (NONE) on 256 bits."""
    config = EncodeConfig(
        case_id="G7",
        modulation="QPSK",
        fec_type="CONVOLUTIONAL",
        interleaver_type="NONE",
        fec_params={"constraint_length": 7, "polynomials": (0o171, 0o133)},
        interleaver_params={},
    )
    rng = np.random.default_rng(7007)
    source_bits = rng.integers(0, 2, size=256)

    enc_res = encode_chain(source_bits, config)
    assert enc_res.source_bit_length == 256
    # 256 info bits -> (256 + 6) * 2 = 524 coded bits -> identity interleaver -> 524 TX bits
    assert enc_res.tx_bit_length == 524
    assert len(enc_res.tx_bits) == 524
    assert np.array_equal(enc_res.fec_bits, enc_res.tx_bits)

    dec_res = decode_chain(enc_res.tx_bits, config, enc_res.interleaver_metadata)
    assert dec_res.decoder_success is True
    assert len(dec_res.recovered_source_bits) == 256
    assert np.array_equal(source_bits, dec_res.recovered_source_bits)
