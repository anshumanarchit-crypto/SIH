"""
tests/unit/test_fec.py

Unit tests for SpectralQ FEC codecs:
- Bit/byte conversion utilities
- Convolutional Codec & Viterbi Decoder (K=7, 171/133 octal)
- Reed-Solomon Codec: RS(255, 223)
- LDPC Codec: Gallager (96, 3, 963) with GF(2) systematic generator & MSA BP
- Concatenated Codec: RS outer + Convolutional inner
- Direct cross-check against CommPy primitives
"""

import pytest
import numpy as np
from spectralq.fec import (
    bits_to_bytes,
    bytes_to_bits,
    ConvolutionalCodec,
    ReedSolomonCodec,
    LDPCCodec,
    ConcatenatedCodec,
    conv_encode,
    viterbi_decode,
)
from commpy.channelcoding.convcode import Trellis, conv_encode as commpy_conv, viterbi_decode as commpy_viterbi


# ============================================================================
# 1. BIT / BYTE CONVERSION TESTS
# ============================================================================

def test_bit_byte_conversions():
    """Verify bit_to_bytes and bytes_to_bits round-trip."""
    rng = np.random.default_rng(42)
    for num_bytes in [1, 2, 10, 223, 255]:
        num_bits = num_bytes * 8
        orig_bits = rng.integers(0, 2, size=num_bits)
        packed = bits_to_bytes(orig_bits)
        assert len(packed) == num_bytes

        unpacked = bytes_to_bits(packed)
        assert len(unpacked) == num_bits
        assert np.array_equal(orig_bits, unpacked)


def test_bit_byte_alignment_validation():
    """Verify bits_to_bytes rejects unaligned bit lengths."""
    with pytest.raises(ValueError):
        bits_to_bytes(np.array([1, 0, 1]))  # len 3 != multiple of 8


# ============================================================================
# 2. CONVOLUTIONAL CODEC & VITERBI DECODER TESTS
# ============================================================================

def test_convolutional_round_trip_standard_frame():
    """Verify zero-noise recovery on standard 256-bit frame."""
    codec = ConvolutionalCodec()
    rng = np.random.default_rng(101)
    source_bits = rng.integers(0, 2, size=256)

    coded = codec.encode(source_bits)
    # Output length must be 2 * (256 + 6) = 524
    assert len(coded) == 2 * (256 + 6)

    recovered, metrics = codec.decode(coded)
    assert metrics["decoder_success"] is True
    assert metrics["tail_bits_trimmed"] == 6
    assert len(recovered) == 256
    assert np.array_equal(source_bits, recovered)


@pytest.mark.parametrize("length", [16, 32, 48, 64, 128, 256])
def test_convolutional_adaptive_traceback_lengths(length):
    """Verify adaptive traceback depth across short and standard frames."""
    codec = ConvolutionalCodec()
    rng = np.random.default_rng(length * 7)
    source_bits = rng.integers(0, 2, size=length)

    coded = codec.encode(source_bits)
    assert len(coded) == 2 * (length + 6)

    recovered, metrics = codec.decode(coded)
    assert metrics["decoder_success"] is True
    assert np.array_equal(source_bits, recovered)


def test_convolutional_cross_check_commpy():
    """Cross-check ConvolutionalCodec against direct CommPy calls."""
    memory = np.array([6])
    g_matrix = np.array([[0o171, 0o133]])
    trellis = Trellis(memory, g_matrix)

    source_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0])
    # 1. Compare encoder output
    codec = ConvolutionalCodec()
    codec_encoded = codec.encode(source_bits)
    commpy_encoded = commpy_conv(source_bits, trellis, termination="term")
    assert np.array_equal(codec_encoded, commpy_encoded)

    # 2. Compare Viterbi decode output
    codec_recovered, _ = codec.decode(codec_encoded)
    commpy_decoded = commpy_viterbi(commpy_encoded.astype(float), trellis, tb_depth=22, decoding_type="hard")
    assert np.array_equal(codec_recovered, commpy_decoded[:len(source_bits)])


# ============================================================================
# 3. REED-SOLOMON RS(255, 223) TESTS
# ============================================================================

def test_rs_clean_round_trip():
    """Verify RS(255, 223) clean encoding and decoding (0 symbol errors)."""
    codec = ReedSolomonCodec()
    rng = np.random.default_rng(223)
    source_bits = rng.integers(0, 2, size=1784)  # 223 bytes

    coded = codec.encode(source_bits)
    assert len(coded) == 2040  # 255 bytes

    recovered, metrics = codec.decode(coded)
    assert metrics["decoder_success"] is True
    assert metrics["errors_corrected"] == 0
    assert np.array_equal(source_bits, recovered)


@pytest.mark.parametrize("err_symbols", [1, 8, 16])
def test_rs_correctable_errors(err_symbols):
    """Verify RS(255, 223) successfully corrects 1, 8, and 16 symbol errors."""
    codec = ReedSolomonCodec()
    rng = np.random.default_rng(42 + err_symbols)
    source_bits = rng.integers(0, 2, size=1784)
    coded = codec.encode(source_bits)

    # Corrupt specified number of distinct bytes
    coded_bytes = bytearray(bits_to_bytes(coded))
    for i in range(err_symbols):
        coded_bytes[i] ^= 0xFF  # Flip all bits in byte

    corrupted_coded = bytes_to_bits(bytes(coded_bytes))
    recovered, metrics = codec.decode(corrupted_coded)

    assert metrics["decoder_success"] is True
    assert metrics["errors_corrected"] == err_symbols
    assert np.array_equal(source_bits, recovered)


def test_rs_uncorrectable_errors_re_encode_verification():
    """Verify RS(255, 223) rejects >16 symbol errors and marks decoder failure."""
    codec = ReedSolomonCodec()
    rng = np.random.default_rng(999)
    source_bits = rng.integers(0, 2, size=1784)
    coded = codec.encode(source_bits)

    # Corrupt 18 bytes (strictly beyond 16-error capability)
    coded_bytes = bytearray(bits_to_bytes(coded))
    for i in range(18):
        coded_bytes[i] ^= 0xAA

    corrupted_coded = bytes_to_bits(bytes(coded_bytes))
    recovered, metrics = codec.decode(corrupted_coded)

    # Must explicitly declare failure and not claim success
    assert metrics["decoder_success"] is False


# ============================================================================
# 4. LDPC GALLAGER (96, 3, 963) TESTS
# ============================================================================

def test_ldpc_systematic_encoding_syndrome():
    """Verify systematic LDPC encoder produces valid codewords H * c = 0 mod 2."""
    codec = LDPCCodec()
    rng = np.random.default_rng(50)
    source_bits = rng.integers(0, 2, size=50)

    coded = codec.encode(source_bits)
    assert len(coded) == 96

    # Verify parity checks
    syndrome = codec.H.dot(coded) % 2
    assert np.sum(syndrome) == 0, "Codeword must satisfy parity check H * c = 0 mod 2"


def test_ldpc_channel_free_round_trip():
    """Verify LDPC exact source bit recovery under channel-free conditions."""
    codec = LDPCCodec()
    rng = np.random.default_rng(88)
    source_bits = rng.integers(0, 2, size=50)

    coded = codec.encode(source_bits)
    recovered, metrics = codec.decode(coded, algo="MSA", n_iters=30)

    assert metrics["decoder_success"] is True
    assert metrics["syndrome_status"] is True
    assert np.array_equal(source_bits, recovered)


def test_ldpc_controlled_bit_flip_correction():
    """Verify LDPC corrects 1-bit flip via Min-Sum Belief Propagation."""
    codec = LDPCCodec()
    rng = np.random.default_rng(123)
    source_bits = rng.integers(0, 2, size=50)
    coded = codec.encode(source_bits)

    # Channel LLR with 1-bit flip
    llr_vec = np.where(coded == 0, 10.0, -10.0).astype(float)
    flip_pos = 10
    llr_vec[flip_pos] = -llr_vec[flip_pos]

    recovered, metrics = codec.decode(coded, soft_llrs=llr_vec, algo="MSA", n_iters=30)
    assert metrics["decoder_success"] is True
    assert np.array_equal(source_bits, recovered)


# ============================================================================
# 5. CONCATENATED CODEC TESTS
# ============================================================================

def test_concatenated_codec_round_trip():
    """Verify RS outer + Convolutional inner concatenated codec round-trip."""
    codec = ConcatenatedCodec()
    rng = np.random.default_rng(555)
    source_bits = rng.integers(0, 2, size=1784)

    coded = codec.encode(source_bits)
    # RS(255, 223) -> 2040 bits. Conv K=7 with tail -> 2 * (2040 + 6) = 4092 bits
    assert len(coded) == 4092

    recovered, metrics = codec.decode(coded)
    assert metrics["decoder_success"] is True
    assert len(recovered) == 1784
    assert np.array_equal(source_bits, recovered)
