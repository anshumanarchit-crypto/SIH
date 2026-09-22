"""Unit tests for core/fec.py and core/deinterleave.py (CSE-3)."""

import numpy as np
import pytest

from core.contracts import ResultStatus
from core.deinterleave import (
    ConvolutionalDeinterleaver,
    ConvolutionalInterleaver,
    block_deinterleave,
    block_interleave,
)
from core.fec import (
    CRC,
    ConvolutionalCodec,
    HammingCodec,
    ReedSolomonCodec,
    decode_fec,
    evaluate_viterbi_pipeline,
)


def test_viterbi_hard_k7_clean():
    codec = ConvolutionalCodec(k=7, g1=0o171, g2=0o133)
    np.random.seed(42)
    tx_bits = np.random.randint(0, 2, 100, dtype=np.uint8)

    encoded = codec.encode(tx_bits, add_tail_bits=True)
    decoded = codec.decode_hard(encoded)

    np.testing.assert_array_equal(decoded, tx_bits)


def test_viterbi_hard_k3_clean():
    codec = ConvolutionalCodec(k=3, g1=0o7, g2=0o5)
    np.random.seed(42)
    tx_bits = np.random.randint(0, 2, 80, dtype=np.uint8)

    encoded = codec.encode(tx_bits, add_tail_bits=True)
    decoded = codec.decode_hard(encoded)

    np.testing.assert_array_equal(decoded, tx_bits)


def test_viterbi_hard_with_controlled_errors():
    codec = ConvolutionalCodec(k=7)
    np.random.seed(42)
    tx_bits = np.random.randint(0, 2, 80, dtype=np.uint8)

    encoded = codec.encode(tx_bits, add_tail_bits=True)

    # Inject 3 bit errors distributed across codeword
    corrupted = encoded.copy()
    corrupted[10] ^= 1
    corrupted[40] ^= 1
    corrupted[90] ^= 1

    decoded, diag = codec.decode_hard(corrupted, return_diagnostics=True)
    # Viterbi must correct these errors
    np.testing.assert_array_equal(decoded, tx_bits)
    assert diag["final_path_metric"] >= 3.0


def test_viterbi_soft_noisy():
    codec = ConvolutionalCodec(k=7)
    np.random.seed(42)
    tx_bits = np.random.randint(0, 2, 60, dtype=np.uint8)

    encoded = codec.encode(tx_bits, add_tail_bits=True)
    # Bipolar mapping: bit 0 -> +1.0, bit 1 -> -1.0
    bipolar = np.where(encoded == 0, 1.0, -1.0)
    # Add Gaussian noise
    noisy_llrs = bipolar + np.random.randn(len(encoded)) * 0.4

    decoded = codec.decode_soft(noisy_llrs)
    np.testing.assert_array_equal(decoded, tx_bits)


def test_evaluate_viterbi_pipeline():
    # Test controlled error recovery evaluation
    res = evaluate_viterbi_pipeline(num_bits=100, error_indices=[12, 50, 110], k=7, soft=False)
    assert res["input_bit_count"] == 100
    assert res["corrupted_bit_count"] == 3
    assert res["bit_errors"] == 0
    assert res["output_bit_error_rate"] == 0.0
    assert res["corrected_bit_count"] == 3
    assert res["success"] is True

    # Test soft-decision evaluation
    res_soft = evaluate_viterbi_pipeline(num_bits=80, k=7, soft=True, noise_std=0.3)
    assert res_soft["output_bit_error_rate"] == 0.0
    assert res_soft["success"] is True


def test_decode_fec_unified_interface():
    np.random.seed(42)
    codec = ConvolutionalCodec(k=7)
    tx_bits = np.random.randint(0, 2, 80, dtype=np.uint8)
    encoded = codec.encode(tx_bits, add_tail_bits=True)

    # 1. Viterbi Hard Decision
    fec_res = decode_fec(encoded, fec_scheme="viterbi_hard", k=7)
    assert fec_res.status == ResultStatus.CONFIRMED
    np.testing.assert_array_equal(fec_res.bits, tx_bits)

    # 2. Hamming(7,4)
    data_nibbles = np.random.randint(0, 2, 40, dtype=np.uint8)
    h_encoded = HammingCodec.encode(data_nibbles)
    # inject single error
    h_encoded[2] ^= 1
    h_res = decode_fec(h_encoded, fec_scheme="hamming")
    assert h_res.status == ResultStatus.CONFIRMED
    assert h_res.corrected_errors == 1
    np.testing.assert_array_equal(h_res.bits, data_nibbles)

    # 3. Reed-Solomon
    rs_codec = ReedSolomonCodec(n=30, k=20)
    msg = b"SPECTRALQ_FEC_TEST01"
    rs_cw = rs_codec.encode(msg)
    # inject byte errors
    cw_arr = bytearray(rs_cw)
    cw_arr[5] ^= 0x55
    cw_arr[12] ^= 0xAA
    cw_bits = np.unpackbits(np.frombuffer(bytes(cw_arr), dtype=np.uint8))

    rs_res = decode_fec(cw_bits, fec_scheme="reed_solomon", rs_n=30, rs_k=20)
    assert rs_res.status == ResultStatus.CONFIRMED
    dec_bytes = np.packbits(rs_res.bits).tobytes()
    assert dec_bytes == msg

    # 4. Passthrough / None
    raw_bits = np.array([1, 0, 1, 1], dtype=np.uint8)
    none_res = decode_fec(raw_bits, fec_scheme="none")
    assert none_res.status == ResultStatus.CONFIRMED
    np.testing.assert_array_equal(none_res.bits, raw_bits)


def test_decode_fec_unsupported_and_degenerate():
    # 1. Unsupported scheme (LDPC)
    bits = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
    ldpc_res = decode_fec(bits, fec_scheme="ldpc")
    assert ldpc_res.status == ResultStatus.UNSUPPORTED
    assert len(ldpc_res.warnings) > 0
    assert ldpc_res.warnings[0]["code"] == "UNSUPPORTED_FEC_SCHEME"
    assert len(ldpc_res.bits) == 0

    # 2. Unsupported scheme (Turbo)
    turbo_res = decode_fec(bits, fec_scheme="turbo")
    assert turbo_res.status == ResultStatus.UNSUPPORTED

    # 3. Empty input
    empty_res = decode_fec(np.array([], dtype=np.uint8), fec_scheme="viterbi_hard")
    assert empty_res.status == ResultStatus.FAILED
    assert empty_res.warnings[0]["code"] == "EMPTY_INPUT"


def test_hamming_7_4():
    np.random.seed(42)
    tx_bits = np.random.randint(0, 2, 40, dtype=np.uint8)  # 10 blocks of 4 bits

    encoded = HammingCodec.encode(tx_bits)
    assert len(encoded) == 70

    # Inject 1 bit error into each of the 10 blocks
    corrupted = encoded.copy()
    for b in range(10):
        err_idx = b * 7 + (b % 7)
        corrupted[err_idx] ^= 1

    decoded, num_corrected = HammingCodec.decode(corrupted)
    assert num_corrected == 10
    np.testing.assert_array_equal(decoded, tx_bits)


def test_reed_solomon():
    rs = ReedSolomonCodec(n=30, k=20)
    msg = b"SPECTRALQ_DATA_PKT01"
    assert len(msg) == 20

    cw = rs.encode(msg)
    assert len(cw) == 30

    # Inject 3 symbol (byte) errors (capacity is t = (30-20)/2 = 5 errors)
    cw_arr = bytearray(cw)
    cw_arr[2] ^= 0x55
    cw_arr[8] ^= 0xAA
    cw_arr[15] ^= 0xFF

    decoded, success = rs.decode(bytes(cw_arr))
    assert success is True
    assert decoded == msg


def test_crc_calculations_and_verification():
    payload = b"SPECTRALQ_TELEMETRY_FRAME_12345"

    # CRC-8
    c8 = CRC.crc8(payload)
    pkt8 = payload + bytes([c8])
    valid8, p8 = CRC.verify(pkt8, "crc8")
    assert valid8 is True
    assert p8 == payload

    # CRC-16
    import struct
    c16 = CRC.crc16(payload)
    pkt16 = payload + struct.pack(">H", c16)
    valid16, p16 = CRC.verify(pkt16, "crc16")
    assert valid16 is True
    assert p16 == payload

    # CRC-32
    c32 = CRC.crc32(payload)
    pkt32 = payload + struct.pack(">I", c32)
    valid32, p32 = CRC.verify(pkt32, "crc32")
    assert valid32 is True
    assert p32 == payload

    # Tampered payload fails CRC
    tampered = bytearray(pkt16)
    tampered[5] ^= 0x01
    valid_bad, _ = CRC.verify(bytes(tampered), "crc16")
    assert valid_bad is False


def test_block_interleaver_roundtrip():
    data = np.arange(64, dtype=np.uint8)
    interleaved = block_interleave(data, num_rows=8, num_cols=8)
    assert not np.array_equal(interleaved, data)

    deinterleaved = block_deinterleave(interleaved, num_rows=8, num_cols=8, original_length=64)
    np.testing.assert_array_equal(deinterleaved, data)


def test_convolutional_interleaver_roundtrip():
    c_intl = ConvolutionalInterleaver(num_branches=4, delay_step=2)
    c_deintl = ConvolutionalDeinterleaver(num_branches=4, delay_step=2)

    data = np.arange(1, 41, dtype=np.int32)
    intl_out = c_intl.process(data)
    deintl_out = c_deintl.process(intl_out)

    total_delay = (4 - 1) * 2 * 4
    np.testing.assert_array_equal(deintl_out[total_delay:], data[:-total_delay])
