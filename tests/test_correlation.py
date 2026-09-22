"""Unit tests for core/correlation.py."""

import numpy as np
import pytest

from core.correlation import (
    KNOWN_SYNC_WORDS,
    bits_to_bytes,
    bytes_to_bits,
    detect_sync_word,
    format_hex_dump,
    parse_packet_frame,
)
from core.fec import CRC
import struct


def test_bits_to_bytes_roundtrip():
    data = b"Hello, SpectralQ!"
    bits = bytes_to_bits(data)
    recovered = bits_to_bytes(bits)
    assert recovered == data


def test_detect_sync_barker_13():
    np.random.seed(42)
    prefix_bits = np.random.randint(0, 2, 100, dtype=np.uint8)
    sync_bits = KNOWN_SYNC_WORDS["BARKER_13"]
    suffix_bits = np.random.randint(0, 2, 150, dtype=np.uint8)

    stream = np.concatenate([prefix_bits, sync_bits, suffix_bits])

    det = detect_sync_word(stream, sync_pattern="BARKER_13")
    assert det.found is True
    assert det.sync_name == "BARKER_13"
    assert det.bit_index == 100
    assert det.is_inverted is False
    assert det.correlation_score == pytest.approx(1.0)


def test_detect_sync_inverted():
    np.random.seed(42)
    prefix_bits = np.random.randint(0, 2, 50, dtype=np.uint8)
    sync_bits = 1 - KNOWN_SYNC_WORDS["CCSDS_32"]  # Inverted sync word
    suffix_bits = np.random.randint(0, 2, 80, dtype=np.uint8)

    stream = np.concatenate([prefix_bits, sync_bits, suffix_bits])

    det = detect_sync_word(stream, sync_pattern="CCSDS_32")
    assert det.found is True
    assert det.bit_index == 50
    assert det.is_inverted is True
    assert det.correlation_score >= 0.99


def test_frame_assembly_and_parsing():
    payload = b"SPECTRALQ_TELEMETRY_PKT_999"
    version = 1
    pkt_type = 10
    seq_num = 4096
    payload_len = len(payload)

    header = struct.pack(">BBHH", version, pkt_type, seq_num, payload_len)
    header_and_payload = header + payload
    crc_val = CRC.crc16(header_and_payload)
    crc_bytes = struct.pack(">H", crc_val)

    sync_bits = KNOWN_SYNC_WORDS["BARKER_11"]
    frame_bits = bytes_to_bits(header_and_payload + crc_bytes)

    # Prepend 20 noise bits
    prefix_noise = np.random.randint(0, 2, 20, dtype=np.uint8)
    stream = np.concatenate([prefix_noise, sync_bits, frame_bits])

    det = detect_sync_word(stream, sync_pattern="BARKER_11")
    assert det.found is True
    assert det.bit_index == 20

    frame = parse_packet_frame(stream, det, crc_type="crc16")
    assert frame is not None
    assert frame.version == version
    assert frame.packet_type == pkt_type
    assert frame.seq_num == seq_num
    assert frame.payload_length == payload_len
    assert frame.raw_payload_bytes == payload
    assert frame.payload_text == "SPECTRALQ_TELEMETRY_PKT_999"
    assert frame.crc_valid is True
    assert frame.crc_actual == crc_val


def test_hex_dump_formatting():
    data = b"0123456789ABCDEF"
    dump = format_hex_dump(data, bytes_per_line=16)
    assert "0000" in dump
    assert "30 31 32 33" in dump
    assert "|0123456789ABCDEF|" in dump
