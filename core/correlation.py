"""
Bitstream Correlation, Frame Synchronization, and Packet Parsing Module for SpectralQ.

Provides:
- Robust bipolar cross-correlation for preamble / sync word detection
- Detection and correction of phase inversion (180 deg carrier slip)
- Barker codes (7, 11, 13), CCSDS (0x1ACFFC1D), AX.25 HDLC (0x7E), and custom sync words
- Packet header parsing, payload extraction, and CRC integrity validation
"""

from __future__ import annotations
import logging
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import correlate

from core.fec import CRC

logger = logging.getLogger("spectralq.correlation")

# Standard Preambles & Sync Words
KNOWN_SYNC_WORDS = {
    "CCSDS_32": np.unpackbits(np.frombuffer(bytes.fromhex("1ACFFC1D"), dtype=np.uint8)),
    "SPECTRALQ_16": np.unpackbits(np.frombuffer(bytes.fromhex("ABCD"), dtype=np.uint8)),
    "AX25_HDLC_16": np.unpackbits(np.frombuffer(bytes.fromhex("7E7E"), dtype=np.uint8)),
    "BARKER_13": np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=np.uint8),
    "BARKER_11": np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=np.uint8),
    "BARKER_7": np.array([1, 1, 1, 0, 0, 1, 0], dtype=np.uint8),
}


from core.contracts import (
    SyncDetection,
    CorrelationResult,
    DecodedFrame,
    PacketHeader,
    ResultStatus,
    make_warning,
    BIT_ORDERING,
    BIT_ARRAY_DTYPE,
)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Convert 1D uint8 bit array to bytes."""
    bits = np.asarray(bits, dtype=np.uint8)
    # Pad to multiple of 8
    pad_len = (8 - (len(bits) % 8)) % 8
    if pad_len > 0:
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert bytes object to 1D uint8 array of 0s and 1s."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr)


def format_hex_dump(data: bytes, bytes_per_line: int = 16) -> str:
    """Produce standard hexdump with offset, hex bytes, and ASCII display."""
    lines = []
    for offset in range(0, len(data), bytes_per_line):
        chunk = data[offset : offset + bytes_per_line]
        hex_bytes = " ".join(f"{b:02X}" for b in chunk)
        ascii_repr = "".join(chr(b) if 32 <= b <= 126 else "." for b in chunk)
        lines.append(f"{offset:04X}  {hex_bytes:<{bytes_per_line * 3}}  |{ascii_repr}|")
    return "\n".join(lines)


def detect_sync_word(
    bits: np.ndarray,
    sync_pattern: Optional[Union[str, np.ndarray, bytes]] = None,
    threshold: float = 0.80,
) -> SyncDetection:
    """
    Search for sync word in bitstream using normalized bipolar cross-correlation.

    Args:
        bits: 1D array of received bits (0s and 1s).
        sync_pattern: Name of known sync word ('BARKER_11', 'CCSDS_32', etc.),
                      or 1D bit array / bytes. If None, tries all KNOWN_SYNC_WORDS.
        threshold: Normalized correlation threshold (0.0 to 1.0).

    Returns:
        SyncDetection object.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n_bits = len(bits)
    if n_bits == 0:
        return SyncDetection(False, "NONE", -1, 0.0, False, 0)

    # Convert bitstream to bipolar {-1, +1}
    bipolar_rx = 2.0 * bits.astype(np.float32) - 1.0

    patterns_to_try = {}
    if sync_pattern is None:
        patterns_to_try = KNOWN_SYNC_WORDS
    elif isinstance(sync_pattern, str) and sync_pattern in KNOWN_SYNC_WORDS:
        patterns_to_try = {sync_pattern: KNOWN_SYNC_WORDS[sync_pattern]}
    elif isinstance(sync_pattern, bytes):
        patterns_to_try = {"CUSTOM_BYTES": bytes_to_bits(sync_pattern)}
    elif isinstance(sync_pattern, np.ndarray):
        patterns_to_try = {"CUSTOM_BITS": np.asarray(sync_pattern, dtype=np.uint8)}
    elif isinstance(sync_pattern, str):
        # Hex string or bit string
        try:
            b = bytes.fromhex(sync_pattern)
            patterns_to_try = {"CUSTOM_HEX": bytes_to_bits(b)}
        except ValueError:
            bit_arr = np.array([int(c) for c in sync_pattern if c in "01"], dtype=np.uint8)
            patterns_to_try = {"CUSTOM_BITS": bit_arr}

    best_det = SyncDetection(False, "NONE", -1, 0.0, False, 0)
    best_weighted_score = 0.0

    # Sort patterns by length descending so longer, more reliable sync words are evaluated first
    sorted_patterns = sorted(patterns_to_try.items(), key=lambda item: len(item[1]), reverse=True)

    for name, pattern_bits in sorted_patterns:
        pat_len = len(pattern_bits)
        if pat_len > n_bits:
            continue

        bipolar_pat = 2.0 * pattern_bits.astype(np.float32) - 1.0

        # Valid mode correlation: output length = n_bits - pat_len + 1
        corr = correlate(bipolar_rx, bipolar_pat, mode="valid") / float(pat_len)

        max_idx = int(np.argmax(corr))
        max_val = float(corr[max_idx])

        min_idx = int(np.argmin(corr))
        min_val = float(corr[min_idx])

        # Longer sync words have higher entropy; weight score by log2(length)
        weight = np.log2(max(pat_len, 4))

        # Check positive correlation
        if max_val >= threshold:
            weighted = max_val * weight
            if weighted > best_weighted_score:
                best_weighted_score = weighted
                best_det = SyncDetection(
                    found=True,
                    sync_name=name,
                    bit_index=max_idx,
                    correlation_score=max_val,
                    is_inverted=False,
                    sync_word_len=pat_len,
                    correlation_curve=corr,
                )

        # Check negative correlation (phase inversion)
        if abs(min_val) >= threshold:
            weighted = abs(min_val) * weight
            if weighted > best_weighted_score:
                best_weighted_score = weighted
                best_det = SyncDetection(
                    found=True,
                    sync_name=name,
                    bit_index=min_idx,
                    correlation_score=abs(min_val),
                    is_inverted=True,
                    sync_word_len=pat_len,
                    correlation_curve=corr,
                )

    return best_det


def parse_packet_frame(
    bits: np.ndarray,
    sync_detection: SyncDetection,
    crc_type: str = "crc16",
) -> Optional[DecodedFrame]:
    """
    Extract and parse structured packet frame starting immediately after the sync word.

    Frame format:
      [SYNC_WORD] (already detected)
      [VERSION: 1B][PKT_TYPE: 1B][SEQ_NUM: 2B BE][PAYLOAD_LEN: 2B BE]
      [PAYLOAD: N bytes]
      [CRC: 2B (CRC-16) or 4B (CRC-32)]
    """
    if not sync_detection.found:
        return None

    # Align bitstream to payload start after sync word
    start_bit = sync_detection.bit_index + sync_detection.sync_word_len
    aligned_bits = bits[start_bit:]

    if sync_detection.is_inverted:
        aligned_bits = 1 - aligned_bits  # invert bits

    raw_bytes = bits_to_bytes(aligned_bits)

    # Need at least 6 bytes of header (Version: 1, Type: 1, Seq: 2, Len: 2)
    crc_len = 2 if crc_type.lower() == "crc16" else (4 if crc_type.lower() == "crc32" else 1)
    min_frame_len = 6 + crc_len

    if len(raw_bytes) < min_frame_len:
        # Fallback: Treat all remaining bytes as raw payload
        text_repr = raw_bytes.decode("utf-8", errors="replace")
        hex_dump = format_hex_dump(raw_bytes)
        return DecodedFrame(
            sync_name=sync_detection.sync_name,
            sync_index=sync_detection.bit_index,
            is_inverted=sync_detection.is_inverted,
            version=0,
            packet_type=0,
            seq_num=0,
            payload_length=len(raw_bytes),
            raw_payload_bytes=raw_bytes,
            payload_text=text_repr,
            hex_dump=hex_dump,
            crc_valid=False,
            crc_type=crc_type,
            crc_expected=0,
            crc_actual=0,
        )

    # Parse header
    version, pkt_type, seq_num, payload_len = struct.unpack(">BBHH", raw_bytes[:6])

    # Check if frame fits in buffer
    total_expected = 6 + payload_len + crc_len
    if len(raw_bytes) >= total_expected and payload_len > 0:
        payload_bytes = raw_bytes[6 : 6 + payload_len]
        crc_bytes = raw_bytes[6 + payload_len : total_expected]

        # Verify CRC
        header_and_payload = raw_bytes[: 6 + payload_len]
        crc_is_valid, _ = CRC.verify(raw_bytes[:total_expected], crc_type=crc_type)

        if crc_type.lower() == "crc16":
            crc_expected = struct.unpack(">H", crc_bytes)[0]
            crc_actual = CRC.crc16(header_and_payload)
        elif crc_type.lower() == "crc32":
            crc_expected = struct.unpack(">I", crc_bytes)[0]
            crc_actual = CRC.crc32(header_and_payload)
        else:
            crc_expected = crc_bytes[0]
            crc_actual = CRC.crc8(header_and_payload)

        text_repr = payload_bytes.decode("utf-8", errors="replace")
        hex_dump = format_hex_dump(payload_bytes)

        return DecodedFrame(
            sync_name=sync_detection.sync_name,
            sync_index=sync_detection.bit_index,
            is_inverted=sync_detection.is_inverted,
            version=version,
            packet_type=pkt_type,
            seq_num=seq_num,
            payload_length=payload_len,
            raw_payload_bytes=payload_bytes,
            payload_text=text_repr,
            hex_dump=hex_dump,
            crc_valid=crc_is_valid,
            crc_type=crc_type,
            crc_expected=crc_expected,
            crc_actual=crc_actual,
            metadata={"total_frame_bytes": total_expected},
        )
    else:
        # If header length doesn't fit, parse as raw buffer with CRC check attempt
        payload_bytes = raw_bytes
        text_repr = payload_bytes.decode("utf-8", errors="replace")
        hex_dump = format_hex_dump(payload_bytes)
        return DecodedFrame(
            sync_name=sync_detection.sync_name,
            sync_index=sync_detection.bit_index,
            is_inverted=sync_detection.is_inverted,
            version=version,
            packet_type=pkt_type,
            seq_num=seq_num,
            payload_length=len(payload_bytes),
            raw_payload_bytes=payload_bytes,
            payload_text=text_repr,
            hex_dump=hex_dump,
            crc_valid=False,
            crc_type=crc_type,
            crc_expected=0,
            crc_actual=0,
        )


def build_synthetic_frame(
    payload_text: str,
    sync_name: str = "BARKER_13",
    version: int = 1,
    packet_type: int = 42,
    seq_num: int = 101,
    crc_type: str = "crc16",
) -> bytes:
    """
    Construct a complete synthetic packet:
      [SYNC_WORD] + [HEADER] + [PAYLOAD] + [CRC]
    """
    sync_bits = KNOWN_SYNC_WORDS[sync_name]
    sync_bytes = bits_to_bytes(sync_bits)

    payload_bytes = payload_text.encode("utf-8")
    payload_len = len(payload_bytes)

    header = struct.pack(">BBHH", version, packet_type, seq_num, payload_len)
    data_to_crc = header + payload_bytes

    if crc_type.lower() == "crc16":
        crc_val = CRC.crc16(data_to_crc)
        crc_bytes = struct.pack(">H", crc_val)
    elif crc_type.lower() == "crc32":
        crc_val = CRC.crc32(data_to_crc)
        crc_bytes = struct.pack(">I", crc_val)
    else:
        crc_val = CRC.crc8(data_to_crc)
        crc_bytes = bytes([crc_val])

    return sync_bytes + data_to_crc + crc_bytes
