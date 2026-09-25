"""
spectralq.bitstream

Deterministic bitstream intelligence and structural analysis subsystem.

Analyzes recovered bitstreams without assuming their semantic meaning.
Produces factual, auditable metrics:
- Bit length, byte alignment, remainder bits
- Bit balance (zero/one density), Shannon entropy
- Run-length distribution (max runs, mean run length)
- Periodicity and autocorrelation indicators
- Byte structure (unique bytes, printable ASCII ratio, control byte ratio, byte entropy)
- Deterministic structural classification (TEXT_LIKE, BYTE_STRUCTURED, BINARY_STRUCTURED, etc.)

TRUTH ISOLATION:
Zero reliance on external golden source files or hardcoded capture identifiers.
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from spectralq.schemas import (
    BitstreamAnalysisResult,
    StructuralClassification,
)


def analyze_bitstream(bits: Union[np.ndarray, List[int], bytes]) -> BitstreamAnalysisResult:
    """Perform deterministic statistical and structural analysis of a bitstream.

    Parameters
    ----------
    bits : np.ndarray, list, or bytes
        Bitstream to analyze (1D array of uint8 binary values 0 or 1).

    Returns
    -------
    result : BitstreamAnalysisResult
        Structured, machine-readable metrics and deterministic classification.
    """
    # 1. Normalize input
    if isinstance(bits, bytes):
        raw_arr = np.frombuffer(bits, dtype=np.uint8)
        arr = np.unpackbits(raw_arr)
    else:
        arr = np.asarray(bits, dtype=np.uint8).ravel()

    bit_count = int(len(arr))

    # A. Length metrics
    byte_count = bit_count // 8
    remainder_bits = bit_count % 8
    is_byte_aligned = (remainder_bits == 0 and bit_count > 0)

    # Empty handling
    if bit_count == 0:
        return BitstreamAnalysisResult(
            bit_count=0,
            byte_count=0,
            is_byte_aligned=False,
            remainder_bits=0,
            zero_count=0,
            one_count=0,
            one_density=0.0,
            empirical_entropy=0.0,
            max_zero_run=0,
            max_one_run=0,
            mean_run_length=0.0,
            total_runs=0,
            run_length_distribution={},
            detected_period=None,
            max_autocorrelation_peak=0.0,
            has_periodic_framing=False,
            autocorrelation_peaks=[],
            repeated_blocks=[],
            aligned_byte_count=0,
            unique_byte_count=0,
            printable_ascii_ratio=0.0,
            ascii_control_ratio=0.0,
            zero_byte_ratio=0.0,
            high_bit_set_ratio=0.0,
            byte_entropy=0.0,
            top_byte_frequencies=[],
            structural_classification=StructuralClassification.PADDING_OR_CONSTANT,
            classification_justification=["Bitstream is empty (0 bits)."],
        )

    # Ensure bits are binary (0 or 1)
    binary_arr = (arr != 0).astype(np.uint8)

    # B. Basic bit statistics
    one_count = int(np.sum(binary_arr))
    zero_count = bit_count - one_count
    one_density = one_count / bit_count

    # Empirical bit Shannon entropy: -p0*log2(p0) - p1*log2(p1)
    p1 = one_density
    p0 = 1.0 - p1
    if p0 <= 0.0 or p1 <= 0.0:
        bit_entropy = 0.0
    else:
        bit_entropy = float(-p0 * math.log2(p0) - p1 * math.log2(p1))

    # C. Run-length structure
    # Calculate transitions
    diffs = np.diff(binary_arr)
    change_indices = np.where(diffs != 0)[0] + 1
    split_indices = np.concatenate(([0], change_indices, [bit_count]))
    run_lengths = np.diff(split_indices)
    run_values = binary_arr[split_indices[:-1]]

    total_runs = int(len(run_lengths))
    mean_run_length = float(np.mean(run_lengths)) if total_runs > 0 else 0.0

    zero_runs = run_lengths[run_values == 0]
    one_runs = run_lengths[run_values == 1]

    max_zero_run = int(np.max(zero_runs)) if len(zero_runs) > 0 else 0
    max_one_run = int(np.max(one_runs)) if len(one_runs) > 0 else 0

    # Run length distribution (bucket counts for lengths up to 32, then >=33)
    run_dist: Dict[int, int] = {}
    for rlen in run_lengths:
        k = int(rlen)
        run_dist[k] = run_dist.get(k, 0) + 1

    # D. Periodicity / Repetition Analysis
    autocorr_peaks: List[Dict[str, Any]] = []
    max_corr_peak = 0.0
    detected_period: Optional[int] = None
    has_periodic_framing = False

    # Compute binary autocorrelation for lags 1 .. min(256, bit_count // 2)
    max_lag = min(256, bit_count // 2)
    if max_lag >= 8:
        # Bipolar representation: 0 -> -1, 1 -> +1
        bipolar = (2 * binary_arr.astype(np.float64)) - 1.0
        # Sample variances
        denom = float(np.sum(bipolar ** 2))
        if denom > 0.0:
            for lag in range(8, max_lag + 1):
                # Normalized linear correlation at lag
                corr = float(np.sum(bipolar[:-lag] * bipolar[lag:]) / (bit_count - lag))
                if corr > 0.70:
                    autocorr_peaks.append({
                        "period_bits": lag,
                        "normalized_correlation": round(corr, 4),
                    })
                    if corr > max_corr_peak:
                        max_corr_peak = corr
                        detected_period = lag

            if max_corr_peak >= 0.80 and detected_period is not None:
                has_periodic_framing = True

    # Repeated block search (e.g. 16-bit or 32-bit blocks repeating)
    repeated_blocks: List[Dict[str, Any]] = []
    if bit_count >= 64:
        for block_size in (16, 32):
            if bit_count >= block_size * 3:
                num_blocks = bit_count // block_size
                block_counts: Dict[str, int] = {}
                for bi in range(num_blocks):
                    chunk = binary_arr[bi * block_size : (bi + 1) * block_size]
                    hex_str = "".join(f"{int(chunk[i]):01b}" for i in range(block_size))
                    # Hex representation
                    int_val = int(hex_str, 2)
                    hex_repr = f"0x{int_val:0{block_size//4}X}"
                    block_counts[hex_repr] = block_counts.get(hex_repr, 0) + 1

                for hex_pat, cnt in sorted(block_counts.items(), key=lambda x: -x[1]):
                    if cnt >= 3:
                        repeated_blocks.append({
                            "pattern_hex": hex_pat,
                            "length_bits": block_size,
                            "occurrences": cnt,
                        })

    # E. Byte Structure (over aligned byte span)
    aligned_byte_count = byte_count
    unique_byte_count = 0
    printable_ascii_ratio = 0.0
    ascii_control_ratio = 0.0
    zero_byte_ratio = 0.0
    high_bit_set_ratio = 0.0
    byte_entropy = 0.0
    top_byte_freqs: List[Dict[str, Any]] = []

    if aligned_byte_count > 0:
        aligned_bits = binary_arr[: aligned_byte_count * 8]
        # Pack into bytes
        packed_bytes = np.packbits(aligned_bits)

        # Count frequencies
        byte_counts = np.bincount(packed_bytes, minlength=256)
        unique_byte_count = int(np.count_nonzero(byte_counts))

        # Printable ASCII: 0x20..0x7E plus \t (0x09), \n (0x0A), \r (0x0D)
        is_printable = np.zeros(256, dtype=bool)
        is_printable[0x20:0x7F] = True
        is_printable[0x09] = True
        is_printable[0x0A] = True
        is_printable[0x0D] = True

        printable_count = int(np.sum(byte_counts[is_printable]))
        printable_ascii_ratio = printable_count / aligned_byte_count

        # Control ASCII: 0x00..0x1F and 0x7F, excluding \t, \n, \r
        is_control = np.zeros(256, dtype=bool)
        is_control[0x00:0x20] = True
        is_control[0x7F] = True
        is_control[0x09] = False
        is_control[0x0A] = False
        is_control[0x0D] = False

        control_count = int(np.sum(byte_counts[is_control]))
        ascii_control_ratio = control_count / aligned_byte_count

        # Zero bytes (0x00)
        zero_byte_ratio = int(byte_counts[0]) / aligned_byte_count

        # High bit set (0x80 .. 0xFF)
        high_bit_set_ratio = int(np.sum(byte_counts[128:256])) / aligned_byte_count

        # Byte Shannon Entropy: sum(-p * log2(p)) for p > 0
        p_bytes = byte_counts[byte_counts > 0] / aligned_byte_count
        byte_entropy = float(-np.sum(p_bytes * np.log2(p_bytes)))

        # Top 5 byte frequencies
        top_indices = np.argsort(-byte_counts)[:5]
        for idx in top_indices:
            cnt = int(byte_counts[idx])
            if cnt > 0:
                top_byte_freqs.append({
                    "byte_hex": f"0x{int(idx):02X}",
                    "byte_int": int(idx),
                    "count": cnt,
                    "frequency": round(cnt / aligned_byte_count, 4),
                })

    # F. Common Structural Signatures
    classification, justifications = _classify_structure(
        bit_count=bit_count,
        is_byte_aligned=is_byte_aligned,
        one_density=one_density,
        bit_entropy=bit_entropy,
        max_zero_run=max_zero_run,
        max_one_run=max_one_run,
        max_corr_peak=max_corr_peak,
        printable_ascii_ratio=printable_ascii_ratio,
        ascii_control_ratio=ascii_control_ratio,
        byte_entropy=byte_entropy,
        unique_byte_count=unique_byte_count,
        aligned_byte_count=aligned_byte_count,
        zero_byte_ratio=zero_byte_ratio,
    )

    return BitstreamAnalysisResult(
        bit_count=bit_count,
        byte_count=byte_count,
        is_byte_aligned=is_byte_aligned,
        remainder_bits=remainder_bits,
        zero_count=zero_count,
        one_count=one_count,
        one_density=round(one_density, 4),
        empirical_entropy=round(bit_entropy, 4),
        max_zero_run=max_zero_run,
        max_one_run=max_one_run,
        mean_run_length=round(mean_run_length, 2),
        total_runs=total_runs,
        run_length_distribution=run_dist,
        detected_period=detected_period,
        max_autocorrelation_peak=round(max_corr_peak, 4),
        has_periodic_framing=has_periodic_framing,
        autocorrelation_peaks=autocorr_peaks,
        repeated_blocks=repeated_blocks,
        aligned_byte_count=aligned_byte_count,
        unique_byte_count=unique_byte_count,
        printable_ascii_ratio=round(printable_ascii_ratio, 4),
        ascii_control_ratio=round(ascii_control_ratio, 4),
        zero_byte_ratio=round(zero_byte_ratio, 4),
        high_bit_set_ratio=round(high_bit_set_ratio, 4),
        byte_entropy=round(byte_entropy, 4),
        top_byte_frequencies=top_byte_freqs,
        structural_classification=classification,
        classification_justification=justifications,
    )


def _classify_structure(
    bit_count: int,
    is_byte_aligned: bool,
    one_density: float,
    bit_entropy: float,
    max_zero_run: int,
    max_one_run: int,
    max_corr_peak: float,
    printable_ascii_ratio: float,
    ascii_control_ratio: float,
    byte_entropy: float,
    unique_byte_count: int,
    aligned_byte_count: int,
    zero_byte_ratio: float = 0.0,
) -> Tuple[StructuralClassification, List[str]]:
    """Determine deterministic structural classification without semantic guessing."""
    justifications: List[str] = []

    # Constant / All padding check
    if one_density <= 0.01 or one_density >= 0.99:
        justifications.append(f"Extreme bit imbalance: one_density={one_density:.4f}")
        return StructuralClassification.PADDING_OR_CONSTANT, justifications

    # Short burst check
    if bit_count < 32:
        justifications.append(f"Short bitstream ({bit_count} bits < 32 bits threshold).")
        return StructuralClassification.UNKNOWN_STRUCTURE, justifications

    # Highly repetitive / periodic check
    if max_zero_run >= 64 or max_one_run >= 64 or max_corr_peak >= 0.85:
        if max_zero_run >= 64:
            justifications.append(f"Long zero run detected: {max_zero_run} consecutive bits.")
        if max_one_run >= 64:
            justifications.append(f"Long one run detected: {max_one_run} consecutive bits.")
        if max_corr_peak >= 0.85:
            justifications.append(f"Strong periodic autocorrelation peak: {max_corr_peak:.4f}.")
        return StructuralClassification.HIGHLY_REPETITIVE, justifications

    # Text-like check
    if (
        is_byte_aligned
        and bit_count >= 64
        and printable_ascii_ratio >= 0.85
        and ascii_control_ratio <= 0.10
    ):
        justifications.append(
            f"High printable ASCII ratio ({printable_ascii_ratio:.2%}) with low control characters ({ascii_control_ratio:.2%}) over byte-aligned payload."
        )
        return StructuralClassification.TEXT_LIKE, justifications

    # Binary structured check (high bit entropy and balanced density, characteristic of coded / LFSR / scrambled bits)
    if bit_entropy >= 0.95 and 0.40 <= one_density <= 0.60 and printable_ascii_ratio < 0.60:
        justifications.append(
            f"High bit entropy ({bit_entropy:.4f}) and balanced bit density ({one_density:.4f}), characteristic of coded or pseudo-random binary payload."
        )
        return StructuralClassification.BINARY_STRUCTURED, justifications

    # Byte-structured check (byte-aligned with non-uniform byte distribution or framing formatting)
    max_theoretical_byte_entropy = math.log2(min(256, max(2, aligned_byte_count)))
    if (
        is_byte_aligned
        and aligned_byte_count >= 8
        and (
            zero_byte_ratio >= 0.05
            or printable_ascii_ratio >= 0.30
            or byte_entropy < 0.85 * max_theoretical_byte_entropy
        )
    ):
        justifications.append(
            f"Byte-aligned with distinct byte-level formatting (zero_bytes={zero_byte_ratio:.2%}, ascii={printable_ascii_ratio:.2%}, byte_entropy={byte_entropy:.2f}/{max_theoretical_byte_entropy:.2f})."
        )
        return StructuralClassification.BYTE_STRUCTURED, justifications

    justifications.append("Structure indeterminate or does not match defined structural signatures.")
    return StructuralClassification.UNKNOWN_STRUCTURE, justifications
