# Bit-Stream Intelligence Output Specification

## 1. Purpose

The `bitintel` module (`spectralq.bitintel`) performs structural and statistical analysis on recovered post-FEC bitstreams. It acts as an evidence producer for Archit's downstream hypothesis/evidence aggregation engine.

> [!NOTE]
> `bitintel` extracts features, candidates, and statistical consistency metrics; it does **not** make the final classification decision.

---

## 2. Evidence Output Schema

The output is structured as a typed dictionary / JSON-serializable artifact embedded into `DecoderResult.crc_info` and `DecoderResult.diagnostics`:

```json
{
  "sync_words": [
    {
      "pattern_hex": "1ACFFC1D",
      "pattern_bits": [0, 0, 0, 1, 1, 0, 1, 0, ...],
      "bit_offset": 0,
      "correlation_score": 1.0,
      "periodicity": 2040,
      "occurrences": 4
    }
  ],
  "framing": {
    "detected_frame_length_bits": 2040,
    "confidence_candidate": 0.95,
    "total_frames_extracted": 4,
    "inferred_structure": "FIXED_LENGTH"
  },
  "crc_candidates": [
    {
      "polynomial_name": "CRC-16-CCITT",
      "polynomial_hex": "0x1021",
      "data_span_bits": [0, 2024],
      "crc_span_bits": [2024, 2040],
      "valid_count": 4,
      "total_tested": 4,
      "syndrome_match_ratio": 1.0
    }
  ],
  "headers": [
    {
      "frame_index": 0,
      "header_bits_hex": "A10408",
      "length_field_value": 255,
      "counter_field_value": 1
    }
  ],
  "cross_burst_metrics": {
    "burst_count": 3,
    "header_stability_score": 0.98,
    "counter_increment_consistent": true,
    "entropy_per_symbol": 7.92
  }
}
```

---

## 3. Analysis Modules

1. **Sync / Preamble Mining**:
   - Detects repetitive sync sequences (e.g. CCSDS `0x1ACFFC1D`, Barker codes, alternating preamble `0xAA` / `0x55`).
   - Computes normalized autocorrelation and cross-correlation peaks.
2. **Frame Candidate Carving**:
   - Tests hypothesis of fixed-length vs variable-length packetization based on periodic sync markers.
3. **CRC Candidate Analysis**:
   - Scans candidate frames against a standardized library of CRC polynomials (CRC-8, CRC-16-IBM, CRC-16-CCITT, CRC-32-IEEE).
   - Reports syndrome match ratios as high-confidence evidence for Archit's engine.
4. **Header / Payload Segmentation**:
   - Segments stable bit positions (headers) from high-entropy variable regions (payloads).
5. **Cross-Burst Consistency**:
   - Tracks sequence numbers, counter increments, and frame length consistency across multiple signal bursts.
