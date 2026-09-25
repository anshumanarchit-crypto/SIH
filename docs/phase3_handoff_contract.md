# Phase 3: Machine-Readable Decoder Evidence Handoff Contract

## 1. Ownership and Integration Boundary

```
[ Sinchana ]
Octave DSP Frontend
  .cf32 + truth metadata
         ↓
[ Arpit - THIS WORKSPACE ]
Phase 1: Codecs & Interleavers
Phase 2: Demodulation & Phase/Mapping Ambiguity
Phase 3: Bitstream Intelligence, Evidence Ledger & Handoff
  Artifact: data/handoff/decoder_evidence.json
         ↓
[ Archit ]
Downstream Cross-Layer Evidence Aggregator
Project-level Multi-Source Confidence
Final UNKNOWN Decision
Final result.json
```

- **Arpit's Responsibility**: Delivers `decoder_evidence.json` (per transmission burst or batched bundle).
- **Archit's Responsibility**: Consumes `decoder_evidence.json` alongside RF front-end metrics and ML classifier benchmarks to produce the final `result.json`.
- **Integrity Rule**: Phase 3 never overwrites `result.json` or claims the project-level UNKNOWN decision.

---

## 2. Handoff JSON Schema Specification

Schema version: `spectralq-decoder-evidence-v1`

```json
{
  "schema_version": "spectralq-decoder-evidence-v1",
  "case_id": "G2",
  "overall_status": "CONFIRMED_BY_MULTIPLE_EVIDENCE",
  "decoder": {
    "status": "SUCCESS",
    "success": true,
    "modulation": "BPSK",
    "fec": "CONV_K7_171_133",
    "interleaver": "BLOCK",
    "bit_count": 266,
    "selected_rotation_deg": 180.0,
    "re_encode_errors": 0
  },
  "candidate_hypotheses": [
    {
      "hypothesis_id": "HYP_ROT_180",
      "category": "PHASE_ROTATION",
      "parameters": {
        "angle_deg": 180.0,
        "mapping_profile": "DEFAULT",
        "fec_type": "CONV_K7_171_133"
      },
      "status": "CONFIRMED_BY_MULTIPLE_EVIDENCE",
      "evidence_refs": ["EVID_FEC_SUCCESS", "EVID_REENCODE_ERRORS"],
      "score_components": {
        "fec_success": 1.0,
        "re_encode_errors": 0.0,
        "re_encode_consistency": 1.0
      },
      "contradictions": [],
      "re_encode_errors": 0,
      "consistency_fraction": 1.0,
      "rank": 1
    }
  ],
  "selected_hypothesis": {
    "hypothesis_id": "HYP_ROT_180",
    "category": "PHASE_ROTATION",
    "parameters": {
      "angle_deg": 180.0,
      "mapping_profile": "DEFAULT",
      "fec_type": "CONV_K7_171_133"
    },
    "status": "CONFIRMED_BY_MULTIPLE_EVIDENCE",
    "evidence_refs": ["EVID_FEC_SUCCESS", "EVID_REENCODE_ERRORS"],
    "score_components": {
      "fec_success": 1.0,
      "re_encode_errors": 0.0,
      "re_encode_consistency": 1.0
    },
    "contradictions": [],
    "re_encode_errors": 0,
    "consistency_fraction": 1.0,
    "rank": 1
  },
  "bitstream_analysis": {
    "bit_count": 266,
    "byte_count": 33,
    "is_byte_aligned": false,
    "remainder_bits": 2,
    "zero_count": 148,
    "one_count": 118,
    "one_density": 0.4436,
    "empirical_entropy": 0.9908,
    "max_zero_run": 8,
    "max_one_run": 8,
    "mean_run_length": 1.99,
    "total_runs": 134,
    "run_length_distribution": {"1": 66, "2": 39, "3": 17, "4": 4, "5": 5, "6": 1, "8": 2},
    "detected_period": null,
    "max_autocorrelation_peak": 0.0,
    "has_periodic_framing": false,
    "autocorrelation_peaks": [],
    "repeated_blocks": [],
    "aligned_byte_count": 33,
    "unique_byte_count": 31,
    "printable_ascii_ratio": 0.303,
    "ascii_control_ratio": 0.0909,
    "zero_byte_ratio": 0.0,
    "high_bit_set_ratio": 0.4242,
    "byte_entropy": 4.8878,
    "top_byte_frequencies": [],
    "structural_classification": "BINARY_STRUCTURED",
    "classification_justification": [
      "High bit entropy (0.9908) and balanced bit density (0.4436), characteristic of coded or pseudo-random binary payload."
    ]
  },
  "evidence": [
    {
      "evidence_id": "EVID_PHYS_SNR_EST",
      "category": "PHYSICAL_SIGNAL",
      "metric": "estimated_snr_db",
      "value": 15.0,
      "unit": "dB",
      "source": "demod.diagnostics",
      "reliability": "MEDIUM",
      "interpretation": "Estimated signal-to-noise ratio: 15.00 dB.",
      "supports": [],
      "contradicts": []
    }
  ],
  "contradictions": [],
  "confidence": {
    "value": 0.9419,
    "method": "deterministic_evidence_weighted_v1",
    "method_version": "phase3-v1",
    "components": {},
    "penalties": {},
    "uncertainty_reasons": []
  },
  "reference": {
    "status": "REFERENCE_BITS_UNAVAILABLE",
    "ber": null,
    "bit_errors": null
  },
  "uncertainty_reasons": [],
  "provenance": {
    "phase": "3",
    "implementation_version": "0.1.0",
    "truth_data_used_in_production": false,
    "deterministic": true
  }
}
```

---

## 3. Official Status Codes

| Status Code | Meaning | Example Occurrence |
| :--- | :--- | :--- |
| `CONFIRMED_BY_MULTIPLE_EVIDENCE` | Synchronization locked, FEC converged, re-encode error = 0. | G2, G3, G4, G5 coded, G6 |
| `SUPPORTED` | Demodulation and FEC pass, but minor inconsistency or partial re-encode. | Clean bursts without ambiguity |
| `TENTATIVE` | Demodulation succeeded, uncoded signal, structural features observed. | Uncoded test signals |
| `AMBIGUOUS` | Multiple competing hypotheses tied with identical top consistency scores. | Degenerate constellation symmetries |
| `REFERENCE_BITS_UNAVAILABLE` | Demodulation succeeded nominally, but external reference bits do not exist to measure BER. | G1, G5 uncoded |
| `FAILED_TO_DECODE_NEAR_THRESHOLD_STRESS_CASE` | Operating near physical noise threshold ($\approx 6$ dB), synchronization degraded, FEC failed to produce a valid codeword. | G7 |
| `DECODER_FAILURE` | Decoder pipeline failure or parity check failure above threshold. | Corrupt captures |

---

## 4. Programmatic Ingestion API

Downstream developers can import and execute the evidence generator directly:

```python
from spectralq.decoder_api import DecoderPipeline, DecoderConfig
from spectralq.bitintel import generate_decoder_evidence

# 1. Run pipeline
pipeline = DecoderPipeline()
result = pipeline.decode(iq_waveform, config)

# 2. Generate evidence handoff object
handoff = generate_decoder_evidence(result, case_id="BURST_001")

# 3. Access machine-readable properties or serialize to JSON
print(f"Status: {handoff.overall_status}")
print(f"Confidence: {handoff.confidence.value}")
json_string = handoff.to_json()
```
