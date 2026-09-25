# Phase 3: Evidence Ledger and Contradiction Model

## 1. Architectural Role

The Evidence Ledger (`spectralq.evidence.EvidenceLedger`) converts raw pipeline diagnostics and measurement artifacts into typed, auditable, and immutable machine-readable evidence items. It enforces a strict factual contract between the physical/FEC decoding layer (Arpit) and the downstream hypothesis aggregation layer (Archit).

```
Demodulator Diagnostics
Timing / Carrier State
FEC Codec Results
Re-encode Consistency
Bitstream Analysis
        ↓
  [ Evidence Ledger ]
  ├── Categorized Evidence Items
  ├── Reliability Grading
  └── Automated Contradiction Detection
        ↓
  Machine-Readable Ledger Artifact
```

---

## 2. Evidence Categories

Every evidence item belongs to a strictly defined `EvidenceCategory`:

1. `PHYSICAL_SIGNAL`: EVM RMS, estimated SNR (dB), symbol count, sample count.
2. `TIMING`: Symbol timing lock status, timing error detector variance (jitter).
3. `CARRIER`: Carrier loop lock status, estimated frequency offset (Hz/rad), residual phase error.
4. `DEMODULATION`: Demodulation status (`SUCCESS`, `NON_CONVERGED`), symbol decision margins.
5. `MAPPING`: Constellation mapping profile utilized (`DEFAULT`, `OCTAVE`, `GRAY`).
6. `PHASE`: Selected carrier phase rotation hypothesis (e.g. 0.0°, 180.0°).
7. `INTERLEAVER`: Interleaving scheme, dimensions/branches, deinterleaver execution.
8. `FEC`: Forward error correction scheme, decoding success flag, iterations, syndrome status.
9. `REENCODE`: Forward re-encoding Hamming distance, consistent channel bit fraction.
10. `BITSTREAM_STRUCTURE`: Recovered bit count, byte alignment, entropy, structural signature.
11. `REFERENCE`: External reference bit comparison metrics (BER, bit errors) if available.
12. `REFERENCE_AVAILABILITY`: Explicit status declaring whether truth reference bits were available.

---

## 3. Evidence Item Data Structure

```json
{
  "evidence_id": "EVID_REENCODE_ERRORS",
  "category": "REENCODE",
  "metric": "re_encode_errors",
  "value": 0,
  "unit": "bits",
  "source": "decoder_result.re_encode_errors",
  "reliability": "HIGH",
  "interpretation": "Forward re-encoding of decoded payload matches channel bits with 0 errors.",
  "supports": ["HYP_ROT_180"],
  "contradicts": ["HYP_ROT_0"]
}
```

### Reliability Grading
- `HIGH`: Deterministic algebraic or cryptographic check (e.g., 0 re-encode error, syndrome match).
- `MEDIUM`: Statistical estimator with standard error bounds (e.g., timing jitter variance, EVM).
- `LOW`: Heuristic or short-burst estimation.
- `UNAVAILABLE`: Data not measurable due to lack of ground reference.

---

## 4. Contradiction Detection Engine

The system actively evaluates recorded evidence for systemic contradictions and anomalies. Contradictions are explicitly modeled and never silently ignored:

| Contradiction Type | Severity | Trigger Condition |
| :--- | :--- | :--- |
| `TIMING_UNLOCKED` | `CRITICAL` | `timing_lock == False` |
| `TIMING_JITTER_ELEVATED` | `HIGH` | `timing_error_variance > 0.15` |
| `CARRIER_UNLOCKED` | `HIGH` | `carrier_lock == False` |
| `NEAR_THRESHOLD_SNR` | `HIGH` | `estimated_snr_db <= 7.0` |
| `HIGH_EVM_DEGRADATION` | `MEDIUM` | `evm_rms > 0.35` |
| `FEC_DECODE_FAILED` | `CRITICAL` | `decoder_success == False` for coded scheme |
| `REENCODE_DISCREPANCY` | `HIGH` | `re_encode_errors > 0` |
| `REFERENCE_UNAVAILABLE` | `INFO` | Reference bits not provided (informational; no penalty) |

```json
{
  "contradiction_id": "CONTRA_NEAR_THRESHOLD_SNR",
  "contradiction_type": "NEAR_THRESHOLD_SNR",
  "severity": "HIGH",
  "details": "Estimated SNR (6.0 dB) indicates near-threshold stress condition (<= 7.0 dB).",
  "metric": "estimated_snr_db",
  "value": 6.0,
  "source_evidence_ids": ["EVID_PHYS_SNR_EST"]
}
```

---

## 5. Principle: Re-Encode Consistency is Evidence, Not Proof

> [!IMPORTANT]
> A re-encode Hamming distance of zero confirms that the recovered source bits, when processed through the forward FEC encoder and interleaver, generate a sequence identical to the demapped channel bits.
>
> While this proves that the recovered payload constitutes a valid, internally self-consistent codeword, it does **not** by itself guarantee that this codeword was the transmitter's intended message (for example, in the presence of undetectable error patterns or uncorrected burst shifts).
>
> External framing, CRC verification, and upper-layer protocol headers are required to confirm semantic authenticity.
