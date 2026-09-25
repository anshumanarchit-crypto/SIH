# SpectralQ Phase 2.6: Official Decoder Alignment and Robustness

## 1. Executive Summary

Phase 2.6 bridges the gap between synthetic validation and official Octave-generated captures delivered by Sinchana (`SIH-main (1).zip`). Without writing capture-specific heuristics or filename conditionals (`if case_id == "G2"` is strictly prohibited), this release introduces generalized architectural capabilities into the demodulator and decoder core:

1. **Rotational Phase Ambiguity Resolution via Candidate Re-Encode Consistency**:
   - For BPSK ($0^\circ, 180^\circ$) and QPSK ($0^\circ, 90^\circ, 180^\circ, 270^\circ$), the demodulator exposes candidates.
   - For coded captures, the decoder core evaluates candidates through the full receive chain (`candidate rotation -> deinterleave -> FEC decode -> encode_chain consistency`).
   - The candidate with zero (or minimum) re-encode errors is deterministically selected without any truth-data leakage.
2. **Explicit Modulation Mapping Profiles**:
   - Decoupled physical sector/level detection from bit labeling.
   - Supported profiles:
     - 8-PSK: `"DEFAULT"` (standard Gray) and `"OCTAVE"` (Octave `pskmod` sector mapping).
     - 16-QAM: `"DEFAULT"` and `"OCTAVE"` (Octave `qamdemod` PAM-4 mapping).
3. **Reference Bit Availability Contract**:
   - Uncoded captures without deterministic reference handoffs (G1, G5 uncoded) transition to `REFERENCE_BITS_UNAVAILABLE`.
   - BER is not marked as zero; external references can be supplied via `external_reference_bits`.
4. **G7 Near-Threshold Diagnostic Breakdown**:
   - Preserves the 6 dB physical-layer stress case safely as `FAILED_TO_DECODE_NEAR_THRESHOLD_STRESS_CASE` without forcing artificial decoding.

---

## 2. Phase Ambiguity Architecture

### 2.1 The Problem: $M$-Fold Costas Ambiguity
A Costas loop or $M$-th power carrier synchronizer for $M$-PSK exhibits $M$-fold rotational phase ambiguity ($\frac{2\pi}{M} \cdot k, k \in \{0, \dots, M-1\}$). In Sinchana's official BPSK captures (G2 and G6), the Costas loop locked to the stable antipodal false lock at $180^\circ$ ($\pi$ radians), causing all demodulated bits to invert ($0 \leftrightarrow 1$).

### 2.2 Generalized Solution: Multi-Candidate Re-Encode Consistency
Instead of manually inverting bits for specific cases, the demodulator and decoder implement an hypothesis candidate engine:
- The demodulator generates candidate bit streams corresponding to the $M$ rotational symmetries:
  - BPSK: $0^\circ, 180^\circ$
  - QPSK / 16-QAM: $0^\circ, 90^\circ, 180^\circ, 270^\circ$
  - 8-PSK: $0^\circ, 45^\circ, \dots, 315^\circ$
- The decoder pipeline iterates through candidates:
  $$\text{Candidate } \theta \to \text{De-interleave} \to \text{FEC Decode} \to \hat{s} \to \text{Re-Encode}(\hat{s}) \to \hat{t}$$
- The re-encoded bits $\hat{t}$ are compared against the candidate hard bits $r_\theta$:
  $$E_{\text{re-encode}}(\theta) = \sum |r_\theta[n] \oplus \hat{t}[n]|$$
- A candidate with $E_{\text{re-encode}} = 0$ is a mathematically self-consistent codeword.
- In official G2 and G6, rotation $0^\circ$ yields 12 and 36 errors respectively, while rotation $180^\circ$ yields **0 errors**, resulting in exact 100% source bit recovery.

---

## 3. Modulation Mapping Profiles

### 3.1 8-PSK Mapping Profiles (`MAPPING_PROFILES_8PSK`)
Sector detection maps complex symbol angle $\theta \in [0, 2\pi)$ to sector $k = \text{round}(\theta / (\pi/4)) \pmod 8$.
Bit labeling is decoupled into configured profiles:

| Sector | Angle | `"DEFAULT"` | `"OCTAVE"` |
| :---: | :---: | :---: | :---: |
| 0 | $0^\circ$ | `000` | `000` |
| 1 | $45^\circ$ | `001` | `001` |
| 2 | $90^\circ$ | `011` | `011` |
| 3 | $135^\circ$ | `010` | `010` |
| 4 | $180^\circ$ | `110` | `111` |
| 5 | $225^\circ$ | `111` | `110` |
| 6 | $270^\circ$ | `101` | `100` |
| 7 | $315^\circ$ | `100` | `101` |

Configuring `mapping_profile="OCTAVE"` for G3 yields 0 TX bit errors and 0 source bit errors across all 1784 bits.

### 3.2 16-QAM Mapping Profiles (`MAPPING_PROFILES_16QAM`)
16-QAM decomposes into independent 4-PAM decision regions on I and Q:
- Region 0: level $> 2.0$ (centered at $+3$)
- Region 1: $0.0 < \text{level} \le 2.0$ (centered at $+1$)
- Region 2: $-2.0 \le \text{level} \le 0.0$ (centered at $-1$)
- Region 3: level $< -2.0$ (centered at $-3$)

| Region | PAM Level | `"DEFAULT"` | `"OCTAVE"` |
| :---: | :---: | :---: | :---: |
| 0 | $+3$ | `(1, 0)` | `(0, 0)` |
| 1 | $+1$ | `(1, 1)` | `(0, 1)` |
| 2 | $-1$ | `(0, 1)` | `(1, 1)` |
| 3 | $-3$ | `(0, 0)` | `(1, 0)` |

Configuring `mapping_profile="OCTAVE"` for G4 yields 0 TX bit errors and 0 source bit errors across all 50 bits.

---

## 4. G1 and G5 Uncoded Reference Status

In Sinchana's capture generation, G1 and uncoded G5 relied on internal Octave PRNG seed generation (`randi`) without providing the generated bit sequences in the handoff.
To maintain cryptographic and scientific rigor:
- No artificial reference bitstreams were fabricated.
- No Octave internal PRNG implementation was guessed.
- The status is explicitly recorded as `REFERENCE_BITS_UNAVAILABLE`.
- BER is recorded as `None` rather than 0.
- `DemodConfig` accepts `external_reference_bits` when Sinchana delivers the missing reference bit files.

---

## 5. G7 Diagnostic Findings (6 dB Stress Case)

Official G7 Parameters:
- Modulation: QPSK, Convolutional $K=7$ ($R=1/2$)
- Length: 262 symbols, 524 TX bits, 256 source bits
- SNR: 6 dB $E_s/N_0$
- Interleaver: `NONE`

Diagnostic Breakdown:
- Raw IQ Power: 1.2755, RRC Output Power: 7.8632
- Gardner Jitter Std: 1.4950 (TED unlocks under severe noise)
- Costas Phase Jitter Std: 0.7303 rad (cycle slipping)
- Constellation Spread: Real/Imag extent $[-3.13, +3.23]$
- Best Demod BER: 0.4580 (240 errors / 524 bits)
- Viterbi Error Pattern: Due to lack of interleaver, burst channel errors overwhelm the constraint length ($K=7$), causing catastrophic trellis diverge (125 errors / 256 source bits).

**Conclusion**: G7 is operating below the un-interleaved cutoff threshold for $K=7$ convolutional coding. The decoder core safely and transparently reports `FAILED_TO_DECODE_NEAR_THRESHOLD_STRESS_CASE` without crashing or falsely asserting success.

---

## 6. Official Results Matrix

| Case | Modulation | Coding | Interleaver | Profile | Rot. Sel. | TX BER | Source Errors | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **G1** | QPSK | Uncoded | None | DEFAULT | $0^\circ$ | N/A | N/A | `REFERENCE_BITS_UNAVAILABLE` |
| **G2** | BPSK | Conv K=7 | Block | DEFAULT | $180^\circ$ | 0.0 | **0 / 256** | **`FULLY_DECODED`** |
| **G3** | 8-PSK | RS(255,223) | Diagonal | OCTAVE | $0^\circ$ | 0.0 | **0 / 1784** | **`FULLY_DECODED`** |
| **G4** | 16-QAM | LDPC | Pseudorandom | OCTAVE | $0^\circ$ | 0.0 | **0 / 50** | **`FULLY_DECODED`** |
| **G5 uncoded** | 2-FSK | Uncoded | None | DEFAULT | $0^\circ$ | N/A | N/A | `REFERENCE_BITS_UNAVAILABLE` |
| **G5 coded** | 2-FSK | RS+Conv | Convolutional | DEFAULT | $0^\circ$ | 0.0 | **0 / 1784** | **`FULLY_DECODED`** |
| **G6** | BPSK | Conv K=7 | Convolutional | DEFAULT | $180^\circ$ | 0.0 | **0 / 256** | **`FULLY_DECODED`** |
| **G7** | QPSK | Conv K=7 | None | DEFAULT | $0^\circ$ | 0.458 | 125 / 256 | `FAILED_TO_DECODE` (6 dB stress) |
