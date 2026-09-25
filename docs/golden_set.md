# SpectralQ Golden-Set Semantics & Validation Rules

## 1. Bitstream Semantics

To ensure rigorous and independent end-to-end validation between Sinchana's GNU Octave signal simulator and Arpit's Python decoder core, a strict two-tier bitstream distinction is established:

### SOURCE BITS
- **Definition**: The ground-truth information bitstream before forward error correction and interleaving.
- **File Location**: `data/golden/Gx/source_bits.txt`
- **Harness Role**: Retained independently in the test harness (`tests/golden/`) as the benchmark against which the recovered output of the Python decoder is evaluated for zero Bit Error Rate (BER = 0.0).
- **Simulator Rule**: The Octave waveform generator must NEVER receive SOURCE BITS directly.

### TX BITS
- **Definition**: The bitstream generated after FEC encoding and interleaving of SOURCE BITS.
- **File Location**: `data/handoff/bitsGx.txt`
- **Harness Role**: Exported to `data/handoff/` as `bitsG2.txt` through `bitsG7.txt`.
- **Simulator Role**: Fed directly into Sinchana's GNU Octave modulator to synthesize synthetic IQ baseband waveforms (`capture.cf32`).

```
[SOURCE BITS] ──(FEC Encode)──> [Coded Bits] ──(Interleave)──> [TX BITS]
      │                                                           │
      │ (Retained in tests/ for truth validation)                  ▼ (Fed to Octave)
      │                                                    [Octave Modulator]
      │                                                           │
      │                                                           ▼
      │                                                   [capture.cf32]
      │                                                           │
      │                                                           ▼
      │                                                    [Python Decoder]
      │                                                           │
      ▼                                                           ▼
[Truth Check BER = 0.0] <───────────────────────────── [Recovered SOURCE BITS]
```

---

## 2. Planned Golden Identities & Frozen Source Lengths

| Identity | Modulation | Forward Error Correction (FEC) | Interleaver Scheme | Frozen Source Length | Handoff Artifact |
|---|---|---|---|---|---|
| **G2** | BPSK | Convolutional Code (K=7, Rate 1/2, polynomials 171/133 octal) | Block Interleaver (Row-Write / Column-Read) | **256 bits** | `data/handoff/bitsG2.txt` |
| **G3** | 8-PSK | Reed-Solomon RS(255, 223) *(Byte-oriented GF(2^8))* | Diagonal Interleaver | **1,784 bits** (223 bytes) | `data/handoff/bitsG3.txt` |
| **G4** | 16-QAM | Low-Density Parity-Check (Gallager 96.3.963, GF(2) RREF) | Pseudo-Random Interleaver (Deterministic Seed) | **50 bits** (1 block) | `data/handoff/bitsG4.txt` |
| **G5** | 2-FSK | Concatenated: Outer RS(255,223) + Inner Conv (K=7, 171/133) | Convolutional Interleaver (Forney shift-register) | **1,784 bits** (223 bytes) | `data/handoff/bitsG5.txt` |
| **G6** | BPSK | Convolutional Code (K=7, Rate 1/2, polynomials 171/133 octal) | Convolutional Interleaver (Forney shift-register) | **256 bits** | `data/handoff/bitsG6.txt` |
| **G7** | QPSK | Convolutional Code (K=7, Rate 1/2, polynomials 171/133 octal) | No Interleaver (Identity / NONE) | **256 bits** | `data/handoff/bitsG7.txt` |

### G7 Golden Identity: QPSK + Convolutional Near-Threshold Channel Test

- **Source Definition**: QPSK + convolutional code near threshold SNR ("Decode or UNKNOWN — never confidently wrong").
- **Phase 1 Implementation Choices**:
  - Source bit length = 256 bits (`data/golden/G7/source_bits.txt`)
  - Deterministic seed = 202607
  - Convolutional code: NASA/ESA standard $K=7$, Rate 1/2, generator polynomials $171_8$ and $133_8$, terminated with 6 zero tail bits $\rightarrow 2 \times (256 + 6) = 524$ coded bits
  - Interleaver: `NONE` (explicit identity mapping, no padding, 524 TX bits)
  - Handoff artifact: `data/handoff/bitsG7.txt` (524 bits)

#### Phased Lifecycle & Responsibilities Boundary
1. **PHASE 1 (Completed in this repo)**:
   G7 digital baseband TX-bit generation and exact no-channel decode validation proving digital baseband correctness.
2. **LATER (Sinchana)**:
   Sinchana's near-threshold QPSK waveform generation (`capture.cf32`).
   *Note*: No exact G7 SNR is specified in the current team plan; the waveform generator owner must document the selected near-threshold SNR and channel impairments.
3. **LATER (Python Demodulation)**:
   Python demodulation (RRC filtering, Costas carrier loop, Gardner timing recovery, QPSK soft LLRs).
4. **LATER (Archit)**:
   Archit's confidence/UNKNOWN logic and hypothesis evaluation engine.

---

## 3. Truth Data Rule (Zero Leakage)

> [!CAUTION]
> **CRITICAL RULE**: `truth.json` is TEST/EVALUATION DATA ONLY.
> Production decoder modules (`python/spectralq/`) MUST NEVER import or consume `truth.json`, `source_bits.txt`, or golden expected outputs.

- `truth.json` and `source_bits.txt` may be consumed strictly by:
  - `tests/`
  - `bench/`
  - Standalone evaluation scripts in `scripts/`
- Decoder modules must operate blindly on the IQ waveform and provided `DecoderConfig` to guarantee that signal blind-recovery metrics are uncompromised.

---

## 4. Reproducibility Standards

1. **Deterministic RNG & Seeds**:
   - Every pseudo-random operation (e.g. PRNG interleaver in G4, source bit generation) must be seeded deterministically.
   - PRNG interleaving MUST always be 100% reproducible strictly from `(bit_length + seed + configuration)`.
2. **Environment & Runtime**:
   - Python runtime: `Python 3.13.7` (in `.venv-spectralq`).
   - Dependencies pinned strictly in `requirements-lock.txt`.
3. **Artifact Integrity & Double Generation**:
   - Running golden generation twice must yield identical source bits, TX bits, metadata, and SHA-256 hashes.
   - No volatile timestamps included in hashed deterministic metadata.
4. **Handoff Tracking**:
   - `data/handoff/` is explicitly tracked in Git (not ignored) to ensure cross-teammate reproducibility.
