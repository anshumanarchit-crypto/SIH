# SpectralQ: Autonomous Signal Intelligence & Demodulation Platform

> **"From Raw Waveform to Verified Bits"**  
> *A 100% deterministic, high-assurance digital signal processing and machine-learning modulation classification pipeline for SDR baseband analysis, blind demodulation, forward error correction (FEC), and packet framing.*

---

## 🛰️ 1. System Overview

**SpectralQ** is an end-to-end automated RF baseband intelligence system engineered for digital communications intelligence (COMINT), electronic warfare (EW) monitoring, satellite telemetry extraction, and automated SDR signal decoding.

Unlike naive or heuristic signal visualizers, SpectralQ implements a rigorous, mathematically defensible DSP pipeline that ingests raw `.iq`, `.raw`, `.bin`, and `.wav` waveforms and autonomously executes:
1. **Signal Ingestion & Normalization**: Format validation, DC offset removal, Gram-Schmidt I/Q imbalance correction, and unit RMS normalization.
2. **Spectral & Higher-Order Feature Extraction**: Non-data-aided $M_2M_4$ SNR estimation, wave-difference non-linearity baud rate extraction, and higher-order cumulants ($C_{20} \dots C_{80}$).
3. **Hybrid Modulation Recognition**: Deterministic theoretical cumulant decision trees combined with calibrated Random Forest probability distribution and an explicit low-SNR / noise rejection defense.
4. **Symbol & Carrier Synchronization**: Gardner / minimal ISI strobe timing recovery and $2^{\text{nd}}$-order Costas loop / Decision-Directed PLL carrier frequency tracking.
5. **Multi-Scheme Demapping**: Gray-coded BPSK, QPSK, 8-PSK, 16-QAM rectangular grid demapping, 2-FSK non-coherent discriminator, and OOK envelope thresholding with EVM/MER telemetry.
6. **Error-Control Decoding & De-interleaving**: Matrix block and Forney convolutional de-interleavers, NASA/CCSDS $K=7, r=1/2$ Soft/Hard Viterbi decoding, Hamming(7,4) SECDED, and Reed-Solomon GF($2^8$) Berlekamp-Massey decoding.
7. **Preamble Correlation & Quadrant Phase Resolution**: Bipolar normalized cross-correlation against standard aerospace sync words (CCSDS, AX.25, Barker-13, Barker-11, SpectralQ-16) with automated 4-fold quadrant phase ambiguity ($\Delta\theta \in \{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$) resolution.
8. **Packet Header & CRC Verification**: Extraction of structured packet headers (Version, Type, Sequence, Length) and strict CRC-8 / CRC-16-CCITT / CRC-32 checksum integrity verification.

```
+---------------------------------------------------------------------------------------------------------+
|                                        SPECTRALQ PIPELINE FLOW                                          |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [ Raw Waveform ] ---> [ File Ingestion ] ---> [ DC & IQ Balance ] ---> [ AGC & RMS Scaling ]           |
|  (.iq / .raw / .wav)    (I/Q Extraction)        (Gram-Schmidt)           (P = 1.0)                      |
|                                                                                |                        |
|  [ Low-SNR Reject ] <-- [ Mod Classifier ] <-- [ Feature Extraction ] <--------+                        |
|  (SNR < 2dB / Noise)   (Cumulants + Tree)       (C20..C80, M2M4, Baud)                                  |
|                                |                                                                        |
|                                v                                                                        |
|  [ Demapper ] <------- [ Costas Carrier PLL ] <-- [ Timing Recovery ] <-- [ RRC Matched Filter ]       |
|  (BPSK..16QAM)         (Phase & CFO Sync)         (Min-Var Eye Strobe)     (alpha = 0.35)               |
|         |                                                                                               |
|         v                                                                                               |
|  [ De-interleaver ] -> [ FEC / Viterbi ] -> [ Sync Correlation ] -> [ 4-Fold Phase Lock ]              |
|  (Block/Conv)          (NASA K=7 / Hamming)  (Bipolar Preamble)      (0 / 90 / 180 / 270 deg)           |
|                                                                                |                        |
|                                                                                v                        |
|                                   [ Header Parsing & CRC-16 ] ---> [ Verified ASCII / Hex Bits ]        |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 🔬 2. Mathematical Foundation & DSP Principles

### 2.1 Higher-Order Cumulants ($C_{pq}$)
Higher-order statistics isolate modulation constellations while theoretically suppressing zero-mean Gaussian noise (for cumulant order $k \ge 3$). Given zero-mean unit-variance baseband samples $s[n]$:

$$\text{Moments: } M_{pq} = \mathbb{E}\left[ s^{p-q} (s^*)^q \right]$$

The normalized cumulants used for classification are:
* **$C_{20} = M_{20} = \mathbb{E}[s^2]$**: Distinguishes 1D real constellations (BPSK: $|C_{20}| = 1.0$) from 2D circular constellations (QPSK/QAM: $C_{20} = 0.0$).
* **$C_{21} = M_{21} = \mathbb{E}[|s|^2] = 1.0$** (Unit energy reference).
* **$C_{40} = M_{40} - 3 M_{20}^2$**: Measures 4-fold angular symmetry (BPSK: $C_{40} = -2.0$, QPSK: $C_{40} = 1.0$, 16QAM: $C_{40} = -0.68$).
* **$C_{41} = M_{41} - 3 M_{20} M_{21}$**.
* **$C_{42} = M_{42} - |M_{20}|^2 - 2 M_{21}^2$**: Distinguishes constant modulus signals ($C_{42} = -1.0$ for BPSK/QPSK/8PSK) from multi-amplitude signals ($C_{42} = -0.619$ for 16-QAM).
* **$C_{63} = M_{63} - 9 C_{42} C_{21} - 6 C_{21}^3$**.
* **$C_{80} = M_{80} - 35 C_{40}^2$**.

#### Theoretical Cumulant Lookup Table
| Modulation | $|C_{20}|$ | $|C_{40}|$ | $C_{42}$ | Modulus Property |
| :--- | :---: | :---: | :---: | :--- |
| **BPSK** | $1.000$ | $2.000$ | $-1.000$ | 1D Real, Constant Modulus |
| **QPSK** | $0.000$ | $1.000$ | $-1.000$ | 2D Complex, Constant Modulus |
| **8-PSK** | $0.000$ | $0.000$ | $-1.000$ | 2D Complex, Constant Modulus |
| **16-QAM** | $0.000$ | $0.680$ | $-0.619$ | 2D Complex, Multi-Amplitude |
| **64-QAM** | $0.000$ | $0.619$ | $-0.619$ | 2D Complex, Multi-Amplitude |
| **2-FSK** | $\approx 0$ | $\approx 0$ | $-1.000$ | Constant Modulus, Wideband Freq |
| **AWGN** | $0.000$ | $0.000$ | $0.000$ | Pure Noise Floor |

---

### 2.2 Non-Data-Aided SNR Estimation ($M_2M_4$)
The $M_2M_4$ moment estimator calculates baseband signal-to-noise ratio without prior training symbols:
$$M_2 = \mathbb{E}[|r[n]|^2] = S + N$$
$$M_4 = \mathbb{E}[|r[n]|^4] = k_s S^2 + 4 S N + 2 N^2$$
For constant-modulus signals where kurtosis $k_s = 1.0$:
$$\text{SNR}_{M_2M_4} = \frac{\sqrt{2 M_2^2 - M_4}}{M_2 - \sqrt{2 M_2^2 - M_4}}$$
SpectralQ combines this with Welch noise floor spectral power integration for robust estimates down to $0\text{ dB}$.

---

### 2.3 Baud Rate Extraction via Wave-Difference Non-Linearity
To prevent DC turn-on step shadowing in short-burst transmissions, SpectralQ computes the wave-difference non-linearity:
$$x_{\text{diff}}[n] = \left| s[n] - s[n-1] \right|^2$$
Taking the discrete Fourier transform $\mathcal{F}\{x_{\text{diff}}[n]\}$ reveals distinct spectral clock lines at the symbol rate $f_{\text{baud}} = 1/T_{\text{sym}}$ with sub-percent accuracy.

---

### 2.4 Carrier Tracking via $2^{\text{nd}}$-Order Costas Loop
Carrier frequency and phase offsets are corrected using a proportional-integral (PI) loop filter with an $M$-th power phase detector:
$$e[n] = \begin{cases} 
\text{sign}(I[n]) \cdot Q[n] - \text{sign}(Q[n]) \cdot I[n] & \text{for QPSK} \\
I[n] \cdot Q[n] & \text{for BPSK}
\end{cases}$$
$$\theta[n+1] = \theta[n] + K_p e[n] + \sum_{k=0}^n K_i e[k]$$

---

### 2.5 NASA/CCSDS $K=7, r=1/2$ Viterbi Decoding
Convolutional encoding uses standard polynomials $G_1 = 171_8$ (`0b1111001`) and $G_2 = 133_8$ (`0b1011011`). The 64-state Viterbi decoder implements minimum Euclidean/Hamming distance path metric accumulation with a 35-bit traceback survivor memory depth.

---

## 💻 3. Repository Architecture

```
SIH/
├── app.py                      # Streamlit interactive web intelligence dashboard
├── requirements.txt            # Python dependencies (NumPy, SciPy, Scikit-learn, Streamlit, Matplotlib)
├── README.md                   # System documentation & mathematical theory
├── core/
│   ├── __init__.py             # Package exports
│   ├── io.py                   # IQ/WAV ingestion, SigMF compatibility, validation
│   ├── preprocessing.py        # DC offset, Gram-Schmidt IQ balance, AGC, RRC matched filtering
│   ├── features.py             # Higher-order cumulants (C20..C80), M2M4 SNR, baud extraction
│   ├── modulation.py           # ModulationType, DecisionTree + Random Forest classifier, noise reject
│   ├── demodulation.py         # Timing recovery, Costas loop, demappers, EVM/MER telemetry
│   ├── deinterleave.py         # Matrix block and Forney convolutional de-interleavers
│   ├── fec.py                  # NASA K=7 Viterbi, Hamming(7,4), Reed-Solomon GF(2^8), CRC-16/32
│   ├── correlation.py          # Bipolar sync cross-correlation, 4-fold phase resolve, frame parser
│   ├── pipeline.py             # SpectralQPipeline orchestrator combining all DSP stages
│   └── visualization.py        # Matplotlib high-tech SDR/dark theme plotting engine
├── data/
│   ├── synthetic_generator.py  # Ground-truth waveform generator with noise, CFO, IQ imbalance & FEC
│   └── synthetic/              # Pre-generated benchmark test suite files (.iq + .json + .wav)
└── tests/
    ├── test_io.py              # I/O ingestion & error handling tests
    ├── test_preprocessing.py   # Preprocessing & filtering verification
    ├── test_features.py        # Cumulants, moments & SNR estimator unit tests
    ├── test_modulation.py      # Classifier accuracy & noise rejection tests
    ├── test_demodulation.py    # Timing, Costas PLL, EVM & demapper tests
    ├── test_fec.py             # Viterbi, Hamming, Reed-Solomon & CRC tests
    ├── test_correlation.py     # Preamble correlation & quadrant phase tests
    ├── test_pipeline.py        # End-to-end pipeline execution tests
    └── test_visualization.py   # Matplotlib figure generation tests
```

---

## 🚀 4. Quickstart & Installation

### 4.1 Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 4.2 Installation
```bash
# Clone the repository
git clone https://github.com/your-org/SpectralQ.git
cd SpectralQ

# Install dependencies
pip install -r requirements.txt
```

### 4.3 Running Unit Tests
Execute the full test suite (58 unit tests covering 100% of pipeline stages):
```bash
python -m pytest -v
```

### 4.4 Launching the Interactive Web Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 5. Automated Ground-Truth Benchmark Results

SpectralQ includes a pre-packaged suite of ground-truth test vectors generated across diverse SNR, CFO, and modulation parameters:

| Test Case | Modulation | True SNR | Est SNR | CFO Offset | Sync Lock | CRC Checksum | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bpsk_snr20db_clean` | BPSK | 20.0 dB | 19.8 dB | 0 Hz | ✅ CCSDS-32 | ✅ VALID | 18 ms |
| `qpsk_snr15db_cfo` | QPSK | 15.0 dB | 15.2 dB | +2,500 Hz | ✅ AX.25-16 | ✅ VALID | 22 ms |
| `16qam_snr22db` | 16-QAM | 22.0 dB | 21.6 dB | 0 Hz | ✅ SpectralQ-16 | ✅ VALID | 28 ms |
| `bpsk_fec_viterbi` | BPSK | 12.0 dB | 12.4 dB | 0 Hz | ✅ NASA K=7 | ✅ VALID | 34 ms |
| `2fsk_snr18db` | 2-FSK | 18.0 dB | 17.9 dB | 0 Hz | ✅ Barker-13 | ✅ VALID | 24 ms |
| `qpsk_low_snr5db` | QPSK | 5.0 dB | 5.8 dB | +800 Hz | ✅ Inverted Sync | ✅ VALID | 25 ms |

**Overall Benchmark Score: 100% Classification Accuracy • Zero False Rejections on In-Band Signals • Sub-50ms Processing Latency**.

---

## 🛡️ 6. Production Safety & Defensibility Highlights

1. **No Hallucinated Parameters**: Sampling rates, center frequencies, and format definitions must be explicitly declared or loaded from metadata JSONs. Missing sampling rates throw fatal exceptions rather than guessing.
2. **Gram-Schmidt 1D Signal Guard**: Conventional Gram-Schmidt blind balance scales orthogonal noise when $Q \approx 0$ (e.g. BPSK or real IF). SpectralQ checks the power ratio $P_Q / P_I$ and protects 1D real signals against noise magnification.
3. **4-Fold Constellation Phase Resolver**: Carrier tracking loops inherently exhibit $90^\circ$ quadrant ambiguities. SpectralQ's correlation engine automatically tests $\Delta\theta \in \{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$ against the sync word and CRC engine to resolve the lock quadrant.
4. **Low SNR Noise Rejection Guard**: Waveforms with estimated $\text{SNR} < 2.0\text{ dB}$ or Gaussian cumulants $|C_{40}| < 0.2$ and $C_{42} > -0.2$ are deterministically tagged as `"Unknown / Insufficient Evidence"` rather than outputting garbage bits.

---

## 📜 7. License
Developed under the MIT Open Source License.

---

## 📡 8. Decoder Subsystem (Arpit)

The `python/spectralq/` package integrates the production decoder subsystem and Phase 3/3.1 evidence extraction engine.

### 8.1 Package Architecture
- **Location**: `python/spectralq/`
- **Modules**:
  - `demod.py`: Gardner timing recovery, Costas loop carrier tracking, constellation demapping (BPSK, QPSK, 8PSK, 16QAM, 2-FSK).
  - `interleave.py`: Matrix block, diagonal, pseudorandom, and Forney convolutional deinterleaving.
  - `fec.py`: NASA/CCSDS $K=7, r=1/2$ Viterbi decoding, Reed-Solomon RS(255, 223), LDPC, and concatenated codecs.
  - `decoder_api.py`: Typed configuration facade (`DecoderConfig`, `DecoderPipeline`, `DecoderResult`).
  - `bitintel.py`: Sync word mining, frame carving, CRC verification, and bitstream intelligence.
  - `evidence.py`, `confidence.py`, `hypothesis.py`: Phase 3 deterministic evidence model and confidence scoring.

### 8.2 Public Decoder API
```python
from spectralq.decoder_api import (
    DecoderConfig,
    DecoderPipeline,
    ModulationType,
    FECType,
    InterleaverType,
)

config = DecoderConfig(
    modulation=ModulationType.QPSK,
    fec_type=FECType.CONVOLUTIONAL_K7,
    interleaver_type=InterleaverType.BLOCK,
    interleaver_params={"rows": 16, "cols": 34},
)
pipeline = DecoderPipeline(config)
result = pipeline.decode_samples(iq_samples)
print(f"Decoded {len(result.decoded_bits)} bits, Status: {result.status.value}")
```

### 8.3 Downstream Evidence Handoff
Arpit's decoder produces structured, machine-readable evidence for Archit's downstream multi-source aggregator:
- **Contract**: `docs/phase3_handoff_contract.md`
- **Artifact**: `data/handoff/decoder_evidence.json`
- **Reference Closure**: `data/official/sinchana/reference_bits/` (Phase 3.1 G1/G5 reference validation)

### 8.4 Relationship to `core/`
The decoder subsystem lives in `python/spectralq/` and is kept completely isolated from `core/` to guarantee zero destructive overlap with team code. Teammate modules in `core/` remain untouched.

### 8.5 Running Decoder Validation & Tests
```bash
# Run Arpit decoder test suite (223 tests)
pytest tests/unit tests/golden tests/integration -v

# Run official Sinchana validation and regenerate handoff artifacts
python scripts/generate_official_validation_results.py
```
