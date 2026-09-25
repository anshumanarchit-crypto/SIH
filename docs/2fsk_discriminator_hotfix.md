# 2-FSK Instantaneous Frequency Discriminator Hotfix Report

**Repository**: `https://github.com/anshumanarchit-crypto/SIH.git`  
**Base Commit**: `9d4ba6e` (origin/main)  
**Hotfix Branch**: `fix/2fsk-discriminator-length`  
**Date**: 2026-09-25  
**Author**: Arpit (Signal Intelligence & Decoder Lead)  

---

## 1. Original Failure

### Failing Test
`tests/test_demodulation.py::test_demodulate_2fsk_matched_filter_and_discriminator`

### Traceback Summary
```python
core/demodulation.py:459: in demodulate_2fsk
    sym_freqs = np.mean(freq[:num_syms * sps].reshape(num_syms, sps), axis=1)
E   ValueError: cannot reshape array of size 2559 into shape (80,32)
```

---

## 2. Root Cause Analysis

In `core/demodulation.py`:
- Input `samples` has length $N = 2560$ (80 symbols $\times$ 32 SPS).
- `phase = np.unwrap(np.angle(samples))` has length $N = 2560$.
- Discrete frequency differentiation was performed via:
  ```python
  freq = np.diff(phase) * (sample_rate / (2.0 * np.pi))
  ```
- Because standard `np.diff` computes forward differences $\Delta \phi[i] = \phi[i+1] - \phi[i]$, the resulting `freq` array had length $N - 1 = 2559$.
- Downstream symbol aggregation expected $N = \text{num\_syms} \times \text{sps} = 2560$ samples:
  ```python
  sym_freqs = np.mean(freq[:num_syms * sps].reshape(num_syms, sps), axis=1)
  ```
- Slicing `freq[:2560]` returned only 2559 elements, causing NumPy to throw a `ValueError` when attempting to reshape into `(80, 32)`.

---

## 3. Exact Minimal Fix

In [`core/demodulation.py`](file:///c:/Users/arpit/Desktop/SIH%20ARPIT/core/demodulation.py#L458), line 458 was updated:

```diff
- freq = np.diff(phase) * (sample_rate / (2.0 * np.pi))
+ freq = np.diff(phase, prepend=phase[0]) * (sample_rate / (2.0 * np.pi))
```

---

## 4. Mathematical & Algorithmic Rationale

1. **Discrete Derivative Invariant**: Instantaneous frequency is defined as the time rate of change of unwrapped phase:
   $$f[n] \approx \frac{f_s}{2\pi} \Delta \phi[n]$$
2. **Boundary Condition at $n=0$**: Prior to the first recorded sample, the phase state is initialized to the boundary value $\phi[-1] = \phi[0]$. Therefore:
   $$\Delta \phi[0] = \phi[0] - \phi[-1] = \phi[0] - \phi[0] = 0$$
   This assigns $f[0] = 0.0\text{ Hz}$ at the start of the burst, preserving sample length $N$ without introducing timing offsets or phase bias.
3. **Symbol Window Robustness**: In a 32 SPS symbol window, having sample 0 at $0\text{ Hz}$ and samples 1..31 at nominal tone frequency $\pm 1000\text{ Hz}$ results in a mean frequency of $\frac{31}{32} \times (\pm 1000) = \pm 968.75\text{ Hz}$. This preserves the exact sign ($>0$ for bit 1, $<0$ for bit 0), achieving **0 bit errors** across all 80 symbols.

---

## 5. Regression Test

In [`tests/test_demodulation.py`](file:///c:/Users/arpit/Desktop/SIH%20ARPIT/tests/test_demodulation.py#L116-L121), the test was strengthened to explicitly assert output array lengths and telemetry without weakening test criteria:

```python
ref_syms_d, rx_bits_d, soft_llrs_d, diag_d = demodulate_2fsk(
    samples, sample_rate=fs, sps=sps, method="discriminator"
)
assert len(rx_bits_d) == n_syms
assert len(ref_syms_d) == n_syms
assert len(soft_llrs_d) == n_syms
assert diag_d["method"] == "instantaneous_frequency_discriminator"
np.testing.assert_array_equal(rx_bits_d, tx_bits)
```

---

## 6. Comprehensive Verification Results

| Verification Suite | Target | Result | Status |
| :--- | :--- | :--- | :--- |
| **Team Core Tests** (`tests/test_*.py`) | 260 tests | **260 passed (100%)** | ✅ RESOLVED |
| **Arpit Decoder Suite** (`tests/unit`, `golden`, `integration`) | 223 tests | **223 passed (100%)** | ✅ PASSED |
| **Full Repository Pytest** | 483 tests | **483 passed (100%)** | ✅ ALL GREEN |
| **Pyright Static Type Check** | 0 errors | **0 errors, 0 warnings** | ✅ CLEAN |
| **Truth Isolation Audit** | 0 dependencies | **Clean (zero truth reads)** | ✅ CLEAN |
| **Git Diff Whitespace Check** | `git diff --check` | **0 errors** | ✅ CLEAN |

---

## 7. Official Decoder & Reference Closure Regression

- **G1 (QPSK Uncoded)**: BER = **0.0000** (0 / 4096 errors, confidence: 0.9352)
- **G2 (BPSK Conv K=7 Block)**: BER = **0.0000** (0 / 544 errors, confidence: 0.9549)
- **G3 (8PSK RS(255,223) Diagonal)**: BER = **0.0000** (0 / 2040 errors, confidence: 0.9549)
- **G4 (16QAM LDPC Pseudorandom)**: BER = **0.0000** (0 / 96 errors, confidence: 0.9549)
- **G5 Uncoded (2-FSK Uncoded)**: BER = **0.0000** (0 / 4096 errors, confidence: 0.9352)
- **G5 Coded (2-FSK Concatenated RS+Conv)**: BER = **0.0000** (0 / 4152 errors, confidence: 0.9030)
- **G6 (BPSK Conv K=7 Convolutional)**: BER = **0.0000** (0 / 548 errors, confidence: 0.9549)
- **G7 (QPSK Conv K=7 Stress Case)**: **`FAILED_TO_DECODE_NEAR_THRESHOLD_STRESS_CASE`** (BER = 0.4580, confidence: 0.1200; strictly preserved as near-threshold failure).

---

## 8. Exact Files Changed

1. `core/demodulation.py` (+1 line, -1 line)
2. `tests/test_demodulation.py` (+4 lines)
3. `docs/2fsk_discriminator_hotfix.md` (this report)
