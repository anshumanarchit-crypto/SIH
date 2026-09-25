# Phase 3: Bitstream Intelligence & Structural Analysis Engine

## 1. Overview and Purpose

The Bitstream Intelligence subsystem (`spectralq.bitstream` and `spectralq.bitintel`) performs deterministic statistical and structural characterization of recovered bitstreams. It acts as an objective, factual evidence producer for downstream hypothesis aggregation without semantic guessing, fabricated payloads, or external language model dependencies.

```
Recovered Bits (1D binary array)
              ↓
  [ Bitstream Intelligence Engine ]
  ├── A. Length & Byte Alignment
  ├── B. Basic Bit Statistics & Entropy
  ├── C. Run-Length Distribution
  ├── D. Periodicity & Autocorrelation
  ├── E. Byte Structure (Aligned Prefix)
  └── F. Deterministic Structural Classification
              ↓
  BitstreamAnalysisResult
```

---

## 2. Statistical and Structural Analysis Metrics

### A. Length Metrics
- `bit_count`: Total number of recovered bits $N$.
- `byte_count`: $\lfloor N / 8 \rfloor$.
- `is_byte_aligned`: Boolean flag indicating $N \pmod 8 == 0$ and $N > 0$.
- `remainder_bits`: Remainder bits $N \pmod 8$.

### B. Bit Statistics & Shannon Entropy
- `zero_count` ($N_0$), `one_count` ($N_1$).
- `one_density`: $p_1 = N_1 / N$.
- `empirical_entropy`: Bit-level Shannon entropy in bits/bit:
  $$H_{\text{bit}} = -p_0 \log_2(p_0) - p_1 \log_2(p_1)$$
  Evaluated with boundary convention $0 \log_2(0) = 0$.

### C. Run-Length Analysis
- Runs of consecutive zeros and ones are segmented deterministically using boundary index transitions.
- `max_zero_run`: Longest sequence of consecutive 0 bits.
- `max_one_run`: Longest sequence of consecutive 1 bits.
- `mean_run_length`: Average length of all alternating runs.
- `run_length_distribution`: Frequency count histogram of run lengths.

### D. Periodicity and Binary Autocorrelation
- Bipolar representation: $s_i = 2 b_i - 1 \in \{-1, +1\}$.
- Normalized linear autocorrelation over lag range $k \in [8, \min(256, \lfloor N / 2 \rfloor)]$:
  $$R(k) = \frac{1}{N - k} \sum_{i=0}^{N - k - 1} s_i \cdot s_{i+k}$$
- `autocorrelation_peaks`: Peaks where $R(k) > 0.70$.
- `has_periodic_framing`: Set to True if $\max_k R(k) \ge 0.80$.

### E. Byte Structure (over Aligned Prefix)
When $N \ge 8$, aligned bits are packed into bytes $\mathbf{B} = \text{packbits}(\mathbf{b}_{0:8 \lfloor N/8 \rfloor})$:
- `unique_byte_count`: Number of distinct byte symbols in $[0, 255]$ observed.
- `printable_ascii_ratio`: Fraction of bytes falling within printable ASCII range `0x20..0x7E` or whitespace control characters (`\t`, `\n`, `\r`).
- `ascii_control_ratio`: Fraction of bytes falling within non-whitespace ASCII control ranges (`0x00..0x1F`, `0x7F`).
- `zero_byte_ratio`: Frequency of `0x00` null bytes.
- `high_bit_set_ratio`: Frequency of bytes with MSB set (`0x80..0xFF`).
- `byte_entropy`: 8-bit symbol Shannon entropy:
  $$H_{\text{byte}} = -\sum_{i=0}^{255} p(B_i) \log_2(p(B_i)) \quad [0.0 \le H_{\text{byte}} \le 8.0 \text{ bits/byte}]$$
- `top_byte_frequencies`: Top 5 most frequent byte values with counts and relative frequencies.

---

## 3. Deterministic Structural Signatures

The engine categorizes bitstreams into explicit, auditable classes using deterministic criteria (no machine learning, no heuristic guesses):

| Classification | Deterministic Criteria | Interpretation |
| :--- | :--- | :--- |
| `TEXT_LIKE` | Byte-aligned, $N \ge 64$, printable ASCII ratio $\ge 0.85$, control ASCII ratio $\le 0.10$. | Payload has text-like character encoding without claiming specific language semantics. |
| `BINARY_STRUCTURED` | Bit entropy $H_{\text{bit}} \ge 0.95$, balanced density $0.40 \le p_1 \le 0.60$, printable ASCII $< 0.60$. | Payload exhibits characteristics of coded, scrambled, or pseudo-random binary data. |
| `BYTE_STRUCTURED` | Byte-aligned, aligned bytes $\ge 8$, non-uniform byte distribution (zero bytes $\ge 5\%$ or ASCII $\ge 30\%$ or byte entropy $< 0.85 \log_2(\min(256, N_{\text{bytes}}))$). | Formatted framing, binary records, or mixed protocol headers. |
| `HIGHLY_REPETITIVE` | $\max(\text{run}_0, \text{run}_1) \ge 64$ or max autocorrelation peak $\ge 0.85$. | Repetitive training sequence, sync marker framing, or idle patterns. |
| `PADDING_OR_CONSTANT` | $p_1 \le 0.01$ or $p_1 \ge 0.99$ or $N = 0$. | Degenerate all-zeros or all-ones unmodulated/padding payload. |
| `UNKNOWN_STRUCTURE` | $N < 32$ bits or indeterminate features. | Insufficient length or inconclusive statistical pattern. |

---

## 4. Truth Isolation Guarantees

The `spectralq.bitstream` module contains zero imports of truth files (`truth.json`, test fixtures, golden manifests). All metrics are derived purely from the incoming bit array passed at runtime.
