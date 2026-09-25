# Phase 3: Deterministic Evidence Confidence Methodology

## 1. Overview and Anti-Arbitrary Design

In traditional heuristic demodulation tools, confidence is often assigned through arbitrary constants (e.g., `confidence = 0.95` on success).

In SpectralQ Phase 3, confidence is strictly defined as **Deterministic Evidence Confidence**:
- It is a deterministic, reproducible mathematical function of factual evidence items.
- It contains zero magic numbers or probabilistic language models.
- It provides a transparent, auditable breakdown of positive component contributions and contradiction penalties.
- It is explicitly termed *deterministic evidence confidence*, **not** a calibrated posterior probability of correctness.

---

## 2. Mathematical Formulation

Let the set of evaluated pipeline components be $\mathcal{K}$. Each component $k \in \mathcal{K}$ contributes a normalized score $c_k \in [0.0, 1.0]$ with an explicit non-negative weight $w_k \ge 0$.

### Step 1: Base Evidence Score
$$\text{Base Score} = \frac{\sum_{k \in \mathcal{K}_{\text{active}}} w_k \cdot c_k}{\sum_{k \in \mathcal{K}_{\text{active}}} w_k}$$

### Step 2: Contradiction Deductions
For each detected contradiction $j$ with severity $S_j$, an explicit penalty $P(S_j)$ is deducted:
$$P(S_j) = \begin{cases}
0.40 & \text{if } S_j = \text{CRITICAL} \\
0.20 & \text{if } S_j = \text{HIGH} \\
0.10 & \text{if } S_j = \text{MEDIUM} \\
0.05 & \text{if } S_j = \text{LOW} \\
0.00 & \text{if } S_j = \text{INFO (e.g. missing external reference)}
\end{cases}$$

$$\text{Total Penalty} = \sum_{j} P(S_j)$$

### Step 3: Bounded Evidence Confidence
$$\text{Confidence} = \max\left(0.0, \min\left(1.0, \text{Base Score} - \text{Total Penalty}\right)\right)$$

### Step 4: Invariant Bounds (Fail-Safes Against False Confidence)
To ensure the decoder is **never confidently wrong**, structural invariants enforce strict upper bounds:
1. **FEC Failure / Major Coded Mismatch**: If FEC is configured and the decoder failed or produced $> 10$ re-encode errors:
   $$\text{Confidence} \le 0.12$$
2. **Timing Recovery Failure**: If the symbol timing loop failed to converge (`timing_lock == False`):
   $$\text{Confidence} \le 0.10$$
3. **Carrier Recovery Failure**: If carrier tracking failed (`carrier_lock == False`):
   $$\text{Confidence} \le 0.15$$
4. **Empty Payload**: If zero bits were recovered:
   $$\text{Confidence} = 0.0$$

---

## 3. Active Component Definitions and Weights

| Component | Default Weight | Raw Metric | Normalization Formula |
| :--- | :--- | :--- | :--- |
| `physical_quality` | 0.15 | Estimated SNR or EVM RMS | $c_{\text{phys}} = \text{clamp}\left(\frac{\text{SNR} - 4.0}{16.0}, 0.0, 1.0\right)$ or $1.0 - 2 \cdot \text{EVM}$ |
| `timing_quality` | 0.15 | Timing lock & Jitter variance | If locked: $\max(0.0, 1.0 - 4.0 \cdot \sigma_{\tau}^2)$; if unlocked: 0.0 |
| `carrier_quality` | 0.15 | Carrier lock & Phase offset | If locked: $\max(0.2, 1.0 - \|\Delta\phi\|/\pi)$; if unlocked: 0.0 |
| `demodulation_quality` | 0.10 | Demodulator status | `SUCCESS` $\to 1.0$; `NON_CONVERGED` $\to 0.35$; failure $\to 0.0$ |
| `fec_consistency` | 0.20 (0.0 if uncoded) | FEC syndrome / Trellis pass | $1.0$ if decoded successfully, else $0.0$ |
| `reencode_consistency` | 0.15 (0.0 if uncoded) | Re-encode bit mismatches | If $0$ errors $\to 1.0$; else $\max(0.0, 1.0 - \text{errors}/100)$ |
| `structural_support` | 0.10 | Bitstream structural class | Valid structure (`TEXT_LIKE`, `BYTE_STRUCT`, `BINARY_STRUCT`) $\to 0.95$; indeterminate $\to 0.40$; constant/padding $\to 0.15$ |
| `reference_support` | Dynamic (0.0 if unavailable) | Truth bit comparison BER | If available: $\max(0.0, 1.0 - 10 \cdot \text{BER})$; if unavailable: weight $0.0$ (neutral) |

---

## 4. Handling of Missing References (G1 / G5 Uncoded)

When external reference bits are unavailable:
- The system does **not** assume 0 errors or fabricate a reference sequence.
- The `reference_support` component is assigned weight $0.0$.
- Contradiction `REFERENCE_UNAVAILABLE` is recorded at severity `INFO`, applying **$0.0$ penalty**.
- Confidence is evaluated purely on observed physical, synchronization, and structural properties.

---

## 5. Output Data Schema

```json
"confidence": {
  "value": 0.935,
  "method": "deterministic_evidence_weighted_v1",
  "method_version": "phase3-v1",
  "components": {
    "physical_quality": {
      "name": "physical_quality",
      "raw_metric": 15.0,
      "normalized_score": 0.6875,
      "weight": 0.15,
      "reason": "Calculated from estimated SNR (15.0 dB).",
      "source_evidence_ids": ["EVID_PHYS_SNR_EST"]
    },
    "reencode_consistency": {
      "name": "reencode_consistency",
      "raw_metric": 0,
      "normalized_score": 1.0,
      "weight": 0.15,
      "reason": "Re-encoded payload identically matches demapped channel bits (0 bit errors).",
      "source_evidence_ids": ["EVID_REENCODE_ERRORS"]
    }
  },
  "penalties": {},
  "uncertainty_reasons": []
}
```
