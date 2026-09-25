# SpectralQ Decoder Interface Contract (Draft)

## 1. Scope & System Boundary

This document defines the draft architectural contract for the Python decoder-core component of the **SpectralQ (SIH26147)** pipeline.

### Team Responsibilities Boundary
- **Arpit (This Component)**: Interleaving/de-interleaving, demodulation, FEC encoding/decoding, stable decoder facade, and bit-stream intelligence.
- **Archit**: Downstream hypothesis engine, evidence aggregation, computed confidence score, UNKNOWN decision, and final `result.json`.
- **Sinchana**: GNU Octave DSP front-end, signal estimation, burst analysis, signal generation, Octave-side feature extraction (`analysis.json`, `capture.cf32`).
- **Harsh**: Modulation-classifier training, benchmark sweeps, accuracy evaluation.
- **Himanshu**: Streamlit GUI.
- **Prince**: Presentation and submission artifacts.

> [!IMPORTANT]
> **Boundary Rule**: The decoder core facade MUST NOT compute the overall confidence metric or make the final UNKNOWN classification. Those decisions belong strictly to Archit's hypothesis/evidence aggregation layer.

---

## 2. Decoder Architecture & Dataflow

The decoder core conceptually exposes a single, stable facade (`DecoderPipeline` in `spectralq.decoder_api`).

```
Complex IQ Waveform (complex64 / cf32) + DecoderConfig
                         │
                         ▼
        ┌──────────────────────────────────┐
        │       Demodulation (demod)       │
        │  RRC Filter, Gardner TED, Costas │
        └──────────────────────────────────┘
                         │
                         ▼
          Demodulated Bits (RECEIVED BITS)
      (Phase 1 testing: ideal channel bits;
       Phase 2 pipeline: demod.py output)
                         │
                         ▼
        ┌──────────────────────────────────┐
        │    De-interleaving (interleave)  │
        │  Block, Diagonal, PRNG, Forney   │
        └──────────────────────────────────┘
                         │
                         ▼
              De-interleaved Coded Bits
                         │
                         ▼
        ┌──────────────────────────────────┐
        │        FEC Decoding (fec)        │
        │  Viterbi (Conv), RS(255,223),    │
        │  LDPC (BP/MSA), Concatenated     │
        └──────────────────────────────────┘
                         │
                         ▼
            Recovered Source Bits (SOURCE BITS)
                         │
                         ▼
        ┌──────────────────────────────────┐
        │   Bit-stream Intel (bitintel)    │
        │  Sync words, frames, CRC scans   │
        └──────────────────────────────────┘
                         │
                         ▼
         DecoderResult (Structured Evidence)
```

### Internal Chain Orchestration APIs (`encode_chain.py`)
- `encode_chain(source_bits, config)`: Orchestrates `source_bits` $\rightarrow$ FEC encoding $\rightarrow$ Interleaving $\rightarrow$ `tx_bits` (TX BITS).
- `decode_chain(received_bits, config)`: Orchestrates `received_bits` (RECEIVED BITS) $\rightarrow$ De-interleaving $\rightarrow$ FEC decoding $\rightarrow$ `recovered_source_bits`.
  *Note*: In Phase 1 verification, no-noise channel tests pass ideal TX bits directly into `received_bits`. In Phase 2, `received_bits` will be produced by demodulation of IQ waveforms.

---

## 3. Typed Result Contract

The public facade exposes typed dataclasses rather than arbitrary tuples or unstructured dictionaries.

78: ### DecoderResult Schema
79: - `status` (`DecoderStatus`): High-level outcome state (`SUCCESS`, `UNSUPPORTED`, `INVALID_INPUT`, `DECODER_FAILURE`, `LOW_QUALITY`, `NON_CONVERGED`).
80: - `modulation` (`ModulationType`): Demodulated modulation scheme (e.g. `BPSK`, `QPSK`, `8-PSK`, `16-QAM`, `64-QAM`, `2-FSK`, `4-FSK`).
81: - `fec_type` (`FECType`): Applied forward error correction scheme.
82: - `interleaver_type` (`InterleaverType`): Applied de-interleaver scheme (`BLOCK`, `DIAGONAL`, `PSEUDORANDOM`, `CONVOLUTIONAL`, `NONE`).
83: - `bit_count` (`int`): Total number of recovered source bits.
84: - `symbols` (`Optional[np.ndarray]`): Complex or real constellation symbols after carrier/timing recovery.
85: - `hard_bits` (`Optional[np.ndarray]`): Demodulated hard decision bits prior to FEC.
86: - `soft_bits` (`Optional[np.ndarray]`): Demodulated soft LLRs (where supported by demodulator).
87: - `recovered_source_bits` (`Optional[np.ndarray]`): Post-FEC recovered information bits.
88: - `timing_status` (`Dict[str, Any]`): Gardner TED lock status, fractional timing offset, jitter metrics.
89: - `carrier_status` (`Dict[str, Any]`): Costas loop lock status, estimated frequency offset, residual phase error.
90: - `decoder_success` (`bool`): Boolean flag indicating whether FEC decoded cleanly without uncorrected errors.
91: - `convergence` (`bool`): Iterative convergence flag (primarily for LDPC/turbo).
92: - `iterations` (`int`): Iteration count reached during decoding.
93: - `syndrome_status` (`Optional[bool]`): Parity check / syndrome validity (`True` = all syndromes zero).
94: - `crc_info` (`Dict[str, Any]`): Discovered CRC matches, polynomial names, syndrome validity, residual bit-error evidence.
95: - `diagnostics` (`Dict[str, Any]`): Execution timings, SNR/EVM estimates, traceback depth, soft metric statistics.
96: - `warnings` (`List[str]`): Traceable non-fatal warnings encountered during processing.
97:
98: ### Demodulation Result Contract (`DemodResult` in `spectralq.demod`)
99: The standalone demodulator output contract provides factual DSP recovery metrics without computing downstream confidence:
100: - `status` (`DemodStatus`): `SUCCESS`, `UNSUPPORTED`, `INVALID_INPUT`, `DECODER_FAILURE`, `LOW_QUALITY`, `NON_CONVERGED`.
101: - `modulation` (`str`): Demodulation scheme string (e.g. `"QPSK"`, `"2-FSK"`).
102: - `input_sample_count` (`int`): Length of input complex sample vector.
103: - `output_symbol_count` (`int`): Number of recovered symbols.
104: - `output_bit_count` (`int`): Number of recovered hard bits.
105: - `samples_per_symbol_used` (`float`): Samples per symbol configured/used.
106: - `timing_status` (`Dict[str, Any]`): `converged`, `jitter`, `mean_step`, `final_delay`.
107: - `carrier_status` (`Dict[str, Any]`): `converged`, `coarse_cfo_rad`, `fine_cfo_rad`, `mean_residual_phase_rad`, `phase_ambiguity_rad`.
108: - `estimated_frequency_offset` (`float`): Estimated normalized carrier frequency offset in radians/sample.
109: - `estimated_phase_offset` (`float`): Estimated residual phase offset in radians.
110: - `timing_error_summary` (`Dict[str, Any]`): `converged`, `mean_error`, `rms_error`.
111: - `symbol_decisions` (`np.ndarray`): Recovered complex or real constellation symbols.
112: - `hard_bits` (`np.ndarray`): Sliced bit stream (uint8).
113: - `soft_bits` (`Optional[np.ndarray]`): Demodulated soft LLR decisions (optional).
114: - `diagnostic_metrics` (`Dict[str, Any]`): Timing jitter, EVM estimate, execution time, etc.
115: - `warnings` (`List[str]`): Diagnostic warnings.
116: - `failure_reason` (`Optional[str]`): Description of failure when status != SUCCESS.
117:
118: ---
119:
120: ## 4. Module Responsibility Matrix
121:
122: | Module | Exact Responsibilities | Strict Exclusions |
123: |---|---|---|
124: | `interleave.py` | ONLY interleaving and de-interleaving algorithms (Block, Diagonal, Deterministic PRNG, Convolutional/Forney, Identity/NONE). | No demodulation, no FEC codecs, no framing logic. |
125: | `fec.py` | ONLY FEC-related codecs and adapters: Convolutional encoding/Viterbi decoding, Reed-Solomon RS(255,223) with byte packing/unpacking, LDPC belief-propagation, and concatenated codecs. | No waveform processing, no interleaving algorithms, no sync word hunting. |
126: | `demod.py` | ONLY waveform/symbol recovery: RRC matched filtering, Gardner timing recovery, Costas carrier recovery, constellation slicing and LLR generation for PSK, QAM, and FSK. | No FEC decoding, no de-interleaving, no truth-data evaluation. |
127: | `decoder_api.py` | Stable public facade orchestrating the decoder pipeline (`DecoderPipeline.decode`); enforces input validation and returns typed `DecoderResult`. | No confidence scoring, no UNKNOWN classification. |
128: | `bitintel.py` | Post-decoding structure analysis: sync/preamble mining, frame candidate carving, multi-polynomial CRC candidate scanning, header/payload mapping, and cross-burst consistency. | No waveform recovery, no raw FEC algorithms. |
129:
130: ---
131:
132: ## 5. Error Semantics & Robustness Rules
133:
134: The decoder must distinguish failures accurately to supply clean evidence to Archit's hypothesis engine:
135:
136: 1. **`SUCCESS`**: Pipeline completed all stages with verified FEC convergence or zero syndrome errors.
137: 2. **`UNSUPPORTED`**: Requested combination of modulation/FEC/interleaver is valid in theory but not supported by the codec catalog.
138: 3. **`INVALID_INPUT`**: Input waveform is malformed (e.g. empty array, NaN/Inf values, incorrect dimensions, incompatible sample rate).
139: 4. **`DECODER_FAILURE`**: Demodulation succeeded, but FEC decoding was unable to correct errors or syndromes failed.
140: 5. **`LOW_QUALITY`**: Waveform SNR or synchronization metrics fall below demodulation operating limits (e.g. Costas loop unlocked, Gardner divergence).
141: 6. **`NON_CONVERGED`**: Demodulator timing or carrier recovery loops failed to lock within tolerance.
142:
143: ### Strict Prohibitions
144: - **DO NOT** represent all failures as empty arrays (`np.array([])`). Set appropriate `DecoderStatus` and provide diagnostic context.
145: - **DO NOT** silently fall back from one codec or interleaver to another without explicit pipeline configuration.
146: - **DO NOT** return the raw demodulated bits as "recovered source bits" when FEC decoding fails.
147: - **DO NOT** compute final confidence scores or output `UNKNOWN` decisions in `DecoderPipeline` (reserved for Archit's hypothesis engine).
