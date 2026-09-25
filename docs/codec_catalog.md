# SpectralQ Codec Catalog & Specifications

This catalog documents the Forward Error Correction (FEC) codes and interleaving schemes implemented in the SpectralQ decoder core.

---

## 1. Convolutional Code (NASA/ESA Standard Rate 1/2)

- **Target Pipelines**: G2, G5 (inner), G6, G7
  - **G7 Configuration**: QPSK + $K=7$ convolutional, identity/no-interleaver configuration (256 source bits $\rightarrow 524$ TX bits).
- **Constraint Length**: $K = 7$ (Memory $m = 6$ registers)
- **Code Rate**: $R = 1/2$
- **Generator Polynomials**:
  - $G_1 = 171_8 = 1111001_2$ ($1 + D + D^2 + D^3 + D^6$)
  - $G_2 = 133_8 = 1011011_2$ ($1 + D^2 + D^3 + D^5 + D^6$)
- **Termination Policy**:
  - Terminated encoding: exactly $K - 1 = 6$ zero tail bits appended to return the encoder to the all-zero state.
  - Output length for message length $L$: $N_{coded} = 2 \times (L + 6)$ bits.
- **Decoder Architecture**: Viterbi decoder
  - Metric options: Hard-decision (Hamming distance) and Soft-decision (Euclidean / LLR)
  - Traceback Depth Strategy:
    - Standard/long frames ($L \ge 64$): $D_{tb} \ge 5 \times K \approx 35 \text{ bits}$.
    - Short frames ($L < 64$): adaptive depth $D_{tb} = \min(35, L + 6)$ without arbitrary slicing.
  - Tail-bit Truncation: Viterbi decoder explicitly trims the 6 tail bits from recovered source bits.
- **Underlying Engine**: `commpy.channelcoding.convcode` (with local wrapper in `spectralq.fec`).

---

## 2. Reed-Solomon Code: RS(255, 223)

- **Target Pipelines**: G3, G5 (outer)
- **Galois Field**: $\text{GF}(2^8)$ with primitive polynomial $p(x) = x^8 + x^4 + x^3 + x^2 + 1$ ($0x11D$)
- **Block Length ($n$)**: 255 symbols (bytes) = 2,040 bits
- **Message Length ($k$)**: 223 symbols (bytes) = 1,784 bits
- **Parity Symbols ($2t$)**: 32 symbols (bytes) = 256 bits
- **Error Correction Capability**: $t = 16$ symbol errors per block
- **Code Rate**: $R = 223/255 \approx 0.8745$
- **Underlying Engine**: `reedsolo.RSCodec(32)`
- **Byte-Orientation & Packing Specification**:
  1. **Bit-to-Byte Packing**: Convert bitstream arrays into `bytes` sequences (MSB-first packing, requiring $L \pmod 8 == 0$) before FEC encoding or decoding.
  2. **Byte-to-Bit Unpacking**: Unpack recovered `bytes` sequences back into binary bit arrays ($[0, 1]$).
- **Validation & Beyond-Capability Verification**:
  - Verified for 0, 1, 8, and 16 symbol errors.
  - For uncorrectable errors (>16 symbol errors): do not rely solely on exceptions. If the backend returns a candidate block, re-encode it and compare against the received word to detect false decodes, marking `decoder_success = False`.

---

## 3. Low-Density Parity-Check (LDPC)

- **Target Pipeline**: G4 (16-QAM)
- **Code Specification**: Gallager $(96, 3, 963)$ parity-check matrix:
  - Block length: $N = 96$ variable nodes
  - Parity-check constraints: $M = 48$ check nodes
  - Parity-check rank over $\text{GF}(2)$: $\text{rank}(H) = 46$ (2 redundant check rows)
  - Information bit capacity: $K = N - \text{rank}(H) = 96 - 46 = 50$ bits
  - Effective Code Rate: $R = 50/96 \approx 0.5208$
- **Systematic Encoder Construction**:
  - CommPy's `triang_ldpc_systematic_encode` requires lower-triangular parity checks and fails on Gallager codes.
  - `spectralq.fec` constructs a systematic $\text{GF}(2)$ RREF generator mapping:
    - Information bits placed in the 50 free non-pivot columns: $c[\text{free\_cols}] = m$.
    - Parity bits in the 46 pivot columns solved via back-substitution: $c[\text{pivot\_cols}] = (M_{\text{RREF}}[\text{free\_cols}] \cdot m) \pmod 2$.
    - Guarantees $H \cdot c = 0 \pmod 2$.
- **Decoder Architecture**: Belief Propagation (BP) via Min-Sum Algorithm (MSA) or Sum-Product Algorithm (SPA)
  - Iteration limit: 20 to 30 iterations
  - Under channel-free conditions: achieves exact bit-for-bit source recovery.
  - Corrects controlled bit-flip channel noise.

---

## 4. Concatenated Code (Outer RS + Inner Convolutional)

- **Target Pipeline**: G5 (2-FSK)
- **Outer Code**: Reed-Solomon RS(255, 223) (223 bytes / 1,784 bits $\rightarrow$ 255 bytes / 2,040 bits)
- **Inner Code**: Convolutional $K=7$, rate 1/2 ($171_8 / 133_8$) with 6 tail bits ($2 \times (2040 + 6) = 4,092$ bits)
- **Interleaver**: Convolutional/cross interleaver between stages to disperse Viterbi burst errors.
- **Decoding Order**: Received bits $\rightarrow$ de-interleave $\rightarrow$ Viterbi decode $\rightarrow$ trim tail bits $\rightarrow$ pack to bytes $\rightarrow$ RS decode $\rightarrow$ recovered source bits.

---

## 5. Interleaver Catalog & Queue Models

| Scheme | Architecture | Parameters | Exact Inversion & Metadata Contract |
|---|---|---|---|
| **Block** | Matrix fill/readout with padding | Rows, Cols | Row-write, Column-read. Preserves `original_length`, `padded_length`, `padding_length`. Inverse writes column-wise, reads row-wise, strips documented padding. |
| **Convolutional / Cross** | Forney multi-branch FIFO shift registers | Branches $B$, Delay step $M$ | Explicit branch delay lines $\text{delay}[r] = r \times M$. Flush length determined dynamically by queue model. Inverse uses complementary branches $(B - 1 - r) \times M$ and removes initial startup delay. |
| **Diagonal** | 2D anti-diagonal grid traversal | Rows, Cols | Row-wise grid fill, anti-diagonal $(r + c = s)$ readout. Preserves `original_length`, `padded_length`, `padding_length`. |
| **Pseudo-Random** | Deterministic permutation | Seed $S$, Length $L$ | Strictly reproducible: `P = rng(seed).permutation(L)`. Inverse permutation $P^{-1}[P[i]] = i$. |
| **None / Identity** | Direct pass-through | None | Strict identity mapping: $\text{TX} = \text{FEC}$. Preserves `original_length`, `padded_length = original_length`, `padding_length = 0`. Inverse returns input bits directly without padding or modification. |
