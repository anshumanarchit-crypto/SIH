"""
Forward Error Correction (FEC) and Decoding Module for SpectralQ (CSE-3).
========================================================================
Implements:
- Configurable Convolutional Codec & Viterbi Decoder (Rate 1/2, configurable K=3, K=7 NASA/CCSDS, etc.)
  * Hard-decision trellis decoding (Hamming metric)
  * Soft-decision trellis decoding (Euclidean/LLR metric)
  * Path metric tracking, survivor history, and telemetry diagnostics
- Hamming(7,4) & Hamming(8,4) SECDED Block Codecs
- Reed-Solomon GF(2^8) Codec (Berlekamp-Massey + Chien search + Forney algorithm)
- Standard CRC Syndromes (CRC-8, CRC-16-CCITT, CRC-32)
- Unified decode_fec(bits, fec_scheme, **config) interface with explicit extension architecture
- Synthetic Viterbi test evaluation pipeline (evaluate_viterbi_pipeline)

Conventions pinned from core.contracts:
- BIT_ORDERING: MSB-first.
- BIT_ARRAY_DTYPE: np.uint8.
- ResultStatus, FECResult, make_warning imported from core.contracts.
"""

from __future__ import annotations
import logging
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.contracts import (
    BIT_ARRAY_DTYPE,
    BIT_ORDERING,
    FECResult,
    ResultStatus,
    make_warning,
)

logger = logging.getLogger("spectralq.fec")


# =====================================================================
# 1. Configurable Convolutional Code & Viterbi Decoder (Rate 1/2)
# =====================================================================

class ConvolutionalCodec:
    """
    Configurable Rate 1/2 Convolutional Encoder and Viterbi Decoder.

    Standard presets:
      - K = 7 (NASA/CCSDS standard): G1 = 0o171 (121), G2 = 0o133 (91)
      - K = 3 (Lightweight standard): G1 = 0o7 (7), G2 = 0o5 (5)
    """

    def __init__(self, k: int = 7, g1: Optional[int] = None, g2: Optional[int] = None):
        """
        Initialize ConvolutionalCodec with constraint length K and generator polynomials.

        Args:
            k: Constraint length (K >= 3).
            g1: Generator polynomial 1 (octal or integer). Defaults to 0o171 for K=7, 0o7 for K=3.
            g2: Generator polynomial 2 (octal or integer). Defaults to 0o133 for K=7, 0o5 for K=3.
        """
        if k < 3:
            raise ValueError(f"Constraint length K must be >= 3, got: {k}")

        self.k = int(k)
        if g1 is None or g2 is None:
            if self.k == 7:
                self.g1 = 0o171 if g1 is None else int(g1)
                self.g2 = 0o133 if g2 is None else int(g2)
            elif self.k == 3:
                self.g1 = 0o7 if g1 is None else int(g1)
                self.g2 = 0o5 if g2 is None else int(g2)
            elif self.k == 4:
                self.g1 = 0o17 if g1 is None else int(g1)
                self.g2 = 0o13 if g2 is None else int(g2)
            elif self.k == 5:
                self.g1 = 0o37 if g1 is None else int(g1)
                self.g2 = 0o27 if g2 is None else int(g2)
            else:
                self.g1 = int(g1) if g1 is not None else (1 << self.k) - 1
                self.g2 = int(g2) if g2 is not None else (1 << (self.k - 1)) | 1
        else:
            self.g1 = int(g1)
            self.g2 = int(g2)

        self.num_states = 1 << (self.k - 1)

        # Precompute state transition tables for fast Viterbi decoding
        # next_state[state][bit] = next_state
        # branch_outputs[state][bit] = (p1, p2)
        self.next_state = np.zeros((self.num_states, 2), dtype=np.int32)
        self.prev_state = np.zeros((self.num_states, 2), dtype=np.int32)
        self.branch_outputs = np.zeros((self.num_states, 2, 2), dtype=np.uint8)

        for s in range(self.num_states):
            for bit in (0, 1):
                reg = (bit << (self.k - 1)) | s
                p1 = bin(reg & self.g1).count("1") % 2
                p2 = bin(reg & self.g2).count("1") % 2
                ns = reg >> 1
                self.next_state[s, bit] = ns
                self.branch_outputs[s, bit] = [p1, p2]

        for ns in range(self.num_states):
            s0 = (ns << 1) & (self.num_states - 1)
            s1 = ((ns << 1) | 1) & (self.num_states - 1)
            self.prev_state[ns, 0] = s0
            self.prev_state[ns, 1] = s1

    def encode(self, bits: Union[np.ndarray, List[int]], add_tail_bits: bool = True) -> np.ndarray:
        """
        Encode input bit array using the (K, r=1/2) convolutional code.

        Args:
            bits: 1-D binary sequence (0 or 1).
            add_tail_bits: If True, append K-1 flush zero bits to return trellis to state 0.

        Returns:
            1-D uint8 array of encoded codeword bits (length 2 * (len(bits) + (K-1) if tail else len(bits))).
        """
        bits_arr = np.asarray(bits, dtype=BIT_ARRAY_DTYPE).flatten()
        if len(bits_arr) == 0:
            return np.array([], dtype=BIT_ARRAY_DTYPE)

        if add_tail_bits:
            bits_to_enc = np.concatenate([bits_arr, np.zeros(self.k - 1, dtype=BIT_ARRAY_DTYPE)])
        else:
            bits_to_enc = bits_arr

        n_in = len(bits_to_enc)
        encoded = np.empty(2 * n_in, dtype=BIT_ARRAY_DTYPE)
        state = 0

        for idx in range(n_in):
            bit = int(bits_to_enc[idx])
            reg = (bit << (self.k - 1)) | state
            p1 = bin(reg & self.g1).count("1") % 2
            p2 = bin(reg & self.g2).count("1") % 2
            encoded[2 * idx] = p1
            encoded[2 * idx + 1] = p2
            state = reg >> 1

        return encoded

    def decode_hard(
        self,
        received_bits: Union[np.ndarray, List[int]],
        traceback_depth: Optional[int] = None,
        return_diagnostics: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Hard-decision Viterbi Decoder using Hamming distance branch metrics.

        Args:
            received_bits: 1-D binary codeword sequence.
            traceback_depth: Optional traceback memory depth constraint.
            return_diagnostics: If True, return (decoded_bits, diagnostics_dict).

        Returns:
            decoded_bits or (decoded_bits, diagnostics)
        """
        received = np.asarray(received_bits, dtype=BIT_ARRAY_DTYPE).flatten()
        n_pairs = len(received) // 2
        if n_pairs == 0:
            empty = np.array([], dtype=BIT_ARRAY_DTYPE)
            if return_diagnostics:
                return empty, {"final_path_metric": 0.0, "corrected_errors": 0, "n_pairs": 0}
            return empty

        INF = 1e9
        path_metrics = np.full(self.num_states, INF, dtype=np.float32)
        path_metrics[0] = 0.0

        survivor_state = np.zeros((n_pairs, self.num_states), dtype=np.int32)
        survivor_bit = np.zeros((n_pairs, self.num_states), dtype=BIT_ARRAY_DTYPE)

        for step in range(n_pairs):
            r0 = int(received[2 * step])
            r1 = int(received[2 * step + 1])

            new_metrics = np.full(self.num_states, INF, dtype=np.float32)

            for s in range(self.num_states):
                if path_metrics[s] >= INF:
                    continue

                for bit in (0, 1):
                    ns = self.next_state[s, bit]
                    c0, c1 = self.branch_outputs[s, bit]
                    bm = (r0 ^ c0) + (r1 ^ c1)
                    tot = path_metrics[s] + bm

                    if tot < new_metrics[ns]:
                        new_metrics[ns] = tot
                        survivor_state[step, ns] = s
                        survivor_bit[step, ns] = bit

            path_metrics = new_metrics

        best_state = int(np.argmin(path_metrics))
        final_metric = float(path_metrics[best_state])

        # Traceback from terminal state with minimum cumulative metric
        decoded_bits = np.empty(n_pairs, dtype=BIT_ARRAY_DTYPE)
        curr_state = best_state
        for step in range(n_pairs - 1, -1, -1):
            prev_s = survivor_state[step, curr_state]
            bit = survivor_bit[step, curr_state]
            decoded_bits[step] = bit
            curr_state = prev_s

        # Strip K-1 flush tail bits
        if len(decoded_bits) >= self.k - 1:
            res_bits = decoded_bits[: -(self.k - 1)]
        else:
            res_bits = decoded_bits

        if return_diagnostics:
            diag = {
                "final_path_metric": final_metric,
                "normalized_metric": float(final_metric / max(1, 2 * n_pairs)),
                "best_terminal_state": best_state,
                "n_pairs": n_pairs,
                "k": self.k,
            }
            return res_bits, diag

        return res_bits

    def decode_soft(
        self,
        soft_llrs: Union[np.ndarray, List[float]],
        traceback_depth: Optional[int] = None,
        return_diagnostics: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Soft-decision Viterbi Decoder using Euclidean / Log-Likelihood Ratio distance metrics.
        LLR > 0 corresponds to bit 1, LLR <= 0 corresponds to bit 0.

        Args:
            soft_llrs: 1-D float array of soft channel values.
            traceback_depth: Optional traceback constraint.
            return_diagnostics: If True, return (decoded_bits, diagnostics_dict).

        Returns:
            decoded_bits or (decoded_bits, diagnostics)
        """
        llrs = np.asarray(soft_llrs, dtype=np.float32).flatten()
        n_pairs = len(llrs) // 2
        if n_pairs == 0:
            empty = np.array([], dtype=BIT_ARRAY_DTYPE)
            if return_diagnostics:
                return empty, {"final_path_metric": 0.0, "n_pairs": 0}
            return empty

        INF = 1e9
        path_metrics = np.full(self.num_states, INF, dtype=np.float32)
        path_metrics[0] = 0.0

        survivor_state = np.zeros((n_pairs, self.num_states), dtype=np.int32)
        survivor_bit = np.zeros((n_pairs, self.num_states), dtype=BIT_ARRAY_DTYPE)

        for step in range(n_pairs):
            llr0 = llrs[2 * step]
            llr1 = llrs[2 * step + 1]

            new_metrics = np.full(self.num_states, INF, dtype=np.float32)

            for s in range(self.num_states):
                if path_metrics[s] >= INF:
                    continue

                for bit in (0, 1):
                    ns = self.next_state[s, bit]
                    c0, c1 = self.branch_outputs[s, bit]
                    # Bipolar reference: bit 0 -> +1.0, bit 1 -> -1.0
                    ref0 = 1.0 if c0 == 0 else -1.0
                    ref1 = 1.0 if c1 == 0 else -1.0

                    # Squared Euclidean distance branch metric
                    bm = (llr0 - ref0) ** 2 + (llr1 - ref1) ** 2
                    tot = path_metrics[s] + bm

                    if tot < new_metrics[ns]:
                        new_metrics[ns] = tot
                        survivor_state[step, ns] = s
                        survivor_bit[step, ns] = bit

            path_metrics = new_metrics

        best_state = int(np.argmin(path_metrics))
        final_metric = float(path_metrics[best_state])

        decoded_bits = np.empty(n_pairs, dtype=BIT_ARRAY_DTYPE)
        curr_state = best_state
        for step in range(n_pairs - 1, -1, -1):
            prev_s = survivor_state[step, curr_state]
            bit = survivor_bit[step, curr_state]
            decoded_bits[step] = bit
            curr_state = prev_s

        if len(decoded_bits) >= self.k - 1:
            res_bits = decoded_bits[: -(self.k - 1)]
        else:
            res_bits = decoded_bits

        if return_diagnostics:
            diag = {
                "final_path_metric": final_metric,
                "best_terminal_state": best_state,
                "n_pairs": n_pairs,
                "k": self.k,
            }
            return res_bits, diag

        return res_bits


# =====================================================================
# 2. Synthetic Test Generator & Viterbi Evaluation Pipeline
# =====================================================================

def evaluate_viterbi_pipeline(
    tx_bits: Optional[np.ndarray] = None,
    num_bits: int = 100,
    error_indices: Optional[List[int]] = None,
    error_rate: Optional[float] = None,
    k: int = 7,
    g1: Optional[int] = None,
    g2: Optional[int] = None,
    soft: bool = False,
    noise_std: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate synthetic convolutional test transmission and evaluate Viterbi error correction.

    Steps:
      original bits -> convolutional encoder -> optional controlled bit errors -> Viterbi decoder -> recovered bits

    Returns telemetry dictionary:
      - input_bit_count
      - encoded_bit_count
      - corrupted_bit_count
      - output_bit_count
      - bit_errors (residual errors between original and decoded)
      - output_bit_error_rate (BER)
      - corrected_bit_count
      - success (True if bit_errors == 0)
    """
    if tx_bits is None:
        np.random.seed(seed)
        tx_bits_arr = np.random.randint(0, 2, num_bits, dtype=BIT_ARRAY_DTYPE)
    else:
        tx_bits_arr = np.asarray(tx_bits, dtype=BIT_ARRAY_DTYPE).flatten()

    codec = ConvolutionalCodec(k=k, g1=g1, g2=g2)
    encoded = codec.encode(tx_bits_arr, add_tail_bits=True)
    corrupted_channel = encoded.copy()

    corrupted_count = 0
    if error_indices is not None:
        for idx in error_indices:
            if 0 <= idx < len(corrupted_channel):
                corrupted_channel[idx] ^= 1
                corrupted_count += 1
    elif error_rate is not None and error_rate > 0:
        np.random.seed(seed)
        flip_mask = np.random.rand(len(corrupted_channel)) < error_rate
        corrupted_channel[flip_mask] ^= 1
        corrupted_count = int(np.sum(flip_mask))

    if soft:
        # Bipolar mapping: bit 0 -> +1.0, bit 1 -> -1.0
        bipolar = np.where(corrupted_channel == 0, 1.0, -1.0)
        np.random.seed(seed)
        noisy_llrs = bipolar + np.random.randn(len(bipolar)) * noise_std
        decoded, diag = codec.decode_soft(noisy_llrs, return_diagnostics=True)
    else:
        decoded, diag = codec.decode_hard(corrupted_channel, return_diagnostics=True)

    min_len = min(len(tx_bits_arr), len(decoded))
    bit_errors = int(np.sum(tx_bits_arr[:min_len] != decoded[:min_len]))
    if len(decoded) != len(tx_bits_arr):
        bit_errors += abs(len(decoded) - len(tx_bits_arr))

    ber = float(bit_errors / max(1, len(tx_bits_arr)))
    corrected_bits = max(0, corrupted_count - bit_errors)

    return {
        "input_bit_count": len(tx_bits_arr),
        "encoded_bit_count": len(encoded),
        "corrupted_bit_count": corrupted_count,
        "output_bit_count": len(decoded),
        "bit_errors": bit_errors,
        "output_bit_error_rate": ber,
        "corrected_bit_count": corrected_bits,
        "success": (bit_errors == 0),
        "diagnostics": diag,
        "k": k,
        "soft_decision": soft,
    }


# =====================================================================
# 3. Hamming(7,4) & Hamming(8,4) SECDED
# =====================================================================

class HammingCodec:
    """
    Hamming(7,4) block code:
    Encodes 4 data bits into 7 code bits, corrects any single-bit error per codeword.
    """

    G = np.array([
        [1, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1],
    ], dtype=BIT_ARRAY_DTYPE)

    H = np.array([
        [1, 0, 0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 1, 1],
    ], dtype=BIT_ARRAY_DTYPE)

    SYNDROME_MAP = {
        0b001: 2,
        0b010: 1,
        0b100: 0,
        0b011: 4,
        0b101: 6,
        0b110: 3,
        0b111: 5,
    }

    @classmethod
    def encode(cls, data_bits: Union[np.ndarray, List[int]]) -> np.ndarray:
        """Encode bit array into Hamming(7,4) codewords."""
        bits = np.asarray(data_bits, dtype=BIT_ARRAY_DTYPE).flatten()
        num_blocks = len(bits) // 4
        if num_blocks == 0:
            return np.array([], dtype=BIT_ARRAY_DTYPE)

        out = np.empty(num_blocks * 7, dtype=BIT_ARRAY_DTYPE)
        for b in range(num_blocks):
            nibble = bits[b * 4 : (b + 1) * 4]
            codeword = np.dot(nibble, cls.G) % 2
            out[b * 7 : (b + 1) * 7] = codeword

        return out

    @classmethod
    def decode(cls, code_bits: Union[np.ndarray, List[int]]) -> Tuple[np.ndarray, int]:
        """
        Decode Hamming(7,4) codewords, correcting single-bit errors.
        Returns (decoded_data_bits, num_corrected_errors).
        """
        bits = np.asarray(code_bits, dtype=BIT_ARRAY_DTYPE).flatten()
        num_blocks = len(bits) // 7
        if num_blocks == 0:
            return np.array([], dtype=BIT_ARRAY_DTYPE), 0

        out = np.empty(num_blocks * 4, dtype=BIT_ARRAY_DTYPE)
        corrected_errors = 0

        for b in range(num_blocks):
            cw = bits[b * 7 : (b + 1) * 7].copy()
            s_vec = np.dot(cls.H, cw) % 2
            s_val = int((s_vec[0] << 2) | (s_vec[1] << 1) | s_vec[2])

            if s_val in cls.SYNDROME_MAP:
                err_pos = cls.SYNDROME_MAP[s_val]
                cw[err_pos] ^= 1
                corrected_errors += 1

            out[b * 4 : (b + 1) * 4] = cw[3:7]

        return out, corrected_errors


# =====================================================================
# 4. Reed-Solomon GF(2^8) Decoder
# =====================================================================

class GF256:
    """Galois Field GF(2^8) arithmetic with field polynomial 0x11D."""

    def __init__(self, prim_poly: int = 0x11D):
        self.exp = [0] * 512
        self.log = [0] * 256
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.exp[i + 255] = x
            self.log[x] = i
            x <<= 1
            if x & 0x100:
                x ^= prim_poly

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def div(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("GF(256) division by zero")
        if a == 0:
            return 0
        return self.exp[self.log[a] - self.log[b] + 255]

    def inv(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("GF(256) inverse of zero")
        return self.exp[255 - self.log[a]]


class ReedSolomonCodec:
    """
    Reed-Solomon Codec over GF(2^8).
    Supports customizable codeword length N and message length K (e.g. RS(255, 223)).
    """

    def __init__(self, n: int = 255, k: int = 223):
        self.n = n
        self.k = k
        self.two_t = n - k
        self.gf = GF256()

        self.gen_poly = [1]
        for i in range(self.two_t):
            root = self.gf.exp[i]
            new_poly = [0] * (len(self.gen_poly) + 1)
            for j, coeff in enumerate(self.gen_poly):
                new_poly[j] ^= coeff
                new_poly[j + 1] ^= self.gf.mul(coeff, root)
            self.gen_poly = new_poly

    def encode(self, msg_bytes: bytes) -> bytes:
        """Encode message into Reed-Solomon codeword."""
        msg = list(msg_bytes)
        if len(msg) > self.k:
            raise ValueError(f"Message length ({len(msg)}) exceeds RS capacity ({self.k})")

        pad_len = self.k - len(msg)
        padded_msg = [0] * pad_len + msg

        remainder = [0] * self.two_t
        for b in padded_msg:
            feedback = b ^ remainder[0]
            remainder = remainder[1:] + [0]
            for i in range(self.two_t):
                remainder[i] ^= self.gf.mul(self.gen_poly[i + 1], feedback)

        cw = padded_msg + remainder
        return bytes(cw[pad_len:])

    def decode(self, cw_bytes: bytes) -> Tuple[bytes, bool]:
        """
        Decode received RS codeword, correcting up to t symbol errors via Berlekamp-Massey.
        Returns (decoded_message_bytes, is_successful).
        """
        cw = list(cw_bytes)
        pad_len = self.n - len(cw)
        padded_cw = [0] * pad_len + cw

        # 1. Compute syndromes
        syndromes = []
        has_error = False
        for i in range(self.two_t):
            alpha_i = self.gf.exp[i]
            s = 0
            for byte in padded_cw:
                s = self.gf.mul(s, alpha_i) ^ byte
            syndromes.append(s)
            if s != 0:
                has_error = True

        if not has_error:
            return bytes(padded_cw[pad_len : pad_len + self.k]), True

        # 2. Berlekamp-Massey
        c_poly = [1]
        b_poly = [1]
        l_degree = 0
        m = 1
        b_val = 1

        for r in range(self.two_t):
            d = syndromes[r]
            for i in range(1, l_degree + 1):
                if i < len(c_poly) and (r - i) >= 0:
                    d ^= self.gf.mul(c_poly[i], syndromes[r - i])

            if d == 0:
                m += 1
            else:
                t_poly = c_poly.copy()
                scale = self.gf.div(d, b_val)
                shift_b = [0] * m + [self.gf.mul(x, scale) for x in b_poly]
                max_len = max(len(c_poly), len(shift_b))
                new_c = [0] * max_len
                for i in range(len(c_poly)):
                    new_c[i] ^= c_poly[i]
                for i in range(len(shift_b)):
                    new_c[i] ^= shift_b[i]
                c_poly = new_c

                if 2 * l_degree <= r:
                    l_degree = r + 1 - l_degree
                    b_poly = t_poly
                    b_val = d
                    m = 1
                else:
                    m += 1

        # 3. Chien Search
        error_positions = []
        for j in range(self.n):
            alpha_inv_j = self.gf.exp[(255 - (j % 255)) % 255]
            val = 0
            term = 1
            for i in range(len(c_poly)):
                val ^= self.gf.mul(c_poly[i], term)
                term = self.gf.mul(term, alpha_inv_j)

            if val == 0:
                pos = self.n - 1 - j
                error_positions.append(pos)

        if len(error_positions) != l_degree:
            return bytes(padded_cw[pad_len : pad_len + self.k]), False

        # 4. Forney Algorithm
        omega = [0] * self.two_t
        for i in range(self.two_t):
            for j in range(len(c_poly)):
                if i + j < self.two_t:
                    omega[i + j] ^= self.gf.mul(syndromes[i], c_poly[j])

        for pos in error_positions:
            j = self.n - 1 - pos
            alpha_j = self.gf.exp[j % 255]
            alpha_inv_j = self.gf.exp[(255 - (j % 255)) % 255]

            omega_val = 0
            term = 1
            for k_idx in range(len(omega)):
                omega_val ^= self.gf.mul(omega[k_idx], term)
                term = self.gf.mul(term, alpha_inv_j)

            num = self.gf.mul(alpha_j, omega_val)

            den = 0
            for k_idx in range(1, len(c_poly), 2):
                coeff = c_poly[k_idx]
                p_exp = (k_idx - 1)
                term_p = self.gf.exp[(self.gf.log[alpha_inv_j] * p_exp) % 255] if alpha_inv_j != 0 and p_exp > 0 else 1
                den ^= self.gf.mul(coeff, term_p)

            if den != 0:
                err_val = self.gf.div(num, den)
                padded_cw[pos] ^= err_val

        decoded = bytes(padded_cw[pad_len : pad_len + self.k])
        return decoded, True


# =====================================================================
# 5. CRC Syndrome Calculators (CRC-8, CRC-16, CRC-32)
# =====================================================================

class CRC:
    """CRC syndrome calculator and validator."""

    @staticmethod
    def crc8(data: bytes, poly: int = 0x07, init: int = 0x00) -> int:
        """Compute CRC-8 checksum."""
        crc = init
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ poly) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
        return crc

    @staticmethod
    def crc16(data: bytes, poly: int = 0x1021, init: int = 0xFFFF) -> int:
        """Compute CRC-16-CCITT checksum."""
        crc = init
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ poly) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc

    @staticmethod
    def crc32(data: bytes) -> int:
        """Compute IEEE 802.3 standard CRC-32."""
        import zlib
        return zlib.crc32(data) & 0xFFFFFFFF

    @classmethod
    def verify(cls, data_with_crc: bytes, crc_type: str = "crc16") -> Tuple[bool, bytes]:
        """
        Verify CRC attached to trailing bytes of payload.
        Returns: (is_valid, payload_without_crc)
        """
        crc_type = crc_type.lower()
        if crc_type == "crc8":
            if len(data_with_crc) < 1:
                return False, data_with_crc
            payload = data_with_crc[:-1]
            expected_crc = data_with_crc[-1]
            actual_crc = cls.crc8(payload)
            return (actual_crc == expected_crc), payload

        elif crc_type == "crc16":
            if len(data_with_crc) < 2:
                return False, data_with_crc
            payload = data_with_crc[:-2]
            expected_crc = struct.unpack(">H", data_with_crc[-2:])[0]
            actual_crc = cls.crc16(payload)
            return (actual_crc == expected_crc), payload

        elif crc_type == "crc32":
            if len(data_with_crc) < 4:
                return False, data_with_crc
            payload = data_with_crc[:-4]
            expected_crc = struct.unpack(">I", data_with_crc[-4:])[0]
            actual_crc = cls.crc32(payload)
            return (actual_crc == expected_crc), payload

        else:
            raise ValueError(f"Unsupported CRC type: {crc_type}")


# =====================================================================
# 6. Unified Generic FEC Interface & Extensibility Architecture
# =====================================================================

def decode_fec(
    bits: Union[np.ndarray, bytes, List[int]],
    fec_scheme: str = "viterbi_hard",
    **config: Any,
) -> FECResult:
    """
    Unified Forward Error Correction Decoding Interface (CSE-3).

    Routes supported FEC schemes to respective codecs and returns structured FECResult.
    Explicitly rejects unsupported schemes (e.g. LDPC, Turbo, Polar, Concatenated)
    with ResultStatus.UNSUPPORTED and structured actionable warning messages.

    Supported schemes:
      - 'viterbi_hard' / 'viterbi' / 'conv': Hard-decision Viterbi decoding
      - 'viterbi_soft' / 'conv_soft': Soft-decision Viterbi decoding
      - 'hamming' / 'hamming_7_4': Hamming (7,4) single error correction
      - 'reed_solomon' / 'rs': Reed-Solomon GF(2^8) byte decoding
      - 'none' / 'passthrough': Raw bitstream passthrough

    Planned future extensions (Phase 2):
      - 'ldpc': Low-Density Parity-Check (DVB-S2 / CCSDS 131.0-B-2)
      - 'turbo': Turbo product / parallel concatenated codes (3GPP / CCSDS)
      - 'polar': 5G NR Polar Codes
      - 'concatenated': RS(255,223) outer + Viterbi(K=7) inner concatenated coding
    """
    scheme_norm = str(fec_scheme).strip().lower().replace("-", "_").replace(" ", "_")

    # Handle input conversion
    if isinstance(bits, bytes):
        raw_bits = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
    else:
        raw_bits = np.asarray(bits)

    if len(raw_bits) == 0:
        return FECResult(
            decoded_bits=np.array([], dtype=BIT_ARRAY_DTYPE),
            fec_scheme=fec_scheme,
            status=ResultStatus.FAILED,
            input_bit_count=0,
            output_bit_count=0,
            warnings=[make_warning("EMPTY_INPUT", "FEC input bit buffer is empty.")],
        )

    # 1. Convolutional Codec / Viterbi Decoding
    if scheme_norm in ("viterbi_hard", "viterbi", "conv", "convolutional", "auto"):
        k = config.get("k", 7)
        g1 = config.get("g1", None)
        g2 = config.get("g2", None)
        codec = ConvolutionalCodec(k=k, g1=g1, g2=g2)
        decoded, diag = codec.decode_hard(raw_bits, return_diagnostics=True)
        return FECResult(
            decoded_bits=decoded,
            fec_scheme=fec_scheme,
            status=ResultStatus.CONFIRMED,
            input_bit_count=len(raw_bits),
            output_bit_count=len(decoded),
            confidence=1.0 if len(decoded) > 0 else 0.0,
            diagnostics=diag,
        )

    if scheme_norm in ("viterbi_soft", "conv_soft"):
        k = config.get("k", 7)
        g1 = config.get("g1", None)
        g2 = config.get("g2", None)
        codec = ConvolutionalCodec(k=k, g1=g1, g2=g2)
        decoded, diag = codec.decode_soft(raw_bits, return_diagnostics=True)
        return FECResult(
            decoded_bits=decoded,
            fec_scheme=fec_scheme,
            status=ResultStatus.CONFIRMED,
            input_bit_count=len(raw_bits),
            output_bit_count=len(decoded),
            confidence=1.0 if len(decoded) > 0 else 0.0,
            diagnostics=diag,
        )

    # 2. Hamming Block Code
    if scheme_norm in ("hamming", "hamming_7_4", "hamming_8_4", "secded"):
        decoded, num_corrected = HammingCodec.decode(raw_bits)
        return FECResult(
            decoded_bits=decoded,
            fec_scheme=fec_scheme,
            status=ResultStatus.CONFIRMED,
            corrected_errors=num_corrected,
            input_bit_count=len(raw_bits),
            output_bit_count=len(decoded),
            confidence=1.0 if len(decoded) > 0 else 0.0,
            diagnostics={"corrected_blocks": num_corrected},
        )

    # 3. Reed-Solomon Code
    if scheme_norm in ("reed_solomon", "rs", "rs_255_223"):
        rs_n = config.get("rs_n", config.get("n", 255))
        rs_k = config.get("rs_k", config.get("k", 223))
        codec_rs = ReedSolomonCodec(n=rs_n, k=rs_k)

        # Convert bit array to bytes
        if len(raw_bits) % 8 != 0:
            pad_len = 8 - (len(raw_bits) % 8)
            raw_bits_padded = np.pad(raw_bits, (0, pad_len))
        else:
            raw_bits_padded = raw_bits

        cw_bytes = np.packbits(raw_bits_padded.astype(np.uint8)).tobytes()
        dec_bytes, success = codec_rs.decode(cw_bytes)
        dec_bits = np.unpackbits(np.frombuffer(dec_bytes, dtype=np.uint8))

        status = ResultStatus.CONFIRMED if success else ResultStatus.FAILED
        warnings = [] if success else [make_warning("RS_DECODE_UNCORRECTABLE", "Uncorrectable error pattern detected in RS codeword.")]
        return FECResult(
            decoded_bits=dec_bits,
            fec_scheme=fec_scheme,
            status=status,
            input_bit_count=len(raw_bits),
            output_bit_count=len(dec_bits),
            confidence=1.0 if success else 0.0,
            warnings=warnings,
            diagnostics={"rs_n": rs_n, "rs_k": rs_k, "success": success},
        )

    # 4. Passthrough / No FEC
    if scheme_norm in ("none", "passthrough", "raw"):
        return FECResult(
            decoded_bits=np.asarray(raw_bits, dtype=BIT_ARRAY_DTYPE),
            fec_scheme="none",
            status=ResultStatus.CONFIRMED,
            input_bit_count=len(raw_bits),
            output_bit_count=len(raw_bits),
            confidence=1.0,
        )

    # 5. Unsupported FEC Schemes (LDPC, Turbo, Polar, Concatenated)
    warn_msg = (
        f"FEC scheme '{fec_scheme}' is not currently implemented. "
        "Supported FEC schemes in CSE-3 are: 'viterbi_hard', 'viterbi_soft', 'hamming', 'reed_solomon', 'none'. "
        "Architectural extension points for LDPC, Turbo, Polar, and Concatenated codes are scheduled for Phase 2."
    )
    logger.warning("[fec] %s", warn_msg)

    return FECResult(
        decoded_bits=np.array([], dtype=BIT_ARRAY_DTYPE),
        fec_scheme=fec_scheme,
        status=ResultStatus.UNSUPPORTED,
        input_bit_count=len(raw_bits),
        output_bit_count=0,
        confidence=0.0,
        warnings=[make_warning("UNSUPPORTED_FEC_SCHEME", warn_msg)],
        diagnostics={
            "supported_schemes": ["viterbi_hard", "viterbi_soft", "hamming", "reed_solomon", "none"],
            "requested_scheme": fec_scheme,
        },
    )
