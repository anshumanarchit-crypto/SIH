"""
spectralq.fec

Forward Error Correction (FEC) codecs and adapters for the SpectralQ decoder core.

Supported Codecs:
1. Convolutional Codec (Rate 1/2, K=7, [171, 133] octal, terminated with 6 tail bits).
2. Reed-Solomon Codec: RS(255, 223) over GF(2^8) byte-oriented with bit/byte packing.
3. LDPC Codec: Systematic GF(2) Gallager (96, 3, 963) codec with Belief Propagation (MSA/SPA).
4. Concatenated Codec: RS(255, 223) outer + Convolutional inner chain.
"""

from abc import ABC, abstractmethod
import os
from typing import Tuple, Dict, Any, Optional, Sequence, Union
import numpy as np
import reedsolo
from commpy.channelcoding.convcode import Trellis, conv_encode as commpy_conv_encode, viterbi_decode as commpy_viterbi_decode
import commpy
import commpy.channelcoding.ldpc as commpy_ldpc


# ============================================================================
# BIT / BYTE CONVERSION UTILITIES
# ============================================================================

def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Convert binary 1D bit array ([0, 1]) to packed bytes (MSB first).

    Parameters
    ----------
    bits : np.ndarray
        1D array of bits. Length must be a multiple of 8.

    Returns
    -------
    data : bytes
        Packed byte sequence.
    """
    bits_arr = np.asarray(bits, dtype=int).ravel()
    if len(bits_arr) % 8 != 0:
        raise ValueError(f"Bit length must be a multiple of 8, got {len(bits_arr)}")
    return np.packbits(bits_arr).tobytes()


def bytes_to_bits(data: Union[bytes, bytearray, Any]) -> np.ndarray:
    """Unpack a byte sequence to a 1D binary numpy array of int (0 and 1).

    Parameters
    ----------
    data : Union[bytes, bytearray, Any]
        Packed byte sequence.

    Returns
    -------
    bits : np.ndarray
        1D array of unpacked bits.
    """
    raw_bytes = bytes(data)
    return np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8)).astype(int)


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================

class BaseFEC(ABC):
    """Abstract base class for all forward error correction codecs."""

    @abstractmethod
    def encode(self, source_bits: np.ndarray) -> np.ndarray:
        """Encode source bits into coded bits."""
        pass

    @abstractmethod
    def decode(
        self,
        coded_bits: np.ndarray,
        soft_llrs: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode coded bits (or soft LLRs) into recovered source bits and diagnostic metrics."""
        pass


# ============================================================================
# CONVOLUTIONAL CODEC & VITERBI DECODER
# ============================================================================

class ConvolutionalCodec(BaseFEC):
    """Rate 1/2, K=7 Convolutional encoder and Viterbi decoder.

    Standard polynomials:
        G1 = 0o171 (1111001)
        G2 = 0o133 (1011011)

    Termination:
        Appends K - 1 = 6 zero tail bits to terminate the trellis into the zero state.
        For source bit length L, coded output length is 2 * (L + 6).
    """

    def __init__(
        self,
        constraint_length: int = 7,
        polynomials: Tuple[int, ...] = (0o171, 0o133)
    ):
        self.constraint_length = constraint_length
        self.polynomials = polynomials
        self.tail_bits = constraint_length - 1

        # CommPy trellis representation
        memory = np.array([self.constraint_length - 1])
        g_matrix = np.array([list(polynomials)])
        self.trellis = Trellis(memory, g_matrix)

    def encode(self, source_bits: np.ndarray) -> np.ndarray:
        """Encode source bits using terminated rate-1/2 convolutional code.

        Parameters
        ----------
        source_bits : np.ndarray
            1D array of source bits.

        Returns
        -------
        coded_bits : np.ndarray
            Terminated coded bits of length 2 * (len(source_bits) + 6).
        """
        bits = np.asarray(source_bits, dtype=int).ravel()
        if len(bits) == 0:
            return np.array([], dtype=int)

        coded = commpy_conv_encode(bits, self.trellis, termination="term")
        return coded.astype(int)

    def decode(
        self,
        coded_bits: np.ndarray,
        soft_llrs: Optional[np.ndarray] = None,
        tb_depth: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode coded bits using hard-decision Viterbi decoding with tail-bit removal.

        Parameters
        ----------
        coded_bits : np.ndarray
            1D array of coded bits. Must have even length.
        soft_llrs : Optional[np.ndarray]
            Optional soft LLR vector (if available).
        tb_depth : Optional[int]
            Traceback depth. If None, selects adaptive depth:
            min(35, info_len + 6) for info_len < 64, else 35.

        Returns
        -------
        recovered_bits : np.ndarray
            Recovered source bits with tail bits trimmed.
        metrics : Dict[str, Any]
            Diagnostic metrics dictionary.
        """
        coded = np.asarray(coded_bits, dtype=int).ravel()
        if len(coded) == 0:
            return np.array([], dtype=int), {
                "decoder_success": False,
                "reason": "Empty coded bitstream",
                "bit_count": 0
            }

        if len(coded) % 2 != 0:
            raise ValueError(f"Coded bitstream length must be even, got {len(coded)}")

        expected_total_len = len(coded) // 2
        info_len = expected_total_len - self.tail_bits
        if info_len <= 0:
            raise ValueError(f"Coded bitstream too short ({len(coded)} bits) for K={self.constraint_length}")

        if tb_depth is None:
            if info_len < 64:
                selected_tb = min(35, max(10, expected_total_len))
            else:
                selected_tb = 35
        else:
            selected_tb = tb_depth

        decoded_raw = commpy_viterbi_decode(
            coded.astype(float),
            self.trellis,
            tb_depth=selected_tb,
            decoding_type="hard"
        )

        recovered_bits = decoded_raw[:info_len]
        metrics = {
            "decoder_success": True,
            "tb_depth": selected_tb,
            "tail_bits_trimmed": self.tail_bits,
            "bit_count": int(len(recovered_bits)),
            "coded_bit_count": int(len(coded)),
        }
        return recovered_bits.astype(int), metrics


def conv_encode(bits: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Module-level wrapper for convolutional encoding."""
    codec = ConvolutionalCodec()
    return codec.encode(bits)


def viterbi_decode(
    coded_bits: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    tb_depth: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Module-level wrapper for Viterbi decoding."""
    codec = ConvolutionalCodec()
    return codec.decode(coded_bits, tb_depth=tb_depth)


# ============================================================================
# REED-SOLOMON RS(255, 223) CODEC
# ============================================================================

class ReedSolomonCodec(BaseFEC):
    """Reed-Solomon RS(255, 223) byte-oriented codec over GF(2^8).

    Parameters:
        n = 255 symbols (bytes) = 2,040 bits
        k = 223 symbols (bytes) = 1,784 bits
        nsym = 32 parity bytes = 256 bits
        t = 16 correctable byte errors per block
    """

    def __init__(self, n: int = 255, k: int = 223):
        self.n = n
        self.k = k
        self.nsym = n - k
        self.codec = reedsolo.RSCodec(self.nsym)
        self.source_bits_per_block = self.k * 8  # 1784
        self.coded_bits_per_block = self.n * 8   # 2040

    def encode(self, source_bits: np.ndarray) -> np.ndarray:
        """Encode source bits into RS(255, 223) codewords.

        Parameters
        ----------
        source_bits : np.ndarray
            1D array of source bits. Length must be a multiple of 1,784 bits (223 bytes).

        Returns
        -------
        coded_bits : np.ndarray
            1D array of coded bits. Length will be multiple of 2,040 bits (255 bytes).
        """
        bits = np.asarray(source_bits, dtype=int).ravel()
        if len(bits) == 0:
            return np.array([], dtype=int)

        if len(bits) % self.source_bits_per_block != 0:
            raise ValueError(
                f"Source bit length {len(bits)} is not a multiple of RS block size {self.source_bits_per_block} bits (223 bytes)"
            )

        num_blocks = len(bits) // self.source_bits_per_block
        coded_chunks = []

        for b in range(num_blocks):
            chunk_bits = bits[b * self.source_bits_per_block : (b + 1) * self.source_bits_per_block]
            msg_bytes = bits_to_bytes(chunk_bits)
            enc_bytes = bytes(self.codec.encode(msg_bytes))
            chunk_coded_bits = bytes_to_bits(enc_bytes)
            coded_chunks.append(chunk_coded_bits)

        return np.concatenate(coded_chunks).astype(int)

    def decode(
        self,
        coded_bits: np.ndarray,
        soft_llrs: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode RS(255, 223) coded bits with re-encoding consistency verification.

        Parameters
        ----------
        coded_bits : np.ndarray
            1D array of coded bits. Length must be a multiple of 2,040 bits (255 bytes).
        soft_llrs : Optional[np.ndarray]
            Unused for hard Reed-Solomon.

        Returns
        -------
        recovered_bits : np.ndarray
            Recovered source bits.
        metrics : Dict[str, Any]
            Diagnostics indicating errors corrected, status, and block count.
        """
        bits = np.asarray(coded_bits, dtype=int).ravel()
        if len(bits) == 0:
            return np.array([], dtype=int), {
                "decoder_success": False,
                "reason": "Empty coded bitstream",
                "bit_count": 0
            }

        if len(bits) % self.coded_bits_per_block != 0:
            raise ValueError(
                f"Coded bit length {len(bits)} is not a multiple of RS codeword size {self.coded_bits_per_block} bits (255 bytes)"
            )

        num_blocks = len(bits) // self.coded_bits_per_block
        recovered_chunks = []
        total_errors_corrected = 0
        overall_success = True
        warnings = []

        for b in range(num_blocks):
            chunk_bits = bits[b * self.coded_bits_per_block : (b + 1) * self.coded_bits_per_block]
            chunk_bytes = bits_to_bytes(chunk_bits)

            try:
                dec_msg_bytes, dec_full_bytes, err_pos = self.codec.decode(chunk_bytes)
                # Re-encoding consistency verification:
                # Re-encode candidate message and compare against received codeword
                re_encoded_bytes = self.codec.encode(dec_msg_bytes)
                # Count symbol differences
                sym_diffs = sum(1 for x, y in zip(chunk_bytes, re_encoded_bytes) if x != y)
                if sym_diffs > self.nsym // 2:  # > 16 symbol errors is beyond capability
                    overall_success = False
                    warnings.append(f"Block {b}: Inconsistent re-encoding (symbol diffs={sym_diffs} > 16)")
                    recovered_chunks.append(np.zeros(self.source_bits_per_block, dtype=int))
                else:
                    total_errors_corrected += len(err_pos)
                    recovered_chunks.append(bytes_to_bits(bytes(dec_msg_bytes)))
            except reedsolo.ReedSolomonError as e:
                overall_success = False
                warnings.append(f"Block {b}: Uncorrectable ReedSolomonError: {e}")
                recovered_chunks.append(np.zeros(self.source_bits_per_block, dtype=int))

        recovered_bits = np.concatenate(recovered_chunks).astype(int)
        metrics = {
            "decoder_success": overall_success,
            "errors_corrected": total_errors_corrected,
            "num_blocks": num_blocks,
            "bit_count": int(len(recovered_bits)),
            "warnings": warnings,
        }
        return recovered_bits, metrics


# ============================================================================
# LOW-DENSITY PARITY-CHECK (LDPC) CODEC
# ============================================================================

class LDPCCodec(BaseFEC):
    """LDPC codec for Gallager (96, 3, 963) code with GF(2) systematic generator.

    Parameters:
        N = 96 codeword bits
        M = 48 check nodes
        rank(H) = 46 (over GF(2))
        K = 50 information bits (free columns)
        Rate R = 50/96 ≈ 0.5208
    """

    def __init__(self, design_filename: Optional[str] = None):
        if design_filename is None:
            designs_dir = os.path.join(
                os.path.dirname(commpy.__file__),
                "channelcoding", "designs", "ldpc", "gallager"
            )
            design_filename = os.path.join(designs_dir, "96.3.963.txt")

        self.design_filename = design_filename
        self.params = self._load_params_with_numpy2_fix(design_filename)
        self.n_vnodes = self.params["n_vnodes"]  # 96
        self.n_cnodes = self.params["n_cnodes"]  # 48

        # Build full parity check matrix H
        cnode_adj = self.params["cnode_adj_list"].reshape((self.n_cnodes, self.params["max_cnode_deg"]))
        self.H = np.zeros((self.n_cnodes, self.n_vnodes), dtype=int)
        for r in range(self.n_cnodes):
            for c in cnode_adj[r]:
                if c >= 0:
                    self.H[r, c] = 1

        # Perform GF(2) Gaussian elimination to construct systematic generator
        self.rank, self.pivot_cols, self.free_cols, self.RREF = self._compute_gf2_rref(self.H)
        self.k_info = len(self.free_cols)  # 50

    @staticmethod
    def _load_params_with_numpy2_fix(filename: str) -> Dict[str, Any]:
        """Load CommPy LDPC code params with NumPy 2.x scalar assignment compatibility."""
        with open(filename, "r") as f:
            [n_vnodes, n_cnodes] = [int(x) for x in f.readline().split(" ")]
            [max_vnode_deg, max_cnode_deg] = [int(x) for x in f.readline().split(" ")]
            vnode_deg_list = np.array([int(x) for x in f.readline().split(" ")[:-1]], np.int32)
            cnode_deg_list = np.array([int(x) for x in f.readline().split(" ")[:-1]], np.int32)
            cnode_adj_list = -np.ones([n_cnodes, max_cnode_deg], int)
            vnode_adj_list = -np.ones([n_vnodes, max_vnode_deg], int)
            for vnode_idx in range(n_vnodes):
                vnode_adj_list[vnode_idx, 0:vnode_deg_list[vnode_idx]] = np.array(
                    [int(x) - 1 for x in f.readline().split("\t")]
                )
            for cnode_idx in range(n_cnodes):
                cnode_adj_list[cnode_idx, 0:cnode_deg_list[cnode_idx]] = np.array(
                    [int(x) - 1 for x in f.readline().split("\t")]
                )

        cnode_vnode_map = -np.ones([n_cnodes, max_cnode_deg], int)
        vnode_cnode_map = -np.ones([n_vnodes, max_vnode_deg], int)
        for cnode in range(n_cnodes):
            for i, vnode in enumerate(cnode_adj_list[cnode, 0:cnode_deg_list[cnode]]):
                m = np.where(vnode_adj_list[vnode, :] == cnode)[0]
                cnode_vnode_map[cnode, i] = m[0] if m.size > 0 else -1
        for vnode in range(n_vnodes):
            for i, cnode in enumerate(vnode_adj_list[vnode, 0:vnode_deg_list[vnode]]):
                m = np.where(cnode_adj_list[cnode, :] == vnode)[0]
                vnode_cnode_map[vnode, i] = m[0] if m.size > 0 else -1

        return {
            "n_vnodes": n_vnodes,
            "n_cnodes": n_cnodes,
            "max_cnode_deg": max_cnode_deg,
            "max_vnode_deg": max_vnode_deg,
            "cnode_adj_list": cnode_adj_list.flatten().astype(np.int32),
            "cnode_vnode_map": cnode_vnode_map.flatten().astype(np.int32),
            "vnode_adj_list": vnode_adj_list.flatten().astype(np.int32),
            "vnode_cnode_map": vnode_cnode_map.flatten().astype(np.int32),
            "cnode_deg_list": cnode_deg_list,
            "vnode_deg_list": vnode_deg_list,
        }

    @staticmethod
    def _compute_gf2_rref(A: np.ndarray) -> Tuple[int, Sequence[int], Sequence[int], np.ndarray]:
        """Perform binary GF(2) Gaussian elimination to compute RREF and pivot columns."""
        M = A.copy() % 2
        nrows, ncols = M.shape
        pivot_row = 0
        pivot_cols = []
        for col in range(ncols):
            cand = np.where(M[pivot_row:, col] == 1)[0]
            if len(cand) == 0:
                continue
            p = pivot_row + cand[0]
            M[[pivot_row, p]] = M[[p, pivot_row]]
            for r in range(nrows):
                if r != pivot_row and M[r, col] == 1:
                    M[r] = (M[r] + M[pivot_row]) % 2
            pivot_cols.append(col)
            pivot_row += 1
            if pivot_row == nrows:
                break

        free_cols = [c for c in range(ncols) if c not in pivot_cols]
        return pivot_row, pivot_cols, free_cols, M

    def encode(self, source_bits: np.ndarray) -> np.ndarray:
        """Encode source bits into Gallager LDPC codewords satisfying H * c = 0 mod 2.

        Parameters
        ----------
        source_bits : np.ndarray
            1D array of source bits. Length must be a multiple of 50 bits.

        Returns
        -------
        coded_bits : np.ndarray
            1D array of coded bits. Length will be multiple of 96 bits.
        """
        bits = np.asarray(source_bits, dtype=int).ravel()
        if len(bits) == 0:
            return np.array([], dtype=int)

        if len(bits) % self.k_info != 0:
            raise ValueError(
                f"Source bit length {len(bits)} is not a multiple of LDPC information block size {self.k_info}"
            )

        num_blocks = len(bits) // self.k_info
        coded_blocks = []

        for b in range(num_blocks):
            msg_chunk = bits[b * self.k_info : (b + 1) * self.k_info]
            codeword = np.zeros(self.n_vnodes, dtype=int)
            codeword[self.free_cols] = msg_chunk
            for r in range(self.rank):
                codeword[self.pivot_cols[r]] = np.sum(self.RREF[r, self.free_cols] * msg_chunk) % 2

            # Parity check verification
            syndrome = self.H.dot(codeword) % 2
            if np.sum(syndrome) != 0:
                raise RuntimeError("Systematic LDPC encoding failed to satisfy parity checks H * c = 0 mod 2")

            coded_blocks.append(codeword)

        return np.concatenate(coded_blocks).astype(int)

    def decode(
        self,
        coded_bits: np.ndarray,
        soft_llrs: Optional[np.ndarray] = None,
        n_iters: int = 30,
        algo: str = "MSA"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode LDPC codewords using CommPy Belief Propagation (MSA/SPA).

        Parameters
        ----------
        coded_bits : np.ndarray
            1D array of coded bits.
        soft_llrs : Optional[np.ndarray]
            Channel LLR values. If None, derived from hard bits (+10 for 0, -10 for 1).
        n_iters : int
            Maximum belief-propagation iterations (default 30).
        algo : str
            Algorithm identifier: 'MSA' (Min-Sum) or 'SPA' (Sum-Product).

        Returns
        -------
        recovered_bits : np.ndarray
            Recovered information bits extracted from free columns.
        metrics : Dict[str, Any]
            Diagnostic metrics dictionary.
        """
        bits = np.asarray(coded_bits, dtype=int).ravel()
        if len(bits) == 0:
            return np.array([], dtype=int), {
                "decoder_success": False,
                "reason": "Empty coded bitstream",
                "bit_count": 0
            }

        if len(bits) % self.n_vnodes != 0:
            raise ValueError(
                f"Coded bit length {len(bits)} is not a multiple of LDPC codeword size {self.n_vnodes}"
            )

        if soft_llrs is None:
            # Map binary {0, 1} to reliable LLR: 0 -> +10.0, 1 -> -10.0
            llr_vec = np.where(bits == 0, 10.0, -10.0).astype(float)
        else:
            llr_vec = np.asarray(soft_llrs, dtype=float).ravel()

        num_blocks = len(bits) // self.n_vnodes
        recovered_blocks = []
        overall_success = True
        total_syndrome_errors = 0

        for b in range(num_blocks):
            block_llrs = llr_vec[b * self.n_vnodes : (b + 1) * self.n_vnodes]
            dec_word, out_llrs = commpy_ldpc.ldpc_bp_decode(
                block_llrs, self.params, algo, n_iters
            )
            # Check parity-check syndrome: H * c mod 2
            syndrome = self.H.dot(dec_word) % 2
            syn_err = int(np.sum(syndrome))
            total_syndrome_errors += syn_err
            if syn_err != 0:
                overall_success = False

            msg_extracted = dec_word[self.free_cols]
            recovered_blocks.append(msg_extracted)

        recovered_bits = np.concatenate(recovered_blocks).astype(int)
        metrics = {
            "decoder_success": overall_success,
            "syndrome_status": (total_syndrome_errors == 0),
            "syndrome_errors": total_syndrome_errors,
            "iterations": n_iters,
            "algorithm": algo,
            "num_blocks": num_blocks,
            "bit_count": int(len(recovered_bits)),
        }
        return recovered_bits, metrics


# ============================================================================
# CONCATENATED CODEC: RS OUTER + CONVOLUTIONAL INNER
# ============================================================================

class ConcatenatedCodec(BaseFEC):
    """Concatenated FEC codec: Reed-Solomon outer + Convolutional inner."""

    def __init__(
        self,
        outer: Optional[ReedSolomonCodec] = None,
        inner: Optional[ConvolutionalCodec] = None
    ):
        self.outer = outer if outer is not None else ReedSolomonCodec()
        self.inner = inner if inner is not None else ConvolutionalCodec()

    def encode(self, source_bits: np.ndarray) -> np.ndarray:
        """Encode source bits through outer RS then inner convolutional.

        Parameters
        ----------
        source_bits : np.ndarray
            Source bits (multiple of 1,784 bits).

        Returns
        -------
        coded_bits : np.ndarray
            Inner convolutional coded bits.
        """
        rs_coded = self.outer.encode(source_bits)
        conv_coded = self.inner.encode(rs_coded)
        return conv_coded

    def decode(
        self,
        coded_bits: np.ndarray,
        soft_llrs: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode inner convolutional (Viterbi) then outer RS.

        Parameters
        ----------
        coded_bits : np.ndarray
            Received coded bits.
        soft_llrs : Optional[np.ndarray]
            Soft LLRs.

        Returns
        -------
        recovered_bits : np.ndarray
            Recovered source bits.
        metrics : Dict[str, Any]
            Diagnostics from both inner and outer stages.
        """
        viterbi_bits, viterbi_meta = self.inner.decode(coded_bits, soft_llrs=soft_llrs)
        rs_bits, rs_meta = self.outer.decode(viterbi_bits)

        success = viterbi_meta.get("decoder_success", False) and rs_meta.get("decoder_success", False)
        metrics = {
            "decoder_success": success,
            "inner_metrics": viterbi_meta,
            "outer_metrics": rs_meta,
            "bit_count": int(len(rs_bits)),
        }
        return rs_bits, metrics
