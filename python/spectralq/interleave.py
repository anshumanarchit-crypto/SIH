"""
spectralq.interleave

Interleaving and de-interleaving algorithms for the SpectralQ decoder core.

Algorithms implemented:
1. Block Interleaver: Row-write, column-read matrix transposition with explicit padding metadata.
2. Convolutional / Cross Interleaver: Forney multi-branch FIFO shift-register model
   with explicit delay lines delay[r] = r * delay_step and dynamic stream flushing.
3. Diagonal Interleaver: 2D anti-diagonal grid traversal with structured padding metadata.
4. Pseudo-Random Interleaver: Deterministic permutation seeded reproducibility and exact inverse.
5. Identity Interleaver (NONE): Pass-through identity mapping preserving bit ordering and metadata.

All algorithms guarantee bit-exact invertibility under channel-free conditions.
"""

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np


@dataclass
class InterleaverResult:
    """Structured container for interleaver execution and metadata."""
    interleaved_bits: np.ndarray
    original_length: int
    padded_length: int
    padding_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def block_interleave(
    bits: np.ndarray,
    rows: int,
    cols: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Interleave bits using a row-write, column-read block interleaver.

    Parameters
    ----------
    bits : np.ndarray
        1D array of binary bits (0 and 1).
    rows : int
        Number of matrix rows.
    cols : int
        Number of matrix columns.

    Returns
    -------
    interleaved_bits : np.ndarray
        Interleaved 1D bit array.
    metadata : Dict[str, Any]
        Dictionary preserving original_length, padded_length, padding_length, rows, cols.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError(f"rows and cols must be positive integers, got rows={rows}, cols={cols}")

    bits_arr = np.asarray(bits, dtype=int).ravel()
    original_length = len(bits_arr)

    capacity = rows * cols
    num_blocks = max(1, math.ceil(original_length / capacity))
    padded_length = num_blocks * capacity
    padding_length = padded_length - original_length

    if padding_length > 0:
        padded = np.pad(bits_arr, (0, padding_length), mode="constant")
    else:
        padded = bits_arr.copy()

    # Reshape into (num_blocks, rows, cols)
    blocks = padded.reshape((num_blocks, rows, cols))
    # Transpose to (num_blocks, cols, rows) for column-read
    transposed = np.transpose(blocks, (0, 2, 1))
    interleaved_bits = transposed.flatten()

    metadata = {
        "interleaver_type": "BLOCK",
        "original_length": int(original_length),
        "padded_length": int(padded_length),
        "padding_length": int(padding_length),
        "rows": int(rows),
        "cols": int(cols),
        "num_blocks": int(num_blocks),
    }

    return interleaved_bits, metadata


def block_deinterleave(
    bits: np.ndarray,
    rows: int,
    cols: int,
    meta_or_length: Optional[Union[Dict[str, Any], int]] = None
) -> np.ndarray:
    """De-interleave bits using a column-write, row-read block de-interleaver.

    Parameters
    ----------
    bits : np.ndarray
        Interleaved 1D bit array.
    rows : int
        Number of matrix rows used during interleaving.
    cols : int
        Number of matrix columns used during interleaving.
    meta_or_length : Optional[Union[Dict[str, Any], int]]
        Metadata dictionary from block_interleave or integer original_length.

    Returns
    -------
    recovered_bits : np.ndarray
        De-interleaved 1D bit array with padding stripped if metadata is provided.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError(f"rows and cols must be positive integers, got rows={rows}, cols={cols}")

    bits_arr = np.asarray(bits, dtype=int).ravel()
    capacity = rows * cols

    if len(bits_arr) % capacity != 0:
        raise ValueError(
            f"Interleaved bit length {len(bits_arr)} is not a multiple of block capacity {capacity} ({rows}x{cols})"
        )

    num_blocks = len(bits_arr) // capacity
    # Reshape to (num_blocks, cols, rows)
    blocks = bits_arr.reshape((num_blocks, cols, rows))
    # Transpose to (num_blocks, rows, cols) for row-read
    transposed = np.transpose(blocks, (0, 2, 1))
    flattened = transposed.flatten()

    orig_length = None
    if isinstance(meta_or_length, dict):
        orig_length = meta_or_length.get("original_length")
    elif isinstance(meta_or_length, (int, np.integer)):
        orig_length = int(meta_or_length)

    if orig_length is not None:
        return flattened[:orig_length]
    return flattened


def diagonal_interleave(
    bits: np.ndarray,
    num_rows: int,
    num_cols: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Interleave bits using a 2D anti-diagonal grid traversal.

    Parameters
    ----------
    bits : np.ndarray
        1D array of binary bits.
    num_rows : int
        Number of grid rows.
    num_cols : int
        Number of grid columns.

    Returns
    -------
    interleaved_bits : np.ndarray
        Diagonal-interleaved 1D bit array.
    metadata : Dict[str, Any]
        Metadata dictionary preserving lengths and grid dimensions.
    """
    if num_rows <= 0 or num_cols <= 0:
        raise ValueError(f"num_rows and num_cols must be positive, got ({num_rows}, {num_cols})")

    bits_arr = np.asarray(bits, dtype=int).ravel()
    original_length = len(bits_arr)

    capacity = num_rows * num_cols
    num_blocks = max(1, math.ceil(original_length / capacity))
    padded_length = num_blocks * capacity
    padding_length = padded_length - original_length

    if padding_length > 0:
        padded = np.pad(bits_arr, (0, padding_length), mode="constant")
    else:
        padded = bits_arr.copy()

    interleaved_out = []
    for b in range(num_blocks):
        block = padded[b * capacity : (b + 1) * capacity].reshape((num_rows, num_cols))
        # Read out along anti-diagonals where sum s = r + c
        for s in range(num_rows + num_cols - 1):
            r_start = max(0, s - num_cols + 1)
            r_end = min(num_rows - 1, s)
            for r in range(r_start, r_end + 1):
                c = s - r
                interleaved_out.append(block[r, c])

    metadata = {
        "interleaver_type": "DIAGONAL",
        "original_length": int(original_length),
        "padded_length": int(padded_length),
        "padding_length": int(padding_length),
        "num_rows": int(num_rows),
        "num_cols": int(num_cols),
        "num_blocks": int(num_blocks),
    }

    return np.array(interleaved_out, dtype=int), metadata


def diagonal_deinterleave(
    bits: np.ndarray,
    num_rows: int,
    num_cols: int,
    meta_or_length: Optional[Union[Dict[str, Any], int]] = None
) -> np.ndarray:
    """De-interleave bits using the exact inverse of diagonal_interleave.

    Parameters
    ----------
    bits : np.ndarray
        Diagonal-interleaved 1D bit array.
    num_rows : int
        Grid rows.
    num_cols : int
        Grid columns.
    meta_or_length : Optional[Union[Dict[str, Any], int]]
        Metadata dictionary or integer original_length.

    Returns
    -------
    recovered_bits : np.ndarray
        De-interleaved 1D bit array with padding removed.
    """
    if num_rows <= 0 or num_cols <= 0:
        raise ValueError(f"num_rows and num_cols must be positive, got ({num_rows}, {num_cols})")

    bits_arr = np.asarray(bits, dtype=int).ravel()
    capacity = num_rows * num_cols

    if len(bits_arr) % capacity != 0:
        raise ValueError(
            f"Interleaved bit length {len(bits_arr)} is not a multiple of capacity {capacity} ({num_rows}x{num_cols})"
        )

    num_blocks = len(bits_arr) // capacity
    idx = 0
    recovered_blocks = []

    for _ in range(num_blocks):
        grid = np.zeros((num_rows, num_cols), dtype=int)
        for s in range(num_rows + num_cols - 1):
            r_start = max(0, s - num_cols + 1)
            r_end = min(num_rows - 1, s)
            for r in range(r_start, r_end + 1):
                c = s - r
                grid[r, c] = bits_arr[idx]
                idx += 1
        recovered_blocks.append(grid.flatten())

    concatenated = np.concatenate(recovered_blocks)

    orig_length = None
    if isinstance(meta_or_length, dict):
        orig_length = meta_or_length.get("original_length")
    elif isinstance(meta_or_length, (int, np.integer)):
        orig_length = int(meta_or_length)

    if orig_length is not None:
        return concatenated[:orig_length]
    return concatenated


def conv_interleave(
    bits: np.ndarray,
    num_branches: int,
    delay_step: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Interleave bits using an explicit multi-branch FIFO shift-register model (Forney architecture).

    Branch r in [0, num_branches - 1] contains a delay line of length:
        delay[r] = r * delay_step

    A dynamic flushing stage appends zeros until all input bits exit the queues,
    preserving exact invertible finite-block transmission.

    Parameters
    ----------
    bits : np.ndarray
        1D array of binary bits.
    num_branches : int
        Number of parallel shift-register branches (B >= 1).
    delay_step : int
        Incremental delay step in branch shifts (M >= 1).

    Returns
    -------
    interleaved_bits : np.ndarray
        Interleaved bitstream including stream flush bits.
    metadata : Dict[str, Any]
        Metadata dictionary recording branches, delay_step, original_length, flush_length.
    """
    if num_branches <= 0 or delay_step < 0:
        raise ValueError(f"num_branches must be >= 1, delay_step >= 0. Got ({num_branches}, {delay_step})")

    bits_arr = np.asarray(bits, dtype=int).ravel()
    original_length = len(bits_arr)

    # Initialize branch FIFO queues with delay[r] = r * delay_step zeros
    fifos = [deque([0] * (r * delay_step)) for r in range(num_branches)]
    output = []

    step = 0
    # Phase 1: Clock all information bits through the branch commutator
    for bit in bits_arr:
        r = step % num_branches
        out_val = fifos[r].popleft() if len(fifos[r]) > 0 else bit
        if len(fifos[r]) > 0 or r > 0:
            fifos[r].append(bit)
        output.append(out_val)
        step += 1

    # Phase 2: Dynamic Flush - clock zero symbols until all FIFOs have cleared their active bits
    # Branch (num_branches - 1) has maximum delay: (num_branches - 1) * delay_step shifts.
    # Since each branch shifts once every num_branches steps, flushing requires
    # exactly (num_branches - 1) * delay_step * num_branches commutator steps.
    max_branch_shifts = (num_branches - 1) * delay_step
    flush_steps = max_branch_shifts * num_branches

    for _ in range(flush_steps):
        r = step % num_branches
        out_val = fifos[r].popleft() if len(fifos[r]) > 0 else 0
        if len(fifos[r]) > 0 or r > 0:
            fifos[r].append(0)
        output.append(out_val)
        step += 1

    metadata = {
        "interleaver_type": "CONVOLUTIONAL",
        "original_length": int(original_length),
        "flush_length": int(flush_steps),
        "interleaved_length": int(len(output)),
        "num_branches": int(num_branches),
        "delay_step": int(delay_step),
        "startup_delay": int(flush_steps),
    }

    return np.array(output, dtype=int), metadata


def conv_deinterleave(
    bits: np.ndarray,
    num_branches: int,
    delay_step: int,
    meta_or_length: Optional[Union[Dict[str, Any], int]] = None
) -> np.ndarray:
    """De-interleave bits using complementary shift-register delay lines.

    Branch r in [0, num_branches - 1] contains a complementary delay of:
        comp_delay[r] = (num_branches - 1 - r) * delay_step

    Together with the interleaver delay, the total end-to-end delay across
    every branch is exactly:
        total_delay = (num_branches - 1) * delay_step * num_branches symbols.

    The first total_delay symbols are startup transients and are discarded;
    the next original_length symbols are the exact original bitstream.

    Parameters
    ----------
    bits : np.ndarray
        Interleaved bitstream.
    num_branches : int
        Number of branches used during interleaving.
    delay_step : int
        Delay step used during interleaving.
    meta_or_length : Optional[Union[Dict[str, Any], int]]
        Metadata dictionary or integer original_length.

    Returns
    -------
    recovered_bits : np.ndarray
        De-interleaved 1D bit array.
    """
    if num_branches <= 0 or delay_step < 0:
        raise ValueError(f"num_branches must be >= 1, delay_step >= 0. Got ({num_branches}, {delay_step})")

    bits_arr = np.asarray(bits, dtype=int).ravel()

    # Complementary FIFOs: branch r has (num_branches - 1 - r) * delay_step zeros
    fifos = [deque([0] * ((num_branches - 1 - r) * delay_step)) for r in range(num_branches)]
    output = []

    step = 0
    for bit in bits_arr:
        r = step % num_branches
        out_val = fifos[r].popleft() if len(fifos[r]) > 0 else bit
        if len(fifos[r]) > 0 or (num_branches - 1 - r) > 0:
            fifos[r].append(bit)
        output.append(out_val)
        step += 1

    total_delay = (num_branches - 1) * delay_step * num_branches

    orig_length = None
    if isinstance(meta_or_length, dict):
        orig_length = meta_or_length.get("original_length")
    elif isinstance(meta_or_length, (int, np.integer)):
        orig_length = int(meta_or_length)

    recovered = np.array(output, dtype=int)[total_delay:]
    if orig_length is not None:
        return recovered[:orig_length]
    return recovered


def pseudorandom_interleave(
    bits: np.ndarray,
    seed: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Interleave bits using a deterministic pseudo-random permutation.

    Reproducibility Rule: Random interleaving is 100% reproducible
    from (bit_length, seed).

    Parameters
    ----------
    bits : np.ndarray
        1D array of binary bits.
    seed : int
        Deterministic random seed.

    Returns
    -------
    interleaved_bits : np.ndarray
        Permuted 1D bit array.
    metadata : Dict[str, Any]
        Metadata recording original_length, seed.
    """
    bits_arr = np.asarray(bits, dtype=int).ravel()
    original_length = len(bits_arr)

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(original_length)

    interleaved_bits = bits_arr[permutation]
    metadata = {
        "interleaver_type": "PSEUDORANDOM",
        "original_length": int(original_length),
        "seed": int(seed),
    }

    return interleaved_bits, metadata


def pseudorandom_deinterleave(
    bits: np.ndarray,
    seed: int,
    meta_or_length: Optional[Union[Dict[str, Any], int]] = None
) -> np.ndarray:
    """De-interleave bits using the exact inverse of pseudorandom_interleave.

    Parameters
    ----------
    bits : np.ndarray
        Interleaved bitstream.
    seed : int
        Deterministic random seed used during interleaving.
    meta_or_length : Optional[Union[Dict[str, Any], int]]
        Metadata dictionary or integer original_length.

    Returns
    -------
    recovered_bits : np.ndarray
        Exact recovered 1D bit array.
    """
    bits_arr = np.asarray(bits, dtype=int).ravel()
    length = len(bits_arr)

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(length)

    # Invert the permutation: inv_perm[permutation[i]] = i
    inv_perm = np.empty_like(permutation)
    inv_perm[permutation] = np.arange(length)

    recovered_bits = bits_arr[inv_perm]

    orig_length = None
    if isinstance(meta_or_length, dict):
        orig_length = meta_or_length.get("original_length")
    elif isinstance(meta_or_length, (int, np.integer)):
        orig_length = int(meta_or_length)

    if orig_length is not None:
        return recovered_bits[:orig_length]
    return recovered_bits


def identity_interleave(
    bits: np.ndarray,
    **kwargs: Any
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Interleave bits using identity mapping (no interleaving / pass-through).

    Parameters
    ----------
    bits : np.ndarray
        1D array of binary bits.
    **kwargs : Any
        Ignored extra arguments for interface compatibility.

    Returns
    -------
    interleaved_bits : np.ndarray
        Exact 1D bit array identical to input.
    metadata : Dict[str, Any]
        Metadata dictionary recording interleaver_type="NONE", original_length,
        padded_length, and padding_length=0.
    """
    bits_arr = np.asarray(bits, dtype=int).ravel()
    original_length = len(bits_arr)

    metadata = {
        "interleaver_type": "NONE",
        "original_length": int(original_length),
        "padded_length": int(original_length),
        "padding_length": 0,
    }

    return bits_arr.copy(), metadata


def identity_deinterleave(
    bits: np.ndarray,
    meta_or_length: Optional[Union[Dict[str, Any], int]] = None,
    **kwargs: Any
) -> np.ndarray:
    """De-interleave bits using identity mapping (no de-interleaving / pass-through).

    Parameters
    ----------
    bits : np.ndarray
        Interleaved 1D bit array.
    meta_or_length : Optional[Union[Dict[str, Any], int]]
        Optional metadata dictionary or integer original_length.
    **kwargs : Any
        Ignored extra arguments for interface compatibility.

    Returns
    -------
    recovered_bits : np.ndarray
        Exact recovered 1D bit array.
    """
    bits_arr = np.asarray(bits, dtype=int).ravel()

    orig_length = None
    if isinstance(meta_or_length, dict):
        orig_length = meta_or_length.get("original_length")
    elif isinstance(meta_or_length, (int, np.integer)):
        orig_length = int(meta_or_length)

    if orig_length is not None:
        return bits_arr[:orig_length].copy()
    return bits_arr.copy()


# Aliases for explicit NONE naming
none_interleave = identity_interleave
none_deinterleave = identity_deinterleave

