"""
De-interleaving Module for SpectralQ.

Provides:
- Block (Matrix) Interleaver and De-interleaver for bitstreams and soft LLRs
- Convolutional (Forney/Ramsey) Interleaver and De-interleaver
- Blind interleaver depth estimation
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("spectralq.deinterleave")


def block_interleave(
    data: np.ndarray,
    num_rows: int,
    num_cols: int,
) -> np.ndarray:
    """
    Block (Matrix) Interleaver:
    Writes data row-wise into an (R x C) matrix, reads column-wise.

    Pads trailing elements with zeros if len(data) is not an exact multiple of R*C.
    """
    block_size = num_rows * num_cols
    if block_size <= 0:
        raise ValueError(f"Invalid block dimensions: {num_rows}x{num_cols}")

    total_len = len(data)
    num_blocks = int(np.ceil(total_len / block_size))
    padded_len = num_blocks * block_size

    if padded_len > total_len:
        pad = np.zeros(padded_len - total_len, dtype=data.dtype)
        padded_data = np.concatenate([data, pad])
    else:
        padded_data = data

    out = []
    for b in range(num_blocks):
        block = padded_data[b * block_size : (b + 1) * block_size]
        mat = block.reshape((num_rows, num_cols))
        # Read column by column
        col_read = mat.T.flatten()
        out.append(col_read)

    return np.concatenate(out)


def block_deinterleave(
    data: np.ndarray,
    num_rows: int,
    num_cols: int,
    original_length: Optional[int] = None,
) -> np.ndarray:
    """
    Block (Matrix) De-interleaver:
    Inverse of block_interleave.
    Writes data column-wise into an (R x C) matrix, reads row-wise.
    """
    block_size = num_rows * num_cols
    if block_size <= 0:
        raise ValueError(f"Invalid block dimensions: {num_rows}x{num_cols}")

    total_len = len(data)
    num_blocks = int(np.ceil(total_len / block_size))
    padded_len = num_blocks * block_size

    if padded_len > total_len:
        pad = np.zeros(padded_len - total_len, dtype=data.dtype)
        padded_data = np.concatenate([data, pad])
    else:
        padded_data = data

    out = []
    for b in range(num_blocks):
        block = padded_data[b * block_size : (b + 1) * block_size]
        # Column-wise write into R x C is equivalent to reshape (C, R).T
        mat = block.reshape((num_cols, num_rows)).T
        # Read row-wise
        row_read = mat.flatten()
        out.append(row_read)

    result = np.concatenate(out)
    if original_length is not None:
        result = result[:original_length]
    return result


class ConvolutionalInterleaver:
    """
    Forney Convolutional Interleaver with I branches and branch step delay M.
    Branch j has delay j * M.
    """

    def __init__(self, num_branches: int = 4, delay_step: int = 2):
        self.num_branches = num_branches
        self.delay_step = delay_step
        self.fifos: List[List[Union[int, float]]] = [
            [0] * (j * delay_step) for j in range(num_branches)
        ]

    def reset(self):
        self.fifos = [
            [0] * (j * self.delay_step) for j in range(self.num_branches)
        ]

    def process(self, data: np.ndarray) -> np.ndarray:
        out = np.empty(len(data), dtype=data.dtype)
        for idx, val in enumerate(data):
            b = idx % self.num_branches
            fifo = self.fifos[b]
            if len(fifo) == 0:
                out[idx] = val
            else:
                out[idx] = fifo.pop(0)
                fifo.append(val)
        return out


class ConvolutionalDeinterleaver:
    """
    Forney Convolutional De-interleaver with I branches and branch step delay M.
    Branch j has complementary delay (I - 1 - j) * M.
    """

    def __init__(self, num_branches: int = 4, delay_step: int = 2):
        self.num_branches = num_branches
        self.delay_step = delay_step
        self.fifos: List[List[Union[int, float]]] = [
            [0] * ((num_branches - 1 - j) * delay_step) for j in range(num_branches)
        ]

    def reset(self):
        self.fifos = [
            [0] * ((self.num_branches - 1 - j) * self.delay_step) for j in range(self.num_branches)
        ]

    def process(self, data: np.ndarray) -> np.ndarray:
        out = np.empty(len(data), dtype=data.dtype)
        for idx, val in enumerate(data):
            b = idx % self.num_branches
            fifo = self.fifos[b]
            if len(fifo) == 0:
                out[idx] = val
            else:
                out[idx] = fifo.pop(0)
                fifo.append(val)
        return out
