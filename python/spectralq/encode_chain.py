"""
spectralq.encode_chain

Encode chain and decode chain orchestrators for the SpectralQ decoder core.

This module orchestrates the execution flow:
    encode_chain:
        source_bits -> FEC encoding -> Interleaving -> tx_bits (TX BITS)

    decode_chain:
        received_bits (RECEIVED BITS) -> De-interleaving -> FEC decoding -> recovered_source_bits

This module does NOT re-implement codec or interleaver algorithms; it imports
and coordinates spectralq.fec and spectralq.interleave.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np

from spectralq.fec import (
    BaseFEC,
    ConvolutionalCodec,
    ReedSolomonCodec,
    LDPCCodec,
    ConcatenatedCodec,
)
from spectralq.interleave import (
    block_interleave,
    block_deinterleave,
    diagonal_interleave,
    diagonal_deinterleave,
    conv_interleave,
    conv_deinterleave,
    pseudorandom_interleave,
    pseudorandom_deinterleave,
    identity_interleave,
    identity_deinterleave,
)


@dataclass
class EncodeConfig:
    """Explicit configuration passed into the encode and decode chain orchestrator."""
    fec_type: Any
    interleaver_type: Any
    modulation: str = "BPSK"
    case_id: str = "GENERIC"
    fec_params: Dict[str, Any] = field(default_factory=dict)
    interleaver_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncodeResult:
    """Structured result returned by encode_chain."""
    tx_bits: np.ndarray
    source_bits: np.ndarray
    fec_bits: np.ndarray
    source_bit_length: int
    tx_bit_length: int
    interleaver_metadata: Dict[str, Any]
    fec_metadata: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodeResult:
    """Structured result returned by decode_chain."""
    recovered_source_bits: np.ndarray
    decoder_success: bool
    deinterleaved_bits: np.ndarray
    fec_metrics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


def _get_fec_codec(fec_type: Any, fec_params: Dict[str, Any]) -> Optional[BaseFEC]:
    """Factory creating the appropriate FEC codec instance."""
    ftype = fec_type.value if hasattr(fec_type, "value") else str(fec_type)
    ftype = ftype.upper()
    if ftype in ("CONV_K7", "CONVOLUTIONAL", "CONV", "CONV_K7_171_133"):
        k = fec_params.get("constraint_length", 7)
        polys = tuple(fec_params.get("polynomials", (0o171, 0o133)))
        return ConvolutionalCodec(constraint_length=k, polynomials=polys)
    elif ftype in ("RS_255_223", "REED_SOLOMON", "RS"):
        n = fec_params.get("n", 255)
        k = fec_params.get("k", 223)
        return ReedSolomonCodec(n=n, k=k)
    elif ftype in ("LDPC", "LDPC_GALLAGER"):
        spec = fec_params.get("design_filename", None)
        return LDPCCodec(design_filename=spec)
    elif ftype in ("CONCAT_RS_CONV", "CONCATENATED"):
        return ConcatenatedCodec()
    elif ftype in ("NONE", "PASSTHROUGH"):
        return None
    else:
        raise ValueError(f"Unknown or unsupported FEC type: {fec_type}")


def encode_chain(
    source_bits: np.ndarray,
    config: EncodeConfig
) -> EncodeResult:
    """Orchestrate source bits through FEC encoding and interleaving into TX bits.

    Parameters
    ----------
    source_bits : np.ndarray
        1D binary array of source bits.
    config : EncodeConfig
        Pipeline configuration specifying FEC and interleaver schemes.

    Returns
    -------
    result : EncodeResult
        Container holding TX bits, intermediary bits, and execution metadata.
    """
    source = np.asarray(source_bits, dtype=int).ravel()
    source_len = len(source)

    # 1. Forward Error Correction Stage
    codec = _get_fec_codec(config.fec_type, config.fec_params)
    if codec is not None:
        fec_bits = codec.encode(source)
        fec_meta = {
            "fec_type": config.fec_type,
            "encoded_bit_length": int(len(fec_bits)),
            "codec_class": codec.__class__.__name__,
        }
    else:
        fec_bits = source.copy()
        fec_meta = {"fec_type": "NONE", "encoded_bit_length": source_len}

    # 2. Interleaving Stage
    itype = config.interleaver_type.value if hasattr(config.interleaver_type, "value") else str(config.interleaver_type)
    itype = itype.upper()
    iparams = config.interleaver_params

    if itype in ("BLOCK", "BLOCK_INTERLEAVER"):
        rows = iparams.get("rows", 16)
        cols = iparams.get("cols", 16)
        tx_bits, intl_meta = block_interleave(fec_bits, rows=rows, cols=cols)
    elif itype in ("DIAGONAL", "DIAGONAL_INTERLEAVER"):
        rows = iparams.get("num_rows", iparams.get("rows", 40))
        cols = iparams.get("num_cols", iparams.get("cols", 51))
        tx_bits, intl_meta = diagonal_interleave(fec_bits, num_rows=rows, num_cols=cols)
    elif itype in ("CONVOLUTIONAL", "CONV", "FORNEY"):
        branches = iparams.get("num_branches", iparams.get("branches", 4))
        delay_step = iparams.get("delay_step", 2)
        tx_bits, intl_meta = conv_interleave(fec_bits, num_branches=branches, delay_step=delay_step)
    elif itype in ("PSEUDORANDOM", "PRNG", "RANDOM"):
        seed = iparams.get("seed", 42)
        tx_bits, intl_meta = pseudorandom_interleave(fec_bits, seed=seed)
    elif itype in ("NONE", "PASSTHROUGH", "IDENTITY"):
        tx_bits, intl_meta = identity_interleave(fec_bits)
    else:
        raise ValueError(f"Unknown or unsupported interleaver type: {config.interleaver_type}")

    # Build comprehensive metadata
    meta = {
        "case_id": config.case_id,
        "modulation": config.modulation,
        "fec_type": config.fec_type,
        "interleaver_type": config.interleaver_type,
        "source_bit_length": int(source_len),
        "tx_bit_length": int(len(tx_bits)),
        "fec_parameters": config.fec_params,
        "interleaver_parameters": config.interleaver_params,
        "interleaver_metadata": intl_meta,
    }

    return EncodeResult(
        tx_bits=tx_bits,
        source_bits=source,
        fec_bits=fec_bits,
        source_bit_length=source_len,
        tx_bit_length=len(tx_bits),
        interleaver_metadata=intl_meta,
        fec_metadata=fec_meta,
        metadata=meta,
    )


def decode_chain(
    received_bits: np.ndarray,
    config: EncodeConfig,
    interleaver_meta: Optional[Dict[str, Any]] = None
) -> DecodeResult:
    """Orchestrate received bits through de-interleaving and FEC decoding into recovered source bits.

    Parameters
    ----------
    received_bits : np.ndarray
        1D binary array of received bits (in Phase 1: ideal channel bits).
    config : EncodeConfig
        Pipeline configuration.
    interleaver_meta : Optional[Dict[str, Any]]
        Interleaver metadata from encode_chain (or reconstructible from config).

    Returns
    -------
    result : DecodeResult
        Recovered source bits and diagnostic metrics.
    """
    received = np.asarray(received_bits, dtype=int).ravel()
    warnings = []

    # 1. De-interleaving Stage
    itype = config.interleaver_type.value if hasattr(config.interleaver_type, "value") else str(config.interleaver_type)
    itype = itype.upper()
    iparams = config.interleaver_params
    meta = interleaver_meta if interleaver_meta is not None else iparams

    if itype in ("BLOCK", "BLOCK_INTERLEAVER"):
        rows = iparams.get("rows", meta.get("rows", 16))
        cols = iparams.get("cols", meta.get("cols", 16))
        deint_bits = block_deinterleave(received, rows=rows, cols=cols, meta_or_length=meta)
    elif itype in ("DIAGONAL", "DIAGONAL_INTERLEAVER"):
        rows = iparams.get("num_rows", iparams.get("rows", meta.get("num_rows", 40)))
        cols = iparams.get("num_cols", iparams.get("cols", meta.get("num_cols", 51)))
        deint_bits = diagonal_deinterleave(received, num_rows=rows, num_cols=cols, meta_or_length=meta)
    elif itype in ("CONVOLUTIONAL", "CONV", "FORNEY"):
        branches = iparams.get("num_branches", iparams.get("branches", meta.get("num_branches", 4)))
        delay_step = iparams.get("delay_step", meta.get("delay_step", 2))
        deint_bits = conv_deinterleave(received, num_branches=branches, delay_step=delay_step, meta_or_length=meta)
    elif itype in ("PSEUDORANDOM", "PRNG", "RANDOM"):
        seed = iparams.get("seed", meta.get("seed", 42))
        deint_bits = pseudorandom_deinterleave(received, seed=seed, meta_or_length=meta)
    elif itype in ("NONE", "PASSTHROUGH", "IDENTITY"):
        deint_bits = identity_deinterleave(received, meta_or_length=meta)
    else:
        raise ValueError(f"Unknown or unsupported interleaver type: {config.interleaver_type}")

    # 2. Forward Error Correction Decoding Stage
    codec = _get_fec_codec(config.fec_type, config.fec_params)
    if codec is not None:
        recovered_bits, fec_metrics = codec.decode(deint_bits)
        success = fec_metrics.get("decoder_success", False)
        if not success:
            warnings.append("FEC decoding reported failure or uncorrected errors.")
    else:
        recovered_bits = deint_bits.copy()
        success = True
        fec_metrics = {"decoder_success": True, "fec_type": "NONE"}

    return DecodeResult(
        recovered_source_bits=recovered_bits,
        decoder_success=success,
        deinterleaved_bits=deint_bits,
        fec_metrics=fec_metrics,
        warnings=warnings,
    )
