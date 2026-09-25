"""
tests/fixtures/official_adapter.py

Official Sinchana Waveform Capture Adapter and Validation Harness.

Provides zero-overhead integration for GNU Octave-generated official golden files:
- capture.cf32 (Raw 32-bit interleaved IEEE-754 complex float IQ samples)
- capture.meta.json (Waveform generator metadata)
- truth.json (Ground-truth bitstream and impairment parameters)

CRITICAL INDEPENDENCE RULE:
- truth.json and Sinchana's golden answers are NEVER imported or consumed by production demodulation code.
- This adapter executes in the test/evaluation harness ONLY.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np

from spectralq.demod import demodulate, DemodConfig, DemodResult


def load_complex64_capture(filepath: Union[str, Path]) -> np.ndarray:
    """Read binary .cf32 capture into a 1D complex64 numpy array.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to .cf32 file.

    Returns
    -------
    samples : np.ndarray
        1D array of dtype np.complex64.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Capture file not found: {path}")
    raw = np.fromfile(str(path), dtype=np.complex64)
    return raw


def evaluate_official_capture(
    capture_dir: Union[str, Path],
    config_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute production demodulator on Sinchana's official capture and evaluate BER.

    Parameters
    ----------
    capture_dir : Union[str, Path]
        Directory containing capture.cf32, capture.meta.json, and truth.json.
    config_override : Optional[Dict[str, Any]]
        Optional manual configuration overrides.

    Returns
    -------
    report : Dict[str, Any]
        Structured evaluation report with BER, demodulation status, and diagnostics.
    """
    cdir = Path(capture_dir)
    cf32_file = cdir / "capture.cf32"
    meta_file = cdir / "capture.meta.json"
    truth_file = cdir / "truth.json"

    if not cf32_file.exists():
        return {
            "status": "NOT_RECEIVED",
            "message": f"Official capture file {cf32_file} does not exist yet.",
        }

    # 1. Load samples
    iq_samples = load_complex64_capture(cf32_file)

    # 2. Load capture metadata (if available)
    capture_meta: Dict[str, Any] = {}
    if meta_file.exists():
        capture_meta = json.loads(meta_file.read_text(encoding="utf-8"))

    # Apply overrides
    if config_override is not None:
        capture_meta.update(config_override)

    # 3. Assemble DemodConfig strictly from capture metadata
    mod = capture_meta.get("modulation", "QPSK")
    sps = int(capture_meta.get("samples_per_symbol", 4))
    fs = float(capture_meta.get("sample_rate", 1.0))
    alpha = float(capture_meta.get("rrc_alpha", 0.35))
    f_dev = capture_meta.get("fsk_deviation")

    # In production evaluation, preamble is extracted from capture metadata if published
    preamble = None
    if "preamble_bits" in capture_meta:
        preamble = np.array(capture_meta["preamble_bits"], dtype=int)

    demod_cfg = DemodConfig(
        modulation=mod,
        sample_rate=fs,
        samples_per_symbol=sps,
        rrc_alpha=alpha,
        fsk_deviation=float(f_dev) if f_dev is not None else None,
        preamble_bits=preamble,
        metadata=capture_meta,
    )

    # 4. Invoke production demodulator blindly on IQ waveform
    demod_result: DemodResult = demodulate(iq_samples, demod_cfg)

    # 5. Evaluate against truth.json ONLY in test harness
    ber: Optional[float] = None
    bit_errors: Optional[int] = None
    truth_bits_len: Optional[int] = None

    if truth_file.exists():
        truth_data = json.loads(truth_file.read_text(encoding="utf-8"))
        if "tx_bits" in truth_data:
            t_bits = np.array(truth_data["tx_bits"], dtype=int)
            truth_bits_len = len(t_bits)
            cmp_len = min(len(demod_result.hard_bits), len(t_bits))
            if cmp_len > 0:
                bit_errors = int(np.sum(demod_result.hard_bits[:cmp_len] != t_bits[:cmp_len]))
                ber = float(bit_errors / cmp_len)

    return {
        "status": demod_result.status.value,
        "modulation": mod,
        "input_sample_count": len(iq_samples),
        "output_bit_count": len(demod_result.hard_bits),
        "truth_bit_count": truth_bits_len,
        "bit_errors": bit_errors,
        "ber": ber,
        "timing_converged": demod_result.timing_status.get("converged", False),
        "carrier_converged": demod_result.carrier_status.get("converged", False),
        "estimated_cfo": demod_result.estimated_frequency_offset,
        "diagnostics": demod_result.diagnostics,
        "warnings": demod_result.warnings,
        "failure_reason": demod_result.failure_reason,
    }
