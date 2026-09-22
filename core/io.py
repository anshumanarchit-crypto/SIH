"""
SpectralQ Signal File Ingestion Module (CSE-1)
===============================================
Production-quality input layer for SpectralQ.

Accepts raw .IQ binary files and .WAV audio/SDR files and converts them into
the shared `SignalData` contract (defined in core.contracts).

Supported formats
-----------------
WAV files:
  - Mono (1-channel): real-valued PCM, converted to complex analytic signal via
    Hilbert transform when ``to_analytic=True`` (default).
  - Stereo (2-channel): Channel 0 = In-Phase (I), Channel 1 = Quadrature (Q).
    Interpreted as complex baseband per the IQ_CONVENTION pinned in core.contracts.
  - PCM integer subtypes: int16, int32, uint8
  - Float subtypes: float32, float64

Raw IQ binary files:
  - Interleaved layouts: IQ (I0,Q0,I1,Q1,...) or QI (Q0,I0,Q1,I1,...)
  - Numeric dtypes: int8, uint8, int16, float32 (complex64), float64 (complex128)
  - On-disk format: little-endian by default (the SDR industry standard)

IQ Convention (pinned)
----------------------
  complex sample = I + jQ
  I = real part (In-Phase), Q = imaginary part (Quadrature).
  Reference: core.contracts.IQ_CONVENTION

Bit Ordering (pinned)
---------------------
  MSB-first (Big-Endian). Reference: core.contracts.BIT_ORDERING

Sample-Rate Policy
------------------
  Raw IQ files carry no intrinsic sample-rate metadata. If neither a companion
  .json/.sigmf-meta is found nor ``sample_rate`` is supplied explicitly, the
  sample_rate field is set to None and a structured warning is emitted. Callers
  must supply sample_rate before invoking any frequency-dependent downstream
  modules (feature extraction, demodulation, etc.).

  WAV files always embed a sample rate in the RIFF header; it is read and
  trusted without modification.

Normalization Behaviour
-----------------------
  ``normalize_signal()`` operates on a *copy* of the samples. It never
  modifies the original file or the original SignalData object in-place.
  Operations (all optional):
    1. NaN/Inf validation (always checked; error on illegal values)
    2. DC offset removal (mean subtraction)
    3. Amplitude normalization to unit RMS power
    4. Gain scaling (multiply by a constant factor)
    5. Clipping detection (reports a warning if any sample exceeds the threshold)

  Note: ``normalize_signal`` does NOT perform modulation classification or
  demodulation. Its sole job is signal conditioning.
"""

from __future__ import annotations

import json
import logging
import os
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert

from core.contracts import (
    IQ_CONVENTION,
    BIT_ORDERING,
    ResultStatus,
    SignalData,
    make_warning,
)

logger = logging.getLogger("spectralq.io")

# --------------------------------------------------------------------------- #
# Module-level constants                                                        #
# --------------------------------------------------------------------------- #

#: Supported raw IQ dtype identifiers → (numpy element dtype, components/sample)
_IQ_DTYPE_MAP: Dict[str, Tuple[np.dtype, int]] = {
    "int8":      (np.dtype("int8"),    2),
    "uint8":     (np.dtype("uint8"),   2),
    "int16":     (np.dtype("<i2"),     2),   # little-endian int16 (SDR default)
    "int16_le":  (np.dtype("<i2"),     2),
    "int16_be":  (np.dtype(">i2"),     2),
    "float32":   (np.dtype("<f4"),     2),   # complex64 on disk
    "complex64":  (np.dtype("<f4"),    2),   # alias
    "float64":   (np.dtype("<f8"),     2),   # complex128 on disk
    "complex128": (np.dtype("<f8"),    2),   # alias
}

#: Supported IQ channel layouts
_SUPPORTED_IQ_FORMATS = ("IQ", "QI")

#: WAV channel limits
_MAX_WAV_CHANNELS = 2


# =========================================================================== #
# Internal Helpers                                                              #
# =========================================================================== #

def _find_companion_metadata(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Search for a companion metadata file beside the signal file.

    Candidates searched in order:
      1. <stem>.json
      2. <stem>.meta.json
      3. <stem>.sigmf-meta
      4. <name>.json  (entire filename + .json)

    Returns parsed dict or None if not found.
    """
    path = Path(file_path)
    candidates = [
        path.with_suffix(".json"),
        path.with_name(f"{path.stem}.meta.json"),
        path.with_suffix(".sigmf-meta"),
        path.with_name(f"{path.name}.json"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            try:
                with open(candidate, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                logger.info("Found companion metadata: %s", candidate)
                return meta
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse companion metadata %s: %s", candidate, exc)
    return None


def _extract_sample_rate_from_meta(meta: Dict[str, Any]) -> Optional[float]:
    """Try all known metadata schemas to extract a sample rate."""
    # SpectralQ native JSON
    for key in ("sample_rate", "sampleRate", "fs"):
        if key in meta:
            return float(meta[key])
    # SigMF
    if "global" in meta:
        for key in ("core:sample_rate", "sample_rate"):
            if key in meta["global"]:
                return float(meta["global"][key])
    return None


def _validate_finite(samples: np.ndarray, context: str = "") -> None:
    """Raise ValueError if samples contains NaN or Inf."""
    if not np.all(np.isfinite(samples)):
        n_nan = int(np.sum(np.isnan(samples)))
        n_inf = int(np.sum(np.isinf(samples)))
        raise ValueError(
            f"Signal contains invalid values [{context}]: "
            f"{n_nan} NaN, {n_inf} Inf out of {samples.size} elements."
        )


def _int_to_float(raw: np.ndarray, elem_dtype: np.dtype) -> np.ndarray:
    """
    Convert raw integer IQ elements to float32 using the canonical scale factors.

    Scale factors follow the standard SDR convention:
      - int16  → divide by 32768.0 (2^15)   → maps [-32768, 32767] to ≈[-1.0, 1.0]
      - int8   → divide by 128.0   (2^7)    → maps [-128, 127] to ≈[-1.0, 1.0]
      - uint8  → subtract 127.5, divide by 127.5 → maps [0, 255] to ≈[-1.0, 1.0]
    """
    kind = elem_dtype.kind
    itemsize = elem_dtype.itemsize

    if kind == "i":  # signed integer
        if itemsize == 2:  # int16
            return raw.astype(np.float32) / 32768.0
        elif itemsize == 1:  # int8
            return raw.astype(np.float32) / 128.0
        else:
            return raw.astype(np.float32) / float(2 ** (8 * itemsize - 1))
    elif kind == "u":  # unsigned integer
        if itemsize == 1:  # uint8
            return (raw.astype(np.float32) - 127.5) / 127.5
        else:
            half = 2 ** (8 * itemsize - 1)
            return (raw.astype(np.float32) - half) / float(half)
    else:
        # floating-point: pass through
        return raw.astype(np.float32)


# =========================================================================== #
# Public API: Signal Loading                                                    #
# =========================================================================== #

def load_wav(
    path: Union[str, Path],
    center_freq: float = 0.0,
    iq_interpretation: bool = True,
    to_analytic: bool = True,
) -> SignalData:
    """
    Load a WAV file and return a ``SignalData`` object.

    Parameters
    ----------
    path:
        Path to the .wav file.
    center_freq:
        Optional RF center frequency in Hz (informational; stored in metadata).
    iq_interpretation:
        If True and the file is stereo, interpret Channel 0 as I and Channel 1
        as Q (the standard SDR convention). If False, returns the raw float
        array with ``is_complex=False``.
    to_analytic:
        For mono files: if True, convert the real waveform to its analytic
        (complex) representation via the Hilbert transform. If False, return
        the raw real samples.

    Returns
    -------
    SignalData
        Validated, normalized-to-float signal container. Integer PCM dtypes
        are normalized to float32 in [-1.0, 1.0].

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is corrupt, empty, or has an unsupported channel count.
    """
    path = Path(path)

    # ------------------------------------------------------------------
    # Validate file existence and size
    # ------------------------------------------------------------------
    if not path.exists():
        raise FileNotFoundError(f"WAV file not found: {path}")
    file_size = path.stat().st_size
    if file_size < 44:  # minimum RIFF/WAV header size
        raise ValueError(f"WAV file is too small to be valid ({file_size} bytes): {path}")

    # ------------------------------------------------------------------
    # Read WAV
    # ------------------------------------------------------------------
    try:
        fs, data = wavfile.read(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to read WAV file '{path}': {exc}") from exc

    if data.size == 0:
        raise ValueError(f"WAV file contains no samples: {path}")

    original_dtype = str(data.dtype)
    fs = float(fs)

    # ------------------------------------------------------------------
    # Normalize integer PCM → float32 ∈ [-1.0, 1.0]
    # ------------------------------------------------------------------
    if data.dtype == np.int16:
        data_float = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data_float = data.astype(np.float32) / 2_147_483_648.0
    elif data.dtype == np.uint8:
        data_float = (data.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.floating):
        data_float = data.astype(np.float32)
    else:
        # Last resort: attempt generic cast, emit warning
        logger.warning("Unknown WAV PCM dtype %s; casting to float32.", data.dtype)
        data_float = data.astype(np.float32)

    warnings_list = []

    # ------------------------------------------------------------------
    # Handle channel layout
    # ------------------------------------------------------------------
    if data_float.ndim == 1:
        # Mono
        channels = 1
        if to_analytic:
            analytic = hilbert(data_float)
            samples: np.ndarray = analytic.astype(np.complex64)
            is_complex = True
        else:
            samples = data_float
            is_complex = False

    elif data_float.ndim == 2:
        n_channels = data_float.shape[1]
        if n_channels > _MAX_WAV_CHANNELS:
            raise ValueError(
                f"WAV file has {n_channels} channels; only mono (1) and "
                f"stereo (2) are supported."
            )
        if n_channels == 1:
            # Edge case: 2-D array with a single column (some encoders)
            data_flat = data_float[:, 0]
            channels = 1
            if to_analytic:
                analytic = hilbert(data_flat)
                samples = analytic.astype(np.complex64)
                is_complex = True
            else:
                samples = data_flat
                is_complex = False
        else:
            # Stereo
            channels = 2
            if iq_interpretation:
                i_comp = data_float[:, 0]
                q_comp = data_float[:, 1]
                samples = (i_comp + 1j * q_comp).astype(np.complex64)
                is_complex = True
            else:
                samples = data_float  # shape (N, 2) real
                is_complex = False
    else:
        raise ValueError(
            f"WAV data has unexpected shape {data_float.shape}; "
            "expected 1-D (mono) or 2-D (stereo)."
        )

    # ------------------------------------------------------------------
    # Validate for NaN/Inf post-conversion
    # ------------------------------------------------------------------
    try:
        _validate_finite(samples, context=str(path.name))
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    metadata = {
        "source_file": str(path.resolve()),
        "original_dtype": original_dtype,
        "channels": channels,
        "converted_to_analytic": (channels == 1 and to_analytic),
        "iq_interpretation": iq_interpretation,
        "iq_convention": IQ_CONVENTION,
    }

    return SignalData(
        samples=samples,
        sample_rate=fs,
        source_path=str(path.resolve()),
        source_format="wav",
        center_freq=float(center_freq),
        is_complex=is_complex,
        metadata=metadata,
        warnings=warnings_list,
    )


def load_iq(
    path: Union[str, Path],
    dtype: str = "float32",
    iq_format: str = "IQ",
    sample_rate: Optional[float] = None,
    center_freq: float = 0.0,
    offset_samples: int = 0,
    max_samples: Optional[int] = None,
) -> SignalData:
    """
    Load a raw binary IQ file and return a ``SignalData`` object.

    Parameters
    ----------
    path:
        Path to the raw IQ/binary file.
    dtype:
        Element data type on disk. Supported values:
        ``'int8'``, ``'uint8'``, ``'int16'`` / ``'int16_le'`` / ``'int16_be'``,
        ``'float32'`` / ``'complex64'``, ``'float64'`` / ``'complex128'``.
    iq_format:
        Interleaving layout. ``'IQ'`` = I₀,Q₀,I₁,Q₁,… (default; SDR industry
        standard). ``'QI'`` = Q₀,I₀,Q₁,I₁,… (some older hardware).
    sample_rate:
        Sampling frequency in Hz. If None, the function attempts to read a
        companion .json or .sigmf-meta file. If still unknown, ``sample_rate``
        is stored as None and a structured warning is appended.
    center_freq:
        RF center frequency in Hz (informational only; default 0.0).
    offset_samples:
        Number of *complex* samples to skip from the start of the file.
    max_samples:
        Maximum number of *complex* samples to read (None reads all).

    Returns
    -------
    SignalData
        Complex64 array of baseband samples. ``sample_rate`` may be None if
        not determinable from file or arguments.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the dtype is unsupported, the file is empty, offset exceeds file
        size, or the raw element count is odd after slicing.
    """
    path = Path(path)

    # ------------------------------------------------------------------
    # Validate file existence and size
    # ------------------------------------------------------------------
    if not path.exists():
        raise FileNotFoundError(f"IQ file not found: {path}")

    file_size_bytes = path.stat().st_size
    if file_size_bytes == 0:
        raise ValueError(f"IQ file is empty: {path}")

    # ------------------------------------------------------------------
    # Dtype lookup
    # ------------------------------------------------------------------
    dtype_key = dtype.lower().strip()
    if dtype_key not in _IQ_DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Supported: {sorted(_IQ_DTYPE_MAP.keys())}"
        )
    elem_dtype, _ = _IQ_DTYPE_MAP[dtype_key]
    elem_size = elem_dtype.itemsize
    bytes_per_complex = elem_size * 2  # 2 components per complex sample

    # ------------------------------------------------------------------
    # IQ format validation
    # ------------------------------------------------------------------
    iq_format_upper = iq_format.upper().strip()
    if iq_format_upper not in _SUPPORTED_IQ_FORMATS:
        raise ValueError(
            f"Unsupported iq_format '{iq_format}'. Supported: {_SUPPORTED_IQ_FORMATS}"
        )

    # ------------------------------------------------------------------
    # Sample count accounting
    # ------------------------------------------------------------------
    total_available = file_size_bytes // bytes_per_complex
    if total_available == 0:
        raise ValueError(
            f"IQ file too small ({file_size_bytes} bytes) to contain even "
            f"one complex sample of type '{dtype}': {path}"
        )

    if offset_samples < 0:
        raise ValueError(f"offset_samples must be >= 0, got {offset_samples}.")
    if offset_samples >= total_available:
        raise ValueError(
            f"offset_samples ({offset_samples}) >= total available samples "
            f"({total_available}) in file."
        )

    remaining = total_available - offset_samples
    samples_to_read = remaining if max_samples is None else min(max_samples, remaining)
    elements_to_read = samples_to_read * 2  # 2 real elements per complex sample

    byte_offset = offset_samples * bytes_per_complex
    raw = np.fromfile(str(path), dtype=elem_dtype, count=elements_to_read, offset=byte_offset)

    # Trim to even count (guard against 1-element truncation from imprecise file sizes)
    if len(raw) % 2 != 0:
        logger.warning(
            "Odd number of IQ elements read from '%s'; discarding last element.", path.name
        )
        raw = raw[:-1]

    if len(raw) == 0:
        raise ValueError(f"No data remaining after reading IQ file (offset may be too large): {path}")

    # ------------------------------------------------------------------
    # Convert integers → float32
    # ------------------------------------------------------------------
    raw_f = _int_to_float(raw, elem_dtype)

    # ------------------------------------------------------------------
    # Deinterleave: IQ or QI
    # ------------------------------------------------------------------
    if iq_format_upper == "IQ":
        i_comp = raw_f[0::2]
        q_comp = raw_f[1::2]
    else:  # QI
        q_comp = raw_f[0::2]
        i_comp = raw_f[1::2]

    samples = (i_comp + 1j * q_comp).astype(np.complex64)

    # ------------------------------------------------------------------
    # Validate for NaN/Inf
    # ------------------------------------------------------------------
    _validate_finite(samples, context=str(path.name))

    # ------------------------------------------------------------------
    # Sample-rate resolution
    # ------------------------------------------------------------------
    warnings_list: list = []
    effective_sr = sample_rate

    if effective_sr is None:
        companion = _find_companion_metadata(path)
        if companion is not None:
            effective_sr = _extract_sample_rate_from_meta(companion)
            if effective_sr is not None:
                logger.info("Sample rate %.1f Hz loaded from companion metadata.", effective_sr)

    if effective_sr is None:
        warnings_list.append(make_warning(
            code="SAMPLE_RATE_UNKNOWN",
            message=(
                f"No sample rate was supplied and none could be inferred from companion "
                f"metadata for '{path.name}'. sample_rate is set to None. "
                "Absolute-frequency-dependent calculations (PSD, CFO estimation, baud rate "
                "estimation, demodulation) will fail or produce incorrect results until "
                "sample_rate is provided by the caller."
            ),
            details={"file": str(path.resolve()), "dtype": dtype, "iq_format": iq_format},
        ))
        logger.warning(
            "Sample rate unknown for '%s'. Provide sample_rate or a companion JSON.", path.name
        )

    if effective_sr is not None and effective_sr <= 0:
        raise ValueError(
            f"Resolved sample_rate ({effective_sr}) must be positive and non-zero."
        )

    # ------------------------------------------------------------------
    # Odd raw-element warning (file may have been truncated)
    # ------------------------------------------------------------------
    raw_elements_in_file = file_size_bytes // elem_size
    if raw_elements_in_file % 2 != 0:
        warnings_list.append(make_warning(
            code="ODD_IQ_ELEMENT_COUNT",
            message=(
                f"File '{path.name}' contains an odd number of raw IQ elements "
                f"({raw_elements_in_file}). The last unpaired element was discarded. "
                "The file may be truncated or corrupt."
            ),
            details={"raw_elements": raw_elements_in_file, "discarded": 1},
        ))

    companion_meta = _find_companion_metadata(path) or {}

    # Extract center_freq from companion metadata when caller did not supply one
    effective_cf = center_freq
    if effective_cf == 0.0 and companion_meta:
        for cf_key in ("center_freq", "centerFreq", "center_frequency"):
            if cf_key in companion_meta:
                effective_cf = float(companion_meta[cf_key])
                break
        # SigMF captures schema
        if effective_cf == 0.0 and "captures" in companion_meta:
            caps = companion_meta["captures"]
            if caps and "core:frequency" in caps[0]:
                effective_cf = float(caps[0]["core:frequency"])

    metadata = {
        "source_file": str(path.resolve()),
        "dtype": dtype,
        "iq_format": iq_format,
        "offset_samples": offset_samples,
        "raw_file_bytes": file_size_bytes,
        "companion_metadata": companion_meta,
        "iq_convention": IQ_CONVENTION,
        "bit_ordering": BIT_ORDERING,
    }

    return SignalData(
        samples=samples,
        sample_rate=effective_sr,
        source_path=str(path.resolve()),
        source_format=f"iq_{dtype}",
        center_freq=effective_cf,
        is_complex=True,
        metadata=metadata,
        warnings=warnings_list,
    )


def load_signal(
    path: Union[str, Path],
    dtype: str = "float32",
    iq_format: str = "IQ",
    sample_rate: Optional[float] = None,
    center_freq: float = 0.0,
    to_analytic: bool = True,
) -> SignalData:
    """
    Unified entry-point that dispatches to ``load_wav`` or ``load_iq`` based
    on file extension.

    Extension dispatch table:
      - ``.wav``              → :func:`load_wav`
      - ``.iq``, ``.raw``,   → :func:`load_iq`
        ``.bin``, ``.dat``,
        ``.sigmf-data``

    Parameters
    ----------
    path:
        Path to the signal file.
    dtype:
        Only used for raw IQ files. See :func:`load_iq`.
    iq_format:
        Only used for raw IQ files (``'IQ'`` or ``'QI'``). See :func:`load_iq`.
    sample_rate:
        Only used for raw IQ files. See :func:`load_iq`.
    center_freq:
        RF center frequency in Hz (passed to both loaders).
    to_analytic:
        Only used for mono WAV files. See :func:`load_wav`.

    Returns
    -------
    SignalData

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".wav":
        return load_wav(path, center_freq=center_freq, to_analytic=to_analytic)
    elif ext in {".iq", ".raw", ".bin", ".dat", ".sigmf-data", ""}:
        return load_iq(
            path,
            dtype=dtype,
            iq_format=iq_format,
            sample_rate=sample_rate,
            center_freq=center_freq,
        )
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}' for '{path.name}'. "
            f"Supported: .wav, .iq, .raw, .bin, .dat, .sigmf-data"
        )


# =========================================================================== #
# Public API: Validation                                                        #
# =========================================================================== #

def validate_signal(sig: SignalData) -> Dict[str, Any]:
    """
    Inspect a ``SignalData`` object and return a validation report.

    Checks performed:
      - Non-empty samples array
      - No NaN or Inf values
      - Sample rate is known and positive (warns if None)
      - Reasonable sample count (warns if < 16)
      - Is-complex flag consistent with actual array dtype
      - Metadata dict is present

    Returns
    -------
    dict with keys:
      ``valid`` (bool), ``issues`` (list of str), ``warnings`` (list of warning dicts)
    """
    issues: list = []
    warnings_out: list = []

    if sig.samples is None or sig.samples.size == 0:
        issues.append("samples array is empty or None.")

    if sig.samples is not None and sig.samples.size > 0:
        if not np.all(np.isfinite(sig.samples)):
            n_bad = int(np.sum(~np.isfinite(sig.samples)))
            issues.append(f"samples contain {n_bad} non-finite (NaN/Inf) values.")

        if sig.num_samples < 16:
            warnings_out.append(make_warning(
                "VERY_SHORT_SIGNAL",
                f"Signal has only {sig.num_samples} samples. "
                "Feature extraction and classification results will be unreliable.",
                {"num_samples": sig.num_samples},
            ))

        is_really_complex = np.iscomplexobj(sig.samples)
        if sig.is_complex and not is_really_complex:
            warnings_out.append(make_warning(
                "IS_COMPLEX_FLAG_MISMATCH",
                "SignalData.is_complex=True but samples array is real dtype.",
            ))
        if not sig.is_complex and is_really_complex:
            warnings_out.append(make_warning(
                "IS_COMPLEX_FLAG_MISMATCH",
                "SignalData.is_complex=False but samples array is complex dtype.",
            ))

    if sig.sample_rate is None:
        warnings_out.append(make_warning(
            "SAMPLE_RATE_UNKNOWN",
            "sample_rate is None. Frequency-dependent calculations will fail.",
        ))
    elif sig.sample_rate <= 0:
        issues.append(f"sample_rate is non-positive: {sig.sample_rate}.")

    if not isinstance(sig.metadata, dict):
        issues.append("metadata is not a dict.")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings_out,
    }


# =========================================================================== #
# Public API: Normalization                                                     #
# =========================================================================== #

def normalize_signal(
    sig: SignalData,
    remove_dc: bool = True,
    normalize_amplitude: bool = True,
    gain: float = 1.0,
    clip_threshold: Optional[float] = None,
) -> SignalData:
    """
    Return a new ``SignalData`` with optional DSP conditioning applied.

    .. note::
        This function never modifies the original ``SignalData`` or the
        underlying file. It always operates on a copy of the samples.

    Parameters
    ----------
    sig:
        Input signal.
    remove_dc:
        If True, subtract the complex mean from the samples (DC offset removal).
    normalize_amplitude:
        If True, scale samples so that RMS power = 1.0 (0 dBFS). No-op if
        signal power is already zero.
    gain:
        Multiplicative gain applied after amplitude normalization (default 1.0).
    clip_threshold:
        If not None, emit a warning if any sample magnitude exceeds this value
        after processing. Typical values: 1.0 for normalized signals.

    Returns
    -------
    SignalData
        A new ``SignalData`` with conditioned samples. All metadata from the
        original is preserved; a ``NORMALIZATION_APPLIED`` entry is added.

    Raises
    ------
    ValueError
        If samples contain NaN or Inf values before processing.
    """
    # Work on copy — never touch the original
    samples = sig.samples.copy().astype(np.complex64 if sig.is_complex else np.float32)

    _validate_finite(samples, context="before normalize_signal")

    warnings_list = list(sig.warnings)  # carry existing warnings forward
    ops_applied: list = []

    if remove_dc:
        dc = np.mean(samples)
        samples = samples - dc
        ops_applied.append(f"dc_removed (offset={dc:.4g})")

    if normalize_amplitude:
        rms = float(np.sqrt(np.mean(np.abs(samples) ** 2)))
        if rms > 1e-12:
            samples = samples / rms
            ops_applied.append(f"amplitude_normalized (rms_before={rms:.4g})")
        else:
            warnings_list.append(make_warning(
                "ZERO_POWER_SIGNAL",
                "Signal RMS power is effectively zero; amplitude normalization skipped.",
                {"rms": rms},
            ))

    if gain != 1.0:
        samples = samples * gain
        ops_applied.append(f"gain_applied ({gain:.4g}x)")

    if clip_threshold is not None:
        magnitudes = np.abs(samples)
        clipped = int(np.sum(magnitudes > clip_threshold))
        if clipped > 0:
            pct = 100.0 * clipped / len(samples)
            warnings_list.append(make_warning(
                "CLIPPING_DETECTED",
                f"{clipped} samples ({pct:.2f}%) exceed clip threshold {clip_threshold:.4g} "
                "after normalization. Signal may be distorted.",
                {
                    "n_clipped": clipped,
                    "pct_clipped": round(pct, 4),
                    "threshold": clip_threshold,
                    "max_magnitude": float(np.max(magnitudes)),
                },
            ))

    new_meta = dict(sig.metadata)
    new_meta["normalization_applied"] = {
        "ops": ops_applied,
        "remove_dc": remove_dc,
        "normalize_amplitude": normalize_amplitude,
        "gain": gain,
        "clip_threshold": clip_threshold,
    }

    return SignalData(
        samples=samples,
        sample_rate=sig.sample_rate,
        source_path=sig.source_path,
        source_format=sig.source_format,
        center_freq=sig.center_freq,
        is_complex=sig.is_complex,
        metadata=new_meta,
        warnings=warnings_list,
    )


# =========================================================================== #
# Public API: Metadata Inference                                                #
# =========================================================================== #

def infer_basic_metadata(sig: SignalData) -> Dict[str, Any]:
    """
    Infer and return basic analytical metadata from a ``SignalData`` object
    without performing any DSP or modulation-aware processing.

    Computed fields:
      - ``num_samples``         : number of samples
      - ``duration_sec``        : duration in seconds (None if sample_rate unknown)
      - ``sample_rate_hz``      : sampling frequency (None if unknown)
      - ``center_freq_hz``      : RF center frequency (0.0 if unset)
      - ``is_complex``          : True/False
      - ``dtype``               : numpy dtype string
      - ``rms_power_dbfs``      : RMS power in dBFS
      - ``peak_magnitude``      : peak sample magnitude
      - ``dynamic_range_db``    : peak/RMS ratio in dB (None if power is 0)
      - ``dc_offset_i``         : mean I component
      - ``dc_offset_q``         : mean Q component (0.0 if real)
      - ``estimated_bandwidth_hz``: approximate signal bandwidth as sample_rate/2
                                   (None if sample_rate unknown)

    Returns
    -------
    dict
    """
    n = sig.num_samples
    samples = sig.samples

    if n == 0:
        return {
            "num_samples": 0,
            "duration_sec": None,
            "sample_rate_hz": sig.sample_rate,
            "center_freq_hz": sig.center_freq,
            "is_complex": sig.is_complex,
            "dtype": sig.dtype,
            "rms_power_dbfs": -120.0,
            "peak_magnitude": 0.0,
            "dynamic_range_db": None,
            "dc_offset_i": 0.0,
            "dc_offset_q": 0.0,
            "estimated_bandwidth_hz": None,
        }

    rms_sq = float(np.mean(np.abs(samples) ** 2))
    rms_power_dbfs = float(10.0 * np.log10(max(rms_sq, 1e-12)))
    peak_mag = float(np.max(np.abs(samples)))
    peak_dbfs = float(20.0 * np.log10(max(peak_mag, 1e-12)))
    rms_dbfs_v = float(10.0 * np.log10(max(np.sqrt(rms_sq), 1e-12)))
    dynamic_range_db = peak_dbfs - rms_dbfs_v if rms_sq > 1e-24 else None

    if np.iscomplexobj(samples):
        dc_i = float(np.mean(np.real(samples)))
        dc_q = float(np.mean(np.imag(samples)))
    else:
        dc_i = float(np.mean(samples))
        dc_q = 0.0

    bw = sig.sample_rate / 2.0 if sig.sample_rate is not None else None

    return {
        "num_samples": n,
        "duration_sec": sig.duration,
        "sample_rate_hz": sig.sample_rate,
        "center_freq_hz": sig.center_freq,
        "is_complex": sig.is_complex,
        "dtype": sig.dtype,
        "rms_power_dbfs": rms_power_dbfs,
        "peak_magnitude": peak_mag,
        "dynamic_range_db": dynamic_range_db,
        "dc_offset_i": dc_i,
        "dc_offset_q": dc_q,
        "estimated_bandwidth_hz": bw,
    }


# =========================================================================== #
# Serialization (carry forward from original io.py)                            #
# =========================================================================== #

def save_iq_file(
    signal_data: SignalData,
    file_path: Union[str, Path],
    data_type: str = "complex64",
    write_metadata_json: bool = True,
) -> Path:
    """
    Serialize ``SignalData`` to a raw IQ binary file and companion metadata JSON.

    Parameters
    ----------
    signal_data:
        Signal to serialize.
    file_path:
        Output path (parent directories are created automatically).
    data_type:
        On-disk format: ``'complex64'`` (float32 I/Q) or ``'int16'``.
    write_metadata_json:
        If True, write a companion .json file with sample_rate, center_freq, etc.

    Returns
    -------
    Path to the written IQ file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    samples = signal_data.samples
    if not np.iscomplexobj(samples):
        samples = samples.astype(np.complex64)

    i_comp = np.real(samples).astype(np.float32)
    q_comp = np.imag(samples).astype(np.float32)

    if data_type.lower() in ("complex64", "float32"):
        interleaved = np.empty(2 * len(samples), dtype=np.float32)
        interleaved[0::2] = i_comp
        interleaved[1::2] = q_comp
        interleaved.tofile(str(path))
    elif data_type.lower() == "int16":
        max_val = max(float(np.max(np.abs(i_comp))), float(np.max(np.abs(q_comp))), 1e-9)
        scale = 32700.0 / max_val if max_val > 1.0 else 32700.0
        i_int = np.clip(i_comp * scale, -32768, 32767).astype(np.int16)
        q_int = np.clip(q_comp * scale, -32768, 32767).astype(np.int16)
        interleaved = np.empty(2 * len(samples), dtype=np.int16)
        interleaved[0::2] = i_int
        interleaved[1::2] = q_int
        interleaved.tofile(str(path))
    else:
        raise ValueError(f"Unsupported save data_type: '{data_type}'")

    if write_metadata_json:
        meta_path = path.with_suffix(".json")
        meta_content: Dict[str, Any] = {
            "sample_rate": signal_data.sample_rate,
            "center_freq": signal_data.center_freq,
            "is_complex": signal_data.is_complex,
            "num_samples": signal_data.num_samples,
            "duration_sec": signal_data.duration_sec,
            "data_type": data_type,
            "source_format": signal_data.source_format,
            "custom_metadata": signal_data.metadata,
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta_content, fh, indent=2)

    return path


def save_wav_file(
    signal_data: SignalData,
    file_path: Union[str, Path],
) -> Path:
    """
    Serialize ``SignalData`` to a WAV file.

    Complex signals are saved as stereo (Ch 0 = I, Ch 1 = Q).
    Real signals are saved as mono.

    Returns
    -------
    Path to the written WAV file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    samples = signal_data.samples
    if signal_data.sample_rate is None:
        raise ValueError("Cannot save WAV file: sample_rate is None.")
    fs = int(signal_data.sample_rate)

    if np.iscomplexobj(samples):
        i_comp = np.real(samples).astype(np.float32)
        q_comp = np.imag(samples).astype(np.float32)
        max_val = max(float(np.max(np.abs(i_comp))), float(np.max(np.abs(q_comp))), 1e-9)
        scale = 0.95 / max_val if max_val > 1.0 else 0.95
        stereo = np.column_stack((i_comp * scale, q_comp * scale)).astype(np.float32)
        wavfile.write(str(path), fs, stereo)
    else:
        mono = samples.astype(np.float32)
        max_val = max(float(np.max(np.abs(mono))), 1e-9)
        scale = 0.95 / max_val if max_val > 1.0 else 0.95
        wavfile.write(str(path), fs, (mono * scale).astype(np.float32))

    return path


# =========================================================================== #
# Backwards-compatibility aliases (used by pipeline, app, tests, generator)   #
# =========================================================================== #
#
# The old functions load_iq_file / load_wav_file had slightly different
# signatures. We preserve them here with compatible defaults so that existing
# callers are not broken.

def load_iq_file(
    file_path: Union[str, Path],
    sample_rate: Optional[float] = None,
    data_type: str = "complex64",
    dtype: Optional[str] = None,
    center_freq: float = 0.0,
    offset_samples: int = 0,
    max_samples: Optional[int] = None,
) -> SignalData:
    """
    Backwards-compatible wrapper around :func:`load_iq`.

    The old API used ``data_type`` as the dtype keyword; both are accepted.
    """
    effective_dtype = dtype if dtype is not None else data_type
    return load_iq(
        path=file_path,
        dtype=effective_dtype,
        iq_format="IQ",
        sample_rate=sample_rate,
        center_freq=center_freq,
        offset_samples=offset_samples,
        max_samples=max_samples,
    )


def load_wav_file(
    file_path: Union[str, Path],
    center_freq: float = 0.0,
    to_analytic: bool = True,
) -> SignalData:
    """
    Backwards-compatible wrapper around :func:`load_wav`.
    """
    return load_wav(path=file_path, center_freq=center_freq, to_analytic=to_analytic)
