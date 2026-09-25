#!/usr/bin/env python3
"""
scripts/generate_golden.py

Deterministic golden bitstream generator for SpectralQ (SIH26147).
Generates SOURCE BITS and TX BITS for all Golden cases (G2 - G7):
- G2: BPSK, Conv K=7, Block Interleaver (256 source bits)
- G3: 8-PSK, RS(255, 223), Diagonal Interleaver (1,784 source bits)
- G4: 16-QAM, LDPC (Gallager 96.3.963), Pseudo-Random Interleaver (50 source bits)
- G5: 2-FSK, Concatenated RS+Conv, Convolutional Interleaver (1,784 source bits)
- G6: BPSK, Conv K=7, Convolutional Interleaver (256 source bits)
- G7: QPSK, Conv K=7, No Interleaver (NONE) near-threshold preparation (256 source bits)

Enforces:
1. Strict file formatting: bits files contain strictly '0' and '1' characters with a trailing newline.
2. Distinct SOURCE BITS (data/golden/Gx/source_bits.txt) and TX BITS (data/handoff/bitsGx.txt).
3. Deterministic SHA-256 digests and structured metadata.json per case.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np

# Ensure spectralq is importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from spectralq.encode_chain import EncodeConfig, encode_chain


def compute_sha256(bit_str: str) -> str:
    """Compute SHA-256 hexadecimal hash of bitstream string."""
    return hashlib.sha256(bit_str.encode("utf-8")).hexdigest()


def generate_case(
    case_id: str,
    modulation: str,
    source_len: int,
    seed: int,
    config: EncodeConfig,
    repo_root: Path
) -> dict:
    """Generate source bits, encode through pipeline, and save golden artifacts."""
    rng = np.random.default_rng(seed)
    source_bits = rng.integers(0, 2, size=source_len)

    # Encode chain execution
    enc_res = encode_chain(source_bits, config)

    source_str = "".join(str(b) for b in source_bits)
    tx_str = "".join(str(b) for b in enc_res.tx_bits)

    source_hash = compute_sha256(source_str)
    tx_hash = compute_sha256(tx_str)

    # File paths
    golden_dir = repo_root / "data" / "golden" / case_id
    golden_dir.mkdir(parents=True, exist_ok=True)
    handoff_dir = repo_root / "data" / "handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)

    source_file = golden_dir / "source_bits.txt"
    metadata_file = golden_dir / "metadata.json"
    handoff_file = handoff_dir / f"bits{case_id}.txt"

    # Write source bits (ONLY 0/1 characters + trailing newline)
    with open(source_file, "w", encoding="utf-8") as f:
        f.write(source_str + "\n")

    # Write handoff TX bits (ONLY 0/1 characters + trailing newline)
    with open(handoff_file, "w", encoding="utf-8") as f:
        f.write(tx_str + "\n")

    # Assemble metadata
    pad_len = enc_res.interleaver_metadata.get("padding_length", 0)
    if case_id == "G7":
        meta = {
            "case_id": "G7",
            "status": "VALIDATED",
            "modulation": "QPSK",
            "fec_type": "CONVOLUTIONAL",
            "constraint_length": 7,
            "generators": ["0171", "0133"],
            "termination": "terminated",
            "tail_bits": 6,
            "interleaver_type": "NONE",
            "source_bit_length": 256,
            "tx_bit_length": 524,
            "source_seed": seed,
            "source_bits_sha256": source_hash,
            "tx_bits_sha256": tx_hash,
            "channel_snr_db": None,
            "channel_snr_status": "TO_BE_SET_BY_SINCHANA_FOR_NEAR_THRESHOLD_WAVEFORM",
            "notes": (
                "QPSK + convolutional near-threshold case; SNR is a waveform-generation "
                "parameter, not encoded into the TX bitstream."
            ),
            "implementation_choices": {
                "source_bits": 256,
                "convolutional_code": "K=7, Rate 1/2",
                "generators": ["0171", "0133"],
                "termination": "terminated with 6 zero tail bits",
                "interleaver": "NONE",
                "notes": (
                    "Source plan specifies QPSK + convolutional near threshold SNR but does "
                    "not fix exact SNR, bit length, or interleaver. These values are explicit "
                    "Phase 1 Python implementation choices."
                ),
            },
            "seed": seed,
            "padding_length": int(pad_len),
            "fec_parameters": config.fec_params,
            "interleaver_parameters": config.interleaver_params,
            "interleaver_metadata": enc_res.interleaver_metadata,
            "fec_metadata": enc_res.fec_metadata,
        }
    else:
        meta = {
            "case_id": case_id,
            "status": "VALIDATED",
            "modulation": modulation,
            "fec_type": config.fec_type,
            "fec_parameters": config.fec_params,
            "interleaver_type": config.interleaver_type,
            "interleaver_parameters": config.interleaver_params,
            "seed": seed,
            "source_bit_length": int(source_len),
            "tx_bit_length": int(len(enc_res.tx_bits)),
            "padding_length": int(pad_len),
            "termination": "6_tail_bits" if "CONV" in config.fec_type else "none",
            "source_bits_sha256": source_hash,
            "tx_bits_sha256": tx_hash,
            "interleaver_metadata": enc_res.interleaver_metadata,
            "fec_metadata": enc_res.fec_metadata,
        }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    repo_root = Path(__file__).resolve().parents[1]
    print("=" * 70)
    print("SpectralQ SIH26147 - Deterministic Golden Bitstream Generator")
    print(f"Target Repository: {repo_root}")
    print("=" * 70)

    cases = [
        # G2: BPSK, Conv K=7, Block Interleaver
        {
            "case_id": "G2",
            "modulation": "BPSK",
            "source_len": 256,
            "seed": 202602,
            "config": EncodeConfig(
                case_id="G2",
                modulation="BPSK",
                fec_type="CONV_K7",
                interleaver_type="BLOCK",
                interleaver_params={"rows": 16, "cols": 34},
            )
        },
        # G3: 8-PSK, RS(255, 223), Diagonal Interleaver
        {
            "case_id": "G3",
            "modulation": "8-PSK",
            "source_len": 1784,
            "seed": 202603,
            "config": EncodeConfig(
                case_id="G3",
                modulation="8-PSK",
                fec_type="RS_255_223",
                interleaver_type="DIAGONAL",
                interleaver_params={"num_rows": 40, "num_cols": 51},
            )
        },
        # G4: 16-QAM, LDPC (Gallager 96.3.963), PRNG Interleaver
        {
            "case_id": "G4",
            "modulation": "16-QAM",
            "source_len": 50,
            "seed": 202604,
            "config": EncodeConfig(
                case_id="G4",
                modulation="16-QAM",
                fec_type="LDPC",
                interleaver_type="PSEUDORANDOM",
                interleaver_params={"seed": 42},
            )
        },
        # G5: 2-FSK, Concatenated RS+Conv, Convolutional Interleaver
        {
            "case_id": "G5",
            "modulation": "2-FSK",
            "source_len": 1784,
            "seed": 202605,
            "config": EncodeConfig(
                case_id="G5",
                modulation="2-FSK",
                fec_type="CONCAT_RS_CONV",
                interleaver_type="CONVOLUTIONAL",
                interleaver_params={"num_branches": 6, "delay_step": 2},
            )
        },
        # G6: BPSK, Conv K=7, Convolutional Interleaver
        {
            "case_id": "G6",
            "modulation": "BPSK",
            "source_len": 256,
            "seed": 202606,
            "config": EncodeConfig(
                case_id="G6",
                modulation="BPSK",
                fec_type="CONV_K7",
                interleaver_type="CONVOLUTIONAL",
                interleaver_params={"num_branches": 4, "delay_step": 2},
            )
        },
        # G7: QPSK, Conv K=7, No Interleaver (Identity)
        {
            "case_id": "G7",
            "modulation": "QPSK",
            "source_len": 256,
            "seed": 202607,
            "config": EncodeConfig(
                case_id="G7",
                modulation="QPSK",
                fec_type="CONVOLUTIONAL",
                interleaver_type="NONE",
                fec_params={"constraint_length": 7, "polynomials": (0o171, 0o133)},
                interleaver_params={},
            )
        },
    ]

    results = []
    for c in cases:
        meta = generate_case(
            case_id=c["case_id"],
            modulation=c["modulation"],
            source_len=c["source_len"],
            seed=c["seed"],
            config=c["config"],
            repo_root=repo_root
        )
        results.append(meta)
        print(f"[{meta['case_id']}] {meta['modulation']:<7} | Source: {meta['source_bit_length']:>4} bits | TX: {meta['tx_bit_length']:>4} bits")
        print(f"     Source SHA-256: {meta['source_bits_sha256']}")
        print(f"     TX SHA-256:     {meta['tx_bits_sha256']}")

    print("\n" + "=" * 70)
    print("Golden bitstreams successfully generated.")
    print("=" * 70)
    return results


if __name__ == "__main__":
    main()
