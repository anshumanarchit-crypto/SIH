"""
tests/golden/test_golden_roundtrip.py

End-to-end golden verification test harness:
Validates that passing the handoff TX bits (data/handoff/bitsGx.txt) through
the decode_chain recovers the original SOURCE BITS (data/golden/Gx/source_bits.txt)
with exact zero Bit Error Rate (BER = 0.0) across all Golden cases G2 through G7.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from spectralq.encode_chain import EncodeConfig, decode_chain

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("case_id", ["G2", "G3", "G4", "G5", "G6", "G7"])
def test_golden_case_roundtrip(case_id):
    """Verify that every golden handoff bitstream decodes to exact source bits (BER = 0.0)."""
    golden_dir = _REPO_ROOT / "data" / "golden" / case_id
    handoff_file = _REPO_ROOT / "data" / "handoff" / f"bits{case_id}.txt"
    source_file = golden_dir / "source_bits.txt"
    metadata_file = golden_dir / "metadata.json"

    assert source_file.exists(), f"Source bits missing for {case_id}"
    assert metadata_file.exists(), f"Metadata missing for {case_id}"
    assert handoff_file.exists(), f"Handoff bits missing for {case_id}"

    # Load source bits
    source_str = source_file.read_text(encoding="utf-8").strip()
    source_bits = np.array([int(c) for c in source_str], dtype=int)

    # Load handoff TX bits
    tx_str = handoff_file.read_text(encoding="utf-8").strip()
    tx_bits = np.array([int(c) for c in tx_str], dtype=int)

    # Load metadata
    meta = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert len(source_bits) == meta["source_bit_length"]
    assert len(tx_bits) == meta["tx_bit_length"]

    config = EncodeConfig(
        case_id=case_id,
        modulation=meta["modulation"],
        fec_type=meta["fec_type"],
        interleaver_type=meta["interleaver_type"],
        fec_params=meta["fec_parameters"],
        interleaver_params=meta["interleaver_parameters"],
    )

    # Execute decode_chain
    dec_res = decode_chain(
        received_bits=tx_bits,
        config=config,
        interleaver_meta=meta.get("interleaver_metadata")
    )

    assert dec_res.decoder_success is True, f"{case_id} decoding failed"
    recovered = dec_res.recovered_source_bits
    assert len(recovered) == len(source_bits), f"{case_id} bit count mismatch: got {len(recovered)}, expected {len(source_bits)}"

    bit_errors = int(np.sum(source_bits != recovered))
    ber = bit_errors / len(source_bits)
    assert ber == 0.0, f"{case_id} BER = {ber} ({bit_errors} bit errors)"
    assert np.array_equal(source_bits, recovered), f"{case_id} recovered bits do not match source bits"
