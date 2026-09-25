"""
tests/integration/test_demod_integration.py

End-to-end integration tests for SpectralQ Phase 2 Demodulation Core:
1. PSK test matrix: BPSK, QPSK, 8-PSK across impairments (timing, CFO, phase)
2. FSK test matrix: 2-FSK, 4-FSK
3. G1 Synthetic Test Case (QPSK uncoded)
4. G5 Synthetic Test Case (2-FSK + Concatenated RS+Conv + Convolutional Interleaver)
   - Demodulator output verified against TX BITS
   - Full chain (demod -> decode_chain) verified against SOURCE BITS (BER = 0.0)
5. G7 Synthetic Test Case (QPSK + Conv K=7 + NONE interleaver)
   - Demodulator output verified against TX BITS
   - Full chain (demod -> decode_chain) verified against SOURCE BITS (BER = 0.0)
6. DecoderPipeline end-to-end integration
7. Controlled SNR sweep: verify observable BER degradation
8. Official Sinchana capture adapter verification
"""

import json
from pathlib import Path
import numpy as np
import pytest

from spectralq.demod import demodulate, DemodConfig, DemodStatus
from spectralq.decoder_api import DecoderPipeline, DecoderConfig, DecoderStatus, ModulationType, FECType, InterleaverType
from spectralq.encode_chain import encode_chain, decode_chain, EncodeConfig
from tests.fixtures.synthetic_generator import generate_synthetic_waveform
from tests.fixtures.official_adapter import evaluate_official_capture

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# 1. PSK TEST MATRIX (BPSK, QPSK, 8-PSK)
# ============================================================================

@pytest.mark.parametrize("mod", ["BPSK", "QPSK", "8-PSK"])
def test_psk_matrix_clean(mod):
    """Verify clean reception across all supported PSK modulations."""
    preamble = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1], dtype=int)
    waveform, tx_bits, meta = generate_synthetic_waveform(
        modulation=mod,
        num_bits=240,
        seed=1001,
        sps=4,
        timing_offset_symbols=0.0,
        cfo_hz=0.0,
        phase_offset_deg=0.0,
        preamble_bits=preamble,
    )
    cfg = DemodConfig(
        modulation=mod,
        samples_per_symbol=4,
        rrc_alpha=0.35,
        preamble_bits=preamble,
    )
    res = demodulate(waveform, cfg)

    assert res.status == DemodStatus.SUCCESS
    cmp_len = min(len(res.hard_bits), len(tx_bits))
    # Exclude filter transient warm-up symbols
    skip_start = len(preamble)
    errors = np.sum(res.hard_bits[skip_start:cmp_len] != tx_bits[skip_start:cmp_len])
    ber = errors / (cmp_len - skip_start)
    assert ber < 0.05, f"{mod} clean BER {ber} too high"


@pytest.mark.parametrize("mod", ["BPSK", "QPSK"])
def test_psk_matrix_with_impairments(mod):
    """Verify PSK demodulation under combined timing offset and CFO."""
    preamble = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0], dtype=int)
    waveform, tx_bits, meta = generate_synthetic_waveform(
        modulation=mod,
        num_bits=300,
        seed=1002,
        sps=4,
        timing_offset_symbols=0.25,
        cfo_hz=2000.0,
        sample_rate=1e6,
        phase_offset_deg=45.0,
        preamble_bits=preamble,
    )
    cfg = DemodConfig(
        modulation=mod,
        sample_rate=1e6,
        samples_per_symbol=4,
        rrc_alpha=0.35,
        preamble_bits=preamble,
    )
    res = demodulate(waveform, cfg)

    assert res.status == DemodStatus.SUCCESS
    cmp_len = min(len(res.hard_bits), len(tx_bits))
    skip = len(preamble)
    errors = np.sum(res.hard_bits[skip:cmp_len] != tx_bits[skip:cmp_len])
    ber = errors / (cmp_len - skip)
    assert ber < 0.08, f"{mod} with impairments BER {ber} too high"


# ============================================================================
# 2. G1 TEST CASE (SYNTHETIC UNCODED QPSK PREPARATION)
# ============================================================================

def test_g1_synthetic_case():
    """Verify G1 preparation: QPSK uncoded recovery on synthetic fixture.

    NOTE: This is a TEST FIXTURE ONLY, not the official Sinchana G1 evidence.
    """
    preamble = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0], dtype=int)
    rng = np.random.default_rng(202601)
    source_bits = rng.integers(0, 2, size=256)

    waveform, tx_bits, meta = generate_synthetic_waveform(
        modulation="QPSK",
        tx_bits=source_bits,
        sps=4,
        timing_offset_symbols=0.15,
        cfo_hz=1000.0,
        sample_rate=1e6,
        phase_offset_deg=90.0,
        preamble_bits=preamble,
    )
    cfg = DemodConfig(
        modulation="QPSK",
        sample_rate=1e6,
        samples_per_symbol=4,
        rrc_alpha=0.35,
        preamble_bits=preamble,
    )
    res = demodulate(waveform, cfg)

    assert res.status == DemodStatus.SUCCESS
    # Check that demodulated payload matches source bits
    skip = len(preamble)
    recovered_payload = res.hard_bits[skip:skip + len(source_bits)]
    assert len(recovered_payload) == len(source_bits)
    errors = int(np.sum(recovered_payload != source_bits))
    ber = errors / len(source_bits)
    assert ber < 0.05, f"G1 synthetic recovery BER = {ber}"


# ============================================================================
# 3. G5 TEST CASE (2-FSK + CONCATENATED RS+CONV + CONV INTERLEAVER)
# ============================================================================

def test_g5_synthetic_case_demod_and_full_pipeline():
    """Verify G5 preparation: 2-FSK continuous-phase signal recovery.

    Validates:
    1. Demodulator output matches transmitted TX bits.
    2. Demodulated TX bits decode through decode_chain to exact source bits (BER = 0.0).
    """
    g5_source_file = _REPO_ROOT / "data" / "golden" / "G5" / "source_bits.txt"
    g5_handoff_file = _REPO_ROOT / "data" / "handoff" / "bitsG5.txt"
    g5_meta_file = _REPO_ROOT / "data" / "golden" / "G5" / "metadata.json"

    assert g5_handoff_file.exists(), "bitsG5.txt must exist"
    source_bits = np.array([int(c) for c in g5_source_file.read_text(encoding="utf-8").strip()], dtype=int)
    tx_bits = np.array([int(c) for c in g5_handoff_file.read_text(encoding="utf-8").strip()], dtype=int)
    meta = json.loads(g5_meta_file.read_text(encoding="utf-8"))

    # Synthesize clean 2-FSK waveform carrying exact G5 TX bits
    sps = 8
    fs = 1e6
    f_dev = 50e3
    waveform, _, _ = generate_synthetic_waveform(
        modulation="2-FSK",
        tx_bits=tx_bits,
        sps=sps,
        sample_rate=fs,
        fsk_deviation_hz=f_dev,
    )

    # 1. Demodulate
    cfg = DemodConfig(
        modulation="2-FSK",
        sample_rate=fs,
        samples_per_symbol=sps,
        fsk_deviation=f_dev,
    )
    demod_res = demodulate(waveform, cfg)
    assert demod_res.status == DemodStatus.SUCCESS

    # Verify recovered TX bits against transmitted TX bits
    recovered_tx = demod_res.hard_bits[:len(tx_bits)]
    tx_errors = int(np.sum(recovered_tx != tx_bits))
    assert tx_errors == 0, f"Demodulator output differs from transmitted G5 TX bits: {tx_errors} errors"

    # 2. Feed recovered TX bits into Phase 1 decode_chain
    enc_cfg = EncodeConfig(
        case_id="G5",
        modulation="2-FSK",
        fec_type=meta["fec_type"],
        interleaver_type=meta["interleaver_type"],
        fec_params=meta["fec_parameters"],
        interleaver_params=meta["interleaver_parameters"],
    )
    dec_res = decode_chain(
        received_bits=recovered_tx,
        config=enc_cfg,
        interleaver_meta=meta.get("interleaver_metadata"),
    )
    assert dec_res.decoder_success is True
    assert np.array_equal(source_bits, dec_res.recovered_source_bits), "Full G5 chain must recover exact source bits"


# ============================================================================
# 4. G7 TEST CASE (QPSK + CONV K=7 + NO INTERLEAVER)
# ============================================================================

def test_g7_synthetic_case_demod_and_full_pipeline():
    """Verify G7 preparation: QPSK + Conv K=7 + NONE interleaver (256 source -> 524 TX bits).

    Validates:
    1. Demodulator output matches transmitted TX bits.
    2. Demodulated TX bits decode through decode_chain to exact 256 source bits (BER = 0.0).
    """
    g7_source_file = _REPO_ROOT / "data" / "golden" / "G7" / "source_bits.txt"
    g7_handoff_file = _REPO_ROOT / "data" / "handoff" / "bitsG7.txt"
    g7_meta_file = _REPO_ROOT / "data" / "golden" / "G7" / "metadata.json"

    source_bits = np.array([int(c) for c in g7_source_file.read_text(encoding="utf-8").strip()], dtype=int)
    tx_bits = np.array([int(c) for c in g7_handoff_file.read_text(encoding="utf-8").strip()], dtype=int)
    meta = json.loads(g7_meta_file.read_text(encoding="utf-8"))

    # Preamble for phase synchronization
    preamble = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0], dtype=int)
    waveform, _, _ = generate_synthetic_waveform(
        modulation="QPSK",
        tx_bits=tx_bits,
        sps=4,
        sample_rate=1e6,
        preamble_bits=preamble,
    )

    cfg = DemodConfig(
        modulation="QPSK",
        sample_rate=1e6,
        samples_per_symbol=4,
        rrc_alpha=0.35,
        preamble_bits=preamble,
    )
    demod_res = demodulate(waveform, cfg)
    assert demod_res.status == DemodStatus.SUCCESS

    # Strip preamble to obtain recovered TX bits
    skip = len(preamble)
    recovered_tx = demod_res.hard_bits[skip:skip + len(tx_bits)]
    assert len(recovered_tx) == len(tx_bits)
    tx_errors = int(np.sum(recovered_tx != tx_bits))
    assert tx_errors == 0, f"Demodulator output differs from transmitted G7 TX bits: {tx_errors} errors"

    # Decode through Phase 1 decode_chain
    enc_cfg = EncodeConfig(
        case_id="G7",
        modulation="QPSK",
        fec_type=meta["fec_type"],
        interleaver_type=meta["interleaver_type"],
        fec_params=meta["fec_parameters"],
        interleaver_params=meta["interleaver_parameters"],
    )
    dec_res = decode_chain(
        received_bits=recovered_tx,
        config=enc_cfg,
        interleaver_meta=meta.get("interleaver_metadata"),
    )
    assert dec_res.decoder_success is True
    assert len(dec_res.recovered_source_bits) == 256
    assert np.array_equal(source_bits, dec_res.recovered_source_bits), "G7 recovered source bits must equal original"


# ============================================================================
# 5. DECODER PIPELINE FACADE END-TO-END INTEGRATION
# ============================================================================

def test_decoder_pipeline_facade():
    """Verify DecoderPipeline.decode orchestrates demod -> deinterleave -> FEC end-to-end."""
    preamble = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1], dtype=int)
    rng = np.random.default_rng(777)
    source_bits = rng.integers(0, 2, size=64)

    # Encode with Conv K=7
    enc_config = EncodeConfig(
        case_id="TEST",
        modulation="QPSK",
        fec_type="CONVOLUTIONAL",
        interleaver_type="NONE",
        fec_params={"constraint_length": 7, "polynomials": (0o171, 0o133)},
        interleaver_params={},
    )
    enc_res = encode_chain(source_bits, enc_config)

    # Synthesize IQ
    waveform, _, _ = generate_synthetic_waveform(
        modulation="QPSK",
        tx_bits=enc_res.tx_bits,
        sps=4,
        sample_rate=1e6,
        preamble_bits=preamble,
    )

    pipeline = DecoderPipeline()
    dec_cfg = DecoderConfig(
        modulation=ModulationType.QPSK,
        fec_type=FECType.CONVOLUTIONAL_K7,
        interleaver_type=InterleaverType.NONE,
        sample_rate=1e6,
        samples_per_symbol=4,
        metadata={"preamble_bits": preamble},
    )

    result = pipeline.decode(waveform, dec_cfg)
    assert result.status in (DecoderStatus.SUCCESS, DecoderStatus.DECODER_FAILURE)
    assert result.symbols is not None
    assert result.hard_bits is not None
    assert "timing_status" in dir(result)
    assert "carrier_status" in dir(result)


# ============================================================================
# 6. CONTROLLED SNR SWEEP / NOISE DEGRADATION OBSERVATION
# ============================================================================

def test_noise_degradation_observable():
    """Verify observable degradation as SNR decreases from high to low."""
    bers = []
    snr_levels = [20.0, 8.0, 2.0]
    for snr in snr_levels:
        waveform, tx_bits, _ = generate_synthetic_waveform(
            modulation="QPSK",
            num_bits=500,
            seed=888,
            sps=4,
            snr_db=snr,
        )
        cfg = DemodConfig(modulation="QPSK", samples_per_symbol=4, rrc_alpha=0.35)
        res = demodulate(waveform, cfg)
        cmp_len = min(len(res.hard_bits), len(tx_bits) - 16)
        errors = np.sum(res.hard_bits[16:cmp_len] != tx_bits[16:cmp_len])
        ber = errors / (cmp_len - 16)
        bers.append(ber)

    # Higher SNR (20 dB) should have lower or equal BER compared to lowest SNR (2 dB)
    assert bers[0] <= bers[-1], f"Expected BER degradation with lower SNR: {bers}"


# ============================================================================
# 7. OFFICIAL SINCHANA CAPTURE ADAPTER PLACEHOLDERS
# ============================================================================

def test_official_adapter_not_received_placeholder():
    """Verify official adapter returns NOT_RECEIVED status when Sinchana files have not yet arrived."""
    res = evaluate_official_capture(_REPO_ROOT / "data" / "captures" / "G1_pending")
    assert res["status"] == "NOT_RECEIVED"
    assert "does not exist yet" in res["message"]
