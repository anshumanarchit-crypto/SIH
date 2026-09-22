"""End-to-End Pipeline tests for SpectralQ."""

import numpy as np
import pytest
from pathlib import Path

from core.io import load_iq_file, load_wav_file
from core.modulation import ModulationType
from core.pipeline import PipelineConfig, SpectralQPipeline
from data.synthetic_generator import SyntheticSignalGenerator


def test_pipeline_bpsk_clean():
    gen = SyntheticSignalGenerator(sample_rate=100_000.0)
    target_payload = "SPECTRALQ_BPSK_E2E_SUCCESS_101"

    sig, gt = gen.generate_signal(
        modulation=ModulationType.BPSK,
        payload_text=target_payload,
        snr_db=25.0,
        sps=8,
        cfo_hz=0.0,
        apply_fec=False,
        sync_name="BARKER_13",
    )

    pipeline = SpectralQPipeline()
    result = pipeline.process_signal(sig)

    assert result.modulation.modulation == ModulationType.BPSK
    assert result.modulation.confidence > 0.85
    assert result.demodulation.evm_percent < 45.0
    assert result.sync_detection.found is True
    assert result.sync_detection.sync_name == "BARKER_13"
    assert result.decoded_frame is not None
    assert result.decoded_frame.crc_valid is True
    assert result.decoded_frame.payload_text == target_payload


def test_pipeline_qpsk_with_cfo():
    gen = SyntheticSignalGenerator(sample_rate=100_000.0)
    target_payload = "SPECTRALQ_QPSK_CARRIER_LOCKED"

    cfo_injected = 1250.0  # 1.25 kHz offset
    sig, gt = gen.generate_signal(
        modulation=ModulationType.QPSK,
        payload_text=target_payload,
        snr_db=22.0,
        sps=8,
        cfo_hz=cfo_injected,
        apply_fec=False,
        sync_name="BARKER_13",
    )

    pipeline = SpectralQPipeline()
    result = pipeline.process_signal(sig)

    assert result.modulation.modulation == ModulationType.QPSK
    assert result.cfo_hz == pytest.approx(cfo_injected, abs=20.0)
    assert result.sync_detection.found is True
    assert result.decoded_frame is not None
    assert result.decoded_frame.crc_valid is True
    assert result.decoded_frame.payload_text == target_payload


def test_pipeline_from_disk_iq():
    # Load pre-generated benchmark file
    iq_path = Path("data/synthetic/bpsk_snr20db_clean.iq")
    if not iq_path.exists():
        pytest.skip("Benchmark file not found")

    sig = load_iq_file(iq_path)
    pipeline = SpectralQPipeline()
    res = pipeline.process_signal(sig)

    assert res.modulation.modulation == ModulationType.BPSK
    assert res.decoded_frame is not None
    assert res.decoded_frame.crc_valid is True
    assert "SPECTRALQ_BPSK_CLEAN_01" in res.decoded_frame.payload_text


def test_pipeline_from_disk_wav():
    wav_path = Path("data/synthetic/bpsk_snr20db_clean.wav")
    if not wav_path.exists():
        pytest.skip("Benchmark file not found")

    sig = load_wav_file(wav_path)
    pipeline = SpectralQPipeline()
    res = pipeline.process_signal(sig)

    assert res.modulation.modulation == ModulationType.BPSK
    assert res.sync_detection.found is True
    assert res.decoded_frame is not None
    assert res.decoded_frame.crc_valid is True


def test_pipeline_noise_rejection():
    np.random.seed(42)
    noise_samples = (np.random.randn(8000) + 1j * np.random.randn(8000)).astype(np.complex64)
    from core.io import SignalData
    sig = SignalData(samples=noise_samples, sample_rate=100_000.0)

    config = PipelineConfig()
    pipeline = SpectralQPipeline(config)
    res = pipeline.process_signal(sig)

    # Engineering Rule 7: Should classify as Unknown
    assert res.modulation.modulation == ModulationType.UNKNOWN
    assert res.modulation.is_unknown is True
