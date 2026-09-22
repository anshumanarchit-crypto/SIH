"""
End-to-End Signal Processing and Bitstream Recovery Pipeline for SpectralQ.

Orchestrates:
  File Ingestion -> Signal Preprocessing -> Feature Extraction -> Modulation Classification ->
  Demodulation & Timing Sync -> De-interleaving -> FEC / Viterbi Decoding -> Sync Correlation -> Packet Framing
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np

from core.correlation import (
    KNOWN_SYNC_WORDS,
    DecodedFrame,
    SyncDetection,
    detect_sync_word,
    parse_packet_frame,
)
from core.deinterleave import (
    ConvolutionalDeinterleaver,
    block_deinterleave,
)
from core.demodulation import (
    DemodulationResult,
    demodulate_16qam,
    demodulate_8psk,
    demodulate_bpsk,
    demodulate_qpsk,
    demodulate_signal,
)
from core.fec import (
    ConvolutionalCodec,
    HammingCodec,
    ReedSolomonCodec,
)
from core.features import (
    SpectralFeatures,
    extract_all_features,
)
from core.io import (
    SignalData,
    load_iq_file,
    load_wav_file,
)
from core.modulation import (
    HybridModulationClassifier,
    ModulationResult,
    ModulationType,
)
from core.preprocessing import (
    apply_rrc_filter,
    correct_iq_imbalance,
    estimate_and_correct_cfo,
    normalize_power,
    remove_dc_offset,
)

logger = logging.getLogger("spectralq.pipeline")


from core.contracts import (
    SignalData,
    SpectralFeatures,
    ModulationType,
    ModulationResult,
    DemodulationResult,
    CorrelationResult,
    SyncDetection,
    DecodedFrame,
    PipelineConfig,
    PipelineResult,
    ResultStatus,
    make_warning,
    IQ_CONVENTION,
    BIT_ORDERING,
)


class SpectralQPipeline:
    """
    Production-grade, extensible DSP pipeline orchestrator.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.classifier = HybridModulationClassifier()
        self.viterbi = ConvolutionalCodec()

    def process_signal(self, signal: SignalData) -> PipelineResult:
        """
        Execute full signal processing and bitstream recovery pipeline on SignalData.
        """
        t_start = time.perf_counter()
        timings = {}

        # -------------------------------------------------------------
        # Step 1: Normalization and Preprocessing
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        samples = signal.samples.copy()
        iq_metrics = {"gain_imbalance_db": 0.0, "phase_error_deg": 0.0}

        if self.config.enable_dc_removal:
            samples = remove_dc_offset(samples)

        if self.config.enable_iq_correction and signal.is_complex:
            samples, iq_metrics = correct_iq_imbalance(samples)

        if self.config.enable_power_norm:
            samples = normalize_power(samples, target_power=1.0)

        cfo_hz = 0.0
        if self.config.enable_cfo_correction and signal.is_complex:
            if self.config.manual_cfo_hz is not None:
                cfo_hz = self.config.manual_cfo_hz
                t_arr = np.arange(len(samples)) / signal.sample_rate
                samples = samples * np.exp(-1j * 2.0 * np.pi * cfo_hz * t_arr)
            else:
                samples, cfo_hz = estimate_and_correct_cfo(samples, sample_rate=signal.sample_rate)

        preprocessed_samples = samples.copy()
        timings["preprocessing_ms"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # Step 2: Feature Extraction
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        features = extract_all_features(
            preprocessed_samples,
            sample_rate=signal.sample_rate,
            center_freq_offset_hz=cfo_hz,
        )
        timings["features_ms"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # Step 3: Modulation Classification
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        if self.config.manual_modulation is not None:
            mod_result = ModulationResult(
                modulation=self.config.manual_modulation,
                confidence=1.0,
                probabilities={self.config.manual_modulation.value: 1.0},
                rationale="Manual user override",
                features=features,
                is_unknown=False,
            )
        else:
            mod_result = self.classifier.classify(features)
        timings["classification_ms"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # Step 4: RRC Matched Filter & Demodulation
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        sps_to_use = self.config.manual_sps if self.config.manual_sps is not None else features.estimated_sps
        sps_int = max(1, int(round(sps_to_use)))

        demod_samples = preprocessed_samples
        if self.config.enable_rrc_filtering and sps_int >= 2 and mod_result.modulation in (
            ModulationType.BPSK, ModulationType.QPSK, ModulationType.PSK8,
            ModulationType.QAM16, ModulationType.QAM64
        ):
            demod_samples = apply_rrc_filter(
                demod_samples,
                sps=sps_int,
                beta=self.config.rrc_beta,
                span=self.config.rrc_span,
            )

        demod_result = demodulate_signal(
            demod_samples,
            sample_rate=signal.sample_rate,
            modulation=mod_result.modulation,
            sps=sps_to_use,
        )
        timings["demodulation_ms"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # Step 5: De-interleaving
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        raw_bits = demod_result.hard_bits
        raw_llrs = demod_result.soft_llrs
        deinterleaved_bits = raw_bits

        if self.config.deinterleave_scheme == "block":
            deinterleaved_bits = block_deinterleave(
                raw_bits,
                num_rows=self.config.block_rows,
                num_cols=self.config.block_cols,
                original_length=len(raw_bits),
            )
            if len(raw_llrs) == len(raw_bits):
                raw_llrs = block_deinterleave(
                    raw_llrs,
                    num_rows=self.config.block_rows,
                    num_cols=self.config.block_cols,
                    original_length=len(raw_llrs),
                )
        elif self.config.deinterleave_scheme == "convolutional":
            c_deintl = ConvolutionalDeinterleaver(
                num_branches=self.config.conv_branches,
                delay_step=self.config.conv_delay_step,
            )
            deinterleaved_bits = c_deintl.process(raw_bits)
        timings["deinterleave_ms"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # Step 6 & 7: Synchronization, FEC Decoding & Packet Framing
        # -------------------------------------------------------------
        t0 = time.perf_counter()

        best_sync_det = SyncDetection(False, "NONE", -1, 0.0, False, 0)
        best_decoded_frame = None
        best_bits = deinterleaved_bits
        best_weighted_metric = -1.0

        is_quad_mod = mod_result.modulation in (ModulationType.QPSK, ModulationType.PSK8, ModulationType.QAM16, ModulationType.BPSK)
        rotations = [0, 1, 2, 3] if (is_quad_mod and len(demod_result.symbols) > 0) else [0]

        # Candidate sync patterns: test longer preambles first
        candidate_patterns = [self.config.sync_pattern] if self.config.sync_pattern else list(KNOWN_SYNC_WORDS.keys())

        found_valid_frame = False

        for rot_k in rotations:
            if found_valid_frame:
                break

            if rot_k == 0:
                cur_raw_bits = deinterleaved_bits
            else:
                rot_syms = demod_result.symbols * np.exp(-1j * rot_k * np.pi / 2.0)
                if mod_result.modulation == ModulationType.BPSK:
                    _, cur_raw_bits, _ = demodulate_bpsk(rot_syms)
                elif mod_result.modulation == ModulationType.QPSK:
                    _, cur_raw_bits, _ = demodulate_qpsk(rot_syms)
                elif mod_result.modulation == ModulationType.PSK8:
                    _, cur_raw_bits, _ = demodulate_8psk(rot_syms)
                elif mod_result.modulation == ModulationType.QAM16:
                    _, cur_raw_bits, _ = demodulate_16qam(rot_syms)
                else:
                    _, cur_raw_bits, _ = demodulate_qpsk(rot_syms)

            # Stream candidates to search: (bits_stream, is_fec_stream)
            streams_to_check = [(cur_raw_bits, False)]

            # Generate FEC stream if applicable
            fec_mode = self.config.fec_scheme.lower()
            if fec_mode in ("viterbi_hard", "viterbi", "auto") and len(cur_raw_bits) >= 64:
                try:
                    v_bits = self.viterbi.decode_hard(cur_raw_bits)
                    if len(v_bits) > 0:
                        streams_to_check.append((v_bits, True))
                except Exception:
                    pass

            if fec_mode == "hamming" and len(cur_raw_bits) >= 7:
                try:
                    h_bits, _ = HammingCodec.decode(cur_raw_bits)
                    if len(h_bits) > 0:
                        streams_to_check.append((h_bits, True))
                except Exception:
                    pass

            for bit_stream, is_fec in streams_to_check:
                if found_valid_frame:
                    break
                for pat_name in candidate_patterns:
                    det = detect_sync_word(
                        bit_stream,
                        sync_pattern=pat_name,
                        threshold=self.config.sync_threshold,
                    )
                    if det.found:
                        frame = parse_packet_frame(bit_stream, det, crc_type=self.config.crc_type)
                        weighted_score = det.correlation_score * det.sync_word_len
                        if frame and frame.crc_valid:
                            best_sync_det = det
                            best_decoded_frame = frame
                            best_bits = bit_stream
                            found_valid_frame = True
                            break
                        elif weighted_score > best_weighted_metric:
                            best_weighted_metric = weighted_score
                            best_sync_det = det
                            best_decoded_frame = frame
                            best_bits = bit_stream

        sync_det = best_sync_det
        decoded_frame = best_decoded_frame
        fec_bits = best_bits

        timings["fec_ms"] = 0.0
        timings["correlation_framing_ms"] = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - t_start) * 1000.0

        status = "RECOVERED_FRAME" if (decoded_frame and decoded_frame.crc_valid) else (
            "FRAME_DETECTED" if decoded_frame else (
                "DEMODULATED_BITS" if len(raw_bits) > 0 else "UNKNOWN_SIGNAL"
            )
        )

        return PipelineResult(
            signal_data=signal,
            preprocessed_samples=preprocessed_samples,
            iq_metrics=iq_metrics,
            cfo_hz=cfo_hz,
            features=features,
            modulation=mod_result,
            demodulation=demod_result,
            deinterleaved_bits=deinterleaved_bits,
            fec_bits=fec_bits,
            sync_detection=sync_det,
            decoded_frame=decoded_frame,
            stage_timings_ms=timings,
            total_time_ms=total_ms,
            status_summary=status,
        )

    def process_file(
        self,
        file_path: Union[str, Path],
        sample_rate: Optional[float] = None,
        data_type: str = "complex64",
        center_freq: float = 0.0,
    ) -> PipelineResult:
        """Convenience method to ingest file from disk and execute pipeline."""
        p = Path(file_path)
        if p.suffix.lower() in (".wav", ".wave"):
            sig = load_wav_file(p, center_freq=center_freq)
        else:
            sig = load_iq_file(p, sample_rate=sample_rate, data_type=data_type, center_freq=center_freq)
        return self.process_signal(sig)
