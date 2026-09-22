"""
Synthetic Signal Generator for SpectralQ.

Generates ground-truth test signals across standard modulation schemes
with known payloads, framing, FEC, interleaving, pulse shaping, and realistic RF impairments.
"""

from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from core.correlation import KNOWN_SYNC_WORDS, bytes_to_bits, format_hex_dump
from core.deinterleave import block_interleave
from core.fec import CRC, ConvolutionalCodec, HammingCodec
from core.io import SignalData, save_iq_file, save_wav_file
from core.modulation import ModulationType
from core.preprocessing import apply_rrc_filter

logger = logging.getLogger("spectralq.generator")


class SyntheticSignalGenerator:
    """
    Generator for realistic synthetic SDR/DSP signals with ground-truth verification.
    """

    def __init__(self, sample_rate: float = 100_000.0):
        self.sample_rate = sample_rate
        self.viterbi = ConvolutionalCodec()

    def generate_packet_bits(
        self,
        payload_text: str = "SPECTRALQ_TELEMETRY_PACKET_001",
        sync_name: str = "BARKER_13",
        version: int = 1,
        pkt_type: int = 42,
        seq_num: int = 1001,
        crc_type: str = "crc16",
        apply_fec: bool = True,
        apply_interleaving: bool = True,
        block_rows: int = 8,
        block_cols: int = 8,
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Build complete packet bitstream with sync word, header, payload, CRC, and optional FEC.
        """
        import struct

        sync_bits = KNOWN_SYNC_WORDS[sync_name]
        payload_bytes = payload_text.encode("utf-8")
        payload_len = len(payload_bytes)

        # 6-byte header: [Version: 1B, Type: 1B, Seq: 2B, Len: 2B]
        header = struct.pack(">BBHH", version, pkt_type, seq_num, payload_len)
        data_to_crc = header + payload_bytes

        if crc_type.lower() == "crc16":
            crc_val = CRC.crc16(data_to_crc)
            crc_bytes = struct.pack(">H", crc_val)
        elif crc_type.lower() == "crc32":
            crc_val = CRC.crc32(data_to_crc)
            crc_bytes = struct.pack(">I", crc_val)
        else:
            crc_val = CRC.crc8(data_to_crc)
            crc_bytes = bytes([crc_val])

        frame_bytes = data_to_crc + crc_bytes
        frame_bits = bytes_to_bits(frame_bytes)

        # Apply FEC and Interleaving to frame bits (excluding sync word)
        if apply_fec:
            encoded_frame_bits = self.viterbi.encode(frame_bits, add_tail_bits=True)
        else:
            encoded_frame_bits = frame_bits

        if apply_interleaving:
            interleaved_frame_bits = block_interleave(
                encoded_frame_bits,
                num_rows=block_rows,
                num_cols=block_cols,
            )
        else:
            interleaved_frame_bits = encoded_frame_bits

        # Concatenate sync word + framed bitstream
        full_bits = np.concatenate([sync_bits, interleaved_frame_bits])

        metadata = {
            "payload_text": payload_text,
            "sync_name": sync_name,
            "version": version,
            "packet_type": pkt_type,
            "seq_num": seq_num,
            "crc_type": crc_type,
            "crc_val": hex(crc_val),
            "applied_fec": apply_fec,
            "applied_interleaving": apply_interleaving,
            "num_raw_bits": len(frame_bits),
            "num_total_bits": len(full_bits),
        }

        return full_bits, metadata

    def map_bits_to_symbols(
        self,
        bits: np.ndarray,
        modulation: ModulationType,
    ) -> np.ndarray:
        """Map bits to complex constellation symbols."""
        bits = np.asarray(bits, dtype=np.uint8)

        if modulation == ModulationType.BPSK:
            return (2.0 * bits - 1.0 + 0j).astype(np.complex64)

        elif modulation == ModulationType.QPSK:
            # Pad to multiple of 2
            if len(bits) % 2 != 0:
                bits = np.append(bits, 0)
            b0 = bits[0::2]
            b1 = bits[1::2]
            return (((2.0 * b0 - 1.0) + 1j * (2.0 * b1 - 1.0)) / np.sqrt(2.0)).astype(np.complex64)

        elif modulation == ModulationType.PSK8:
            # Pad to multiple of 3
            pad = (3 - (len(bits) % 3)) % 3
            if pad > 0:
                bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
            chunks = bits.reshape(-1, 3)
            gray_map = {
                (0, 0, 0): 0, (0, 0, 1): 1, (0, 1, 1): 2, (0, 1, 0): 3,
                (1, 1, 0): 4, (1, 1, 1): 5, (1, 0, 1): 6, (1, 0, 0): 7
            }
            symbols = []
            for ch in chunks:
                sec = gray_map.get(tuple(ch), 0)
                symbols.append(np.exp(1j * sec * np.pi / 4.0))
            return np.array(symbols, dtype=np.complex64)

        elif modulation == ModulationType.QAM16:
            # Pad to multiple of 4
            pad = (4 - (len(bits) % 4)) % 4
            if pad > 0:
                bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
            norm = np.sqrt(10.0)

            def bit_pair_to_level(msb, lsb):
                return np.where(msb == 1, np.where(lsb == 1, 1.0, 3.0), np.where(lsb == 1, -1.0, -3.0))

            b0, b1 = bits[0::4], bits[1::4]
            b2, b3 = bits[2::4], bits[3::4]
            i_lvl = bit_pair_to_level(b0, b1) / norm
            q_lvl = bit_pair_to_level(b2, b3) / norm
            return (i_lvl + 1j * q_lvl).astype(np.complex64)

        elif modulation in (ModulationType.OOK, ModulationType.ASK2):
            return bits.astype(np.complex64)

        else:
            return (2.0 * bits - 1.0 + 0j).astype(np.complex64)

    def generate_signal(
        self,
        modulation: ModulationType = ModulationType.QPSK,
        payload_text: str = "SPECTRALQ_TELEMETRY_PACKET_001",
        snr_db: float = 20.0,
        sps: int = 8,
        cfo_hz: float = 0.0,
        gain_imbalance_db: float = 0.0,
        phase_imbalance_deg: float = 0.0,
        apply_rrc: bool = True,
        rrc_beta: float = 0.35,
        apply_fec: bool = False,
        apply_interleaving: bool = False,
        sync_name: str = "BARKER_13",
        prefix_noise_samples: int = 200,
        suffix_noise_samples: int = 200,
    ) -> Tuple[SignalData, Dict[str, any]]:
        """
        Synthesize a complete RF signal recording with known ground-truth.
        """
        # 1. Generate bitstream
        bits, pkt_meta = self.generate_packet_bits(
            payload_text=payload_text,
            sync_name=sync_name,
            apply_fec=apply_fec,
            apply_interleaving=apply_interleaving,
        )

        # 2. Modulate to baseband waveform
        if modulation in (ModulationType.FSK2, ModulationType.FSK4):
            # FSK continuous phase generation
            f_dev = self.sample_rate / (4.0 * sps)
            freqs = np.repeat((2.0 * bits - 1.0) * f_dev, sps)
            phase = 2.0 * np.pi * np.cumsum(freqs) / self.sample_rate
            tx_wave = np.exp(1j * phase).astype(np.complex64)
        else:
            symbols = self.map_bits_to_symbols(bits, modulation=modulation)
            # Upsample
            upsampled = np.zeros(len(symbols) * sps, dtype=np.complex64)
            upsampled[::sps] = symbols

            if apply_rrc:
                tx_wave = apply_rrc_filter(upsampled, sps=sps, beta=rrc_beta, span=8)
            else:
                tx_wave = np.repeat(symbols, sps)

        # 3. Add Pre/Post guard noise
        total_len = prefix_noise_samples + len(tx_wave) + suffix_noise_samples
        padded_wave = np.zeros(total_len, dtype=np.complex64)
        padded_wave[prefix_noise_samples : prefix_noise_samples + len(tx_wave)] = tx_wave

        # 4. RF Impairments: Carrier Frequency Offset (CFO)
        if abs(cfo_hz) > 1e-3:
            t = np.arange(total_len) / self.sample_rate
            cfo_rotator = np.exp(1j * 2.0 * np.pi * cfo_hz * t).astype(np.complex64)
            padded_wave = padded_wave * cfo_rotator

        # 5. RF Impairments: IQ Imbalance
        if abs(gain_imbalance_db) > 1e-3 or abs(phase_imbalance_deg) > 1e-3:
            alpha = 10.0 ** (gain_imbalance_db / 20.0)
            phi = np.deg2rad(phase_imbalance_deg)
            i_comp = np.real(padded_wave)
            q_comp = np.imag(padded_wave)
            q_imb = alpha * (np.sin(phi) * i_comp + np.cos(phi) * q_comp)
            padded_wave = (i_comp + 1j * q_imb).astype(np.complex64)

        # 6. AWGN Channel
        # Signal power during transmission
        p_sig = np.mean(np.abs(tx_wave) ** 2) if len(tx_wave) > 0 else 1.0
        p_noise = p_sig / (10.0 ** (snr_db / 10.0))
        noise = (np.random.randn(total_len) + 1j * np.random.randn(total_len)) * np.sqrt(p_noise / 2.0)
        rx_samples = (padded_wave + noise).astype(np.complex64)

        ground_truth = {
            "modulation": modulation.value,
            "snr_db": snr_db,
            "sps": sps,
            "baud_rate": self.sample_rate / sps,
            "cfo_hz": cfo_hz,
            "gain_imbalance_db": gain_imbalance_db,
            "phase_imbalance_deg": phase_imbalance_deg,
            "sample_rate": self.sample_rate,
            "packet_metadata": pkt_meta,
            "num_symbols": len(bits) if modulation == ModulationType.BPSK else len(bits) // 2,
        }

        sig_data = SignalData(
            samples=rx_samples,
            sample_rate=self.sample_rate,
            center_freq=0.0,
            is_complex=True,
            metadata={"ground_truth": ground_truth},
        )

        return sig_data, ground_truth


def generate_benchmark_suite(output_dir: Union[str, Path] = "data/synthetic") -> Dict[str, Path]:
    """
    Generate standard benchmark dataset across multiple modulations and SNRs.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    gen = SyntheticSignalGenerator(sample_rate=100_000.0)
    created_files = {}

    benchmarks = [
        ("bpsk_snr20db_clean", ModulationType.BPSK, 20.0, 8, 0.0, False, "SPECTRALQ_BPSK_CLEAN_01"),
        ("qpsk_snr15db_cfo", ModulationType.QPSK, 15.0, 8, 1250.0, False, "SPECTRALQ_QPSK_NAV_TELEMETRY"),
        ("16qam_snr22db", ModulationType.QAM16, 22.0, 8, 0.0, False, "SPECTRALQ_16QAM_HIGH_THROUGHPUT"),
        ("bpsk_fec_viterbi", ModulationType.BPSK, 12.0, 8, 0.0, True, "SPECTRALQ_DEEP_SPACE_CCSDS_PKT"),
        ("2fsk_snr18db", ModulationType.FSK2, 18.0, 16, 0.0, False, "SPECTRALQ_FSK_SENSOR_BEACON"),
        ("qpsk_low_snr5db", ModulationType.QPSK, 5.0, 8, 0.0, False, "SPECTRALQ_LOW_SNR_STRESS_TEST"),
    ]

    for name, mod, snr, sps, cfo, fec, payload in benchmarks:
        sig, gt = gen.generate_signal(
            modulation=mod,
            payload_text=payload,
            snr_db=snr,
            sps=sps,
            cfo_hz=cfo,
            apply_fec=fec,
            sync_name="BARKER_13" if not fec else "CCSDS_32",
        )

        iq_file = out_path / f"{name}.iq"
        save_iq_file(sig, iq_file, data_type="complex64")

        wav_file = out_path / f"{name}.wav"
        save_wav_file(sig, wav_file)

        created_files[name] = iq_file
        logger.info("Generated benchmark file %s -> %s", name, iq_file)

    return created_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    files = generate_benchmark_suite()
    print(f"Generated {len(files)} benchmark files in data/synthetic/")
