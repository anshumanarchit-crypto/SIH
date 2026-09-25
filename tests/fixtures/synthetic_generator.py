"""
tests/fixtures/synthetic_generator.py

TEST FIXTURE ONLY — NOT OFFICIAL GOLDEN DATA.

Deterministic synthetic complex baseband waveform generator for unit and
integration testing of the SpectralQ Phase 2 demodulator core.

DO NOT USE THESE WAVEFORMS AS OFFICIAL GOLDEN DATA.
Official golden waveforms (.cf32) are produced independently by Sinchana
via GNU Octave simulations.

Supported Modulations:
- BPSK (NASA/ESA standard, order 2)
- QPSK (Gray mapped, order 4)
- 8-PSK (Gray mapped, order 8)
- 16-QAM (Square Gray mapped, order 16)
- 64-QAM (Square Gray mapped, order 64)
- 2-FSK (Continuous-phase, binary frequency deviation)
- 4-FSK (Continuous-phase, 4-level frequency deviation)

Channel Impairments Model:
- RRC pulse shaping (configurable alpha and filter span)
- Configurable samples-per-symbol (sps)
- Continuous fractional timing offset (via bandlimited FFT phase shift)
- Carrier Frequency Offset (CFO in Hz or normalized)
- Initial carrier phase rotation (degrees)
- Additive White Gaussian Noise (AWGN) at specified SNR (dB)
"""

import math
from typing import Dict, Any, Tuple, Optional
import numpy as np

from spectralq.demod import rrc_filter_taps


def _generate_deterministic_bits(num_bits: int, seed: int) -> np.ndarray:
    """Generate deterministic pseudorandom bit sequence."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=num_bits)


def _map_bits_to_symbols(bits: np.ndarray, modulation: str) -> np.ndarray:
    """Map binary bit array to ideal complex constellation symbols."""
    mod = modulation.upper().replace("-", "").replace("_", "")
    b = np.asarray(bits, dtype=int).ravel()

    if "BPSK" in mod:
        # 0 -> +1.0, 1 -> -1.0
        return np.where(b == 0, 1.0 + 0.0j, -1.0 + 0.0j)

    elif "QPSK" in mod:
        # Gray QPSK: bit0 -> Re, bit1 -> Im
        if len(b) % 2 != 0:
            b = np.pad(b, (0, 1), mode="constant")
        b0 = b[0::2]
        b1 = b[1::2]
        re = np.where(b0 == 0, 1.0, -1.0)
        im = np.where(b1 == 0, 1.0, -1.0)
        return (re + 1j * im) / np.sqrt(2.0)

    elif "8PSK" in mod or "PSK8" in mod:
        # Gray 8-PSK: 3 bits per symbol
        rem = len(b) % 3
        if rem != 0:
            b = np.pad(b, (0, 3 - rem), mode="constant")
        mapping = {
            (0, 0, 0): 0.0,
            (0, 0, 1): np.pi / 4.0,
            (0, 1, 1): np.pi / 2.0,
            (0, 1, 0): 3.0 * np.pi / 4.0,
            (1, 1, 0): np.pi,
            (1, 1, 1): 5.0 * np.pi / 4.0,
            (1, 0, 1): 3.0 * np.pi / 2.0,
            (1, 0, 0): 7.0 * np.pi / 4.0,
        }
        symbols = []
        for i in range(0, len(b), 3):
            triple = (int(b[i]), int(b[i + 1]), int(b[i + 2]))
            theta = mapping.get(triple, 0.0)
            symbols.append(np.exp(1j * theta))
        return np.array(symbols, dtype=complex)

    elif "16QAM" in mod or "QAM16" in mod:
        # Square Gray 16-QAM: 4 bits per symbol (2 for I, 2 for Q)
        rem = len(b) % 4
        if rem != 0:
            b = np.pad(b, (0, 4 - rem), mode="constant")
        pam4_map = {
            (0, 0): -3.0,
            (0, 1): -1.0,
            (1, 1): +1.0,
            (1, 0): +3.0,
        }
        symbols = []
        for i in range(0, len(b), 4):
            i_pair = (int(b[i]), int(b[i + 1]))
            q_pair = (int(b[i + 2]), int(b[i + 3]))
            re = pam4_map[i_pair]
            im = pam4_map[q_pair]
            symbols.append((re + 1j * im) / np.sqrt(10.0))
        return np.array(symbols, dtype=complex)

    elif "64QAM" in mod or "QAM64" in mod:
        # 6 bits per symbol (3 for I, 3 for Q)
        rem = len(b) % 6
        if rem != 0:
            b = np.pad(b, (0, 6 - rem), mode="constant")
        pam8_map = {
            (0, 0, 0): -7.0, (0, 0, 1): -5.0, (0, 1, 1): -3.0, (0, 1, 0): -1.0,
            (1, 1, 0): +1.0, (1, 1, 1): +3.0, (1, 0, 1): +5.0, (1, 0, 0): +7.0,
        }
        symbols = []
        for i in range(0, len(b), 6):
            i_triple = (int(b[i]), int(b[i + 1]), int(b[i + 2]))
            q_triple = (int(b[i + 3]), int(b[i + 4]), int(b[i + 5]))
            re = pam8_map[i_triple]
            im = pam8_map[q_triple]
            symbols.append((re + 1j * im) / np.sqrt(42.0))
        return np.array(symbols, dtype=complex)

    else:
        raise ValueError(f"Unsupported modulation for direct symbol mapping: {modulation}")


def generate_synthetic_waveform(
    modulation: str,
    num_bits: int = 256,
    seed: int = 12345,
    tx_bits: Optional[np.ndarray] = None,
    sps: int = 4,
    rrc_alpha: float = 0.35,
    filter_span: int = 8,
    sample_rate: float = 1e6,
    timing_offset_symbols: float = 0.0,
    cfo_hz: float = 0.0,
    phase_offset_deg: float = 0.0,
    snr_db: Optional[float] = None,
    fsk_deviation_hz: Optional[float] = None,
    preamble_bits: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Synthesize a deterministic complex IQ baseband test fixture.

    Parameters
    ----------
    modulation : str
        Modulation scheme name ('BPSK', 'QPSK', '8-PSK', '16-QAM', '2-FSK', '4-FSK').
    num_bits : int
        Number of bits to generate if tx_bits is not explicitly supplied.
    seed : int
        Deterministic RNG seed for bit and noise generation.
    tx_bits : Optional[np.ndarray]
        Explicit transmission bitstream to synthesize. If None, generated via seed.
    sps : int
        Samples per symbol.
    rrc_alpha : float
        RRC filter excess bandwidth roll-off.
    filter_span : int
        RRC filter span in symbols.
    sample_rate : float
        Baseband sampling rate in Hz.
    timing_offset_symbols : float
        Fractional timing delay in symbols (e.g. 0.25).
    cfo_hz : float
        Carrier frequency offset in Hz.
    phase_offset_deg : float
        Initial carrier phase offset in degrees.
    snr_db : Optional[float]
        Signal-to-Noise Ratio in dB. None indicates noiseless condition.
    fsk_deviation_hz : Optional[float]
        Peak frequency deviation for FSK modulations in Hz.
    preamble_bits : Optional[np.ndarray]
        Known preamble bits prepended to the payload.

    Returns
    -------
    waveform : np.ndarray
        1D complex64/complex128 array of baseband IQ samples.
    final_tx_bits : np.ndarray
        1D integer binary array of transmitted bits (preamble + payload).
    metadata : Dict[str, Any]
        Synthesis parameters and ground truth metadata for test evaluation.
    """
    rng = np.random.default_rng(seed)

    # 1. Establish Transmit Bits
    if tx_bits is None:
        payload_bits = _generate_deterministic_bits(num_bits, seed)
    else:
        payload_bits = np.asarray(tx_bits, dtype=int).ravel()

    if preamble_bits is not None:
        p_bits = np.asarray(preamble_bits, dtype=int).ravel()
        final_tx_bits = np.concatenate([p_bits, payload_bits])
    else:
        final_tx_bits = payload_bits

    mod_str = modulation.upper().replace("-", "").replace("_", "")

    # 2. Modulate Signal
    if "FSK" in mod_str:
        # FSK Continuous-Phase Modulation
        order = 4 if "4" in mod_str else 2
        f_dev = fsk_deviation_hz if fsk_deviation_hz is not None else (sample_rate / (sps * 4.0))

        if order == 2:
            # 0 -> -f_dev, 1 -> +f_dev
            freq_devs = np.where(final_tx_bits == 0, -f_dev, +f_dev)
        else:
            # 4-FSK: 2 bits per symbol
            rem = len(final_tx_bits) % 2
            padded_bits = np.pad(final_tx_bits, (0, 2 - rem), mode="constant") if rem != 0 else final_tx_bits
            df = f_dev / 3.0
            freq_table = {
                (0, 0): -3.0 * df,
                (0, 1): -1.0 * df,
                (1, 1): +1.0 * df,
                (1, 0): +3.0 * df,
            }
            freq_devs_list = []
            for i in range(0, len(padded_bits), 2):
                pair = (int(padded_bits[i]), int(padded_bits[i + 1]))
                freq_devs_list.append(freq_table[pair])
            freq_devs = np.array(freq_devs_list, dtype=float)

        # Upsample frequency deviations to sample rate
        sample_freqs = np.repeat(freq_devs, sps)
        # Integrate phase: phi[n] = 2 * pi * sum(f[k] * dt)
        dt = 1.0 / sample_rate
        phase_accum = 2.0 * np.pi * np.cumsum(sample_freqs) * dt
        base_waveform = np.exp(1j * phase_accum)

    else:
        # PSK / QAM Linear Modulation with RRC Pulse Shaping
        symbols = _map_bits_to_symbols(final_tx_bits, modulation)
        # Upsample by sps
        upsampled = np.zeros(len(symbols) * sps, dtype=complex)
        upsampled[::sps] = symbols

        # RRC Pulse Shaping Filter
        rrc = rrc_filter_taps(sps=sps, alpha=rrc_alpha, span=filter_span)
        base_waveform = np.convolve(upsampled, rrc, mode="same")

    # 3. Apply Fractional Timing Offset (via frequency-domain linear phase ramp)
    if abs(timing_offset_symbols) > 1e-6:
        n_samples = len(base_waveform)
        delay_samples = timing_offset_symbols * sps
        # FFT linear phase shift
        w_fft = np.fft.fft(base_waveform)
        freqs = np.fft.fftfreq(n_samples)
        phase_shift = np.exp(-1j * 2.0 * np.pi * freqs * delay_samples)
        base_waveform = np.fft.ifft(w_fft * phase_shift)

    # 4. Apply Carrier Frequency Offset (CFO) and Phase Rotation
    t_axis = np.arange(len(base_waveform), dtype=float) / sample_rate
    initial_phase_rad = np.radians(phase_offset_deg)
    carrier_multiplier = np.exp(1j * (2.0 * np.pi * cfo_hz * t_axis + initial_phase_rad))
    impaired_waveform = base_waveform * carrier_multiplier

    # 5. Apply AWGN
    if snr_db is not None:
        p_signal = float(np.mean(np.abs(impaired_waveform) ** 2))
        snr_linear = 10.0 ** (snr_db / 10.0)
        p_noise = p_signal / snr_linear
        noise_std = np.sqrt(p_noise / 2.0)
        noise = rng.normal(0.0, noise_std, size=len(impaired_waveform)) + 1j * rng.normal(
            0.0, noise_std, size=len(impaired_waveform)
        )
        final_waveform = impaired_waveform + noise
    else:
        final_waveform = impaired_waveform

    metadata = {
        "fixture_label": "TEST FIXTURE ONLY — NOT OFFICIAL GOLDEN DATA",
        "modulation": modulation,
        "sample_rate": float(sample_rate),
        "sps": int(sps),
        "rrc_alpha": float(rrc_alpha),
        "filter_span": int(filter_span),
        "timing_offset_symbols": float(timing_offset_symbols),
        "cfo_hz": float(cfo_hz),
        "phase_offset_deg": float(phase_offset_deg),
        "snr_db": snr_db,
        "tx_bit_length": int(len(final_tx_bits)),
        "seed": int(seed),
    }

    return final_waveform, final_tx_bits, metadata
