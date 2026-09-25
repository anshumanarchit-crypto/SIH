"""Comprehensive diagnostic analysis for G7 official capture at near-threshold SNR."""

import numpy as np
import json
import pathlib
from spectralq.demod import apply_rrc_filter, gardner_timing_recovery, estimate_cfo_mth_power, costas_carrier_recovery
from spectralq.fec import ConvolutionalCodec

def diagnose_g7():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    golden_dir = repo_root / "data" / "official" / "sinchana" / "golden"
    handoff_dir = repo_root / "data" / "handoff"
    golden_base_dir = repo_root / "data" / "golden"

    cf32_path = golden_dir / "G7_QPSK_conv_near_threshold.cf32"
    truth_path = golden_dir / "G7_QPSK_conv_near_threshold.truth.json"
    meta = json.loads(truth_path.read_text(encoding="utf-8"))

    raw = np.fromfile(cf32_path, dtype=np.float32)
    iq = raw[0::2] + 1j * raw[1::2]
    tx_bits = np.array([int(c) for c in (handoff_dir / "bitsG7.txt").read_text(encoding="utf-8").strip()], dtype=np.uint8)
    src_bits = np.array([int(c) for c in (golden_base_dir / "G7" / "source_bits.txt").read_text(encoding="utf-8").strip()], dtype=np.uint8)

    print("=== 1. RAW IQ POWER ===")
    raw_power = float(np.mean(np.abs(iq)**2))
    raw_rms = float(np.sqrt(raw_power))
    raw_peak = float(np.max(np.abs(iq)))
    print(f"Mean Power: {raw_power:.4f}, RMS: {raw_rms:.4f}, Peak: {raw_peak:.4f}, PAPR: {raw_peak**2/raw_power:.2f}")

    print("=== 2. RRC FILTER OUTPUT ===")
    filtered = apply_rrc_filter(iq, sps=8, alpha=0.35, span=8)
    filt_power = float(np.mean(np.abs(filtered)**2))
    print(f"Filtered Power: {filt_power:.4f}, Filtered RMS: {np.sqrt(filt_power):.4f}")

    print("=== 3. TIMING RECOVERY & JITTER ===")
    syms_gardner, t_diag = gardner_timing_recovery(filtered, sps=8)
    print(f"Gardner Output Symbols: {len(syms_gardner)}, Converged: {t_diag.get('converged')}")
    print(f"Mean TED Error: {t_diag.get('mean_error'):.6f}, Jitter Std: {t_diag.get('jitter_std'):.4f}")

    print("=== 4. CARRIER RECOVERY (COARSE & FINE) ===")
    cfo_coarse = estimate_cfo_mth_power(filtered[::8], order=4)
    syms_costas, c_diag = costas_carrier_recovery(filtered[::8], order=4)
    print(f"Coarse CFO rad: {cfo_coarse:.6f}")
    print(f"Costas Converged: {c_diag.get('converged')}, Est CFO rad: {c_diag.get('estimated_cfo'):.6f}, Final Phase rad: {c_diag.get('final_phase'):.4f}")

    print("=== 5. CONSTELLATION SPREAD & HARD MARGINS ===")
    direct_syms = filtered[::8]
    print(f"Direct downsampled symbols mean power: {np.mean(np.abs(direct_syms)**2):.4f}")
    re = np.real(direct_syms)
    im = np.imag(direct_syms)
    print(f"Real min/mean/max: {np.min(re):.2f}, {np.mean(re):.2f}, {np.max(re):.2f}")
    print(f"Imag min/mean/max: {np.min(im):.2f}, {np.mean(im):.2f}, {np.max(im):.2f}")
    margin_re = float(np.mean(np.abs(re)))
    margin_im = float(np.mean(np.abs(im)))
    print(f"Average Decision Margin Real: {margin_re:.4f}, Imag: {margin_im:.4f}")

    print("=== 6. DEMOD BER ACROSS 4 ROTATIONS & DELAYS ===")
    best_ber = 1.0
    best_rot = 0
    best_delay = 0
    for d in range(8):
        for rot in [0, 90, 180, 270]:
            s = filtered[d::8] * np.exp(-1j * np.radians(rot))
            b0 = (np.real(s) > 0).astype(int)
            b1 = (np.imag(s) > 0).astype(int)
            b = np.empty(len(s)*2, dtype=int)
            b[0::2] = b0
            b[1::2] = b1
            min_l = min(len(b), len(tx_bits))
            err = int(np.sum(b[:min_l] != tx_bits[:min_l]))
            ber = err / min_l
            if ber < best_ber:
                best_ber = ber
                best_rot = rot
                best_delay = d

    print(f"Best Demod BER: {best_ber:.4f} at delay={best_delay}, rot={best_rot} deg")

    print("=== 7. VITERBI ERROR PATTERN ===")
    s_best = filtered[best_delay::8] * np.exp(-1j * np.radians(best_rot))
    b0 = (np.real(s_best) > 0).astype(int)
    b1 = (np.imag(s_best) > 0).astype(int)
    b_best = np.empty(len(s_best)*2, dtype=int)
    b_best[0::2] = b0
    b_best[1::2] = b1
    codec = ConvolutionalCodec(constraint_length=7, polynomials=(0o171, 0o133))
    rec_src, v_diag = codec.decode(b_best[:524])
    src_errs = int(np.sum(rec_src[:256] != src_bits[:256]))
    print(f"Viterbi Output Bits: {len(rec_src)}, Bit Errors vs Source: {src_errs}/256 ({src_errs/256:.4f})")
    print(f"Viterbi Metrics: {v_diag}")

if __name__ == "__main__":
    diagnose_g7()
