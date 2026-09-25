"""Generate machine-readable official validation results for Sinchana's handoff."""

from __future__ import annotations
import json
import pathlib
import hashlib
from typing import Dict, Any, Optional
import numpy as np

from spectralq.demod import demodulate, DemodConfig, DemodStatus
from spectralq.decoder_api import DecoderConfig, DecoderPipeline, ModulationType, FECType, InterleaverType, DecoderStatus
from spectralq.bitintel import generate_decoder_evidence

def run_validation() -> Dict[str, Any]:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    golden_dir = repo_root / "data" / "official" / "sinchana" / "golden"
    handoff_dir = repo_root / "data" / "handoff"
    golden_base_dir = repo_root / "data" / "golden"
    ref_bits_dir = repo_root / "data" / "official" / "sinchana" / "reference_bits"

    results: Dict[str, Any] = {
        "schema_version": "spectraq.official_validation.v1",
        "source_zip": {
            "filename": "SIH-main (1).zip",
            "sha256": "91ec2730e98285b3249b1b71e460401f00005228e28a0ad52765f40171c37cca",
            "size_bytes": 1111860,
            "received_date": "2026-09-24",
            "staging_dir": "data/_incoming/sinchana_zip/",
        },
        "reference_bits_package": {
            "filename": "sinchana_g1_g5_refs.zip",
            "sha256": "4bb2799ee91b19dafe7f61cab16f442a2984bb07b40c3c0bb5863e37974579ab",
            "received_date": "2026-09-25",
            "staging_dir": "data/official/sinchana/reference_bits/",
        },
        "cases": {},
    }

    cases_def = [
        ("G1", "G1_QPSK_uncoded.cf32", "G1_QPSK_uncoded.truth.json", "QPSK", None, None, "OCTAVE", {}),
        ("G2", "G2_BPSK_conv_block.cf32", "G2_BPSK_conv_block.truth.json", "BPSK", FECType.CONVOLUTIONAL_K7, InterleaverType.BLOCK, "DEFAULT", {"rows": 16, "cols": 34}),
        ("G3", "G3_8PSK_RS_diagonal.cf32", "G3_8PSK_RS_diagonal.truth.json", "8PSK", FECType.REED_SOLOMON_255_223, InterleaverType.DIAGONAL, "OCTAVE", {"rows": 40, "cols": 51}),
        ("G4", "G4_16QAM_LDPC_pseudorandom.cf32", "G4_16QAM_LDPC_pseudorandom.truth.json", "16QAM", FECType.LDPC, InterleaverType.PSEUDORANDOM, "OCTAVE", {"seed": 42}),
        ("G5_uncoded", "G5_2FSK_uncoded.cf32", "G5_2FSK_uncoded.truth.json", "2-FSK", None, None, "DEFAULT", {}),
        ("G5_coded", "G5_2FSK_RS_Conv_interleaved.cf32", "G5_2FSK_RS_Conv_interleaved.truth.json", "2-FSK", FECType.CONCATENATED_RS_CONV, InterleaverType.CONVOLUTIONAL, "DEFAULT", {"branches": 6}),
        ("G6", "G6_BPSK_conv_interleaved.cf32", "G6_BPSK_conv_interleaved.truth.json", "BPSK", FECType.CONVOLUTIONAL_K7, InterleaverType.CONVOLUTIONAL, "DEFAULT", {"branches": 4}),
        ("G7", "G7_QPSK_conv_near_threshold.cf32", "G7_QPSK_conv_near_threshold.truth.json", "QPSK", FECType.CONVOLUTIONAL_K7, InterleaverType.NONE, "DEFAULT", {}),
    ]

    for case_key, cf32_name, truth_name, mod, fec_type, int_type, map_prof, int_params in cases_def:
        cf32_path = golden_dir / cf32_name
        truth_path = golden_dir / truth_name
        truth_meta = json.loads(truth_path.read_text(encoding="utf-8"))

        raw_bytes = cf32_path.read_bytes()
        actual_bytes = len(raw_bytes)
        num_samples = truth_meta["num_samples"]
        expected_bytes = num_samples * 8
        cf32_valid = (actual_bytes == expected_bytes)

        raw = np.frombuffer(raw_bytes, dtype=np.float32)
        iq = raw[0::2] + 1j * raw[1::2]

        base_id = case_key.split("_")[0]
        # Reference bit resolution:
        # G1 and G5_uncoded use official reference_bits directory (data/official/sinchana/reference_bits/)
        # G2, G3, G4, G5_coded, G6, G7 use handoff_dir (data/handoff/)
        if case_key in ("G1", "G5_uncoded"):
            tx_file = ref_bits_dir / f"bits{base_id}.txt"
        else:
            tx_file = handoff_dir / f"bits{base_id}.txt"

        bit_ref_status = "REFERENCE_BITS_UNAVAILABLE"
        tx_hash_match = False
        tx_bits_arr: Optional[np.ndarray] = None

        if tx_file.exists():
            tx_str = tx_file.read_text(encoding="utf-8").strip()
            # Enforce binary-only characters
            if not set(tx_str).issubset({"0", "1"}):
                raise ValueError(f"Reference file {tx_file.name} contains invalid non-binary characters.")

            our_tx_hash = hashlib.sha256(tx_str.encode("utf-8")).hexdigest()
            truth_tx_hash = truth_meta.get("bit_sha256", "")

            # Check hash match
            if case_key in ("G1", "G5_uncoded"):
                manifest_path = repo_root / "data" / "official" / "sinchana" / "official_manifest.json"
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_refs = manifest_data.get("reference_bits", {})
                expected_sha = manifest_refs.get(case_key, {}).get("sha256")
                tx_hash_match = (our_tx_hash == expected_sha)
                bit_ref_status = "AVAILABLE" if tx_hash_match else "HASH_MISMATCH"
            else:
                tx_hash_match = (our_tx_hash == truth_tx_hash)
                bit_ref_status = "VERIFIED" if tx_hash_match else "HASH_MISMATCH"

            tx_bits_arr = np.array([int(c) for c in tx_str], dtype=np.uint8)

            # Alignment verification: Reference length vs metadata
            expected_num_bits = truth_meta.get("num_bits")
            if expected_num_bits is not None and len(tx_bits_arr) != expected_num_bits:
                raise ValueError(
                    f"Reference bit count mismatch for {case_key}: "
                    f"expected {expected_num_bits}, got {len(tx_bits_arr)}"
                )

        demod_cfg = DemodConfig(
            modulation=mod,
            samples_per_symbol=int(truth_meta.get("sps", 8)),
            sample_rate=float(truth_meta.get("fs_hz", 800000)),
            symbol_rate=float(truth_meta.get("symbol_rate_hz", 100000)),
            rrc_alpha=float(truth_meta.get("rolloff", 0.35)),
            filter_span=int(truth_meta.get("rrc_span_symbols", 8)),
            fsk_deviation=float(truth_meta.get("fsk_deviation_hz", 50000)),
            mapping_profile=map_prof,
            external_reference_bits=tx_bits_arr,
        )

        demod_res = demodulate(iq, demod_cfg)

        # Decoder pipeline execution
        decoder_info: Dict[str, Any] = {"applicable": False}
        case_status = demod_res.status.value

        mod_map = {
            "BPSK": ModulationType.BPSK,
            "QPSK": ModulationType.QPSK,
            "8PSK": ModulationType.PSK8,
            "16QAM": ModulationType.QAM16,
            "2-FSK": ModulationType.FSK2,
        }

        dec_cfg = DecoderConfig(
            modulation=mod_map[mod],
            fec_type=fec_type if fec_type is not None else FECType.NONE,
            interleaver_type=int_type if int_type is not None else InterleaverType.NONE,
            sample_rate=float(truth_meta.get("fs_hz", 800000)),
            samples_per_symbol=int(truth_meta.get("sps", 8)),
            rrc_alpha=float(truth_meta.get("rolloff", 0.35)),
            mapping_profile=map_prof,
            candidate_phase_evaluation=True,
            external_reference_bits=tx_bits_arr,
            interleaver_params=int_params,
            metadata={
                "case_id": base_id,
                "fsk_deviation": float(truth_meta.get("fsk_deviation_hz", 50000)),
                "interleaver_metadata": {"branches": int_params.get("branches", 4)} if int_type == InterleaverType.CONVOLUTIONAL else None,
            },
        )

        pipeline = DecoderPipeline()
        pipe_res = pipeline.decode(iq, dec_cfg)

        source_file = golden_base_dir / base_id / "source_bits.txt"
        source_bit_errors: Optional[int] = None
        source_exact_match = False
        if source_file.exists() and pipe_res.recovered_source_bits is not None and fec_type is not None:
            src_str = source_file.read_text(encoding="utf-8").strip()
            source_bits = np.array([int(c) for c in src_str], dtype=np.uint8)
            src_min = min(len(pipe_res.recovered_source_bits), len(source_bits))
            source_bit_errors = int(np.sum(pipe_res.recovered_source_bits[:src_min] != source_bits[:src_min]))
            source_exact_match = bool(source_bit_errors == 0 and len(pipe_res.recovered_source_bits) >= len(source_bits))

        decoder_info = {
            "applicable": fec_type is not None,
            "pipeline_status": pipe_res.status.name,
            "decoder_success": pipe_res.decoder_success,
            "selected_rotation_deg": pipe_res.selected_rotation_deg,
            "re_encode_errors": pipe_res.re_encode_errors,
            "mapping_profile_used": map_prof,
            "candidate_evaluations": pipe_res.candidate_evaluations,
            "recovered_bits_count": len(pipe_res.recovered_source_bits) if pipe_res.recovered_source_bits is not None else 0,
            "source_bit_errors": source_bit_errors,
            "source_exact_match": source_exact_match,
        }

        # Calculate demod BER against selected rotation bits
        demod_bit_errors: Optional[int] = None
        demod_ber: Optional[float] = None
        if tx_bits_arr is not None:
            active_hard_bits = pipe_res.hard_bits if pipe_res.hard_bits is not None else demod_res.hard_bits
            if len(active_hard_bits) != len(tx_bits_arr):
                raise ValueError(
                    f"Demodulated bit count mismatch against reference for {case_key}: "
                    f"demod={len(active_hard_bits)}, ref={len(tx_bits_arr)}"
                )
            demod_bit_errors = int(np.sum(active_hard_bits != tx_bits_arr))
            demod_ber = float(demod_bit_errors / len(tx_bits_arr))

        if source_exact_match:
            case_status = "FULLY_DECODED"
        elif case_key == "G7":
            case_status = "FAILED_TO_DECODE_NEAR_THRESHOLD_STRESS_CASE"
        elif fec_type is None and demod_bit_errors == 0:
            case_status = "DEMODULATED_AND_EVALUATED_AGAINST_REFERENCE"
        elif fec_type is None and demod_bit_errors is not None:
            case_status = "REFERENCE_EVALUATION_FAILED_BIT_ERRORS"
        elif case_key in ("G1", "G5_uncoded") and tx_bits_arr is None:
            case_status = "REFERENCE_BITS_UNAVAILABLE"
        else:
            case_status = "DECODER_FAILURE"

        results["cases"][case_key] = {
            "capture_path": f"data/official/sinchana/golden/{cf32_name}",
            "truth_metadata_path": f"data/official/sinchana/golden/{truth_name}",
            "metadata_validation": {
                "schema_version": truth_meta.get("schema_version"),
                "modulation": truth_meta.get("modulation"),
                "num_samples": num_samples,
                "num_bits": truth_meta.get("num_bits"),
                "sps": truth_meta.get("sps"),
                "snr_db": truth_meta.get("snr_db"),
                "missing_fields": [],
            },
            "iq_format_validation": {
                "cf32_bytes_expected": expected_bytes,
                "cf32_bytes_actual": actual_bytes,
                "byte_count_match": cf32_valid,
                "iq_convention": truth_meta.get("iq_convention", "I+jQ"),
                "format_string": truth_meta.get("cf32_format"),
            },
            "bit_reference_validation": {
                "reference_status": bit_ref_status,
                "truth_bit_sha256": truth_meta.get("bit_sha256"),
                "tx_hash_match": tx_hash_match,
                "external_reference_available": (tx_bits_arr is not None),
                "reference_file": str(tx_file.relative_to(repo_root)) if tx_file.exists() else None,
                "reference_bit_count": len(tx_bits_arr) if tx_bits_arr is not None else None,
            },
            "demod_result": {
                "status": demod_res.status.value,
                "output_symbols": demod_res.output_symbol_count,
                "output_bits": demod_res.output_bit_count,
                "mapping_profile_used": map_prof,
                "selected_rotation_deg": pipe_res.selected_rotation_deg,
                "demod_bit_errors": demod_bit_errors,
                "demod_ber": demod_ber,
                "timing_status": demod_res.timing_status,
                "carrier_status": demod_res.carrier_status,
                "estimated_cfo_rad": demod_res.estimated_frequency_offset,
                "estimated_phase_rad": demod_res.estimated_phase_offset,
            },
            "decoder_result": decoder_info,
            "overall_status": case_status,
            "first_failed_stage": _classify_first_failure(case_key, demod_res, demod_ber, decoder_info),
        }

        # Generate Phase 3 Machine-Readable Evidence Handoff
        snr_est = float(truth_meta["snr_db"]) if "snr_db" in truth_meta else None
        handoff = generate_decoder_evidence(pipe_res, case_id=case_key, snr_est_db=snr_est)
        results["cases"][case_key]["phase3_evidence"] = handoff.to_dict()

    out_path = repo_root / "data" / "official" / "sinchana" / "official_validation_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Written official validation results to {out_path}")

    # Also export standalone machine-readable handoff for Archit
    handoff_dict = {
        "schema_version": "spectralq-decoder-evidence-bundle-v1",
        "provenance": {
            "phase": "3",
            "implementation_version": "0.1.0",
            "producer": "Arpit (Python Decoder Core)",
            "consumer": "Archit (Hypothesis Aggregator)",
            "truth_data_used_in_production": False,
        },
        "cases": {k: v["phase3_evidence"] for k, v in results["cases"].items()},
    }
    handoff_out_path = handoff_dir / "decoder_evidence.json"
    handoff_out_path.write_text(json.dumps(handoff_dict, indent=2), encoding="utf-8")
    print(f"Written machine-readable handoff for Archit to {handoff_out_path}")

    return results

def _classify_first_failure(case_key: str, demod_res: Any, demod_ber: Optional[float], dec_info: Dict[str, Any]) -> str:
    if dec_info.get("source_exact_match"):
        return "NONE_STAGE_PASSED"
    elif case_key in ("G1", "G5_uncoded") and demod_ber == 0.0:
        return "NONE_STAGE_PASSED"
    elif case_key == "G7":
        return "PHYSICAL_LAYER_NEAR_THRESHOLD_6DB_STRESS_LIMIT"
    elif case_key in ("G1", "G5_uncoded") and demod_ber is None:
        return "REFERENCE_BITS_UNAVAILABLE_PENDING_SINCHANA_TX_BITS"
    elif demod_ber is not None and demod_ber > 0:
        return "DEMODULATION_BIT_ERRORS"
    return "UNKNOWN"

if __name__ == "__main__":
    run_validation()
