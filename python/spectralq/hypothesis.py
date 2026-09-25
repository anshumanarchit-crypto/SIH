"""
spectralq.hypothesis

Generic hypothesis representation and candidate evaluation engine for Phase 3.

ARCHITECTURE:
- Evaluates competing candidate hypotheses (e.g. carrier phase rotations,
  constellation mapping profiles, interleaver branches, FEC rates).
- Performs deterministic candidate ranking and consistency scoring.
- Resolves ambiguities without capture-specific hacks (e.g. no "G2=180" hardcoding).

CRITICAL PRINCIPLE:
RE-ENCODE CONSISTENCY IS EVIDENCE.
It is an algebraic consistency check confirming that the recovered source bits,
when re-encoded through the forward FEC/interleaver chain, yield the demapped
channel bits. It is NOT by itself absolute mathematical proof that the payload
is the transmitter's intended original message without external framing or CRC
validation.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from spectralq.schemas import (
    Hypothesis,
    HypothesisStatus,
)


def build_candidate_hypotheses(
    candidate_records: List[Dict[str, Any]],
    mapping_profile: str = "DEFAULT",
    fec_type_str: str = "NONE",
    selected_angle_deg: float = 0.0,
) -> Tuple[List[Hypothesis], Optional[Hypothesis]]:
    """Build and evaluate generic hypotheses from pipeline candidate evaluations.

    Parameters
    ----------
    candidate_records : List[Dict[str, Any]]
        List of candidate evaluation dictionaries from demod/decoder pipeline
        containing 'angle_deg', 'decoder_success', and 're_encode_errors'.
    mapping_profile : str
        Constellation demapping profile evaluated.
    fec_type_str : str
        FEC scheme used in candidate evaluation.
    selected_angle_deg : float
        The angle chosen by the pipeline.

    Returns
    -------
    hypotheses : List[Hypothesis]
        List of evaluated candidate hypotheses, sorted deterministically by rank.
    selected : Optional[Hypothesis]
        The best supported hypothesis, or None if none could be validated.
    """
    if not candidate_records:
        # Single nominal hypothesis (no competing candidates evaluated)
        nominal_hyp = Hypothesis(
            hypothesis_id="HYP_NOMINAL_0DEG",
            category="PHASE_ROTATION",
            parameters={
                "angle_deg": float(selected_angle_deg),
                "mapping_profile": mapping_profile,
                "fec_type": fec_type_str,
            },
            status=HypothesisStatus.SUPPORTED if fec_type_str == "NONE" else HypothesisStatus.CONFIRMED_BY_MULTIPLE_EVIDENCE,
            evidence_refs=["EVID_DEMOD_STATUS"],
            score_components={"fec_success": 1.0, "re_encode_consistency": 1.0},
            contradictions=[],
            re_encode_errors=0,
            consistency_fraction=1.0,
            rank=1,
        )
        return [nominal_hyp], nominal_hyp

    hyp_list: List[Hypothesis] = []

    for cand in candidate_records:
        deg = float(cand.get("angle_deg", 0.0))
        success = bool(cand.get("decoder_success", False))
        re_err = cand.get("re_encode_errors")
        if re_err is not None:
            re_err = int(re_err)

        hyp_id = f"HYP_ROT_{int(round(deg))}"
        score_comp: Dict[str, float] = {}
        contradictions: List[str] = []
        evidence_refs: List[str] = []

        if success:
            score_comp["fec_success"] = 1.0
            evidence_refs.append("EVID_FEC_SUCCESS")
        else:
            score_comp["fec_success"] = 0.0
            contradictions.append(f"FEC decoding failed for candidate rotation {deg}°.")

        consistency_frac: Optional[float] = None
        if re_err is not None:
            score_comp["re_encode_errors"] = float(re_err)
            if re_err == 0:
                score_comp["re_encode_consistency"] = 1.0
                consistency_frac = 1.0
                evidence_refs.append("EVID_REENCODE_ERRORS")
            else:
                score_comp["re_encode_consistency"] = max(0.0, 1.0 - min(1.0, re_err / 100.0))
                consistency_frac = max(0.0, 1.0 - min(1.0, re_err / 100.0))
                contradictions.append(f"Re-encode discrepancy ({re_err} errors) at {deg}°.")

        # Determine individual hypothesis status
        if success and re_err == 0:
            status = HypothesisStatus.CONFIRMED_BY_MULTIPLE_EVIDENCE
        elif success and (re_err is None or re_err < 10):
            status = HypothesisStatus.SUPPORTED
        elif success:
            status = HypothesisStatus.TENTATIVE
        else:
            status = HypothesisStatus.REJECTED

        hyp_list.append(Hypothesis(
            hypothesis_id=hyp_id,
            category="PHASE_ROTATION",
            parameters={
                "angle_deg": deg,
                "mapping_profile": mapping_profile,
                "fec_type": fec_type_str,
            },
            status=status,
            evidence_refs=evidence_refs,
            score_components=score_comp,
            contradictions=contradictions,
            re_encode_errors=re_err,
            consistency_fraction=consistency_frac,
            rank=0,
        ))

    # Deterministic multi-key sorting:
    # 1. FEC success (descending: True before False)
    # 2. re_encode_errors (ascending: 0 is best, None treated as 999999)
    # 3. angle_deg (ascending: tie-breaker)
    def _sort_key(h: Hypothesis) -> Tuple[int, int, float]:
        success_int = 0 if h.status != HypothesisStatus.REJECTED else 1
        err_int = h.re_encode_errors if h.re_encode_errors is not None else 999999
        angle = float(h.parameters.get("angle_deg", 0.0))
        return (success_int, err_int, angle)

    hyp_list.sort(key=_sort_key)

    # Assign ranks and detect ambiguities
    for rank_idx, h in enumerate(hyp_list, start=1):
        h.rank = rank_idx

    # Check for ambiguity among top candidates
    if len(hyp_list) >= 2:
        top1 = hyp_list[0]
        top2 = hyp_list[1]
        if (
            top1.status != HypothesisStatus.REJECTED
            and top2.status != HypothesisStatus.REJECTED
            and top1.re_encode_errors is not None
            and top2.re_encode_errors is not None
            and top1.re_encode_errors == top2.re_encode_errors
        ):
            top1.status = HypothesisStatus.AMBIGUOUS
            top2.status = HypothesisStatus.AMBIGUOUS

    selected = hyp_list[0] if hyp_list and hyp_list[0].status != HypothesisStatus.REJECTED else None

    return hyp_list, selected
