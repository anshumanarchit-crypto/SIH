"""
spectralq.evidence

Explicit evidence ledger and contradiction detection subsystem for Phase 3.

Provides an auditable, machine-readable record of all factual metrics gathered
across the physical, synchronization, demodulation, decoding, and bitstream
analysis stages.

RULES:
- Every important conclusion must have explicit evidence attached.
- Contradictory evidence must never be silently suppressed.
- Zero truth-data leakage into production.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np

from spectralq.schemas import (
    EvidenceCategory,
    EvidenceReliability,
    EvidenceItem,
    Contradiction,
    ContradictionSeverity,
    BitstreamAnalysisResult,
)


class EvidenceLedger:
    """Auditable ledger of factual evidence and detected contradictions."""

    def __init__(self) -> None:
        self._items: List[EvidenceItem] = []
        self._contradictions: List[Contradiction] = []
        self._item_map: Dict[str, EvidenceItem] = {}

    def add_item(self, item: EvidenceItem) -> None:
        """Add an evidence item to the ledger."""
        self._items.append(item)
        self._item_map[item.evidence_id] = item

    def add_contradiction(self, contradiction: Contradiction) -> None:
        """Record an explicit contradiction."""
        self._contradictions.append(contradiction)

    def get_items(self) -> List[EvidenceItem]:
        """Return all recorded evidence items."""
        return list(self._items)

    def get_contradictions(self) -> List[Contradiction]:
        """Return all recorded contradictions."""
        return list(self._contradictions)

    def get_by_category(self, category: EvidenceCategory) -> List[EvidenceItem]:
        """Query evidence items belonging to a specific category."""
        return [it for it in self._items if it.category == category]

    def get_by_id(self, evidence_id: str) -> Optional[EvidenceItem]:
        """Retrieve an evidence item by its ID."""
        return self._item_map.get(evidence_id)

    def detect_contradictions(self) -> List[Contradiction]:
        """Scan recorded evidence for systemic contradictions and anomalies.
        
        Evaluates physical synchronization, FEC integrity, re-encode consistency,
        and reference availability.
        """
        existing_types = {c.contradiction_type for c in self._contradictions}

        # 1. Timing synchronization check
        timing_lock = self.get_by_id("EVID_TIMING_LOCK")
        timing_jitter = self.get_by_id("EVID_TIMING_JITTER_VAR")
        if timing_lock and not bool(timing_lock.value):
            if "TIMING_UNLOCKED" not in existing_types:
                self.add_contradiction(Contradiction(
                    contradiction_id="CONTRA_TIMING_UNLOCKED",
                    contradiction_type="TIMING_UNLOCKED",
                    severity=ContradictionSeverity.CRITICAL,
                    details="Symbol timing recovery loop failed to converge/lock.",
                    metric="timing_lock",
                    value=False,
                    source_evidence_ids=["EVID_TIMING_LOCK"],
                ))
        elif timing_jitter and timing_jitter.value is not None:
            try:
                j_val = float(timing_jitter.value)
                if j_val > 0.15 and "TIMING_JITTER_ELEVATED" not in existing_types:
                    self.add_contradiction(Contradiction(
                        contradiction_id="CONTRA_TIMING_JITTER_ELEVATED",
                        contradiction_type="TIMING_JITTER_ELEVATED",
                        severity=ContradictionSeverity.HIGH,
                        details=f"Timing error variance is severely elevated ({j_val:.4f} > 0.15 threshold).",
                        metric="timing_jitter_variance",
                        value=j_val,
                        source_evidence_ids=["EVID_TIMING_JITTER_VAR"],
                    ))
            except (ValueError, TypeError):
                pass

        # 2. Carrier synchronization check
        carrier_lock = self.get_by_id("EVID_CARRIER_LOCK")
        if carrier_lock and not bool(carrier_lock.value):
            if "CARRIER_UNLOCKED" not in existing_types:
                self.add_contradiction(Contradiction(
                    contradiction_id="CONTRA_CARRIER_UNLOCKED",
                    contradiction_type="CARRIER_UNLOCKED",
                    severity=ContradictionSeverity.HIGH,
                    details="Carrier frequency/phase recovery loop failed to converge.",
                    metric="carrier_lock",
                    value=False,
                    source_evidence_ids=["EVID_CARRIER_LOCK"],
                ))

        # 3. Physical SNR / EVM check (near-threshold condition)
        snr_est = self.get_by_id("EVID_PHYS_SNR_EST")
        evm_rms = self.get_by_id("EVID_PHYS_EVM_RMS")
        if snr_est and snr_est.value is not None:
            try:
                snr_val = float(snr_est.value)
                if snr_val <= 7.0 and "NEAR_THRESHOLD_SNR" not in existing_types:
                    self.add_contradiction(Contradiction(
                        contradiction_id="CONTRA_NEAR_THRESHOLD_SNR",
                        contradiction_type="NEAR_THRESHOLD_SNR",
                        severity=ContradictionSeverity.HIGH,
                        details=f"Estimated SNR ({snr_val:.1f} dB) indicates near-threshold stress condition (<= 7.0 dB).",
                        metric="estimated_snr_db",
                        value=snr_val,
                        source_evidence_ids=["EVID_PHYS_SNR_EST"],
                    ))
            except (ValueError, TypeError):
                pass
        elif evm_rms and evm_rms.value is not None:
            try:
                evm_val = float(evm_rms.value)
                if evm_val > 0.35 and "HIGH_EVM_DEGRADATION" not in existing_types:
                    self.add_contradiction(Contradiction(
                        contradiction_id="CONTRA_HIGH_EVM_DEGRADATION",
                        contradiction_type="HIGH_EVM_DEGRADATION",
                        severity=ContradictionSeverity.MEDIUM,
                        details=f"RMS Error Vector Magnitude is elevated ({evm_val:.3f} > 0.35 threshold).",
                        metric="evm_rms",
                        value=evm_val,
                        source_evidence_ids=["EVID_PHYS_EVM_RMS"],
                    ))
            except (ValueError, TypeError):
                pass

        # 4. FEC Decoding success
        fec_success = self.get_by_id("EVID_FEC_SUCCESS")
        if fec_success and not bool(fec_success.value):
            if "FEC_DECODE_FAILED" not in existing_types:
                self.add_contradiction(Contradiction(
                    contradiction_id="CONTRA_FEC_DECODE_FAILED",
                    contradiction_type="FEC_DECODE_FAILED",
                    severity=ContradictionSeverity.CRITICAL,
                    details="FEC decoder could not recover a valid codeword / trellis termination failed.",
                    metric="fec_success",
                    value=False,
                    source_evidence_ids=["EVID_FEC_SUCCESS"],
                ))

        # 5. Re-encode consistency
        reencode_err = self.get_by_id("EVID_REENCODE_ERRORS")
        if reencode_err and reencode_err.value is not None:
            try:
                r_err = int(reencode_err.value)
                if r_err > 0 and "REENCODE_DISCREPANCY" not in existing_types:
                    self.add_contradiction(Contradiction(
                        contradiction_id="CONTRA_REENCODE_DISCREPANCY",
                        contradiction_type="REENCODE_DISCREPANCY",
                        severity=ContradictionSeverity.HIGH,
                        details=f"Re-encoded payload does not match demapped bitstream ({r_err} bit mismatches).",
                        metric="re_encode_errors",
                        value=r_err,
                        source_evidence_ids=["EVID_REENCODE_ERRORS"],
                    ))
            except (ValueError, TypeError):
                pass

        # 6. Reference availability
        ref_status = self.get_by_id("EVID_REF_STATUS")
        if ref_status and ref_status.value == "REFERENCE_BITS_UNAVAILABLE":
            if "REFERENCE_UNAVAILABLE" not in existing_types:
                self.add_contradiction(Contradiction(
                    contradiction_id="CONTRA_REFERENCE_UNAVAILABLE",
                    contradiction_type="REFERENCE_UNAVAILABLE",
                    severity=ContradictionSeverity.INFO,
                    details="No external reference bits provided; true BER cannot be evaluated objectively.",
                    metric="reference_status",
                    value="REFERENCE_BITS_UNAVAILABLE",
                    source_evidence_ids=["EVID_REF_STATUS"],
                ))

        return list(self._contradictions)
