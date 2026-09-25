# Pull Request: Integrate Arpit Decoder and Phase 3.1 Subsystem

**Target Repository**: `https://github.com/anshumanarchit-crypto/SIH.git`  
**Base Branch**: `origin/main` (commit `f76b83e`)  
**Feature Branch**: `arpit/decoder-phase3.1-integration`  
**Author**: Arpit (Decoder Subsystem Lead)  
**Status**: Ready for Pull Request Review  

---

## 1. Purpose & Scope

This PR integrates Arpit's completed Python Decoder Subsystem (Phases 1, 2, 3, and 3.1 Reference Closure) into the team's SpectralQ repository without modifying, overwriting, or breaking existing teammate code.

### Core Objectives Delivered:
- **Clean Subsystem Isolation**: Placed entirely within `python/spectralq/`, preventing namespace collisions with `core/`.
- **Phase 3 Deterministic Evidence Extraction**: Generates structured, typed evidence schemas (`data/handoff/decoder_evidence.json`) for Archit's multi-source hypothesis engine.
- **Phase 3.1 Reference Closure**: Full integration and validation of Sinchana's G1 (QPSK uncoded) and G5 (2-FSK uncoded) external reference bit packages with 0.0 BER.
- **Strict Truth Isolation**: Zero production decoder code accesses ground-truth metadata or reference bits.
- **Zero Force Push / Non-Destructive**: Built cleanly on top of `origin/main`.

---

## 2. File Changes Summary

### 2.1 Files Added (Arpit Subsystem)
- **Production Decoder Package (`python/spectralq/`)**:
  - `__init__.py`, `demod.py`, `decoder_api.py`, `encode_chain.py`, `fec.py`, `interleave.py`, `schemas.py`, `bitstream.py`, `evidence.py`, `confidence.py`, `hypothesis.py`, `bitintel.py`.
- **Packaging & Tooling Configuration**:
  - `pyproject.toml`, `requirements-dev.txt`, `requirements-lock.txt`.
- **Scripts**:
  - `scripts/generate_official_validation_results.py`, `scripts/diagnose_g7.py`, `scripts/generate_golden.py`, `scripts/verify_commpy.py`.
- **Tests (`tests/`)**:
  - `tests/fixtures/` (`official_adapter.py`, `synthetic_generator.py`, `__init__.py`)
  - `tests/unit/` (13 unit test modules covering RRC, Gardner, Costas, FSK, Interleave, FEC, Bitstream Intelligence, etc.)
  - `tests/golden/` (`test_official_validation.py`, `test_official_g1_g5_reference_closure.py`, `test_phase3_official_handoff.py`, `test_golden_roundtrip.py`)
  - `tests/integration/` (`test_demod_integration.py`, `test_encode_chain.py`)
- **Data & Artifacts**:
  - `data/golden/G2..G7/` (source bits and metadata for roundtrip tests)
  - `data/handoff/` (`decoder_evidence.json`, `bitsG2.txt` through `bitsG7.txt`)
  - `data/official/sinchana/` (`official_manifest.json`, `official_validation_results.json`, `README.md`, `golden/`, `reference_bits/bitsG1.txt`, `reference_bits/bitsG5.txt`)
- **Documentation**:
  - `docs/phase3*` (5 Phase 3/3.1 engineering specifications)
  - `docs/arpit_dependency_integration.md`
  - `docs/arpit_integration_conflict_report.md`
  - `docs/arpit_pull_request_summary.md`

### 2.2 Files Modified (Minimal & Additive)
- `.gitignore`: Added `.venv-spectralq/`, `data/_incoming/`, and `*.local_bak`.
- `README.md`: Added Section 8 "Decoder Subsystem (Arpit)" detailing package layout, public API, handoff artifacts, and validation commands.
- `requirements.txt`: Added `scikit-commpy>=0.8.0` and `reedsolo>=1.7.0` (all original 6 team dependencies strictly preserved).

### 2.3 Files Intentionally Untouched (Protected Teammate Ownership)
- `app.py` (Himanshu's Streamlit dashboard)
- `core/*` (12 team DSP and pipeline modules)
- `octave/*` (3 Sinchana DSP scripts)
- `data/golden/*.cf32` and `*.truth.json` (canonical team golden captures)
- `data/synthetic/*` and `data/synthetic_generator.py` (team synthetic data)
- `docs/octave_environment.txt` and `docs/real_dataset_provenance.md`
- `tests/test_*.py` (10 team test modules)

---

## 3. Validation Results

| Test Category | Target Result | Pre-Integration | Post-Integration | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Arpit Pytest Suite** | 223 tests | 223 passed | **223 passed (100%)** | ✅ PASSED |
| **Pyright Type Check** | 0 errors | 0 errors | **0 errors, 0 warnings** | ✅ CLEAN |
| **Target Repository Tests** | 260 tests | 259 passed, 1 fail* | **259 passed, 1 fail* (unchanged)** | ✅ UNREGRESSED |
| **Official Validation Run** | 8 captures | Clean generation | **Clean generation** | ✅ PASSED |
| **Truth Isolation Audit** | 0 production dependencies | Clean | **0 production dependencies** | ✅ CLEAN |

*\* Note: The single failure in `tests/test_demodulation.py` is pre-existing in `origin/main` commit `f76b83e`. See Section 6.*

---

## 4. Regression & Case Verification Contract

| Case | Modulation | Impairment / Coding | Evaluated BER | Bit Errors | Confidence Score | Decision / Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **G1** | QPSK | Uncoded (Octave ref closure) | **0.0000** | 0 / 4096 | 0.9352 | `CONFIRMED_BY_MULTIPLE_EVIDENCE` |
| **G2** | BPSK | Conv K=7, Block Interleaved | **0.0000** | 0 / 544 | 0.9352 | `CONFIRMED_BY_MULTIPLE_EVIDENCE` |
| **G3** | 8PSK | RS(255,223), Diagonal Interleaved | **0.0000** | 0 / 2040 | 0.9352 | `CONFIRMED_BY_MULTIPLE_EVIDENCE` |
| **G4** | 16QAM | LDPC, Pseudorandom Interleaved | **0.0000** | 0 / 96 | 0.9352 | `CONFIRMED_BY_MULTIPLE_EVIDENCE` |
| **G5 Uncoded** | 2-FSK | Uncoded (Octave ref closure) | **0.0000** | 0 / 4096 | 0.9352 | `CONFIRMED_BY_MULTIPLE_EVIDENCE` |
| **G5 Coded** | 2-FSK | RS + Conv, Convolutional Interleaved | **0.0000** | 0 / 4152 | 0.8852 | `SUPPORTED` |
| **G6** | BPSK | Conv K=7, Convolutional Interleaved | **0.0000** | 0 / 548 | 0.9352 | `CONFIRMED_BY_MULTIPLE_EVIDENCE` |
| **G7** | QPSK | Conv K=7 (6.0 dB SNR stress case) | **0.4580** | 240 / 524 | 0.1200 | `FAILED_TO_DECODE_NEAR_THRESHOLD_STRESS_CASE` |

**Zero Fabrication / Defensibility Highlights**:
- G1 and G5 uncoded bit references are evaluated post-Costas phase-ambiguity resolution with zero hardcoded values or shortcuts.
- G7 is verified as a near-threshold stress failure and is strictly **not** promoted to a false success.

---

## 5. Team Ownership Boundaries

| Teammate | Subsystem / Directory | Responsibility Boundary |
| :--- | :--- | :--- |
| **Arpit** | `python/spectralq/`, `tests/unit/`, `tests/golden/`, `data/handoff/`, `docs/phase3*` | Decoder Core, FEC, Demodulation, Bitstream Intelligence, Evidence Generation. |
| **Archit** | Downstream Aggregator / `result.json` | Cross-source evidence aggregation, hypothesis arbitration, UNKNOWN classifier. Consumes `data/handoff/decoder_evidence.json`. |
| **Sinchana** | `octave/`, `data/official/sinchana/`, `data/golden/*.cf32` | GNU Octave DSP front-end, burst detection, RF golden dataset. |
| **Harsh** | Future ML Classifier Output | Modulation classification telemetry. |
| **Himanshu** | `app.py` | Streamlit user interface and visualization dashboard. |

---

## 6. Action Items for Teammates

1. **Archit (Multi-Source Aggregator)**:
   - Ingest `data/handoff/decoder_evidence.json` using the contract schema in `docs/phase3_handoff_contract.md`.
   - Fuse decoder evidence with RF cumulant features and ML classifier telemetry.
2. **Team Review on `core/demodulation.py:459`**:
   - `core/demodulation.py` was left untouched to avoid altering teammate files.
   - In `test_demodulate_2fsk_matched_filter_and_discriminator`, updating line 458 to `freq = np.diff(phase, prepend=phase[0]) * ...` resolves the pre-existing 1-test failure with zero bit errors.
3. **Himanshu (GUI)**:
   - When ready, display decoder evidence metrics (`confidence.value`, `demod_quality`, `carrier_quality`) on the Streamlit dashboard alongside RF plots.
