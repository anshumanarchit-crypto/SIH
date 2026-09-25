# Arpit Integration Conflict & Provenance Report

**Target Remote**: `https://github.com/anshumanarchit-crypto/SIH.git`  
**Base Commit**: `f76b83e` (origin/main)  
**Integration Branch**: `arpit/decoder-phase3.1-integration`  
**Timestamp**: 2026-09-25  
**Author**: Arpit (Decoder Subsystem Lead)  

---

## 1. Classification Categories

- **A. SAFE ADD**: New file belonging to Arpit's subsystem; does not collide with any target path.
- **B. SAFE UPDATE**: Existing target file updated minimally and non-destructively.
- **C. EXISTING TEAM FILE — DO NOT TOUCH**: Existing team-owned file preserved with zero modifications.
- **D. CONFLICT REQUIRES REVIEW**: Implementation differences or pre-existing failures noted for team review.
- **E. UNNECESSARY / DO NOT COPY**: Build caches, virtual environments, or temporary archive files.
- **F. LARGE BINARY / VALIDATION ARTIFACT**: Raw IQ captures and validation bitstreams.
- **G. SECRET / PRIVATE — NEVER COPY**: Secrets, credentials, private environment files (None present).

---

## 2. File-by-File Classification Matrix

| Local Path | Target Path | Action | Reason | Ownership | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `python/spectralq/__init__.py` | `python/spectralq/__init__.py` | **A. SAFE ADD** | Subsystem package init and export metadata | Arpit | Low |
| `python/spectralq/demod.py` | `python/spectralq/demod.py` | **A. SAFE ADD** | Production demodulator (Gardner, Costas, Slicers) | Arpit | Low |
| `python/spectralq/decoder_api.py` | `python/spectralq/decoder_api.py` | **A. SAFE ADD** | Facade API (`DecoderConfig`, `DecoderPipeline`) | Arpit | Low |
| `python/spectralq/encode_chain.py` | `python/spectralq/encode_chain.py` | **A. SAFE ADD** | End-to-end codec chain harness for golden testing | Arpit | Low |
| `python/spectralq/fec.py` | `python/spectralq/fec.py` | **A. SAFE ADD** | Viterbi K=7, Reed-Solomon RS(255,223), LDPC | Arpit | Low |
| `python/spectralq/interleave.py` | `python/spectralq/interleave.py` | **A. SAFE ADD** | Block, diagonal, pseudorandom, convolutional interleavers | Arpit | Low |
| `python/spectralq/schemas.py` | `python/spectralq/schemas.py` | **A. SAFE ADD** | Typed enum schemas and configuration models | Arpit | Low |
| `python/spectralq/bitstream.py` | `python/spectralq/bitstream.py` | **A. SAFE ADD** | Bitstream analysis and characterization models | Arpit | Low |
| `python/spectralq/evidence.py` | `python/spectralq/evidence.py` | **A. SAFE ADD** | Phase 3 deterministic evidence evaluation engine | Arpit | Low |
| `python/spectralq/confidence.py` | `python/spectralq/confidence.py` | **A. SAFE ADD** | Phase 3 confidence scoring algorithm | Arpit | Low |
| `python/spectralq/hypothesis.py` | `python/spectralq/hypothesis.py` | **A. SAFE ADD** | Multi-hypothesis ranking model | Arpit | Low |
| `python/spectralq/bitintel.py` | `python/spectralq/bitintel.py` | **A. SAFE ADD** | Sync word mining, frame carving, CRC scanner | Arpit | Low |
| `pyproject.toml` | `pyproject.toml` | **A. SAFE ADD** | Standard packaging & pytest/pyright root configuration | Arpit / Team | Low |
| `requirements-dev.txt` | `requirements-dev.txt` | **A. SAFE ADD** | Development dependencies (pytest) | Arpit | Low |
| `requirements-lock.txt` | `requirements-lock.txt` | **A. SAFE ADD** | Frozen verified local dependencies | Arpit | Low |
| `scripts/diagnose_g7.py` | `scripts/diagnose_g7.py` | **A. SAFE ADD** | Diagnostic harness for G7 near-threshold case | Arpit | Low |
| `scripts/generate_golden.py` | `scripts/generate_golden.py` | **A. SAFE ADD** | Synthetic golden vector generator | Arpit | Low |
| `scripts/generate_official_validation_results.py` | `scripts/generate_official_validation_results.py` | **A. SAFE ADD** | Generates official validation JSON and handoff evidence | Arpit | Low |
| `scripts/verify_commpy.py` | `scripts/verify_commpy.py` | **A. SAFE ADD** | CommPy compatibility verification script | Arpit | Low |
| `docs/bitintel_output.md` | `docs/bitintel_output.md` | **A. SAFE ADD** | Bitstream intelligence reference documentation | Arpit | Low |
| `docs/codec_catalog.md` | `docs/codec_catalog.md` | **A. SAFE ADD** | Codec specifications and polynomial tables | Arpit | Low |
| `docs/decoder_interface.md` | `docs/decoder_interface.md` | **A. SAFE ADD** | Public API contract reference | Arpit | Low |
| `docs/demodulator.md` | `docs/demodulator.md` | **A. SAFE ADD** | DSP demodulator architecture documentation | Arpit | Low |
| `docs/golden_set.md` | `docs/golden_set.md` | **A. SAFE ADD** | Golden set descriptions and telemetry | Arpit | Low |
| `docs/official_decoder_alignment.md` | `docs/official_decoder_alignment.md` | **A. SAFE ADD** | Octave-to-Python decoder alignment specifications | Arpit | Low |
| `docs/official_validation_report.md` | `docs/official_validation_report.md` | **A. SAFE ADD** | Official capture validation report | Arpit | Low |
| `docs/phase3_1_reference_closure.md` | `docs/phase3_1_reference_closure.md` | **A. SAFE ADD** | Phase 3.1 G1/G5 reference closure report | Arpit | Low |
| `docs/phase3_bitstream_intelligence.md` | `docs/phase3_bitstream_intelligence.md` | **A. SAFE ADD** | Phase 3 bitstream intelligence methodology | Arpit | Low |
| `docs/phase3_confidence_method.md` | `docs/phase3_confidence_method.md` | **A. SAFE ADD** | Phase 3 confidence scoring documentation | Arpit | Low |
| `docs/phase3_evidence_model.md` | `docs/phase3_evidence_model.md` | **A. SAFE ADD** | Phase 3 evidence taxonomy and schemas | Arpit | Low |
| `docs/phase3_handoff_contract.md` | `docs/phase3_handoff_contract.md` | **A. SAFE ADD** | Downstream contract for Archit's evidence aggregation | Arpit | Low |
| `docs/arpit_dependency_integration.md` | `docs/arpit_dependency_integration.md` | **A. SAFE ADD** | Dependency classification and compatibility report | Arpit | Low |
| `docs/arpit_integration_conflict_report.md` | `docs/arpit_integration_conflict_report.md` | **A. SAFE ADD** | This integration and conflict report | Arpit | Low |
| `docs/arpit_pull_request_summary.md` | `docs/arpit_pull_request_summary.md` | **A. SAFE ADD** | Comprehensive PR summary | Arpit | Low |
| `tests/fixtures/__init__.py` | `tests/fixtures/__init__.py` | **A. SAFE ADD** | Test fixtures package init | Arpit | Low |
| `tests/fixtures/official_adapter.py` | `tests/fixtures/official_adapter.py` | **A. SAFE ADD** | Official capture evaluation fixture | Arpit | Low |
| `tests/fixtures/synthetic_generator.py` | `tests/fixtures/synthetic_generator.py` | **A. SAFE ADD** | Synthetic generator fixture for unit tests | Arpit | Low |
| `tests/golden/test_golden_roundtrip.py` | `tests/golden/test_golden_roundtrip.py` | **A. SAFE ADD** | Golden roundtrip verification tests | Arpit | Low |
| `tests/golden/test_official_g1_g5_reference_closure.py` | `tests/golden/test_official_g1_g5_reference_closure.py` | **A. SAFE ADD** | G1 and G5 reference closure test suite (10 tests) | Arpit | Low |
| `tests/golden/test_official_validation.py` | `tests/golden/test_official_validation.py` | **A. SAFE ADD** | Official Octave capture validation test suite | Arpit | Low |
| `tests/golden/test_phase3_official_handoff.py` | `tests/golden/test_phase3_official_handoff.py` | **A. SAFE ADD** | Phase 3 handoff contract verification tests | Arpit | Low |
| `tests/integration/test_demod_integration.py` | `tests/integration/test_demod_integration.py` | **A. SAFE ADD** | End-to-end demodulator integration tests | Arpit | Low |
| `tests/integration/test_encode_chain.py` | `tests/integration/test_encode_chain.py` | **A. SAFE ADD** | Codec chain integration tests | Arpit | Low |
| `tests/unit/test_bitstream_intelligence.py` | `tests/unit/test_bitstream_intelligence.py` | **A. SAFE ADD** | Bitstream intelligence unit tests | Arpit | Low |
| `tests/unit/test_carrier_recovery.py` | `tests/unit/test_carrier_recovery.py` | **A. SAFE ADD** | Costas loop carrier tracking tests | Arpit | Low |
| `tests/unit/test_external_reference.py` | `tests/unit/test_external_reference.py` | **A. SAFE ADD** | External reference evaluation tests | Arpit | Low |
| `tests/unit/test_fec.py` | `tests/unit/test_fec.py` | **A. SAFE ADD** | Viterbi, RS, LDPC unit tests | Arpit | Low |
| `tests/unit/test_fsk.py` | `tests/unit/test_fsk.py` | **A. SAFE ADD** | 2-FSK matched filter unit tests | Arpit | Low |
| `tests/unit/test_g7.py` | `tests/unit/test_g7.py` | **A. SAFE ADD** | G7 near-threshold behavior tests | Arpit | Low |
| `tests/unit/test_gardner.py` | `tests/unit/test_gardner.py` | **A. SAFE ADD** | Gardner timing recovery unit tests | Arpit | Low |
| `tests/unit/test_interleave.py` | `tests/unit/test_interleave.py` | **A. SAFE ADD** | Interleaver/deinterleaver unit tests | Arpit | Low |
| `tests/unit/test_mapping_profiles.py` | `tests/unit/test_mapping_profiles.py` | **A. SAFE ADD** | Octave vs Default constellation mapping tests | Arpit | Low |
| `tests/unit/test_package_structure.py` | `tests/unit/test_package_structure.py` | **A. SAFE ADD** | Package exports and namespace tests | Arpit | Low |
| `tests/unit/test_phase3_evidence_confidence.py` | `tests/unit/test_phase3_evidence_confidence.py` | **A. SAFE ADD** | Evidence & confidence calculation tests | Arpit | Low |
| `tests/unit/test_phase_candidates.py` | `tests/unit/test_phase_candidates.py` | **A. SAFE ADD** | Phase ambiguity resolution tests | Arpit | Low |
| `tests/unit/test_rrc.py` | `tests/unit/test_rrc.py` | **A. SAFE ADD** | RRC matched filter tests | Arpit | Low |
| `data/golden/G2..G7/` | `data/golden/G2..G7/` | **A. SAFE ADD** | Source bits and metadata for roundtrip tests | Arpit | Low |
| `data/handoff/bitsG2..G7.txt` | `data/handoff/bitsG2..G7.txt` | **A. SAFE ADD** | Validated handoff bitstreams | Arpit | Low |
| `data/handoff/decoder_evidence.json` | `data/handoff/decoder_evidence.json` | **A. SAFE ADD** | Machine-readable evidence handoff for Archit | Arpit | Low |
| `data/official/sinchana/official_manifest.json` | `data/official/sinchana/official_manifest.json` | **A. SAFE ADD** | Cryptographic manifest of official captures | Arpit / Sinchana | Low |
| `data/official/sinchana/official_validation_results.json` | `data/official/sinchana/official_validation_results.json` | **A. SAFE ADD** | Official validation results artifact | Arpit | Low |
| `data/official/sinchana/README.md` | `data/official/sinchana/README.md` | **A. SAFE ADD** | Official dataset provenance documentation | Arpit | Low |
| `data/official/sinchana/reference_bits/bitsG1.txt` | `data/official/sinchana/reference_bits/bitsG1.txt` | **A. SAFE ADD** | Phase 3.1 Sinchana G1 QPSK uncoded reference | Sinchana | Low |
| `data/official/sinchana/reference_bits/bitsG5.txt` | `data/official/sinchana/reference_bits/bitsG5.txt` | **A. SAFE ADD** | Phase 3.1 Sinchana G5 2-FSK uncoded reference | Sinchana | Low |
| `.gitignore` | `.gitignore` | **B. SAFE UPDATE** | Appended Arpit exclusions (`.venv-spectralq/`, `data/_incoming/`) | Shared | Low |
| `README.md` | `README.md` | **B. SAFE UPDATE** | Appended section 8 describing Decoder Subsystem | Shared | Low |
| `requirements.txt` | `requirements.txt` | **B. SAFE UPDATE** | Appended `scikit-commpy>=0.8.0` and `reedsolo>=1.7.0` | Shared | Low |
| `app.py` | `app.py` | **C. PROTECTED** | Streamlit web application (Himanshu) | Himanshu | None |
| `core/*` (12 files) | `core/*` | **C. PROTECTED** | Existing team DSP and pipeline modules | Archit/Team | None |
| `octave/*` (3 files) | `octave/*` | **C. PROTECTED** | Sinchana Octave signal processing scripts | Sinchana | None |
| `data/golden/*.cf32` (8 files) | `data/golden/*.cf32` | **C. PROTECTED** | Canonical team golden IQ waveforms | Team | None |
| `data/golden/*.truth.json` (8 files) | `data/golden/*.truth.json` | **C. PROTECTED** | Canonical team golden truth metadata | Team | None |
| `data/synthetic/*` (12 files) | `data/synthetic/*` | **C. PROTECTED** | Synthetic benchmark waveforms | Team | None |
| `data/synthetic_generator.py` | `data/synthetic_generator.py` | **C. PROTECTED** | Team synthetic waveform generator | Team | None |
| `docs/octave_environment.txt` | `docs/octave_environment.txt` | **C. PROTECTED** | Team Octave environment documentation | Sinchana | None |
| `docs/real_dataset_provenance.md` | `docs/real_dataset_provenance.md` | **C. PROTECTED** | Team real dataset provenance | Sinchana | None |
| `tests/test_*.py` (10 files) | `tests/test_*.py` | **C. PROTECTED** | Team test suite covering `core/` pipeline | Team | None |
| `data/official/sinchana/golden/*.cf32` | `data/official/sinchana/golden/*.cf32` | **F. VALIDATION ARTIFACT** | Exact SHA-256 match with `data/golden/*.cf32` | Sinchana | Low |
| `data/official/sinchana/golden/*.truth.json` | `data/official/sinchana/golden/*.truth.json` | **F. VALIDATION ARTIFACT** | Exact SHA-256 match with `data/golden/*.truth.json` | Sinchana | Low |
| `data/_incoming/` | *(excluded)* | **E. UNNECESSARY** | Raw input zip archives (`sinchana_g1_g5_refs.zip`) | Staging | None |
| `.venv*` | *(excluded)* | **E. UNNECESSARY** | Local Python virtual environments | Dev | None |
| `core/demodulation.py:459` | `core/demodulation.py:459` | **D. FOR REVIEW** | Pre-existing 1-test failure in `test_demodulate_2fsk_matched_filter_and_discriminator` | Team | Review |

---

## 3. Provenance & Duplicate Artifact Verification

The 8 `.cf32` files and 8 `.truth.json` files under `data/official/sinchana/golden/` were compared against `data/golden/` on `origin/main`:

| Waveform Capture | File Size (bytes) | SHA-256 Checksum | Match Status | Canonical Origin |
| :--- | :--- | :--- | :--- | :--- |
| `G1_QPSK_uncoded.cf32` | 131,072 | `e2becef449e81ea48309b0b815fbbc4ef31ade7ef975ad3209042fd022c2b0c9` | **100% IDENTICAL** | Sinchana Octave |
| `G1_QPSK_uncoded.truth.json` | 582 | `60d34238b0e0152d4bd2bc8e2a840483c7a16699d5f21d7ebcfc6378ada8891a` | **100% IDENTICAL** | Sinchana Octave |
| `G2_BPSK_conv_block.cf32` | 34,816 | `e8648923f8f1c9b1c7612f6841022347b0a2d80781e7eff1c8b417fbe6164e69` | **100% IDENTICAL** | Sinchana Octave |
| `G2_BPSK_conv_block.truth.json` | 628 | `e8b69b0e2305026973ec8727c170998717c0aa1a186459350564236a564dd4d4` | **100% IDENTICAL** | Sinchana Octave |
| `G3_8PSK_RS_diagonal.cf32` | 43,520 | `97b154c3e557a49e81e3305f1c433b32d1601026c2a6e56a3467033a28407871` | **100% IDENTICAL** | Sinchana Octave |
| `G3_8PSK_RS_diagonal.truth.json` | 626 | `4a9b360a4cea429510b6279456bdd426feabd9c024a4648b30be238a5873bd95` | **100% IDENTICAL** | Sinchana Octave |
| `G4_16QAM_LDPC_pseudorandom.cf32` | 1,536 | `f468df6c1a95ea7c87ba94e314514220c65a769562a2e3f49ae93ec9208f74e2` | **100% IDENTICAL** | Sinchana Octave |
| `G4_16QAM_LDPC_pseudorandom.truth.json` | 646 | `77fb7c8301147c7366bcff94427cae78102cb3b4a9c88b5b77d4a5695a6c22e8` | **100% IDENTICAL** | Sinchana Octave |
| `G5_2FSK_RS_Conv_interleaved.cf32` | 265,728 | `b69f616917d79c6e9a5c6cefc59e707f00e5beb2af730b6b1ce0edd37cc6cd7e` | **100% IDENTICAL** | Sinchana Octave |
| `G5_2FSK_RS_Conv_interleaved.truth.json` | 643 | `b4b2ade1c6f55edb0050173d1943e1fb35b2066055bfe94fca273214a822543f` | **100% IDENTICAL** | Sinchana Octave |
| `G5_2FSK_uncoded.cf32` | 262,144 | `234a8390be4a60bf1fcc420513128dc1c3939a1fa8dcd15ac40cd057b88b30b9` | **100% IDENTICAL** | Sinchana Octave |
| `G5_2FSK_uncoded.truth.json` | 583 | `666dc31c67aff5c943ffe52aa398e55fcac8200372b21aec2d9362e05e35a419` | **100% IDENTICAL** | Sinchana Octave |
| `G6_BPSK_conv_interleaved.cf32` | 35,072 | `e2d39c7370b33bc5dc0b76be84f6ecdc6571d411983463d16bcde0bfa7bac30b` | **100% IDENTICAL** | Sinchana Octave |
| `G6_BPSK_conv_interleaved.truth.json` | 641 | `15469851b612b3c788dc4aaef8ab497eba0cd83572368409654d9d5e877c2c43` | **100% IDENTICAL** | Sinchana Octave |
| `G7_QPSK_conv_near_threshold.cf32` | 131,072 | `3dd4a0dd37d259c4033c5e4fc9668d27376c7255476aebbeab64bfb5638c4dae` | **100% IDENTICAL** | Sinchana Octave |
| `G7_QPSK_conv_near_threshold.truth.json` | 629 | `08487e8cb40af5a589cf87948a474dd31a5ebc69f06df5f8a0026e6ef10f9247` | **100% IDENTICAL** | Sinchana Octave |

**Decision**:
Because Git uses content-addressable storage, tracking these identical files under `data/official/sinchana/golden/` incurs **0 bytes of duplicate blob storage** in Git object storage, while preserving strict compatibility with `official_manifest.json` and `test_official_validation.py`.

---

## 4. Conflict Flagged for Teammate Review (Category D)

- **Location**: `core/demodulation.py:459`
- **Issue**: Pre-existing unit test failure in `tests/test_demodulation.py::test_demodulate_2fsk_matched_filter_and_discriminator`.
- **Cause**: In `core/demodulation.py`, `np.diff(phase)` returns $N-1$ points (2559 points for 80 symbols $\times$ 32 SPS = 2560 samples). Attempting to reshape `freq[:2560]` into `(80, 32)` triggers `ValueError: cannot reshape array of size 2559 into shape (80,32)`.
- **Action Taken**: In accordance with Section 9 instructions to protect teammate files in `core/*` and specifically `core/demodulation.py`, this file was **NOT modified**.
- **Recommended Team Patch**: Update line 458 to `freq = np.diff(phase, prepend=phase[0]) * (sample_rate / (2.0 * np.pi))`. Tested locally: resolves the test with 0 bit errors.
