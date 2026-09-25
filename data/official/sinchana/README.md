# Official Sinchana Waveform Captures (Phase 2.5 Handoff)

## Provenance Information
- **ZIP Filename**: `SIH-main (1).zip`
- **ZIP SHA-256**: `91ec2730e98285b3249b1b71e460401f00005228e28a0ad52765f40171c37cca`
- **Source Folder Inside ZIP**: `SIH-main/data/golden/`
- **Date Received**: 24 September 2026
- **Staging Location**: `data/_incoming/sinchana_zip/`
- **Official Golden Location**: `data/official/sinchana/golden/`

## Ingested Artifacts
The following 16 files (8 complex baseband captures and 8 ground-truth metadata files) were staged and verified:
1. `G1_QPSK_uncoded.cf32` (131,072 bytes) + `G1_QPSK_uncoded.truth.json`
2. `G2_BPSK_conv_block.cf32` (34,816 bytes) + `G2_BPSK_conv_block.truth.json`
3. `G3_8PSK_RS_diagonal.cf32` (43,520 bytes) + `G3_8PSK_RS_diagonal.truth.json`
4. `G4_16QAM_LDPC_pseudorandom.cf32` (1,536 bytes) + `G4_16QAM_LDPC_pseudorandom.truth.json`
5. `G5_2FSK_RS_Conv_interleaved.cf32` (265,728 bytes) + `G5_2FSK_RS_Conv_interleaved.truth.json`
6. `G5_2FSK_uncoded.cf32` (262,144 bytes) + `G5_2FSK_uncoded.truth.json`
7. `G6_BPSK_conv_interleaved.cf32` (35,072 bytes) + `G6_BPSK_conv_interleaved.truth.json`
8. `G7_QPSK_conv_near_threshold.cf32` (16,768 bytes) + `G7_QPSK_conv_near_threshold.truth.json`

## Immutability Statement
These files are external validation artifacts generated independently by Sinchana.
They are strictly **immutable validation inputs**.
They must never be overwritten, modified, or consumed directly by production decoder/demodulator runtime logic (`truth.json` is strictly restricted to test harnesses and offline evaluation).
