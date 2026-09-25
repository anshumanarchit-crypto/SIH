"""
SpectralQ Python Decoder Core (SIH26147)

Workspace: Arpit (Python Decoder Core)
Responsibilities:
- Interleaving / de-interleaving (interleave.py)
- Demodulation / waveform recovery (demod.py)
- Forward Error Correction (fec.py)
- Stable Decoder Facade & Interface (decoder_api.py)
- Bit-stream Intelligence & Structural Analysis (bitstream.py, bitintel.py)
- Evidence Ledger & Contradiction Detection (evidence.py)
- Hypothesis Generation & Candidate Evaluation (hypothesis.py)
- Deterministic Evidence Confidence (confidence.py)
- Data Schemas & Downstream Handoff Contracts (schemas.py)
- Pipeline Encode / Decode Chain Orchestration (encode_chain.py)

Note: Archit handles downstream cross-layer evidence aggregation and final UNKNOWN decision;
      Sinchana handles Octave DSP frontend and raw-signal captures;
      Harsh handles modulation classification benchmarks;
      Himanshu handles Streamlit GUI.
"""

from spectralq import decoder_api
from spectralq import demod
from spectralq import fec
from spectralq import interleave
from spectralq import bitstream
from spectralq import evidence
from spectralq import hypothesis
from spectralq import confidence
from spectralq import bitintel
from spectralq import schemas
from spectralq import encode_chain

__version__ = "0.1.0"
__author__ = "Arpit"
__all__ = [
    "decoder_api",
    "demod",
    "fec",
    "interleave",
    "bitstream",
    "evidence",
    "hypothesis",
    "confidence",
    "bitintel",
    "schemas",
    "encode_chain",
]
