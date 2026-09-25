# Arpit Dependency Integration Report

**Target Remote**: `https://github.com/anshumanarchit-crypto/SIH.git`
**Branch**: `arpit/decoder-phase3.1-integration`
**Date**: 2026-09-25
**Author**: Arpit (Decoder Subsystem Owner)

---

## 1. Executive Summary

This report analyzes the dependency compatibility between the target team repository (`requirements.txt`) and Arpit's local decoder subsystem (`python/spectralq/`, `pyproject.toml`, `requirements.txt`).

The dependency merge was achieved with **zero version conflicts** and **minimal additive changes**:
- All existing team dependencies are preserved with their exact version constraints.
- Exactly two non-conflicting packages required for forward error correction (`scikit-commpy` and `reedsolo`) were added.
- Packaging configuration was added in `pyproject.toml` to support the isolated `spectralq` package without perturbing team scripts.

---

## 2. Dependency Classification Table

| Package | Classification | Target Constraint | Arpit Requirement | Final Integrated Constraint | Rationale & Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`numpy`** | `ALREADY_PRESENT` | `>=2.0.0` | `numpy` | `numpy>=2.0.0` | Core tensor math, signal arrays, FFT, IQ sample processing. |
| **`scipy`** | `ALREADY_PRESENT` | `>=1.15.0` | `scipy` | `scipy>=1.15.0` | Filter design, signal processing, convolution, Gardner loop. |
| **`scikit-learn`** | `ALREADY_PRESENT` | `>=1.6.0` | *(unused by decoder)* | `scikit-learn>=1.6.0` | Used by team RF modulation classification (`core/modulation.py`). |
| **`matplotlib`** | `ALREADY_PRESENT` | `>=3.10.0` | *(test visualizer)* | `matplotlib>=3.10.0` | Used by team visualization (`core/visualization.py`). |
| **`streamlit`** | `ALREADY_PRESENT` | `>=1.40.0` | *(unused by decoder)* | `streamlit>=1.40.0` | Used by Himanshu's web GUI (`app.py`). |
| **`pytest`** | `ALREADY_PRESENT` | `>=8.0.0` | `>=8.0.0` | `pytest>=8.0.0` | Test suite runner across all test directories. |
| **`scikit-commpy`** | `NEEDED` | *(absent)* | `>=0.8.0` | `scikit-commpy>=0.8.0` | NASA/CCSDS $K=7$ Trellis structure, Viterbi traceback decoding, LDPC matrix operations (`spectralq.fec`). |
| **`reedsolo`** | `NEEDED` | *(absent)* | `>=1.7.0` | `reedsolo>=1.7.0` | Galois Field GF($2^8$) Reed-Solomon RS(255, 223) codec (`spectralq.fec`). |

---

## 3. Dependency Conflict Analysis

### 3.1 Version Range Compatibility
- **`scikit-commpy`**: Compatible with NumPy 2.x and Python 3.10–3.13. Tested locally on Python 3.13.7 with zero import issues.
- **`reedsolo`**: Pure Python / Cython implementation with zero strict upper bounds. Tested locally with 100% pass on RS(255, 223) golden tests.
- **`numpy` & `scipy`**: Both local decoder code and target team code run cleanly on NumPy 2.x and SciPy 1.15+.

### 3.2 Result
- **Conflicting Dependencies**: **0**
- **Downgrades Required**: **0**
- **Invasive Changes**: **0**

---

## 4. Packaging Configuration (`pyproject.toml`)

The target repository lacked a `pyproject.toml`. A clean, standardized configuration was introduced:

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "spectralq"
version = "0.1.0"
description = "SpectralQ SIH26147 - Signal Intelligence & Decoder System"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "numpy>=2.0.0",
    "scipy>=1.15.0",
    "scikit-learn>=1.6.0",
    "matplotlib>=3.10.0",
    "streamlit>=1.40.0",
    "scikit-commpy>=0.8.0",
    "reedsolo>=1.7.0",
]

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
pythonpath = ["python", "."]
addopts = "-ra -q"

[tool.pyright]
include = ["python", "tests/unit", "tests/golden", "tests/integration", "tests/fixtures", "scripts"]
exclude = ["data/_incoming", "**/__pycache__", ".venv*"]
extraPaths = ["python"]
```

### Key Packaging Benefits:
1. `pythonpath = ["python", "."]` enables direct execution of both `from core...` and `from spectralq...` without path manipulation hacks.
2. `[tool.pyright]` scopes static type analysis to Arpit's verified typed packages, resulting in 0 errors and 0 warnings.
