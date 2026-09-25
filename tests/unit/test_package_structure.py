"""
tests/unit/test_package_structure.py

Phase 0 verification tests:
- Package can be imported
- Submodules can be imported without errors
- No circular imports
- Decoder facade dataclasses / enums instantiate cleanly
- Documented project files exist
"""

from pathlib import Path
import pytest


def test_package_import():
    """Verify spectralq package is importable and defines version metadata."""
    import spectralq
    assert hasattr(spectralq, "__version__")
    assert spectralq.__version__ == "0.1.0"


def test_module_imports():
    """Verify all five decoder core submodules import cleanly."""
    from spectralq import interleave
    from spectralq import fec
    from spectralq import demod
    from spectralq import decoder_api
    from spectralq import bitintel

    assert interleave is not None
    assert fec is not None
    assert demod is not None
    assert decoder_api is not None
    assert bitintel is not None


def test_decoder_api_types():
    """Verify typed enum and dataclass contracts in decoder_api."""
    from spectralq.decoder_api import (
        DecoderStatus,
        ModulationType,
        FECType,
        InterleaverType,
        DecoderConfig,
        DecoderResult,
        DecoderPipeline,
    )

    config = DecoderConfig(
        modulation=ModulationType.BPSK,
        fec_type=FECType.CONVOLUTIONAL_K7,
        interleaver_type=InterleaverType.BLOCK,
        sample_rate=1e6,
    )

    assert config.sample_rate == 1e6
    assert config.modulation == ModulationType.BPSK

    result = DecoderResult(
        status=DecoderStatus.SUCCESS,
        modulation=ModulationType.BPSK,
        fec_type=FECType.CONVOLUTIONAL_K7,
        interleaver_type=InterleaverType.BLOCK,
        bit_count=100,
        decoder_success=True,
    )

    assert result.status == DecoderStatus.SUCCESS
    assert result.bit_count == 100
    assert result.decoder_success is True

    pipeline = DecoderPipeline()
    assert pipeline is not None


def test_documented_files_exist():
    """Verify that all required Phase 0 documentation and contract files exist."""
    workspace_root = Path(__file__).resolve().parents[2]

    required_files = [
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-lock.txt",
        "README.md",
        ".gitignore",
        "docs/decoder_interface.md",
        "docs/codec_catalog.md",
        "docs/golden_set.md",
        "docs/bitintel_output.md",
        "scripts/verify_commpy.py",
        "python/spectralq/__init__.py",
        "python/spectralq/interleave.py",
        "python/spectralq/fec.py",
        "python/spectralq/demod.py",
        "python/spectralq/decoder_api.py",
        "python/spectralq/bitintel.py",
    ]

    for rel_path in required_files:
        file_path = workspace_root / rel_path
        assert file_path.exists(), f"Expected file does not exist: {rel_path}"


def test_handoff_directory_not_ignored():
    """Verify that data/handoff is not ignored and is present in workspace."""
    workspace_root = Path(__file__).resolve().parents[2]
    handoff_dir = workspace_root / "data" / "handoff"
    assert handoff_dir.exists(), "data/handoff directory must exist"

    gitignore_path = workspace_root / ".gitignore"
    assert gitignore_path.exists(), ".gitignore must exist"

    # Check that no active (non-comment) rule ignores data/handoff
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            assert "data/handoff" not in line, f"data/handoff must NOT be an active rule in .gitignore: {line}"

