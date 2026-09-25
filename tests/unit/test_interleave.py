"""
tests/unit/test_interleave.py

Unit tests for the SpectralQ deterministic interleavers:
1. Block Interleaver
2. Convolutional / Cross Interleaver (Forney)
3. Diagonal Interleaver
4. Pseudo-Random Interleaver
5. Identity Interleaver (NONE)

Tests non-trivial patterns, multiple lengths, edge cases, and invalid configurations.
"""

import pytest
import numpy as np
from spectralq.interleave import (
    block_interleave,
    block_deinterleave,
    diagonal_interleave,
    diagonal_deinterleave,
    conv_interleave,
    conv_deinterleave,
    pseudorandom_interleave,
    pseudorandom_deinterleave,
    identity_interleave,
    identity_deinterleave,
    none_interleave,
    none_deinterleave,
)

# Mandatory test lengths specified in requirements
TEST_LENGTHS = [1, 2, 3, 7, 16, 31, 64, 100, 127, 256]


def generate_patterns(length: int):
    """Generate multiple non-trivial test bit patterns for a given length."""
    patterns = []
    # 1. Alternating bits: [0, 1, 0, 1, ...]
    alt = np.array([i % 2 for i in range(length)], dtype=int)
    patterns.append(("alternating", alt))

    # 2. All zeros
    zeros = np.zeros(length, dtype=int)
    patterns.append(("zeros", zeros))

    # 3. All ones
    ones = np.ones(length, dtype=int)
    patterns.append(("ones", ones))

    # 4. Deterministic random bits (fixed seed per length)
    rng = np.random.default_rng(length * 101)
    rand_bits = rng.integers(0, 2, size=length)
    patterns.append(("random", rand_bits))

    return patterns


# ============================================================================
# 1. BLOCK INTERLEAVER TESTS
# ============================================================================

@pytest.mark.parametrize("length", TEST_LENGTHS)
def test_block_interleave_lengths_and_patterns(length):
    """Verify block interleaver round-trip across all test lengths and bit patterns."""
    rows, cols = 8, 8
    for name, pattern in generate_patterns(length):
        interleaved, meta = block_interleave(pattern, rows, cols)
        assert meta["original_length"] == length
        assert meta["padded_length"] >= length
        assert meta["padding_length"] == meta["padded_length"] - length
        assert len(interleaved) == meta["padded_length"]

        recovered = block_deinterleave(interleaved, rows, cols, meta)
        assert np.array_equal(pattern, recovered), f"Failed pattern '{name}' at length {length}"


def test_block_interleave_non_square():
    """Verify block interleaver with non-square matrix configurations."""
    for rows, cols in [(4, 16), (16, 4), (5, 9), (7, 13)]:
        data = np.random.default_rng(42).integers(0, 2, size=77)
        interleaved, meta = block_interleave(data, rows, cols)
        recovered = block_deinterleave(interleaved, rows, cols, meta)
        assert np.array_equal(data, recovered)


def test_block_interleave_invalid_params():
    """Verify block interleaver rejects invalid rows and cols."""
    data = np.array([1, 0, 1])
    with pytest.raises(ValueError):
        block_interleave(data, 0, 8)
    with pytest.raises(ValueError):
        block_interleave(data, 8, -1)
    with pytest.raises(ValueError):
        block_deinterleave(np.array([1, 0]), 3, 3)  # length not multiple of capacity


# ============================================================================
# 2. CONVOLUTIONAL / CROSS INTERLEAVER TESTS
# ============================================================================

@pytest.mark.parametrize("length", TEST_LENGTHS)
def test_conv_interleave_lengths_and_patterns(length):
    """Verify convolutional interleaver round-trip across all test lengths and patterns."""
    branches = 4
    delay_step = 2
    for name, pattern in generate_patterns(length):
        interleaved, meta = conv_interleave(pattern, branches, delay_step)
        assert meta["original_length"] == length
        assert meta["num_branches"] == branches
        assert meta["delay_step"] == delay_step

        recovered = conv_deinterleave(interleaved, branches, delay_step, meta)
        assert np.array_equal(pattern, recovered), f"Conv interleaver failed on '{name}' at length {length}"


def test_conv_interleave_branch_variations():
    """Verify convolutional interleaver across diverse branch and delay configurations."""
    test_configs = [
        (1, 0),   # 1 branch, 0 delay (trivial identity)
        (2, 1),
        (4, 2),
        (6, 2),
        (8, 3),
        (12, 1),
    ]
    data = np.random.default_rng(1234).integers(0, 2, size=150)
    for b, m in test_configs:
        interleaved, meta = conv_interleave(data, b, m)
        recovered = conv_deinterleave(interleaved, b, m, meta)
        assert np.array_equal(data, recovered), f"Failed for branches={b}, delay_step={m}"


def test_conv_interleave_invalid_params():
    """Verify convolutional interleaver raises on invalid arguments."""
    data = np.array([1, 0, 1])
    with pytest.raises(ValueError):
        conv_interleave(data, 0, 2)
    with pytest.raises(ValueError):
        conv_interleave(data, 4, -1)


# ============================================================================
# 3. DIAGONAL INTERLEAVER TESTS
# ============================================================================

@pytest.mark.parametrize("length", TEST_LENGTHS)
def test_diagonal_interleave_lengths_and_patterns(length):
    """Verify diagonal interleaver round-trip across all test lengths and patterns."""
    rows, cols = 8, 8
    for name, pattern in generate_patterns(length):
        interleaved, meta = diagonal_interleave(pattern, rows, cols)
        assert meta["original_length"] == length
        assert meta["padded_length"] >= length
        assert meta["padding_length"] == meta["padded_length"] - length
        assert len(interleaved) == meta["padded_length"]

        recovered = diagonal_deinterleave(interleaved, rows, cols, meta)
        assert np.array_equal(pattern, recovered), f"Diagonal failed on '{name}' at length {length}"


def test_diagonal_interleave_non_square_grids():
    """Verify diagonal interleaver with non-square and prime dimensions."""
    for rows, cols in [(3, 7), (11, 5), (40, 51), (17, 19)]:
        data = np.random.default_rng(999).integers(0, 2, size=231)
        interleaved, meta = diagonal_interleave(data, rows, cols)
        recovered = diagonal_deinterleave(interleaved, rows, cols, meta)
        assert np.array_equal(data, recovered)


def test_diagonal_interleave_invalid_params():
    """Verify diagonal interleaver rejects invalid row/col dimensions."""
    data = np.array([1, 0, 1])
    with pytest.raises(ValueError):
        diagonal_interleave(data, -4, 4)
    with pytest.raises(ValueError):
        diagonal_deinterleave(np.array([1, 0]), 4, 4)


# ============================================================================
# 4. PSEUDO-RANDOM INTERLEAVER TESTS
# ============================================================================

@pytest.mark.parametrize("seed", [0, 1, 42, 12345])
@pytest.mark.parametrize("length", [7, 16, 64, 100, 256])
def test_pseudorandom_interleave_seeds_and_lengths(seed, length):
    """Verify pseudo-random interleaver round-trip across mandatory seeds and lengths."""
    for name, pattern in generate_patterns(length):
        interleaved, meta = pseudorandom_interleave(pattern, seed)
        assert meta["original_length"] == length
        assert meta["seed"] == seed
        assert len(interleaved) == length

        recovered = pseudorandom_deinterleave(interleaved, seed, meta)
        assert np.array_equal(pattern, recovered), f"PRNG failed for seed {seed} on '{name}'"


def test_pseudorandom_determinism():
    """Verify pseudo-random interleaver produces identical permutations on identical seeds."""
    data = np.random.default_rng(777).integers(0, 2, size=128)
    int1, meta1 = pseudorandom_interleave(data, seed=42)
    int2, meta2 = pseudorandom_interleave(data, seed=42)
    assert np.array_equal(int1, int2), "PRNG interleaver must be strictly deterministic"

    # Distinct seeds should yield different permutations
    int3, meta3 = pseudorandom_interleave(data, seed=43)
    assert not np.array_equal(int1, int3), "Different seeds should produce different interleavings"


# ============================================================================
# 5. IDENTITY INTERLEAVER (NONE) TESTS
# ============================================================================

@pytest.mark.parametrize("length", TEST_LENGTHS)
def test_identity_interleave_lengths_and_patterns(length):
    """Verify identity interleaver round-trip across all test lengths and patterns."""
    for name, pattern in generate_patterns(length):
        interleaved, meta = identity_interleave(pattern)
        assert meta["interleaver_type"] == "NONE"
        assert meta["original_length"] == length
        assert meta["padded_length"] == length
        assert meta["padding_length"] == 0
        assert len(interleaved) == length
        assert np.array_equal(interleaved, pattern), f"Identity changed bits on '{name}' at length {length}"

        recovered = identity_deinterleave(interleaved, meta)
        assert np.array_equal(pattern, recovered), f"Identity de-interleave failed on '{name}' at length {length}"


def test_identity_interleave_exactness_and_contract():
    """Verify strict mathematical identity properties required by Step 3."""
    rng = np.random.default_rng(9999)
    test_bits = rng.integers(0, 2, size=524)

    # 1. interleave(input) == input exactly
    interleaved, meta = identity_interleave(test_bits)
    assert np.array_equal(interleaved, test_bits)

    # 2. input == deinterleave(interleave(input)) exactly
    recovered = identity_deinterleave(interleaved, meta)
    assert np.array_equal(recovered, test_bits)

    # 3. Aliases none_interleave and none_deinterleave
    interleaved_alias, meta_alias = none_interleave(test_bits)
    assert np.array_equal(interleaved_alias, test_bits)
    recovered_alias = none_deinterleave(interleaved_alias, meta_alias)
    assert np.array_equal(recovered_alias, test_bits)

    # 4. No padding, no randomization
    assert meta["padding_length"] == 0
    assert meta["original_length"] == 524
    assert meta["padded_length"] == 524
