"""Kerdock mutually-unbiased-basis cubature in R^256.

The binary Kerdock code K(7) is generated from the quadratic forms

    c_{a,w,k}(x) = x P_a x^T + 2 <w,x> + k  (mod 4),
    (P_a)_{ij} = Tr(a alpha^i alpha^j),

then mapped from Z4 to two binary coordinates by the Gray map. Mapping the
binary words to signs gives 128 mutually unbiased orthonormal bases of R^256.
Adding the standard basis gives the maximal 129 real MUBs.

The full antipodal union has 66,048 nodes and is a spherical 5-design. Any
subset of complete bases remains antipodal and has exact spherical moments
through degree three (in particular, exact covariance). This makes the basis
count a clean accuracy/compute knob for WhestBench.

Only plain NumPy is used here. Estimators call this from setup(), before
WhestBench starts FLOP accounting.
"""

from __future__ import annotations

import math
from typing import Final, Iterable

import numpy as np

_M: Final[int] = 7
_FIELD_SIZE: Final[int] = 1 << _M
_DIM: Final[int] = 1 << (_M + 1)  # 256 after the Gray map
_NUM_BASES: Final[int] = _FIELD_SIZE + 1
# x^7 + x + 1. It is irreducible; because 2^7-1=127 is prime, its root is
# automatically primitive.
_MODULUS: Final[int] = (1 << 7) | (1 << 1) | 1
_GRAY_SIGNS: Final[np.ndarray] = np.asarray(
    [[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=np.int8
)
_RADIUS_MEAN: Final[float] = math.exp(
    0.5 * math.log(2.0)
    + math.lgamma((_DIM + 1.0) / 2.0)
    - math.lgamma(_DIM / 2.0)
)


def _gf_mul(a: int, b: int) -> int:
    """Multiply in GF(2^7), polynomial basis, returning an integer in [0,127]."""
    result = 0
    x = int(a)
    y = int(b)
    while y:
        if y & 1:
            result ^= x
        y >>= 1
        x <<= 1
        if x & _FIELD_SIZE:
            x ^= _MODULUS
    return result & (_FIELD_SIZE - 1)


def _gf_trace(a: int) -> int:
    """Absolute trace GF(2^7)->GF(2)."""
    total = 0
    value = int(a)
    for _ in range(_M):
        total ^= value
        value = _gf_mul(value, value)
    if total not in (0, 1):
        raise ArithmeticError("field trace did not land in GF(2)")
    return total


def _quadratic_tables(bits: np.ndarray) -> np.ndarray:
    """Return Q[a,x] = x P_a x^T mod 4 for all a,x in GF(2^7)."""
    basis = [1 << i for i in range(_M)]
    tables = np.empty((_FIELD_SIZE, _FIELD_SIZE), dtype=np.uint8)
    for a in range(_FIELD_SIZE):
        p = np.empty((_M, _M), dtype=np.int16)
        for i in range(_M):
            for j in range(_M):
                p[i, j] = _gf_trace(_gf_mul(a, _gf_mul(basis[i], basis[j])))
        xp = bits @ p
        tables[a] = np.remainder(np.sum(xp * bits, axis=1), 4).astype(np.uint8)
    return tables


def build_unit_bases() -> np.ndarray:
    """Return all 129 real MUBs, shape (129,256,256), with unit-norm rows."""
    ids = np.arange(_FIELD_SIZE, dtype=np.uint16)
    bits = ((ids[:, None] >> np.arange(_M, dtype=np.uint16)) & 1).astype(np.int16)
    quadratic = _quadratic_tables(bits)
    parity = np.remainder(bits @ bits.T, 2).astype(np.uint8)  # x,w
    linear = (2 * parity.T).astype(np.uint8)  # w,x

    bases = np.empty((_NUM_BASES, _DIM, _DIM), dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(_DIM))
    for a in range(_FIELD_SIZE):
        code = np.remainder(quadratic[a][None, :] + linear, 4)
        k0 = _GRAY_SIGNS[code].reshape(_FIELD_SIZE, _DIM)
        k1 = _GRAY_SIGNS[np.remainder(code + 1, 4)].reshape(_FIELD_SIZE, _DIM)
        bases[a] = np.concatenate((k0, k1), axis=0).astype(np.float32) * scale

    # The 129th MUB is the standard basis.
    bases[-1] = np.eye(_DIM, dtype=np.float32)
    return np.ascontiguousarray(bases)


def _normalise_indices(indices: Iterable[int]) -> np.ndarray:
    selected = np.asarray(tuple(int(i) for i in indices), dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("basis_indices must be a non-empty one-dimensional sequence")
    if np.any(selected < 0) or np.any(selected >= _NUM_BASES):
        raise ValueError(f"basis indices must lie in [0,{_NUM_BASES})")
    if np.unique(selected).size != selected.size:
        raise ValueError("basis_indices must not contain duplicates")
    return selected


def build_half_design(
    seed: int | None = 0,
    shuffle: bool = True,
    num_bases: int | None = None,
    basis_indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Return one representative from every antipodal pair in selected MUBs.

    Each selected basis contributes 256 rows; the complete estimator evaluates
    their negatives algebraically, so the full node count is ``512*num_bases``.
    Rows are scaled by E[Chi_256]. Therefore averaging a positively homogeneous
    degree-one function over these rows and their negatives directly estimates
    its standard-Gaussian expectation; no sampled radii or radial weights remain.

    Subsets are nested as ``num_bases`` grows: one seeded permutation of all 129
    bases is formed, and the first ``num_bases`` entries are taken. This makes a
    compute sweep interpretable rather than confounding it with different nodes.
    """
    if num_bases is not None and basis_indices is not None:
        raise ValueError("specify either num_bases or basis_indices, not both")

    seed_value = 0 if seed is None else int(seed)
    if basis_indices is not None:
        selected = _normalise_indices(basis_indices)
    else:
        count = _NUM_BASES if num_bases is None else int(num_bases)
        if count < 1 or count > _NUM_BASES:
            raise ValueError(f"num_bases must lie in [1,{_NUM_BASES}]")
        rng_basis = np.random.default_rng(seed_value ^ 0x4D55424241534553)
        selected = rng_basis.permutation(_NUM_BASES)[:count]

    bases = build_unit_bases()
    half = bases[selected].reshape(-1, _DIM).copy()
    half *= np.float32(_RADIUS_MEAN)

    if shuffle:
        rng_rows = np.random.default_rng(seed_value ^ 0x4B4552444F434B)
        half = half[rng_rows.permutation(half.shape[0])]
    return np.ascontiguousarray(half, dtype=np.float32)


def _moment_errors(unit_half: np.ndarray) -> tuple[float, float]:
    n_full = 2 * unit_half.shape[0]
    gram2 = (2.0 / n_full) * (unit_half.T @ unit_half)
    target2 = np.eye(_DIM, dtype=np.float64) / _DIM
    second_error = float(np.max(np.abs(gram2 - target2)))

    rng = np.random.default_rng(20260810)
    fourth_error = 0.0
    target4 = 3.0 / (_DIM * (_DIM + 2.0))
    for _ in range(16):
        direction = rng.standard_normal(_DIM)
        direction /= np.linalg.norm(direction)
        projection = unit_half @ direction
        observed = float(2.0 * np.sum(projection**4) / n_full)
        fourth_error = max(fourth_error, abs(observed - target4))
    return second_error, fourth_error


def validate_design() -> dict[str, float | int]:
    """Construct and verify full-design and representative subset identities."""
    bases = build_unit_bases().astype(np.float64)
    b0 = bases[0]
    b1 = bases[1]
    within_error = float(np.max(np.abs(b0 @ b0.T - np.eye(_DIM))))
    cross_abs_error = float(np.max(np.abs(np.abs(b0 @ b1.T) - 1.0 / 16.0)))

    full = build_half_design(seed=0, shuffle=False).astype(np.float64) / _RADIUS_MEAN
    full_second, full_fourth = _moment_errors(full)

    subset = build_half_design(seed=0, shuffle=False, num_bases=16).astype(np.float64)
    subset /= _RADIUS_MEAN
    subset_second, subset_fourth = _moment_errors(subset)

    return {
        "dimension": _DIM,
        "num_bases": _NUM_BASES,
        "full_nodes": int(2 * full.shape[0]),
        "full_second_moment_max_error": full_second,
        "full_fourth_moment_max_error": full_fourth,
        "subset16_nodes": int(2 * subset.shape[0]),
        "subset16_second_moment_max_error": subset_second,
        "subset16_fourth_moment_max_error": subset_fourth,
        "basis_orthogonality_max_error": within_error,
        "mutual_unbiasedness_max_error": cross_abs_error,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(validate_design(), indent=2, sort_keys=True))
