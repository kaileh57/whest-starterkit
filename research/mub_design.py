"""Exact Kerdock/MUB spherical 5-design in R^256.

The binary Kerdock code K(7) is generated from the quadratic forms

    c_{a,w,k}(x) = x P_a x^T + 2 <w,x> + k  (mod 4),
    (P_a)_{ij} = Tr(a alpha^i alpha^j),

then mapped from Z4 to two binary coordinates by the Gray map.  Mapping the
binary words to signs gives 128 mutually unbiased orthonormal bases of R^256.
Adding the standard basis gives the maximal 129 real MUBs.  Including both
signs yields 66,048 unit vectors, an antipodal spherical 5-design.

Only plain NumPy is used here.  Estimators call this from setup(), before
WhestBench starts FLOP accounting.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

_M: Final[int] = 7
_FIELD_SIZE: Final[int] = 1 << _M
_DIM: Final[int] = 1 << (_M + 1)  # 256 after the Gray map
# x^7 + x + 1.  It is irreducible; because 2^7-1=127 is prime, its root is
# automatically primitive.
_MODULUS: Final[int] = (1 << 7) | (1 << 1) | 1
_GRAY_SIGNS: Final[np.ndarray] = np.asarray(
    [[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=np.int8
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


def build_half_design(seed: int = 0, shuffle: bool = True) -> np.ndarray:
    """Return one representative from each antipodal pair, shape (33024,256).

    Rows are scaled by E[Chi_256].  Therefore averaging a positively homogeneous
    degree-one function over these rows and their negatives directly estimates
    its standard-Gaussian expectation; no per-row radial weights are needed.
    """
    ids = np.arange(_FIELD_SIZE, dtype=np.uint16)
    bits = ((ids[:, None] >> np.arange(_M, dtype=np.uint16)) & 1).astype(np.int16)
    quadratic = _quadratic_tables(bits)
    parity = np.remainder(bits @ bits.T, 2).astype(np.uint8)  # x,w
    linear = (2 * parity.T).astype(np.uint8)  # w,x

    # For fixed a, (w,kappa) with kappa in {0,1} gives 256 orthogonal
    # representatives.  kappa+2 supplies exactly their antipodes.
    bases: list[np.ndarray] = []
    for a in range(_FIELD_SIZE):
        code = np.remainder(quadratic[a][None, :] + linear, 4)
        k0 = _GRAY_SIGNS[code].reshape(_FIELD_SIZE, _DIM)
        k1 = _GRAY_SIGNS[np.remainder(code + 1, 4)].reshape(_FIELD_SIZE, _DIM)
        bases.extend((k0, k1))
    dense = np.concatenate(bases, axis=0).astype(np.float32)
    dense *= np.float32(1.0 / math.sqrt(_DIM))

    # The 129th MUB is the standard basis.  Positive axes are representatives;
    # the estimator's antithetic mirror supplies the negative axes.
    axes = np.eye(_DIM, dtype=np.float32)
    half = np.concatenate((dense, axes), axis=0)
    if half.shape != (33024, 256):
        raise AssertionError(f"unexpected Kerdock design shape {half.shape}")

    radius_mean = math.exp(
        0.5 * math.log(2.0)
        + math.lgamma((_DIM + 1.0) / 2.0)
        - math.lgamma(_DIM / 2.0)
    )
    half *= np.float32(radius_mean)

    if shuffle:
        rng = np.random.default_rng(int(seed) ^ 0x4B4552444F434B)
        half = half[rng.permutation(half.shape[0])]
    return np.ascontiguousarray(half, dtype=np.float32)


def validate_design() -> dict[str, float | int]:
    """Construct and verify the exact second/fourth-moment identities."""
    half = build_half_design(seed=0, shuffle=False)
    radius_mean = math.exp(
        0.5 * math.log(2.0)
        + math.lgamma((_DIM + 1.0) / 2.0)
        - math.lgamma(_DIM / 2.0)
    )
    unit_half = half.astype(np.float64) / radius_mean
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

    # Check two complete bases and their mutual unbiasedness exactly enough to
    # detect any field/Gray-map construction error.
    dense = unit_half[:-_DIM]
    b0 = dense[:_DIM]
    b1 = dense[_DIM : 2 * _DIM]
    within_error = float(np.max(np.abs(b0 @ b0.T - np.eye(_DIM))))
    cross_abs_error = float(np.max(np.abs(np.abs(b0 @ b1.T) - 1.0 / 16.0)))

    return {
        "dimension": _DIM,
        "full_nodes": n_full,
        "half_nodes": int(unit_half.shape[0]),
        "second_moment_max_error": second_error,
        "fourth_moment_max_error": fourth_error,
        "basis_orthogonality_max_error": within_error,
        "mutual_unbiasedness_max_error": cross_abs_error,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(validate_design(), indent=2, sort_keys=True))
