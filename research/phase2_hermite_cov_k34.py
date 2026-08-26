"""Hermite-exact Gaussian covariance plus leading connected K3/K4 diagrams.

Price's theorem gives the exact off-diagonal covariance series for a Gaussian
preactivation pair:

  Cov[ReLU(z_i),ReLU(z_j)] = sum_{k>=1} C_ij^k g_k(i)g_k(j)/k!,

where g_k=E[ReLU^(k)(z)].  The starter covariance closure keeps only k=1.
This file evaluates the series through k=2,4,or 6, replaces the diagonal by
the exact marginal variance, and optionally stacks the goal-directed connected
K3/K4 mean correction.  All numerical work uses flopscope primitives.
"""
from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, MLP

_EPS = fnp.float32(1e-12)
_ZERO = fnp.float32(0.0)
_TWO = fnp.float32(2.0)
_THREE = fnp.float32(3.0)
_FOUR = fnp.float32(4.0)
_SIX = fnp.float32(6.0)
_TWELVE = fnp.float32(12.0)


def _normal(alpha):
    return (
        flops.stats.norm.pdf(alpha).astype(fnp.float32),
        flops.stats.norm.cdf(alpha).astype(fnp.float32),
    )


def _derivatives(alpha, sigma, pdf):
    inv = fnp.float32(1.0) / sigma
    inv2 = inv * inv
    a2 = alpha * alpha
    g2 = pdf * inv
    g3 = -alpha * pdf * inv2
    g4 = (a2 - fnp.float32(1.0)) * pdf * inv2 * inv
    g5 = -(alpha * a2 - _THREE * alpha) * pdf * inv2 * inv2
    g6 = (a2 * a2 - _SIX * a2 + _THREE) * pdf * inv2 * inv2 * inv
    return g2, g3, g4, g5, g6


def _relu_moments(mu, var, sigma, pdf, cdf):
    mu2 = mu * mu
    mu3 = mu2 * mu
    mu4 = mu2 * mu2
    var2 = var * var
    m1 = mu * cdf + sigma * pdf
    m2 = (mu2 + var) * cdf + mu * sigma * pdf
    m3 = (mu3 + _THREE * mu * var) * cdf + (
        mu2 * sigma + _TWO * sigma * var
    ) * pdf
    m4 = (mu4 + _SIX * mu2 * var + _THREE * var2) * cdf + (
        mu3 * sigma + fnp.float32(5.0) * mu * sigma * var
    ) * pdf
    variance = fnp.maximum(m2 - m1 * m1, _ZERO)
    k3 = m3 - _THREE * m1 * m2 + _TWO * m1 * m1 * m1
    central4 = m4 - _FOUR * m1 * m3 + _SIX * m1 * m1 * m2 - _THREE * m1**4
    k4 = central4 - _THREE * variance * variance
    return m1, variance, k3, k4


def _covariance_series(cov_pre, variance, g1, gs, order):
    cov = fnp.multiply(fnp.outer(g1, g1), cov_pre)
    power = cov_pre * cov_pre
    if order >= 2:
        cov = cov + fnp.float32(1.0 / 2.0) * fnp.outer(gs[0], gs[0]) * power
    if order >= 3:
        power = power * cov_pre
        cov = cov + fnp.float32(1.0 / 6.0) * fnp.outer(gs[1], gs[1]) * power
    if order >= 4:
        power = power * cov_pre
        cov = cov + fnp.float32(1.0 / 24.0) * fnp.outer(gs[2], gs[2]) * power
    if order >= 5:
        power = power * cov_pre
        cov = cov + fnp.float32(1.0 / 120.0) * fnp.outer(gs[3], gs[3]) * power
    if order >= 6:
        power = power * cov_pre
        cov = cov + fnp.float32(1.0 / 720.0) * fnp.outer(gs[4], gs[4]) * power
    fnp.fill_diagonal(cov, variance)
    return flops.as_symmetric(cov, symmetry=(0, 1))


def _contract(previous, weight):
    cov, g1, g2, g3, residual3, residual4 = previous
    projected = fnp.einsum("ij,iq->jq", cov, g1[:, None] * weight)

    k3 = _THREE * fnp.sum(g2[:, None] * weight * projected * projected, axis=0)
    w2 = weight * weight
    k3 = k3 + fnp.einsum("iq,i->q", w2 * weight, residual3)

    star = _FOUR * fnp.sum(
        g3[:, None] * weight * projected * projected * projected,
        axis=0,
    )
    path_factor = g2[:, None] * weight * projected
    path = _TWELVE * fnp.sum(
        path_factor * fnp.einsum("ij,iq->jq", cov, path_factor),
        axis=0,
    )
    k4 = star + path + fnp.einsum("iq,i->q", w2 * w2, residual4)
    return k3, k4


def _local_mean(alpha, var, sigma, pdf, k3, k4):
    a2 = alpha * alpha
    f3 = -alpha * pdf / var
    f4 = (a2 - fnp.float32(1.0)) * pdf / (var * sigma)
    f6 = (a2 * a2 - _SIX * a2 + _THREE) * pdf / (var * var * sigma)
    return (
        fnp.float32(1.0 / 6.0) * k3 * f3
        + fnp.float32(1.0 / 24.0) * k4 * f4
        + fnp.float32(1.0 / 72.0) * k3 * k3 * f6
    )


def _predict(mlp: MLP, *, cov_order: int, cross: bool):
    width = mlp.width
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
    previous = None
    rows = []

    for source_weight in mlp.weights:
        weight = fnp.asarray(source_weight, dtype=fnp.float32)
        mu_pre = fnp.einsum("ij,i->j", weight, mu)
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, weight, weight)
        cov_pre = flops.as_symmetric(cov_pre, symmetry=(0, 1))
        var_pre = fnp.maximum(fnp.diag(cov_pre), _EPS)
        sigma = fnp.sqrt(var_pre)
        alpha = mu_pre / sigma
        pdf, cdf = _normal(alpha)
        gs = _derivatives(alpha, sigma, pdf)
        mean, variance, marginal3, marginal4 = _relu_moments(
            mu_pre, var_pre, sigma, pdf, cdf
        )

        if cross and previous is not None:
            k3_pre, k4_pre = _contract(previous, weight)
            mean = mean + _local_mean(alpha, var_pre, sigma, pdf, k3_pre, k4_pre)
        rows.append(mean)

        var2 = var_pre * var_pre
        var3 = var2 * var_pre
        approx3 = _THREE * cdf * cdf * gs[0] * var2
        approx4 = (
            _FOUR * cdf * cdf * cdf * gs[1] * var3
            + _TWELVE * cdf * cdf * gs[0] * gs[0] * var3
        )
        previous = (
            cov_pre,
            cdf,
            gs[0],
            gs[1],
            marginal3 - approx3,
            marginal4 - approx4,
        )

        cov = _covariance_series(cov_pre, variance, cdf, gs, cov_order)
        mu = mean

    return fnp.stack(rows, axis=0)


class EstimatorCov2(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, cov_order=2, cross=False)


class EstimatorCov4(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, cov_order=4, cross=False)


class EstimatorCov6(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, cov_order=6, cross=False)


class EstimatorCov2CrossK34(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, cov_order=2, cross=True)


class EstimatorCov4CrossK34(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, cov_order=4, cross=True)


class EstimatorCov6CrossK34(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, cov_order=6, cross=True)
