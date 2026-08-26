"""Goal-directed leading K=3/K=4 connected-Hermite correction.

The estimator contracts the leading connected Gaussian diagrams directly
against every next-layer weight column, avoiding explicit n^3/n^4 tensors.
For K=3 it keeps the (1,1,2) tree. For K=4 it keeps all minimum-edge trees:
(1,1,1,3) stars and (1,1,2,2) paths. Exact marginal cumulants replace each
approximation's all-equal diagonal. The final Edgeworth mean includes kappa3,
kappa4, and the kappa3^2 Bell term. All numerics use flopscope primitives.
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


def _contract(previous, weight):
    cov, g1, g2, g3, residual3, residual4 = previous
    projected = fnp.einsum("ij,iq->jq", cov, g1[:, None] * weight)

    k3 = _THREE * fnp.sum(
        g2[:, None] * weight * projected * projected,
        axis=0,
    )
    w2 = weight * weight
    k3 = k3 + fnp.einsum("iq,i->q", w2 * weight, residual3)

    star = _FOUR * fnp.sum(
        g3[:, None] * weight * projected * projected * projected,
        axis=0,
    )
    path_factor = g2[:, None] * weight * projected
    path_projected = fnp.einsum("ij,iq->jq", cov, path_factor)
    path = _TWELVE * fnp.sum(path_factor * path_projected, axis=0)
    k4 = star + path + fnp.einsum("iq,i->q", w2 * w2, residual4)
    return k3, k4


def _local_mean(alpha, var, sigma, pdf, k3, k4):
    a2 = alpha * alpha
    f3 = -alpha * pdf / var
    f4 = (a2 - fnp.float32(1.0)) * pdf / (var * sigma)
    h4 = a2 * a2 - _SIX * a2 + _THREE
    f6 = h4 * pdf / (var * var * sigma)
    return (
        fnp.float32(1.0 / 6.0) * k3 * f3
        + fnp.float32(1.0 / 24.0) * k4 * f4
        + fnp.float32(1.0 / 72.0) * k3 * k3 * f6
    )


def _predict(mlp: MLP, *, mode: str):
    width = mlp.width
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
    delta = fnp.zeros(width, dtype=fnp.float32)
    previous = None
    rows = []

    for layer, source_weight in enumerate(mlp.weights):
        weight = fnp.asarray(source_weight, dtype=fnp.float32)
        mu_pre = fnp.einsum("ij,i->j", weight, mu)
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, weight, weight)
        cov_pre = flops.as_symmetric(cov_pre, symmetry=(0, 1))
        var_pre = fnp.maximum(fnp.diag(cov_pre), _EPS)
        sigma = fnp.sqrt(var_pre)
        alpha = mu_pre / sigma
        pdf, cdf = _normal(alpha)
        mean_base, var_post, marginal3, marginal4 = _relu_moments(
            mu_pre, var_pre, sigma, pdf, cdf
        )

        is_last = layer == mlp.depth - 1
        need_local = previous is not None and (
            mode in ("source_sum", "recursive") or (mode == "final" and is_last)
        )
        local = fnp.zeros(width, dtype=fnp.float32)
        if need_local:
            k3_pre, k4_pre = _contract(previous, weight)
            local = _local_mean(alpha, var_pre, sigma, pdf, k3_pre, k4_pre)

        if mode == "final":
            delta_next = local if is_last else fnp.zeros(width, dtype=fnp.float32)
            mean_next = mean_base
            mean_out = mean_base + delta_next
        elif mode == "source_sum":
            propagated = cdf * fnp.einsum("ij,i->j", weight, delta)
            delta_next = propagated + local
            mean_next = mean_base
            mean_out = mean_base + delta_next
        elif mode == "recursive":
            delta_next = fnp.zeros(width, dtype=fnp.float32)
            mean_next = mean_base + local
            mean_out = mean_next
        else:
            delta_next = fnp.zeros(width, dtype=fnp.float32)
            mean_next = mean_base
            mean_out = mean_base
        rows.append(mean_out)

        g2 = pdf / sigma
        g3 = -alpha * pdf / var_pre
        var2 = var_pre * var_pre
        var3 = var2 * var_pre
        approx3_diag = _THREE * cdf * cdf * g2 * var2
        approx4_diag = (
            _FOUR * cdf * cdf * cdf * g3 * var3
            + _TWELVE * cdf * cdf * g2 * g2 * var3
        )
        previous = (
            cov_pre,
            cdf,
            g2,
            g3,
            marginal3 - approx3_diag,
            marginal4 - approx4_diag,
        )

        cov = fnp.multiply(fnp.outer(cdf, cdf), cov_pre)
        fnp.fill_diagonal(cov, var_post)
        cov = flops.as_symmetric(cov, symmetry=(0, 1))
        mu = mean_next
        delta = delta_next

    return fnp.stack(rows, axis=0)


class EstimatorK2(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, mode="k2")


class EstimatorCrossK34Final(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, mode="final")


class EstimatorCrossK34SourceSum(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, mode="source_sum")


class EstimatorCrossK34Recursive(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, mode="recursive")
