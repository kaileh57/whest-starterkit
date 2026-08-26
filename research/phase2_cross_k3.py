"""Goal-directed leading off-diagonal K=3 correction for Phase 2.

For z jointly Gaussian and x_i=ReLU(z_i), the dominant distinct-index third
cumulant is the connected Hermite (1,1,2) sector

  kappa_ijk ~= g1_i g1_j g2_k C_ik C_jk + two permutations,

where g1=E[ReLU'(z)]=Phi(alpha), g2=E[ReLU''(z)]=phi(alpha)/sigma.
Contracting this tensor with every next-layer weight column does not require
materializing an n^3 tensor:

  <kappa, w_q^3> = 3 sum_k g2_k w_kq [C(g1*w_q)]_k^2
                    + sum_k residual_diag_k w_kq^3.

The residual term replaces the approximation's diagonal with the exact
marginal third cumulant.  This is the full leading O(correlation^2) connected
sector, evaluated for all outputs with one n-by-n product per layer.  All
numerics use flopscope primitives.
"""
from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, MLP

_EPS = fnp.float32(1e-12)
_ZERO = fnp.float32(0.0)
_ONE_SIXTH = fnp.float32(1.0 / 6.0)
_THREE = fnp.float32(3.0)
_TWO = fnp.float32(2.0)


def _normal(alpha):
    pdf = flops.stats.norm.pdf(alpha).astype(fnp.float32)
    cdf = flops.stats.norm.cdf(alpha).astype(fnp.float32)
    return pdf, cdf


def _relu_moments(mu, var, sigma, pdf, cdf):
    mu2 = mu * mu
    m1 = mu * cdf + sigma * pdf
    m2 = (mu2 + var) * cdf + mu * sigma * pdf
    m3 = (mu2 * mu + _THREE * mu * var) * cdf + (
        mu2 * sigma + _TWO * sigma * var
    ) * pdf
    variance = fnp.maximum(m2 - m1 * m1, _ZERO)
    k3 = m3 - _THREE * m1 * m2 + _TWO * m1 * m1 * m1
    return m1, variance, k3


def _contract_cross_k3(cov_pre, g1, g2, diag_residual, weight):
    weighted = g1[:, None] * weight
    projected = fnp.einsum("ij,iq->jq", cov_pre, weighted)
    cross = _THREE * fnp.sum(
        g2[:, None] * weight * projected * projected,
        axis=0,
    )
    w2 = weight * weight
    diagonal = fnp.einsum("iq,i->q", w2 * weight, diag_residual)
    return cross + diagonal


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
        mean_post, var_post, marginal_k3 = _relu_moments(
            mu_pre, var_pre, sigma, pdf, cdf
        )

        propagated = cdf * fnp.einsum("ij,i->j", weight, delta)
        local = fnp.zeros(width, dtype=fnp.float32)
        if previous is not None:
            prev_cov, prev_g1, prev_g2, prev_residual = previous
            k3_pre = _contract_cross_k3(
                prev_cov, prev_g1, prev_g2, prev_residual, weight
            )
            f3 = -alpha * pdf / var_pre
            local = _ONE_SIXTH * k3_pre * f3

        is_last = layer == mlp.depth - 1
        if mode == "final":
            delta_next = local if is_last else fnp.zeros(width, dtype=fnp.float32)
        elif mode == "source_sum":
            delta_next = propagated + local
        elif mode == "recursive":
            delta_next = propagated + local
            mean_post = mean_post + delta_next
        else:
            delta_next = fnp.zeros(width, dtype=fnp.float32)

        rows.append(mean_post + delta_next if mode != "recursive" else mean_post)

        g2 = pdf / sigma
        approx_diag = _THREE * cdf * cdf * g2 * var_pre * var_pre
        diag_residual = marginal_k3 - approx_diag
        previous = (cov_pre, cdf, g2, diag_residual)

        cov = fnp.multiply(fnp.outer(cdf, cdf), cov_pre)
        fnp.fill_diagonal(cov, var_post)
        cov = flops.as_symmetric(cov, symmetry=(0, 1))
        mu = mean_post
        delta = delta_next

    return fnp.stack(rows, axis=0)


class EstimatorK2(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, mode="k2")


class EstimatorCrossK3Final(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, mode="final")


class EstimatorCrossK3SourceSum(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, mode="source_sum")


class EstimatorCrossK3Recursive(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, mode="recursive")
