"""Cheap diagonal cumulant corrections on top of full-covariance K=2.

These estimators keep the exact K=2 covariance geometry while propagating the
marginal third/fourth cumulants with power kernels.  The final ReLU mean then
uses the first Edgeworth/Bell terms.  Every numerical operation is a flopscope
primitive; these are valid Phase-2 estimator candidates, not research-only
wrappers.
"""
from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, MLP

_EPS = fnp.float32(1e-12)
_ZERO = fnp.float32(0.0)


def _normal(alpha):
    pdf = flops.stats.norm.pdf(alpha).astype(fnp.float32)
    cdf = flops.stats.norm.cdf(alpha).astype(fnp.float32)
    return pdf, cdf


def _gaussian_relu_raw(mu, var, sigma, pdf, cdf):
    mu2 = mu * mu
    mu3 = mu2 * mu
    mu4 = mu2 * mu2
    var2 = var * var
    m1 = mu * cdf + sigma * pdf
    m2 = (mu2 + var) * cdf + mu * sigma * pdf
    m3 = (mu3 + fnp.float32(3.0) * mu * var) * cdf + (
        mu2 * sigma + fnp.float32(2.0) * sigma * var
    ) * pdf
    m4 = (
        mu4 + fnp.float32(6.0) * mu2 * var + fnp.float32(3.0) * var2
    ) * cdf + (
        mu3 * sigma + fnp.float32(5.0) * mu * sigma * var
    ) * pdf
    return m1, m2, m3, m4


def _post_cumulants(m1, m2, m3, m4):
    m1_2 = m1 * m1
    variance = fnp.maximum(m2 - m1_2, _ZERO)
    k3 = m3 - fnp.float32(3.0) * m1 * m2 + fnp.float32(2.0) * m1_2 * m1
    central4 = (
        m4
        - fnp.float32(4.0) * m1 * m3
        + fnp.float32(6.0) * m1_2 * m2
        - fnp.float32(3.0) * m1_2 * m1_2
    )
    k4 = central4 - fnp.float32(3.0) * variance * variance
    return variance, k3, k4


def _edgeworth_mean(mu_gauss, alpha, var, sigma, pdf, k3, k4, order):
    inv_var = fnp.float32(1.0) / var
    f3 = -alpha * pdf * inv_var
    out = mu_gauss + fnp.float32(1.0 / 6.0) * k3 * f3
    if order >= 4:
        f4 = (alpha * alpha - fnp.float32(1.0)) * pdf * inv_var / sigma
        out = out + fnp.float32(1.0 / 24.0) * k4 * f4
    if order >= 6:
        a2 = alpha * alpha
        h4 = a2 * a2 - fnp.float32(6.0) * a2 + fnp.float32(3.0)
        f6 = h4 * pdf * inv_var * inv_var / sigma
        out = out + fnp.float32(1.0 / 72.0) * k3 * k3 * f6
    return out


def _predict(mlp: MLP, *, order: int, recursive: bool):
    width = mlp.width
    mu = fnp.zeros(width, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
    k3 = fnp.zeros(width, dtype=fnp.float32)
    k4 = fnp.zeros(width, dtype=fnp.float32)
    rows = []

    for layer, weight in enumerate(mlp.weights):
        w = fnp.asarray(weight, dtype=fnp.float32)
        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), _EPS)
        sigma = fnp.sqrt(var_pre)

        w2 = w * w
        k3_pre = fnp.einsum("ij,i->j", w2 * w, k3)
        k4_pre = fnp.einsum("ij,i->j", w2 * w2, k4)

        alpha = mu_pre / sigma
        pdf, cdf = _normal(alpha)
        m1, m2, m3, m4 = _gaussian_relu_raw(mu_pre, var_pre, sigma, pdf, cdf)
        var_post, k3_post, k4_post = _post_cumulants(m1, m2, m3, m4)

        corrected = _edgeworth_mean(
            m1, alpha, var_pre, sigma, pdf, k3_pre, k4_pre, order
        )
        is_last = layer == mlp.depth - 1
        mu_next = corrected if recursive else m1
        rows.append(corrected if is_last else mu_next)

        cov = fnp.multiply(fnp.outer(cdf, cdf), cov_pre)
        fnp.fill_diagonal(cov, var_post)
        cov = flops.as_symmetric(cov, symmetry=(0, 1))
        mu = mu_next
        k3 = k3_post
        k4 = k4_post

    return fnp.stack(rows, axis=0)


class EstimatorK2(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, order=0, recursive=False)


class EstimatorDiagK3Final(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, order=3, recursive=False)


class EstimatorDiagK34Final(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, order=4, recursive=False)


class EstimatorDiagK346Final(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, order=6, recursive=False)


class EstimatorDiagK3Recursive(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, order=3, recursive=True)


class EstimatorDiagK346Recursive(BaseEstimator):
    def predict(self, mlp: MLP, budget: int):
        _ = budget
        return _predict(mlp, order=6, recursive=True)
