"""Research-only Phase 2 stability frontier for the public factored K=3 recurrence.

The K3 classes import a pinned copy of Paul Rosu's public Phase 1 reference and
exercise only its generic SIMPLE recurrence with explicit factor-rank caps.
That reference still contains direct NumPy operations inside predict(), so this
file is an accuracy/stability probe, not a prize-eligible submission.
"""
from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, MLP

import paul_k3_estimator as reference


class EstimatorK2(BaseEstimator):
    """Current-stack full-covariance Gaussian closure baseline."""

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        width = mlp.width
        mu = fnp.zeros(width, dtype=fnp.float32)
        cov = flops.as_symmetric(
            fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1)
        )
        rows = []
        for weight in mlp.weights:
            w = fnp.asarray(weight, dtype=fnp.float32)
            mu_pre = w.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
            var_pre = fnp.maximum(fnp.diag(cov_pre), fnp.float32(1e-12))
            sigma = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma
            pdf = flops.stats.norm.pdf(alpha)
            cdf = flops.stats.norm.cdf(alpha)
            mu = mu_pre * cdf + sigma * pdf
            second = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma * pdf
            var_post = fnp.maximum(second - mu * mu, fnp.float32(0.0))
            cov = fnp.multiply(fnp.outer(cdf, cdf), cov_pre)
            fnp.fill_diagonal(cov, var_post)
            cov = flops.as_symmetric(cov, symmetry=(0, 1))
            rows.append(mu)
        return fnp.stack(rows, axis=0)


class _RankCappedK3(BaseEstimator):
    J_MAX = 64

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        return reference.predict_k3_factored_simple_drops(
            mlp.weights,
            drops=True,
            lite_last=True,
            J_max=self.J_MAX,
            eps_factor=0.0,
        )


class EstimatorJ16(_RankCappedK3):
    J_MAX = 16


class EstimatorJ32(_RankCappedK3):
    J_MAX = 32


class EstimatorJ64(_RankCappedK3):
    J_MAX = 64


class EstimatorJ128(_RankCappedK3):
    J_MAX = 128


class EstimatorJ256(_RankCappedK3):
    J_MAX = 256
