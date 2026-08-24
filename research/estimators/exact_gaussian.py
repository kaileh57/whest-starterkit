"""Exact Gaussian moment closure for bias-free ReLU MLPs.

The linear step is exact. The nonlinear step replaces the current joint law
by a Gaussian with matching first two moments, but computes every bivariate
ReLU moment rather than applying the usual derivative-gain approximation.
The bivariate normal CDF is evaluated by fixed Gauss-Legendre quadrature of
Plackett's correlation integral, entirely through flopscope operations.
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_TWO_PI = 2.0 * math.pi
_RHO_LIMIT = 0.9995
_EPS = 1.0e-12

_GL = {
    4: (
        (-0.8611363115940526, -0.33998104358485626, 0.33998104358485626, 0.8611363115940526),
        (0.34785484513745357, 0.6521451548625464, 0.6521451548625464, 0.34785484513745357),
    ),
    6: (
        (-0.9324695142031519, -0.6612093864662645, -0.2386191860831969, 0.2386191860831969, 0.6612093864662645, 0.9324695142031519),
        (0.17132449237917027, 0.3607615730481387, 0.46791393457269104, 0.46791393457269104, 0.3607615730481387, 0.17132449237917027),
    ),
    8: (
        (-0.9602898564975362, -0.7966664774136267, -0.525532409916329, -0.18343464249564978, 0.18343464249564978, 0.525532409916329, 0.7966664774136267, 0.9602898564975362),
        (0.10122853629037706, 0.22238103445337443, 0.3137066458778869, 0.36268378337836166, 0.36268378337836166, 0.3137066458778869, 0.22238103445337443, 0.10122853629037706),
    ),
    12: (
        (-0.9815606342467192, -0.9041172563704748, -0.7699026741943047, -0.5873179542866175, -0.3678314989981802, -0.1252334085114689, 0.1252334085114689, 0.3678314989981802, 0.5873179542866175, 0.7699026741943047, 0.9041172563704748, 0.9815606342467192),
        (0.04717533638651141, 0.10693932599531907, 0.16007832854334642, 0.20316742672306573, 0.2334925365383546, 0.2491470458134027, 0.2491470458134027, 0.2334925365383546, 0.20316742672306573, 0.16007832854334642, 0.10693932599531907, 0.04717533638651141),
    ),
    16: (
        (-0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.755404408355003, -0.6178762444026438, -0.45801677765722737, -0.2816035507792589, -0.09501250983763744, 0.09501250983763744, 0.2816035507792589, 0.45801677765722737, 0.6178762444026438, 0.755404408355003, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499),
        (0.027152459411754176, 0.062253523938647456, 0.0951585116824926, 0.12462897125553407, 0.1495959888165767, 0.16915651939500265, 0.18260341504492364, 0.18945061045506864, 0.18945061045506864, 0.18260341504492364, 0.16915651939500265, 0.1495959888165767, 0.12462897125553407, 0.0951585116824926, 0.062253523938647456, 0.027152459411754176),
    ),
}


def _bvn_cdf(a_col, a_row, rho, Phi_col, Phi_row, order: int):
    """Phi_2(a_i, a_j; rho_ij) from Plackett's integral."""
    nodes, weights = _GL[order]
    integral = fnp.zeros_like(rho)
    aa = a_col * a_col
    bb = a_row * a_row
    ab2 = 2.0 * a_col * a_row
    for node, weight in zip(nodes, weights):
        t = 0.5 * rho * (node + 1.0)
        one_minus = fnp.maximum(1.0 - t * t, _EPS)
        exponent = -(aa - ab2 * t + bb) / (2.0 * one_minus)
        integral = integral + weight * fnp.exp(exponent) / fnp.sqrt(one_minus)
    return fnp.clip(Phi_col * Phi_row + 0.5 * rho * integral / _TWO_PI, 0.0, 1.0)


def _relu_gaussian_moments(mu, cov, order: int):
    var = fnp.maximum(fnp.diag(cov), _EPS)
    sigma = fnp.sqrt(var)
    a = mu / sigma
    phi = flops.stats.norm.pdf(a)
    Phi = flops.stats.norm.cdf(a)

    mean = mu * Phi + sigma * phi
    second = (mu * mu + var) * Phi + mu * sigma * phi
    variance = fnp.maximum(second - mean * mean, 0.0)

    sigma_col = sigma[:, None]
    sigma_row = sigma[None, :]
    a_col = a[:, None]
    a_row = a[None, :]
    mu_col = mu[:, None]
    mu_row = mu[None, :]
    phi_col = phi[:, None]
    phi_row = phi[None, :]
    Phi_col = Phi[:, None]
    Phi_row = Phi[None, :]

    rho = cov / (sigma_col * sigma_row)
    rho = fnp.clip(rho, -_RHO_LIMIT, _RHO_LIMIT)
    one_minus_rho2 = fnp.maximum(1.0 - rho * rho, _EPS)
    root = fnp.sqrt(one_minus_rho2)

    joint_cdf = _bvn_cdf(a_col, a_row, rho, Phi_col, Phi_row, order)
    cond_a = flops.stats.norm.cdf((a_row - rho * a_col) / root)
    cond_b = flops.stats.norm.cdf((a_col - rho * a_row) / root)
    exponent = -(a_col * a_col - 2.0 * rho * a_col * a_row + a_row * a_row) / (
        2.0 * one_minus_rho2
    )
    joint_pdf = fnp.exp(exponent) / (_TWO_PI * root)

    cross = (
        (mu_col * mu_row + cov) * joint_cdf
        + mu_col * sigma_row * phi_row * cond_b
        + mu_row * sigma_col * phi_col * cond_a
        + sigma_col * sigma_row * one_minus_rho2 * joint_pdf
    )
    post_cov = cross - fnp.outer(mean, mean)
    post_cov = 0.5 * (post_cov + post_cov.T)
    fnp.fill_diagonal(post_cov, variance)
    post_cov = flops.as_symmetric(post_cov, symmetry=(0, 1))
    return mean, post_cov


class _ExactGaussianBase(BaseEstimator):
    ORDER = 8

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        mu = fnp.zeros(mlp.width, dtype=fnp.float64)
        cov = flops.as_symmetric(fnp.eye(mlp.width, dtype=fnp.float64), symmetry=(0, 1))
        rows = []
        for weight in mlp.weights:
            mu_pre = weight.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, weight, weight)
            mu, cov = _relu_gaussian_moments(mu_pre, cov_pre, self.ORDER)
            rows.append(mu)
        return fnp.stack(rows, axis=0)


class EstimatorGL4(_ExactGaussianBase):
    ORDER = 4


class EstimatorGL6(_ExactGaussianBase):
    ORDER = 6


class EstimatorGL8(_ExactGaussianBase):
    ORDER = 8


class EstimatorGL12(_ExactGaussianBase):
    ORDER = 12


class EstimatorGL16(_ExactGaussianBase):
    ORDER = 16


class Estimator(EstimatorGL8):
    pass
