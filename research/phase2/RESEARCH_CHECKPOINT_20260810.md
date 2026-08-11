# Phase 2 research checkpoint — 2026-08-10

Status: **active research; not yet a submission candidate**.

## Reproduced third-order result

For a centered hidden activation `Y` with covariance `C` and third cumulant
`K3`, write the repeated-index block

`S[i,j] = K3[i,i,j] = E[Y_i^2 Y_j]`.

The deep all-distinct third cumulant is well approximated by the
covariance-polynomial module

`K3 ~= sum_m sym(a_m tensor A_m)`, with `A_m in {C, C^2, C^3}`.

The vectors `a_m` are fitted from `S` through

`S[i,j] ~= sum_m (2 a_m[i] A_m[i,j] + A_m[i,i] a_m[j])`.

This gives the downstream scalar identity

`K3(w,w,w) ~= 3 sum_m (a_m^T w) (w^T A_m w)`

and repeated-index transport

`K3'(a,a,b) ~= sum_m [2 aw_m[a] G_m[a,b] + aw_m[b] G_m[a,a]]`,
where `aw_m = W^T a_m` and `G_m = W^T A_m W`.

The original saved audits were reproduced exactly after recovering the correct
MLP RNG convention (`standard_normal` in float64, scale, then cast float32).
For seed 1701, powers `{C,C^2,C^3}`, ridge `1e-4`:

| activation layer | all-distinct downstream K3 MSE / zero MSE | correlation |
|---:|---:|---:|
| 16 | 0.1467698 | 0.9230 |
| 24 | 0.0472416 | 0.9760 |
| 31 | 0.0285375 | 0.9857 |

Fresh seeds at layer 31:

- seed 1702: ratio `0.0400`, correlation `0.9800`;
- seed 1703: ratio `0.0130`, correlation `0.9936`.

## Partition-aware bivariate Edgeworth map

A bivariate Edgeworth ReLU map was reconstructed using triangular Gaussian
whitening and split Gauss-Legendre integration at the ReLU kink.  The split
quadrature makes the independent-Gaussian covariance error approximately
machine precision, unlike ordinary Gauss-Hermite quadrature at the kink.

On seed 1701, transition into activation layer 17, using oracle preactivation
moments:

| nonlinear closure | covariance off-diagonal error ratio | `S_ij=K3_iij` off-diagonal error ratio |
|---|---:|---:|
| Gaussian | 4.466e-3 | 8.950e-1 |
| K3 | 4.818e-5 | 3.585e-3 |
| K3 + K4 | 6.641e-6 | 1.690e-4 |
| K3 + K4 + K3^2 | 5.999e-6 | 1.288e-4 |

The K3+K4+K3^2 mixed-skew correlation was `0.9999433`.

## Important falsification: K3-only free rollout

A complete matrix-sized K3 recurrence was implemented.  Exact repeated-index
residuals were transported while the covariance-polynomial module supplied the
all-distinct block.  It is locally accurate but recursively unstable:

- direct use from early layers produces non-PSD covariance states;
- realizability damping prevents blow-up but the final mean MSE remains around
  `1e-5` to `1e-4` against complete-MUB references;
- partial-MUB prefixes followed by the K3 analytic tail also remain around
  `1e-5` final MSE.

Therefore **third order alone is closed as a winning end-to-end route**.
The fourth-order state is load-bearing.

## New fourth-order covariance-volatility module

For fourth cumulant `K4`, define repeated blocks

- `k40[i] = K4[i,i,i,i]`,
- `k31[i,j] = K4[i,i,i,j]`,
- `k22[i,j] = K4[i,i,j,j]`.

A new matrix-sized ansatz is

`K4 ~= Sym_6(M tensor C)`,

where

`K4[i,j,k,l] = M[i,j]C[k,l] + M[i,k]C[j,l] + M[i,l]C[j,k]
              + C[i,j]M[k,l] + C[i,k]M[j,l] + C[i,l]M[j,k]`.

Its repeated blocks are

- `k40[i] = 6 M[ii] C[ii]`,
- `k31[i,j] = 3 (M[ii] C[ij] + C[ii] M[ij])`,
- `k22[i,j] = M[ii] C[jj] + C[ii] M[jj] + 4 M[ij] C[ij]`.

`M` is recovered from the observed repeated blocks by a closed, pairwise
least-squares solve.  Under a linear layer, the representation is closed:

`C -> W^T C W`, `M -> W^T M W`.

On seed 1701, predicting the next preactivation repeated K4 state:

| source activation layer | k31 error ratio | k22 error ratio | fitted optimal amplitude (roughly) |
|---:|---:|---:|---:|
| 16 | 0.674 | 1.868 | 0.4–0.6 |
| 24 | 0.208 | 0.278 | 0.69–0.74 |
| 31 | 0.111 | 0.116 | 0.84–0.86 |

At layer 31 the correlations are about `0.957` for k31 and `0.949` for k22.
This is the first demonstrated network-dependent, matrix-sized fourth-cumulant
transport with strong deep-layer out-of-orientation prediction.

A larger `{C,C^2,C^3}` fourth-order basis can interpolate repeated blocks almost
perfectly but catastrophically fails after the next random linear map.  This is
a useful negative result: repeated-index fit alone is not the right objective;
the single covariance-coupled mode generalizes because it respects the linear
transport geometry.

## Current mathematical target

The next step is not another static cubature sweep.  It is a realizable,
transport-stable **Gaussian location-scale mixture closure** whose low-order
cumulants jointly satisfy

- `K3 = Sym(a tensor C)` (or a small covariance-polynomial extension),
- `K4 = Sym(M tensor C)`,
- positivity/realizability constraints linking `a`, `M`, and `C`,
- exact closure under `W^T (.) W`,
- a ReLU nonlinear projection back into the same module.

The empirical evidence suggests deep hidden activations are approaching a
low-dimensional location/volatility mixture rather than merely a generic
low-rank tensor.  Deriving the joint realizability relations is the main open
research problem.

## Evaluation discipline

No public target was used to select these formulas.  Development used generated
width-256, depth-32 networks; fresh seeds were used for transfer checks.  No
leaderboard score is claimed until a complete 32-layer free rollout passes
fresh generated holdouts and exact `whestbench`/`flopscope` accounting.
