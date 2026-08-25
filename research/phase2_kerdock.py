"""Phase-2 real-Kerdock spherical cubature for 1024x16 ReLU MLPs.

All estimator arithmetic uses flopscope.  The packed bytes are fixed Kerdock
phase data, decoded during setup.  Complete bases are evaluated with antipodes
at E[Chi_1024], making the radial integral exact by positive homogeneity.
"""
from __future__ import annotations

import base64
import zlib

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, MLP, SetupContext

_WIDTH = 1024
_HALF = 512
_RADIUS = 31.9921884548318
_SCALE = 0.9997558892134938
_PACKED = 'c-pPlOKu!73`J3${ck#h1Ta~*3rQog2Q=(<l|;Uubk3hI=lPTTO8zDvl26OO$^XlP<cacVd564A-YTz`b;yEbWwJzBt1MbpFWZp)$i`%cvQ^o$Y+kw`?U24mgQQc^GU=T(QMxH@mHtYjrQ_0inSl&LrXu5!Imy6eaxy}hr3_W3D`S><%isk80tLZ@fJ0Csa1q=HfCNbbDZ!S2OwcB<6Z{Df1(5<x!Ki>$P%H2i91EZY*#dFFx`1BLFEkK72r+~qLKWeRkVjY~bP`?(!Gvi-IpLm=P}nH66n+X(g|R|i;joZdSS|Dxo(tiH`62<L1tJQf4k8nx7a|;@AR;BADIzYSG9ovkJ0d`$L?TI|O(IgFRw7%XUm|3pXd-Q*aUyo2dLn<KgCdBcj3SYul_Hv=o+6{7ry{JPup+ghxgx%z!Xn3_%OcRC)FRoU-6G<m<|6B&?;`Y~_#*w{0b&N?3Stl96Ji+R9AY8jC1NV#E@Cs{H)1^EKw?GWNn%doQes!)TVi11WMXOJZDMlbc4B+te`18<h+>W6kz$tOnqr^gqhhGytYWd^wPL#BzGB1T$70Ol&|=l%*<#+};$r9G>tgWY^kVtq{SpEs4M;4Id>}zV5`#ns$q*7IBvnYfkenfbLz0I?5XmAEN+g{~Op&}I!9@~`L>b965^f~rNZgU!BLV2vmvx2oxGs6CC)K<@xbmElVBQ`gbzIk+;oj?J=G<J^KD>>PeYT05uj;j(8kY>7e;)K1w<V9qTbDME%bWji-I=GE^(ODtmi)Q=kv=ZXVk^kE^`wsPd&B(+gsU%7&)MpG9`?2BJ+zF}wY2m@J>S+V@H@j){`#43s#|z|oWTcZ#>wG{xx(vHKdupE-5Am8z1I51E$A)At#in1`IgC5&p6%8o98v2TNM5|V!aM)U;e)+RqHKHn5UUN-jnK^dQoJ~3SQ@5>ZjZ8rmuUd_4`G?03qiM=>'


def _hadamard():
    h = fnp.asarray(((1.0,),), dtype=fnp.float32)
    for _ in range(9):
        h = fnp.concatenate((
            fnp.concatenate((h, h), axis=1),
            fnp.concatenate((h, fnp.negative(h)), axis=1),
        ), axis=0)
    return h


def _tables():
    raw = zlib.decompress(base64.b85decode(_PACKED))
    bits = fnp.reshape(
        fnp.asarray(tuple(raw[:4608]), dtype=fnp.int32), (_HALF, 9)
    )
    phases = fnp.reshape(
        fnp.asarray(tuple(raw[4608:]), dtype=fnp.int32), (31, 9, 9)
    )
    return bits, phases


def _fallback(mlp: MLP):
    mu = fnp.zeros(mlp.width, dtype=fnp.float32)
    var = fnp.ones(mlp.width, dtype=fnp.float32)
    rows = []
    for weight in mlp.weights:
        w = fnp.asarray(weight, dtype=fnp.float32)
        pre = w.T @ mu
        pre_var = fnp.maximum((w * w).T @ var, 1e-20)
        sigma = fnp.sqrt(pre_var)
        alpha = pre / sigma
        cdf = flops.stats.norm.cdf(alpha)
        pdf = flops.stats.norm.pdf(alpha)
        mu = pre * cdf + sigma * pdf
        second = (pre * pre + pre_var) * cdf + pre * sigma * pdf
        var = fnp.maximum(second - mu * mu, 1e-20)
        rows.append(mu)
    return fnp.stack(rows, axis=0)


class _Kerdock(BaseEstimator):
    BASIS_COUNT = 20
    NORMALIZER = 0.0000244140625

    def __init__(self):
        self._bases = None

    def setup(self, ctx: SetupContext) -> None:
        _ = ctx
        h = _hadamard()
        bits, phases = _tables()
        gray = fnp.asarray(
            ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)),
            dtype=fnp.float32,
        )
        bases = []
        for p in phases[: self.BASIS_COUNT - 1]:
            q = fnp.remainder(fnp.sum((bits @ p) * bits, axis=1), 4)
            s0 = gray[q]
            s1 = gray[fnp.remainder(q + 1, 4)]
            b0 = fnp.reshape(h[:, :, None] * s0[None, :, :], (_HALF, _WIDTH))
            b1 = fnp.reshape(h[:, :, None] * s1[None, :, :], (_HALF, _WIDTH))
            bases.append(fnp.concatenate((b0, b1), axis=0) * _SCALE)
        self._bases = tuple(bases)

    def predict(self, mlp: MLP, budget: int):
        _ = budget
        if mlp.width != _WIDTH or mlp.depth != 16 or self._bases is None:
            return _fallback(mlp)

        weights = tuple(fnp.asarray(w, dtype=fnp.float32) for w in mlp.weights)
        zero = fnp.asarray(0.0, dtype=fnp.float32)
        total = fnp.zeros(_WIDTH, dtype=fnp.float32)

        z = weights[0] * _RADIUS
        a = fnp.maximum(z, zero)
        y = fnp.concatenate((a, a - z), axis=0)
        for w in weights[1:]:
            y = fnp.maximum(y @ w, zero)
        total = total + fnp.sum(y, axis=0)

        for basis in self._bases:
            z = basis @ weights[0]
            a = fnp.maximum(z, zero)
            y = fnp.concatenate((a, a - z), axis=0)
            for w in weights[1:]:
                y = fnp.maximum(y @ w, zero)
            total = total + fnp.sum(y, axis=0)

        final = total * self.NORMALIZER
        blank = fnp.zeros(_WIDTH, dtype=fnp.float32)
        return fnp.stack([blank] * 15 + [final], axis=0)


class EstimatorK8(_Kerdock):
    BASIS_COUNT = 8
    NORMALIZER = 0.00006103515625


class EstimatorK12(_Kerdock):
    BASIS_COUNT = 12
    NORMALIZER = 0.000040690104166666664


class EstimatorK16(_Kerdock):
    BASIS_COUNT = 16
    NORMALIZER = 0.000030517578125


class EstimatorK20(_Kerdock):
    BASIS_COUNT = 20
    NORMALIZER = 0.0000244140625


class EstimatorK24(_Kerdock):
    BASIS_COUNT = 24
    NORMALIZER = 0.000020345052083333332


class EstimatorK32(_Kerdock):
    BASIS_COUNT = 32
    NORMALIZER = 0.0000152587890625


class Estimator(EstimatorK20):
    pass
