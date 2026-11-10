"""Options.QuadraticZeeman (0.7.1): field-quadratic/cubic Sys.Ham terms in levels,
the resonance search (resfields_batch / resfreqs_batch) and pepper / pepper_autograd.

References used here:
- the full-field Hamiltonian ham(sys, B·n) (exact diagonalization at each field);
- a closed form for S=1 with isotropic g and a Telser-type w·(μB B·S)² term;
- a brute-force field-point powder simulation (diagonalize H(B) on the field axis,
  Gaussian line in frequency);
- finite differences (dB/dE, gradients);
- EasySpin pepper (tests/data/ref_pepper_quadzeeman.mat, generate_pepper_quadzeeman_refs.m).
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from torchspin import Experiment, Options, SpinSystem, ham, pepper
from torchspin.constants import BMAGN, PLANCK
from torchspin.ham_ezho import (ham_ezho, field_nonlinear_zeeman_ops,
                                project_field_nonlinear_zeeman)
from torchspin.hamsymm import hamsymm
from torchspin.levels import levels
from torchspin.pepper_autograd import pepper_autograd
from torchspin.resfields_batch import resfields_batch
from torchspin.resfreqs_matrix import resfreqs_batch, resfreqs_matrix
from torchspin.sphgrid import sphgrid

BETA = BMAGN / PLANCK * 1e-9          # MHz/mT
REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_pepper_quadzeeman.mat'


def _nvec(phi, theta):
    return np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])


def _random_ham(rng, scale2=2e-4, scale3=0.0):
    """Random lB=2 (and optionally lB=3) Sys.Ham terms (valid keys: lB+lS even, l even,
    |lB-lS| <= l <= lB+lS; lS <= 2S, so lS <= 2 for S=1)."""
    h = {'202': (scale2 * rng.normal(size=(1, 5))).tolist(), '220': [[scale2 * rng.normal()]],
         '222': (scale2 * rng.normal(size=(1, 5))).tolist(),
         '224': (scale2 * rng.normal(size=(1, 9))).tolist()}
    if scale3:
        h['312'] = (scale3 * rng.normal(size=(1, 5))).tolist()
        h['314'] = (scale3 * rng.normal(size=(1, 9))).tolist()
    return h


def _sys_S1(Ham, **kw):
    base = dict(S=[1.0], g=[2.0, 2.02, 2.04], D=[[300.0, 30.0]], Ham=Ham)
    base.update(kw)
    return SpinSystem(**base)


def _cos(a, b):
    return float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))


# ---------------------------------------------------------------------------
# (a) default unchanged
# ---------------------------------------------------------------------------

def test_default_drops_quadratic_terms_bit_identical():
    """Default QuadraticZeeman=False: lB=2 terms are dropped exactly as before
    (bit-identical to the system without them); QuadraticZeeman=True without
    lB>=2 terms is bit-identical to the default."""
    rng = np.random.default_rng(3)
    Ham2 = _random_ham(rng)
    s_q = _sys_S1(Ham2, lwpp=1.0)
    s_l = _sys_S1(None, lwpp=1.0)
    exp = Experiment(mwFreq=9.5, Range=[150, 500], nPoints=1024)
    _, y_def = pepper(s_q, exp, Options(GridSize=[10, 4]))
    _, y_off = pepper(s_q, exp, Options(GridSize=[10, 4], QuadraticZeeman=False))
    _, y_lin = pepper(s_l, exp, Options(GridSize=[10, 4], GridSymmetry='C1'))
    _, y_lin_on = pepper(s_l, exp, Options(GridSize=[10, 4], GridSymmetry='C1', QuadraticZeeman=True))
    _, y_q_c1 = pepper(s_q, exp, Options(GridSize=[10, 4], GridSymmetry='C1'))
    assert torch.equal(y_def, y_off)
    assert torch.equal(y_q_c1, y_lin)
    assert torch.equal(y_lin_on, y_lin)
    B = torch.linspace(0, 600, 31)
    _, E0 = levels(s_q, 0.4, 1.1, B)
    _, E1 = levels(s_l, 0.4, 1.1, B)
    _, E2 = levels(s_l, 0.4, 1.1, B, quadratic_zeeman=True)
    assert torch.equal(E0, E1) and torch.equal(E1, E2)


def test_options_field_default():
    assert Options().QuadraticZeeman is False


# ---------------------------------------------------------------------------
# helpers: tensor decomposition
# ---------------------------------------------------------------------------

def test_projection_equals_ham_ezho_orders():
    """n·G2·n and G3[n,n,n] reproduce ham_ezho's lB=2 and lB=3 parts at B·n."""
    rng = np.random.default_rng(5)
    s = SpinSystem(S=[1.5], g=[2.0, 2.0, 2.0], Ham=_random_ham(rng, 1.0, 1.0))
    ops = field_nonlinear_zeeman_ops(s)
    for _ in range(4):
        n = rng.normal(size=3); n /= np.linalg.norm(n); b = rng.uniform(0.5, 3.0)
        Q2, Q3 = project_field_nonlinear_zeeman(ops, torch.tensor(n[None]))
        H2 = ham_ezho(s, B0=(b * n).tolist(), lB_select=2)
        H3 = ham_ezho(s, B0=(b * n).tolist(), lB_select=3)
        assert float((b ** 2 * Q2[0] - H2).abs().max()) < 1e-12 * max(1.0, float(H2.abs().max()))
        assert float((b ** 3 * Q3[0] - H3).abs().max()) < 1e-12 * max(1.0, float(H3.abs().max()))


# ---------------------------------------------------------------------------
# (b) levels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('with_cubic', [False, True])
def test_levels_equal_full_field_hamiltonian(with_cubic):
    rng = np.random.default_rng(11 + with_cubic)
    s = _sys_S1(_random_ham(rng, 2e-2, 2e-5 if with_cubic else 0.0))
    for _ in range(4):
        ph, th = rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi)
        B = rng.uniform(5, 600, size=7)
        _, E = levels(s, ph, th, torch.tensor(B), quadratic_zeeman=True)
        n = _nvec(ph, th)
        Eref = torch.stack([torch.linalg.eigvalsh(ham(s, (b * n).tolist())) for b in B], 1)
        assert float((E - Eref).abs().max()) <= 1e-12 * float(Eref.abs().max())
        # and the default really drops them
        _, E0 = levels(s, ph, th, torch.tensor(B))
        assert float((E0 - Eref).abs().max()) > 1.0


# ---------------------------------------------------------------------------
# (c) resonance fields, dB/dE and intensities vs the full H
# ---------------------------------------------------------------------------

def _rate_perp(s, B, n, i, j):
    """chi-averaged perpendicular transition rate (|mu|^2 - |n.mu|^2)/2 at B·n (full H)."""
    H0, mux, muy, muz = ham(s)
    _, V = torch.linalg.eigh(ham(s, (B * n).tolist()))
    me = [complex(V[:, i].conj() @ op @ V[:, j]) for op in (mux, muy, muz)]
    mu2 = sum(abs(m) ** 2 for m in me)
    mzl = abs(sum(n[k] * me[k] for k in range(3))) ** 2
    return (mu2 - mzl) / 2.0


@pytest.mark.parametrize('with_cubic', [False, True])
def test_resfields_batch_resonance_condition_and_dBdE(with_cubic):
    rng = np.random.default_rng(21 + with_cubic)
    s = _sys_S1(_random_ham(rng, 3e-4, 3e-7 if with_cubic else 0.0))
    H0, mux, muy, muz = ham(s)
    exp = Experiment(mwFreq=9.5, Range=[100, 600], nPoints=1024)
    nu = 9500.0
    phi = torch.tensor(rng.uniform(0, 2 * np.pi, 6), dtype=torch.float64)
    theta = torch.tensor(rng.uniform(0, np.pi, 6), dtype=torch.float64)
    Bl, Il, _, Pl = resfields_batch(H0, mux, muy, muz, phi, theta, exp, Options(Threshold=0.0), sys=s,
                                    return_pairs=True, zeeman_ho=field_nonlinear_zeeman_ops(s))
    n_checked = 0
    for m in range(6):
        n = _nvec(float(phi[m]), float(theta[m]))
        if Bl[m].numel() == 0:
            continue
        for B, I, (i, j) in zip(Bl[m].tolist(), Il[m].tolist(), Pl[m].tolist()):
            E = torch.linalg.eigvalsh(ham(s, (B * n).tolist()))
            assert abs(float(E[j] - E[i]) - nu) <= 1e-9 * nu
            h = 1e-3
            Ep = torch.linalg.eigvalsh(ham(s, ((B + h) * n).tolist()))
            Em = torch.linalg.eigvalsh(ham(s, ((B - h) * n).tolist()))
            slope = float((Ep[j] - Ep[i]) - (Em[j] - Em[i])) / (2 * h)
            rate = _rate_perp(s, B, n, i, j)
            if rate < 1e-6 * BETA ** 2:
                continue
            I_ref = rate / abs(slope)                # infinite temperature: polarization 1, no nuclei
            assert abs(I - I_ref) <= 1e-6 * abs(I_ref), (m, B, I, I_ref)
            n_checked += 1
    assert n_checked >= 6


# ---------------------------------------------------------------------------
# (d) closed form: S=1, isotropic g, Telser-type w (mu_B B.S)^2
# ---------------------------------------------------------------------------

def _calibrate_bs2(S, w):
    """Ham coefficients for w·(μB B·S)² by least-squares projection onto the
    lB=2 basis (keys 202, 220, 222, 224), checked to reproduce the target.

    The field enters as the rank-lB spherical tensor of B, so the rank-0 part
    w β² B² S(S+1)/3 (a common shift of all levels) is not representable and is
    subtracted from the target."""
    keys = [('202', 5), ('220', 1), ('222', 5), ('224', 9)]
    basis = []
    for k, n in keys:
        for q in range(n):
            row = [0.0] * n; row[q] = 1.0
            basis.append((k, n, q, SpinSystem(S=[S], g=[2.0, 2.0, 2.0], Ham={k: [row]})))
    from torchspin.spinops import sop
    Sx, Sy, Sz = (sop([S], [[1, c]]) for c in (1, 2, 3))
    rng = np.random.default_rng(0)
    rows, rhs = [], []
    for _ in range(6):
        B = rng.normal(size=3) * 100
        BS = B[0] * Sx + B[1] * Sy + B[2] * Sz
        target = w * BETA ** 2 * (BS @ BS - float(B @ B) * S * (S + 1) / 3 * torch.eye(BS.shape[0], dtype=torch.float64))
        cols = [ham_ezho(bs, B0=B.tolist(), lB_select=2) for (_, _, _, bs) in basis]
        rows.append(torch.stack([c.reshape(-1) for c in cols], 1))
        rhs.append(target.reshape(-1))
    A = torch.cat(rows); y = torch.cat(rhs)
    Ar = torch.cat([A.real, A.imag]); yr = torch.cat([y.real, y.imag])
    coef = torch.linalg.lstsq(Ar, yr.unsqueeze(1), driver='gelsd').solution.squeeze(1)
    assert float((Ar @ coef - yr).abs().max()) < 1e-9 * float(yr.abs().max())
    Ham = {}
    for c, (k, n, q, _) in zip(coef.tolist(), basis):
        Ham.setdefault(k, [[0.0] * n])[0][q] = c
    return Ham


def test_closed_form_S1_isotropic_BS_squared():
    """E(m) = m g β B + c (m² − 2/3) B² (c = w β²): resonance fields
    B+ = (−a + √(a² + 4cν)) / 2c (0→+1) and B− = (a − √(a² − 4cν)) / 2c (−1→0),
    dB/dE = 1/(a ± 2cB), any orientation."""
    w = 1e-5                               # 1/MHz
    Ham = _calibrate_bs2(1.0, w)
    # isotropic: only the l = 0 coupling (220) survives the projection
    assert all(abs(v) < 1e-12 for v in Ham['202'][0] + Ham['222'][0] + Ham['224'][0])
    s = SpinSystem(S=[1.0], g=[2.0, 2.0, 2.0], Ham=Ham)
    a, c, nu = 2.0 * BETA, w * BETA ** 2, 9500.0
    Bp = (-a + math.sqrt(a * a + 4 * c * nu)) / (2 * c)
    Bm = (a - math.sqrt(a * a - 4 * c * nu)) / (2 * c)
    assert Bm - Bp > 10.0                  # well separated (the default gives one line)
    H0, mux, muy, muz = ham(s)
    exp = Experiment(mwFreq=9.5, Range=[200, 500], nPoints=1024)
    ph = torch.tensor([0.0, 0.7, 2.1], dtype=torch.float64)
    th = torch.tensor([0.0, 1.0, 2.5], dtype=torch.float64)
    Bl, Il, _, Pl = resfields_batch(H0, mux, muy, muz, ph, th, exp, Options(), sys=s,
                                    return_pairs=True, zeeman_ho=field_nonlinear_zeeman_ops(s))
    for m in range(3):
        allowed = [(B, I) for B, I, p in zip(Bl[m].tolist(), Il[m].tolist(), Pl[m].tolist()) if p[1] - p[0] == 1]
        allowed.sort()
        assert len(allowed) == 2, Bl[m]
        (B1, I1), (B2, I2) = allowed
        assert abs(B1 - Bp) < 1e-8 * Bp and abs(B2 - Bm) < 1e-8 * Bm
        # equal transition rates (|<1|S+|0>|² = |<0|S+|-1>|²), so I1/I2 = dBdE1/dBdE2
        assert abs((I1 / I2) - (a - 2 * c * Bm) / (a + 2 * c * Bp)) < 1e-8
    # levels: closed-form energies
    B = torch.linspace(0, 600, 13, dtype=torch.float64)
    _, E = levels(s, 0.3, 0.9, B, quadratic_zeeman=True)
    Ecf = torch.stack([-a * B + c * B ** 2 / 3, -2 * c * B ** 2 / 3, a * B + c * B ** 2 / 3])
    Ecf = torch.sort(Ecf, dim=0).values
    assert float((E - Ecf).abs().max()) < 1e-9 * float(Ecf.abs().max())


# ---------------------------------------------------------------------------
# (e) pepper vs brute-force field-point simulation
# ---------------------------------------------------------------------------

def _brute_force_powder(s, exp, N, HStrain):
    """Σ_orientations w · 2π Σ_pairs rate(B) · Gauss_ν(E_j(B)−E_i(B)−hν; FWHM_HStrain(n)),
    diagonalizing the full H(B) at every field point of the axis."""
    x = torch.linspace(exp.Range[0], exp.Range[1], exp.nPoints, dtype=torch.float64)
    phi, theta, wts, _ = sphgrid('D2h', N)
    H0, mux, muy, muz = ham(s)
    ops = field_nonlinear_zeeman_ops(s)
    nv = torch.stack([torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)], 1)
    Q2, Q3 = project_field_nonlinear_zeeman(ops, nv)
    nu = exp.mwFreq * 1e3
    Hs = torch.tensor(HStrain, dtype=torch.float64)
    spec = torch.zeros_like(x)
    for m in range(phi.numel()):
        muzL = nv[m, 0] * mux + nv[m, 1] * muy + nv[m, 2] * muz
        H = H0 - x[:, None, None] * muzL + x[:, None, None] ** 2 * Q2[m]
        if Q3 is not None:
            H = H + x[:, None, None] ** 3 * Q3[m]
        E, V = torch.linalg.eigh(H)
        fwhm = float(torch.sqrt((Hs ** 2 * nv[m] ** 2).sum()))
        sig = fwhm / math.sqrt(8 * math.log(2))
        n_ = E.shape[1]
        for i in range(n_):
            for j in range(i + 1, n_):
                me = [(V[:, :, i].conj() * (op @ V[:, :, j].T).T).sum(1) for op in (mux, muy, muz)]
                mu2 = sum(t.abs() ** 2 for t in me)
                mzl = (nv[m, 0] * me[0] + nv[m, 1] * me[1] + nv[m, 2] * me[2]).abs() ** 2
                rate = (mu2 - mzl) / 2
                d = E[:, j] - E[:, i] - nu
                spec += float(wts[m]) * rate * torch.exp(-0.5 * (d / sig) ** 2) / (sig * math.sqrt(2 * math.pi))
    return x, spec * 2 * math.pi


def test_pepper_vs_brute_force_field_points():
    Ham = {'220': [[2e-4]], '222': [[2e-4, 0.0, 3e-4, 0.0, 0.0]]}   # D2h-compatible
    HS = [60.0, 70.0, 80.0]
    s = _sys_S1(Ham, HStrain=HS)
    assert hamsymm(s)[0] == 'D2h'
    exp = Experiment(mwFreq=9.5, Range=[150, 500], nPoints=700, Harmonic=0)
    x, y = pepper(s, exp, Options(QuadraticZeeman=True))
    _, y_off = pepper(s, exp, Options())
    _, ref = _brute_force_powder(s, exp, 46, HS)
    y, y_off, ref = y.numpy(), y_off.numpy(), ref.numpy()
    assert _cos(y, ref) > 0.999
    assert abs(y.max() / ref.max() - 1) < 0.03
    assert abs(y.sum() / ref.sum() - 1) < 0.03
    assert _cos(y_off, ref) < 0.995         # the effect is resolved


# ---------------------------------------------------------------------------
# frequency sweeps (fixed field): add B² n·G2·n to H0
# ---------------------------------------------------------------------------

def test_freq_sweep_positions_equal_full_hamiltonian():
    rng = np.random.default_rng(31)
    s = _sys_S1(_random_ham(rng, 3e-4))
    H0, mux, muy, muz = ham(s)
    B0 = 340.0
    exp = Experiment(Field=B0, mwRange=[1.0, 20.0], nPoints=512)
    ph = torch.tensor(rng.uniform(0, 2 * np.pi, 4), dtype=torch.float64)
    th = torch.tensor(rng.uniform(0, np.pi, 4), dtype=torch.float64)
    P, _, _, Pr = resfreqs_batch(H0, mux, muy, muz, ph, th, exp, Options(Threshold=0.0), sys=s,
                                 return_pairs=True, zeeman_ho=field_nonlinear_zeeman_ops(s))
    for m in range(4):
        n = _nvec(float(ph[m]), float(th[m]))
        E = torch.linalg.eigvalsh(ham(s, (B0 * n).tolist()))
        for f, (i, j) in zip(P[m].tolist(), Pr[m].tolist()):
            assert abs(f * 1e3 - float(E[j] - E[i])) < 1e-9 * float(E.abs().max())
    f1, _ = resfreqs_matrix(s, float(ph[0]), float(th[0]), B0, threshold=0.0, opt=Options(QuadraticZeeman=True))
    n = _nvec(float(ph[0]), float(th[0]))
    E = torch.linalg.eigvalsh(ham(s, (B0 * n).tolist()))
    dE = sorted(float(E[j] - E[i]) for i in range(3) for j in range(i + 1, 3))
    assert np.allclose(sorted(f1.tolist()), dE, rtol=1e-12)
    # pepper frequency sweep runs and differs from the default
    s2 = replace(s, lw=[0.0, 20.0])
    exp2 = Experiment(Field=B0, mwRange=[8.0, 11.0], nPoints=1024, Harmonic=0)
    _, ya = pepper(s2, exp2, Options(GridSize=[10, 4], QuadraticZeeman=True))
    _, yb = pepper(s2, exp2, Options(GridSize=[10, 4]))
    assert torch.isfinite(ya).all() and float((ya - yb).abs().max()) > 1e-3 * float(yb.abs().max())


# ---------------------------------------------------------------------------
# crystal path
# ---------------------------------------------------------------------------

def test_crystal_lines_at_full_hamiltonian_resonances():
    from torchspin.rotations import erot
    Ham = {'220': [[1e-3]], '222': [[1e-3, 0.0, 1.5e-3, 0.0, 0.0]]}
    s = _sys_S1(Ham, lw=[0.3, 0.0])
    mf = [0.3, 0.7, 0.1]
    exp = Experiment(mwFreq=9.5, Range=[150, 500], nPoints=8192, Harmonic=0, MolFrame=mf, CrystalSymmetry='P1')
    x, y = pepper(s, exp, Options(QuadraticZeeman=True))
    # brute force along the lab z direction in the molecular frame
    from torchspin.sitetransforms import crystal_orientations
    ang = crystal_orientations(np.zeros((1, 3)), 'P1', mf)[0]
    n = erot(list(ang)).numpy()[2]
    xb = x.numpy()
    H0, mux, muy, muz = ham(s)
    Q2, _ = project_field_nonlinear_zeeman(field_nonlinear_zeeman_ops(s), torch.tensor(n[None]))
    muzL = n[0] * mux + n[1] * muy + n[2] * muz
    E = torch.linalg.eigvalsh(H0 - x[:, None, None] * muzL + x[:, None, None] ** 2 * Q2[0]).numpy()
    Eh = torch.linalg.eigvalsh(ham(s, (float(x[4000]) * n).tolist())).numpy()
    assert np.allclose(E[4000], Eh, rtol=1e-12, atol=1e-9)
    yn = y.numpy()
    n_checked, checked = 0, []
    for (i, j) in ((0, 1), (1, 2), (0, 2)):
        d = E[:, j] - E[:, i] - 9500.0
        for kk in np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]:
            Bres = xb[kk] - d[kk] * (xb[kk + 1] - xb[kk]) / (d[kk + 1] - d[kk])
            win = np.abs(xb - Bres) < 1.0
            if yn[win].max() > 0.05 * yn.max():
                assert abs(xb[win][np.argmax(yn[win])] - Bres) < 2 * (xb[1] - xb[0])
                n_checked += 1
                checked.append(Bres)
    assert n_checked >= 2
    # without the quadratic term the strong lines sit elsewhere (shift > 2 bins)
    E0 = torch.linalg.eigvalsh(H0 - x[:, None, None] * muzL).numpy()
    B_no = []
    for (i, j) in ((0, 1), (1, 2), (0, 2)):
        d0 = E0[:, j] - E0[:, i] - 9500.0
        B_no += [xb[k] for k in np.where(np.sign(d0[:-1]) != np.sign(d0[1:]))[0]]
    assert max(min(abs(b - b0) for b0 in B_no) for b in checked) > 2 * (xb[1] - xb[0])


# ---------------------------------------------------------------------------
# (f) gradients with the option on
# ---------------------------------------------------------------------------

def test_pepper_autograd_gradient_vs_central_differences():
    Ham = {'220': [[2e-4]], '222': [[2e-4, 0.0, 3e-4, 0.0, 0.0]]}
    exp = Experiment(mwFreq=9.5, Range=[150, 500], nPoints=512, Harmonic=1)
    opt = Options(GridSize=[19, 4], QuadraticZeeman=True)
    D = torch.tensor([[300.0, 30.0]], dtype=torch.float64, requires_grad=True)
    g = torch.tensor([2.0, 2.02, 2.04], dtype=torch.float64, requires_grad=True)
    s = SpinSystem(S=[1.0], g=g, D=D, Ham=Ham, lw=[1.5, 0.0])
    _, y = pepper_autograd(s, exp, opt)
    wv = torch.tensor(np.random.default_rng(0).normal(size=y.numel()))
    L = (y * wv).sum()
    gD, gg = torch.autograd.grad(L, [D, g])

    def loss(Dv, gv):
        ss = SpinSystem(S=[1.0], g=gv, D=Dv, Ham=Ham, lw=[1.5, 0.0])
        return float((pepper(ss, exp, opt)[1] * wv).sum())
    D0 = [[300.0, 30.0]]; g0 = [2.0, 2.02, 2.04]
    for idx, an, (hs, mk) in (
        (0, float(gD[0, 0]), ((1e-2, 3e-3, 1e-3), lambda h: ([[300.0 + h, 30.0]], g0))),
        (1, float(gD[0, 1]), ((1e-2, 3e-3, 1e-3), lambda h: ([[300.0, 30.0 + h]], g0))),
        (2, float(gg[2]), ((1e-6, 3e-7, 1e-7), lambda h: (D0, [2.0, 2.02, 2.04 + h]))),
    ):
        errs = []
        for h in hs:
            fd = (loss(*mk(h)) - loss(*mk(-h))) / (2 * h)
            errs.append(abs(fd - an) / max(abs(fd), 1e-30))
        assert min(errs) < 1e-6, (idx, an, errs)
    # forward value equals pepper (no grad)
    _, y_np = pepper(SpinSystem(S=[1.0], g=g0, D=D0, Ham=Ham, lw=[1.5, 0.0]), exp, opt)
    assert float((y.detach() - y_np).abs().max()) <= 1e-9 * float(y_np.abs().max())


# ---------------------------------------------------------------------------
# unsupported paths raise
# ---------------------------------------------------------------------------

def test_unsupported_paths_raise():
    from torchspin.garlic import garlic
    from torchspin.resfields import resfields
    from torchspin.resfields_perturb import resfields_perturb_batch
    Ham = {'220': [[2e-4]]}
    s1 = SpinSystem(S=[1.0], g=[2.0, 2.0, 2.0], D=[[300.0, 30.0]], Ham=Ham, lw=[1.0, 0.0])
    s_half = SpinSystem(S=[0.5], g=[2.0, 2.0, 2.0], Nucs='1H', A=[[10.0, 10.0, 10.0]],
                        Ham={'202': [[1e-4, 0.0, 2e-4, 0.0, 0.0]]}, lw=[0.5, 0.0])
    exp = Experiment(mwFreq=9.5, Range=[300, 380], nPoints=256)
    on = Options(QuadraticZeeman=True)
    for meth in ('perturb', 'perturb1'):
        with pytest.raises(NotImplementedError, match='QuadraticZeeman'):
            pepper(s_half, exp, replace(on, Method=meth))
    with pytest.raises(NotImplementedError, match='QuadraticZeeman'):
        pepper(s_half, exp, replace(on, Method='hybrid'))
    with pytest.raises(NotImplementedError, match='QuadraticZeeman'):
        garlic(s_half, exp, on)
    with pytest.raises(NotImplementedError, match='QuadraticZeeman'):
        resfields_perturb_batch(s_half, torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64), exp, on)
    H0, mux, muy, muz = ham(s1)
    with pytest.raises(NotImplementedError, match='QuadraticZeeman'):
        resfields(H0, mux, muy, muz, 0.0, 0.0, exp, on, sys=s1)
    from torchspin.salt import salt
    with pytest.raises(NotImplementedError, match='QuadraticZeeman'):
        salt(s_half, Experiment(mwFreq=9.5, Field=340.0, Range=[1, 30], nPoints=128), opt=on)
    with pytest.raises(NotImplementedError, match='QuadraticZeeman'):
        pepper_autograd(SpinSystem(S=[0.5], g=torch.tensor([2.0, 2.0, 2.1], requires_grad=True),
                                   Ham={'202': [[1e-4, 0.0, 2e-4, 0.0, 0.0]]}, lw=[0.5, 0.0]), exp, replace(on, Method='perturb'))
    # lB >= 4 terms are not in the decomposition
    s4 = SpinSystem(S=[1.0], g=[2.0, 2.0, 2.0], Ham={'404': [[1e-8] * 9]})
    with pytest.raises(NotImplementedError, match='lB >= 4'):
        field_nonlinear_zeeman_ops(s4)
    # option off: the same calls are fine
    pepper(s_half, exp, Options(Method='perturb'))


# ---------------------------------------------------------------------------
# (g) EasySpin
# ---------------------------------------------------------------------------

def _load_cases():
    from scipy.io import loadmat
    m = loadmat(REF, squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(m['cases'])}


def _case_names():
    return list(_load_cases()) if REF.exists() else []


@pytest.mark.parametrize('name', _case_names())
def test_pepper_quadratic_zeeman_vs_easyspin(name):
    """EasySpin's resfields builds the level model from ham(Sys,B) at every field,
    so Ham2xx terms enter its resonance fields. Its dB/dE uses the linear moment
    only (torchspin: the full slope), a ~0.3 % amplitude difference here."""
    c = _load_cases()[name]
    S = c.Sys; f = S._fieldnames
    kw = dict(S=[float(S.S)], g=np.atleast_1d(S.g).astype(float).tolist(),
              D=[np.atleast_1d(S.D).astype(float).tolist()],
              Ham={k[3:]: np.atleast_1d(getattr(S, k)).astype(float).reshape(1, -1).tolist()
                   for k in f if k.startswith('Ham')})
    if 'lwpp' in f:
        kw['lwpp'] = [float(S.lwpp), 0.0]
    if 'HStrain' in f:
        kw['HStrain'] = np.atleast_1d(S.HStrain).astype(float).tolist()
    sys = SpinSystem(**kw)
    E = c.Exp; ef = E._fieldnames
    exp = Experiment(mwFreq=float(E.mwFreq), Range=np.atleast_1d(E.Range).astype(float).tolist(),
                     nPoints=int(E.nPoints), Harmonic=int(E.Harmonic) if 'Harmonic' in ef else 1)
    ref = np.asarray(c.spc, dtype=float)
    x, y = pepper(sys, exp, Options(QuadraticZeeman=True))
    assert np.allclose(x.numpy(), np.asarray(c.x, dtype=float), rtol=1e-9)
    y = y.numpy()
    assert _cos(y, ref) > 0.999
    assert abs(np.abs(y).max() / np.abs(ref).max() - 1) < 0.02


def test_levels_forward_and_reverse_mode_vs_central_differences():
    """levels(quadratic_zeeman=True) keeps the graph in D (forward and reverse mode)."""
    import torch.autograd.forward_ad as fwAD
    Ham = {'220': [[2e-4]], '222': [[2e-4, 0.0, 3e-4, 0.0, 0.0]]}
    B = torch.linspace(50, 500, 7, dtype=torch.float64)

    def f(Dv):
        E = levels(SpinSystem(S=[1.0], g=[2.0, 2.02, 2.04], D=Dv, Ham=Ham), 0.4, 1.1, B,
                   quadratic_zeeman=True)[1]
        return E[2] - E[0]
    D0 = torch.tensor([[300.0, 30.0]], dtype=torch.float64)
    for t in (torch.tensor([[1.0, 0.0]], dtype=torch.float64), torch.tensor([[0.0, 1.0]], dtype=torch.float64)):
        with fwAD.dual_level():
            jv = fwAD.unpack_dual(f(fwAD.make_dual(D0, t))).tangent
        errs = []
        for h in (1e-2, 1e-3, 1e-4):
            fd = (f(D0 + h * t) - f(D0 - h * t)) / (2 * h)
            errs.append(float((jv - fd).abs().max() / fd.abs().max()))
        assert min(errs) < 1e-7, errs
        Dg = D0.clone().requires_grad_(True)
        gr = torch.autograd.grad(f(Dg).sum(), Dg)[0]
        assert abs(float((gr * t).sum()) - float(jv.sum())) < 1e-9 * float(jv.abs().sum())
