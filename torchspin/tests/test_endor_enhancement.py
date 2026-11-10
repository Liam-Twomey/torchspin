"""ENDOR hyperfine enhancement (``Options.Enhancement``, EasySpin ``Opt.Enhancement``).

With ``Enhancement=True`` the rf transition operator of ``endorfrq`` and the
matrix paths of ``salt`` is the total magnetic moment (electron + nuclear,
``ham()``'s ``mux/muy/muz``) perpendicular to the field, chi-averaged over the
lab x and y axes, instead of the nuclear moment.  Selection (threshold) uses
the nuclear-only rates and electronic level pairs are excluded, as in EasySpin
``endorfrq.m``.

Checks: default unchanged; intensities vs a NumPy brute force from the
eigenvectors; first-order enhancement factor (nu_ENDOR/nu_I)^2 for an
isotropic 1H; salt vs a brute-force orientation sum; gradients (reverse and
forward mode) vs central differences; perturbation paths raise; EasySpin
references (tests/data/endor_enhancement_refs.mat from
tests/data/generate_endor_enhancement_refs.m).
"""
import math
import os

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwAD

from torchspin import Experiment, Options, SpinSystem
from torchspin.constants import NMAGN, PLANCK
from torchspin.endorfrq import endorfrq
from torchspin.endorfrq_perturb import endorfrq_perturb
from torchspin.ham import ham
from torchspin.salt import salt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data")
ON = Options(Enhancement=True)
DT = torch.float64


def _sys_1H():
    return SpinSystem(S=[0.5], g=[[2.0, 2.01, 2.02]], Nucs='1H', A=[[6.0, 9.0, 16.0]])


def _sys_14N_1H():
    return SpinSystem(S=[0.5], g=[[2.0088, 2.0061, 2.0027]], Nucs='14N,1H',
                      A=[[17.0, 17.0, 98.0], [2.0, 3.0, 5.0]],
                      Q=[[1.0, 0.5, -1.5], [0.0, 0.0, 0.0]])


def _sys_14N():
    return SpinSystem(S=[0.5], g=[[2.25, 2.25, 2.0]], Nucs='14N', A=[[44.0, 49.0, 54.0]],
                      Q=[[0.84, 0.84, -1.68]])


def _frame(phi, theta):
    z = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])
    x = np.array([math.cos(theta) * math.cos(phi), math.cos(theta) * math.sin(phi), -math.sin(theta)])
    y = np.array([-math.sin(phi), math.cos(phi), 0.0])
    return x, y, z


def _brute_endor(sys, B, phi, theta):
    """NumPy: all intra-block level pairs, frequency and chi-averaged |<i|mu_perp|j>|^2."""
    H0, mux, muy, muz = (m.detach().numpy() for m in ham(sys))
    mu = (mux, muy, muz)
    x, y, z = _frame(phi, theta)
    E, V = np.linalg.eigh(H0 - B * sum(z[k] * mu[k] for k in range(3)))
    Mx = V.conj().T @ sum(x[k] * mu[k] for k in range(3)) @ V
    My = V.conj().T @ sum(y[k] * mu[k] for k in range(3)) @ V
    nnuc = int(round(np.prod([2 * i + 1 for i in sys.I])))
    out = {}
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            if i // nnuc == j // nnuc:
                out[(i, j)] = (E[j] - E[i], (abs(Mx[i, j]) ** 2 + abs(My[i, j]) ** 2) / 2)
    return out


# ---------------------------------------------------------------------------
# Default unchanged / API
# ---------------------------------------------------------------------------

def test_default_is_off():
    assert Options().Enhancement is False


@pytest.mark.parametrize("mk", [_sys_1H, _sys_14N_1H])
def test_endorfrq_default_identical(mk):
    s = mk()
    a = endorfrq(s, 340.0, 0.3, 0.7)
    for opt in (Options(), Options(Enhancement=False)):
        b = endorfrq(s, 340.0, 0.3, 0.7, opt=opt)
        assert all(torch.equal(u, v) for u, v in zip(a, b))


def test_salt_default_identical():
    s = _sys_14N_1H()
    exp = Experiment(Field=340.0, mwFreq=9.55)
    a = salt(s, exp, Options(GridSize=[11, 3]), freq_range=(0, 60), n_points=300)
    b = salt(s, exp, Options(GridSize=[11, 3], Enhancement=False), freq_range=(0, 60), n_points=300)
    assert all(torch.equal(u, v) for u, v in zip(a, b))


def test_endorfrq_perturb_raises():
    s = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], Nucs='1H', A=[[3.0, 7.0, 12.0]])
    endorfrq_perturb(s, 350.0, 0.3, 0.7, opt=Options())       # off: fine
    with pytest.raises(NotImplementedError):
        endorfrq_perturb(s, 350.0, 0.3, 0.7, opt=ON)


# ---------------------------------------------------------------------------
# endorfrq vs brute force
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mk,B", [(_sys_1H, 326.5), (_sys_14N_1H, 340.0), (_sys_14N, 326.5)])
@pytest.mark.parametrize("phi,theta", [(0.0, 0.0), (0.3, 0.7), (1.1, 1.3), (2.5, 2.2)])
def test_endorfrq_enhanced_matches_brute_force(mk, B, phi, theta):
    s = mk()
    ref = _brute_endor(s, B, phi, theta)
    f, I, tr = endorfrq(s, B, phi, theta, threshold=0.0, opt=ON)
    got = {tuple(t): (fi, ii) for t, fi, ii in zip(tr.tolist(), f.tolist(), I.tolist())}
    assert set(got) == set(ref)          # all intra-block pairs, electronic ones excluded
    Imax = max(v[1] for v in ref.values())
    for k, (fr, ir) in ref.items():
        assert abs(got[k][0] - fr) < 1e-9
        assert abs(got[k][1] - ir) < 1e-10 * Imax


def test_endorfrq_enhanced_selection_uses_nuclear_rates():
    """Same transitions as the default minus electronic pairs; only intensities change."""
    s = _sys_14N_1H()
    nnuc = 6
    f0, I0, t0 = endorfrq(s, 340.0, 0.3, 0.7, threshold=1e-3)
    f1, I1, t1 = endorfrq(s, 340.0, 0.3, 0.7, threshold=1e-3, opt=ON)
    keep0 = {tuple(t) for t in t0.tolist() if t[0] // nnuc == t[1] // nnuc}
    assert keep0 == {tuple(t) for t in t1.tolist()}
    d0 = {tuple(t): (f, i) for t, f, i in zip(t0.tolist(), f0.tolist(), I0.tolist())}
    d1 = {tuple(t): (f, i) for t, f, i in zip(t1.tolist(), f1.tolist(), I1.tolist())}
    assert all(d0[k][0] == d1[k][0] for k in d1)                       # same frequencies
    assert max(abs(d1[k][1] / d0[k][1] - 1) for k in d1) > 0.05        # enhanced intensities


def test_enhancement_factor_isotropic_1H():
    """First-order hyperfine enhancement: I_on/I_off = (nu_ENDOR/nu_I)^2 (O(nu_I/nu_S) corrections)."""
    s = SpinSystem(S=[0.5], g=[[2.0023] * 3], Nucs='1H', A=[[10.0] * 3])
    B = 350.0
    f0, I0, t0 = endorfrq(s, B, 0.3, 0.5)
    f1, I1, t1 = endorfrq(s, B, 0.3, 0.5, opt=ON)
    assert torch.equal(t0, t1) and torch.equal(f0, f1) and len(f1) == 2
    nuI = s.gn[0] * NMAGN * B * 1e-3 / PLANCK * 1e-6
    ratio = (I1 / I0).numpy()
    closed = (f1.numpy() / nuI) ** 2           # (|nu_I +- A/2| / nu_I)^2
    np.testing.assert_allclose(ratio, closed, rtol=3e-3)
    assert ratio[1] > 1.7 and ratio[0] < 0.45  # high-frequency branch enhanced


# ---------------------------------------------------------------------------
# salt vs brute-force orientation sum
# ---------------------------------------------------------------------------

def _fibonacci(n):
    k = np.arange(n) + 0.5
    theta = np.arccos(1 - 2 * k / n)
    phi = (np.pi * (1 + 5 ** 0.5) * k) % (2 * np.pi)
    return phi, theta


@pytest.mark.parametrize("mk,B,rng,lw", [(_sys_1H, 326.5, (2.0, 25.0), 0.3),
                                         (_sys_14N, 326.5, (16.0, 65.0), 0.5)])
def test_salt_enhanced_vs_brute_force(mk, B, rng, lw):
    s = mk()
    n_points = 600
    x = np.linspace(rng[0], rng[1], n_points)
    dx = x[1] - x[0]
    H0, mux, muy, muz = (m.detach().numpy() for m in ham(s))
    mu = np.stack([mux, muy, muz])
    phi, theta = _fibonacci(6000)
    nnuc = int(round(np.prod([2 * i + 1 for i in s.I])))
    dim = H0.shape[0]
    iu = [(i, j) for i in range(dim) for j in range(i + 1, dim) if i // nnuc == j // nnuc]
    ii = np.array([p[0] for p in iu]); jj = np.array([p[1] for p in iu])
    spec = np.zeros(n_points)
    sig = lw / (2 * math.sqrt(2 * math.log(2)))
    for ph, th in zip(phi, theta):
        xL, yL, zL = _frame(ph, th)
        E, V = np.linalg.eigh(H0 - B * np.tensordot(zL, mu, 1))
        Mx = V.conj().T @ np.tensordot(xL, mu, 1) @ V
        My = V.conj().T @ np.tensordot(yL, mu, 1) @ V
        f = E[jj] - E[ii]
        w = (abs(Mx[ii, jj]) ** 2 + abs(My[ii, jj]) ** 2) / 2
        # salt's projection puts lines half a bin (dx/2) below their frequency;
        # EasySpin salt does the same (measured against this brute force), so the
        # reference is shifted by -dx/2.
        spec += (w[:, None] * np.exp(-0.5 * ((x[None, :] - f[:, None] + dx / 2) / sig) ** 2)).sum(0)
    xs, ys = salt(s, Experiment(Field=B, mwFreq=9.7), Options(GridSize=[31, 6], Enhancement=True),
                  freq_range=rng, n_points=n_points, lw_mhz=lw)
    ys = ys.numpy()
    cos = ys @ spec / np.linalg.norm(ys) / np.linalg.norm(spec)
    sc = (ys @ spec) / (ys @ ys)
    assert cos > 0.99999, cos
    assert np.abs(sc * ys - spec).max() / spec.max() < 5e-3


def test_salt_enhancement_bypasses_perturb_path():
    """1H, one electron, no Q: default uses first-order perturbation theory; Enhancement uses the matrix path."""
    s = _sys_1H()
    exp = Experiment(Field=326.5, mwFreq=9.7)
    _, y_off = salt(s, exp, Options(GridSize=[11, 3]), freq_range=(2, 25), n_points=300)
    _, y_on = salt(s, exp, Options(GridSize=[11, 3], Enhancement=True), freq_range=(2, 25), n_points=300)
    assert torch.isfinite(y_on).all() and y_on.max() > 0
    # first-order path: unit intensities; matrix path: |mu|^2 rates (MHz/mT)^2, far smaller
    assert float(y_on.max()) < 1e-2 * float(y_off.max())


def test_salt_field_swept_enhancement_runs():
    s = _sys_14N_1H()
    exp = Experiment(Range=[330, 350], mwFreq=9.55)
    _, y0 = salt(s, exp, Options(GridSize=7), freq_range=(0, 60), n_points=200)
    _, y1 = salt(s, exp, Options(GridSize=7, Enhancement=True), freq_range=(0, 60), n_points=200)
    assert torch.isfinite(y1).all() and y1.max() > 0
    assert not torch.allclose(y0 / y0.max(), y1 / y1.max())


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

def _endor_I(Av, gv=(2.0, 2.0, 2.0)):
    s = SpinSystem(S=[0.5], g=[list(gv)] if not torch.is_tensor(gv) else gv, Nucs='1H', A=Av)
    return endorfrq(s, 350.0, 0.3, 0.7, threshold=0.0, opt=ON)[1]


def test_endorfrq_enhanced_gradients_reverse_and_forward():
    A0 = [[3.0, 7.0, 12.0]]
    A = torch.tensor(A0, dtype=DT, requires_grad=True)
    I = _endor_I(A)
    J = torch.stack([torch.autograd.grad(I[k], A, retain_graph=True)[0].reshape(-1) for k in range(I.numel())])
    for c in range(3):
        best = math.inf
        for h in (1e-3, 1e-4, 1e-5):
            Ap = [list(A0[0])]; Am = [list(A0[0])]
            Ap[0][c] += h; Am[0][c] -= h
            fd = (_endor_I(Ap) - _endor_I(Am)) / (2 * h)
            best = min(best, float((J[:, c] - fd).abs().max() / fd.abs().max()))
        assert best < 1e-6, (c, best)
        # forward mode
        tang = torch.zeros(1, 3, dtype=DT); tang[0, c] = 1.0
        with fwAD.dual_level():
            Ad = fwAD.make_dual(torch.tensor(A0, dtype=DT), tang)
            jv = fwAD.unpack_dual(_endor_I(Ad)).tangent
        torch.testing.assert_close(jv, J[:, c], rtol=1e-9, atol=1e-14)


def test_endorfrq_enhanced_gradient_g():
    g0 = [2.0, 2.01, 2.02]
    g = torch.tensor([g0], dtype=DT, requires_grad=True)
    I = _endor_I([[3.0, 7.0, 12.0]], g)
    (dg,) = torch.autograd.grad(I.sum(), g)
    best = math.inf
    for h in (1e-5, 1e-6, 1e-7):
        gp = list(g0); gm = list(g0); gp[2] += h; gm[2] -= h
        fd = float((_endor_I([[3.0, 7.0, 12.0]], gp).sum() - _endor_I([[3.0, 7.0, 12.0]], gm).sum()) / (2 * h))
        best = min(best, abs(float(dg[0, 2]) - fd) / abs(fd))
    assert best < 1e-5, best


def test_salt_enhanced_gradients():
    """Fixed-field salt matrix path with enhancement: d spec / dA vs central differences."""
    exp = Experiment(Field=326.5, mwFreq=9.7)
    opt = Options(GridSize=[11, 3], Enhancement=True)
    A0 = [6.0, 9.0, 16.0]

    def run(Av):
        s = SpinSystem(S=[0.5], g=[[2.0, 2.01, 2.02]], Nucs='1H', A=Av)
        return salt(s, exp, opt, freq_range=(2, 25), n_points=256, lw_mhz=0.4)[1]
    A = torch.tensor([A0], dtype=DT, requires_grad=True)
    y = run(A)
    assert y.requires_grad
    w = torch.linspace(0.0, 1.0, y.numel(), dtype=DT)
    (g,) = torch.autograd.grad((w * y).sum(), A)
    for c in range(3):
        best = math.inf
        for h in (1e-4, 1e-5, 1e-6):
            Ap = [list(A0)]; Am = [list(A0)]
            Ap[0][c] += h; Am[0][c] -= h
            fd = float(((w * run(Ap)).sum() - (w * run(Am)).sum()) / (2 * h))
            best = min(best, abs(float(g[0, c]) - fd) / abs(fd))
        assert best < 1e-5, (c, best)


# ---------------------------------------------------------------------------
# EasySpin references
# ---------------------------------------------------------------------------

def _refs():
    try:
        from scipy.io import loadmat
    except ImportError:  # pragma: no cover
        pytest.skip("scipy not installed")
    path = os.path.join(DATA_DIR, "endor_enhancement_refs.mat")
    if not os.path.exists(path):
        pytest.skip(f"Reference file not found: {path}")
    return loadmat(path, simplify_cells=True)["ref"]


def _sys_from(S):
    kw = dict(S=[0.5], g=[np.atleast_1d(S['g']).astype(float).tolist()], Nucs=S['Nucs'],
              A=np.atleast_2d(S['A']).astype(float).tolist())
    if 'Q' in S:
        kw['Q'] = np.atleast_2d(S['Q']).astype(float).tolist()
    return SpinSystem(**kw)


def test_endorfrq_vs_easyspin():
    """EasySpin endorfrq (powder mode), Enhancement off and on: per-pair positions and intensities."""
    refs = _refs()["endorfrq"]
    assert len(refs) == 12
    for r in refs:
        s = _sys_from(r['Sys'])
        tr = np.atleast_2d(r['Tr']) - 1
        for opt, P_ref, I_ref in ((Options(), r['Poff'], r['Ioff']), (ON, r['Pon'], r['Ion'])):
            f, I, t = endorfrq(s, float(r['Field']), float(r['phi']), float(r['theta']), threshold=0.0, opt=opt)
            got = {tuple(x): (fi, ii) for x, fi, ii in zip(t.tolist(), f.tolist(), I.tolist())}
            P_ref, I_ref = np.atleast_1d(P_ref), np.atleast_1d(I_ref)
            ours = np.array([got[(u, v)] for u, v in tr])
            np.testing.assert_allclose(ours[:, 0], P_ref, rtol=0, atol=1e-9)
            assert np.abs(ours[:, 1] - I_ref).max() < 1e-10 * I_ref.max()


def test_salt_vs_easyspin():
    """EasySpin salt (fixed field), Enhancement off and on: shape and on/off scale."""
    for r in _refs()["salt"]:
        S, E = r['Sys'], r['Exp']
        s = _sys_from(S)
        scales = {}
        for enh, yref in ((False, r['yoff']), (True, r['yon'])):
            _, y = salt(s, Experiment(Field=float(E['Field']), mwFreq=float(E['mwFreq'])),
                        Options(GridSize=[31, 6], Threshold=1e-4, Enhancement=enh),
                        freq_range=tuple(E['Range']), n_points=int(E['nPoints']), lw_mhz=float(S['lwEndor']))
            y, yref = y.numpy(), np.asarray(yref, float)
            cos = abs(y @ yref) / np.linalg.norm(y) / np.linalg.norm(yref)
            sc = (y @ yref) / (y @ y)
            scales[enh] = sc
            if enh or 'Q' in S:     # matrix path (the default 1H path is first-order perturbation theory)
                assert cos > 0.99995, (S['Nucs'], enh, cos)
                assert np.abs(sc * y - yref).max() / np.abs(yref).max() < 1e-2
        if 'Q' in S:   # both settings on the matrix path: on/off intensity scale as in EasySpin
            assert abs(scales[True] / scales[False] - 1) < 1e-3
