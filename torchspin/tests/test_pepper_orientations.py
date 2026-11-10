"""Exp.Orientations / Exp.OrientationWeights: pepper on user-supplied field
directions (molecular frame) with chi-averaged transition rates (0.7.6).

Checks
------
* default (``None``) leaves pepper unchanged;
* one orientation equals a hand-built stick spectrum whose chi average is taken
  explicitly from two resfields_batch calls with B1 along the two lab axes
  perpendicular to B0;
* a Fibonacci quadrature of the sphere reproduces the powder spectrum (field and
  frequency sweeps), and a restricted theta band reproduces the matching part of
  an axial powder with closed-form field limits;
* pepper_autograd: forward equals pepper, gradients w.r.t. g and the weights
  agree with central differences of pepper;
* EasySpin: tests/data/ref_pepper_orientations.mat (EasySpin crystal spectra
  summed over two sample rotations about B0, an exact chi average);
* error cases.

Note on comparing with the powder: pepper's powder spectrum comes from the SOPHE
projection, which (like EasySpin's projectzones/projecttriangles) assigns bin k
the field interval [x_k, x_k + dx), while the stick/template paths (crystal, O3
and Orientations; EasySpin lisum1i) centre bin k on x_k.  The powder references
below are therefore computed on the axis shifted by -dx/2, which puts their bin
centres on the same fields.
"""
from pathlib import Path

import math

import numpy as np
import pytest
import torch
from scipy.io import loadmat
from scipy.special import erf

from torchspin import SpinSystem, Experiment, Options, pepper
from torchspin.ham import ham
from torchspin.resfields_batch import resfields_batch
from torchspin.pepper_autograd import pepper_autograd
from torchspin.constants import PLANCK, BMAGN

REF_FILE = Path(__file__).parent.parent.parent / "tests" / "data" / "ref_pepper_orientations.mat"


def fibonacci(n):
    """Fibonacci points on the sphere, [phi, theta]; equal solid angles 4π/n."""
    i = np.arange(n) + 0.5
    return np.stack([(math.pi * (1 + 5 ** 0.5) * i) % (2 * math.pi), np.arccos(1 - 2 * i / n)], axis=1)


def _relmax(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.abs(a - b).max() / np.abs(b).max())


def _halfbin_powder(sys, expk, opt=None, key='Range'):
    """Powder spectrum with bin centres on the fields of Experiment(**expk)."""
    r = expk[key]
    dx = (r[1] - r[0]) / (expk['nPoints'] - 1)
    e = Experiment(**{**expk, key: [r[0] - dx / 2, r[1] - dx / 2]})
    return pepper(sys, e, opt or Options(GridSize=[61, 8]))[1]


NITRO = dict(S=[0.5], g=[[2.0, 2.05, 2.2]], gFrame=[[0.2, 0.4, 0.1]], Nucs=['14N'], A=[[20, 30, 90]])


# ---------------------------------------------------------------------------
# default unchanged
# ---------------------------------------------------------------------------

def test_default_is_none_and_unchanged():
    e = Experiment(mwFreq=9.5, Range=[290, 350], nPoints=400)
    assert e.Orientations is None and e.OrientationWeights is None
    sys = SpinSystem(**NITRO, lw=[1.0, 0.0])
    _, a = pepper(sys, e)
    _, b = pepper(sys, Experiment(mwFreq=9.5, Range=[290, 350], nPoints=400, Orientations=None))
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# single orientation vs hand-built chi average
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('phi,theta', [(0.8, 1.0), (2.3, 0.3), (4.4, 2.0)])
def test_single_orientation_equals_hand_chi_average(phi, theta):
    lw = 0.5
    sys = SpinSystem(**NITRO, lw=[lw, 0.0])
    expk = dict(mwFreq=9.5, Range=[290, 350], nPoints=700, Harmonic=0)
    opt = Options(Threshold=0.0)
    x, spec = pepper(sys, Experiment(**expk, Orientations=[[phi, theta]]), opt)

    # by hand: rates with B1 along the two lab axes perpendicular to B0, averaged
    e = Experiment(**expk)
    H0, mx, my, mz = ham(sys)
    ph = torch.tensor([phi], dtype=torch.float64); th = torch.tensor([theta], dtype=torch.float64)
    xL = torch.tensor([[math.cos(theta) * math.cos(phi), math.cos(theta) * math.sin(phi), -math.sin(theta)]], dtype=torch.float64)
    yL = torch.tensor([[-math.sin(phi), math.cos(phi), 0.0]], dtype=torch.float64)
    Bx, Ix, _ = resfields_batch(H0, mx, my, mz, ph, th, e, opt, sys, xlab_batch=xL)
    By, Iy, _ = resfields_batch(H0, mx, my, mz, ph, th, e, opt, sys, xlab_batch=yL)
    Bc, Ic, _ = resfields_batch(H0, mx, my, mz, ph, th, e, opt, sys)
    assert torch.allclose(Bx[0], By[0], rtol=0, atol=1e-9)
    assert torch.allclose(Bx[0], Bc[0], rtol=0, atol=1e-9)
    I_avg = 0.5 * (Ix[0] + Iy[0]).numpy()
    assert np.allclose(I_avg, Ic[0].numpy(), rtol=1e-9, atol=0)
    Bn = Bx[0].numpy()
    xn = x.numpy(); dx = xn[1] - xn[0]
    sg = lw / math.sqrt(8 * math.log(2)) * math.sqrt(2)
    ref = np.zeros_like(xn)
    for b, i in zip(Bn, I_avg):
        ref += 0.5 * i * 4 * math.pi * (erf((xn + dx / 2 - b) / sg) - erf((xn - dx / 2 - b) / sg))
    ref = ref / dx * 2 * math.pi
    assert _relmax(spec, ref) < 1e-9


def test_scalar_pair_and_weights_scale():
    sys = SpinSystem(**NITRO, lw=[1.0, 0.0])
    expk = dict(mwFreq=9.5, Range=[290, 350], nPoints=400)
    _, a = pepper(sys, Experiment(**expk, Orientations=[0.4, 1.2]))
    _, b = pepper(sys, Experiment(**expk, Orientations=[[0.4, 1.2]], OrientationWeights=[0.25]))
    assert torch.allclose(0.25 * a, b, rtol=1e-12, atol=0)
    # default weights are 1/N
    ori = [[0.4, 1.2], [1.0, 0.5]]
    _, c = pepper(sys, Experiment(**expk, Orientations=ori))
    _, d = pepper(sys, Experiment(**expk, Orientations=ori, OrientationWeights=[0.5, 0.5]))
    assert torch.allclose(c, d, rtol=1e-12, atol=0)


def test_separate_orientations_rows_sum_to_total():
    sys = SpinSystem(**NITRO, lw=[1.0, 0.0])
    expk = dict(mwFreq=9.5, Range=[290, 350], nPoints=400, Orientations=[[0.4, 1.2], [1.0, 0.5], [3.0, 2.0]],
                OrientationWeights=[0.2, 0.3, 0.5])
    _, tot = pepper(sys, Experiment(**expk))
    _, rows = pepper(sys, Experiment(**expk), Options(separate='orientations'))
    assert rows.shape == (3, 400)
    assert torch.allclose(rows.sum(0), tot, rtol=0, atol=1e-10 * float(tot.abs().max()))


# ---------------------------------------------------------------------------
# quadratures of the sphere reproduce the powder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('harmonic', [0, 1])
def test_fibonacci_sphere_reproduces_powder_field(harmonic):
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.05, 2.2]], Nucs=['14N'], A=[[20, 30, 90]], lw=[1.0, 0.0])
    expk = dict(mwFreq=9.5, Range=[290, 350], nPoints=1024, Harmonic=harmonic)
    ref = _halfbin_powder(sys, expk)
    _, s = pepper(sys, Experiment(**expk, Orientations=fibonacci(20000)))
    # measured: 1.0e-4 (absorption), 2.1e-3 (first harmonic)
    assert _relmax(s, ref) < (3e-4 if harmonic == 0 else 3e-3)
    if harmonic == 0:
        assert abs(float(s.sum() / ref.sum()) - 1) < 1e-5


def test_fibonacci_sphere_reproduces_powder_freqsweep():
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.05, 2.2]], lw=[20.0, 0.0])
    expk = dict(Field=330, mwRange=[9.0, 10.5], nPoints=1000, Harmonic=0)
    ref = _halfbin_powder(sys, expk, key='mwRange')
    x, s = pepper(sys, Experiment(**expk, Orientations=fibonacci(20000)))
    assert float(x[0]) == 9.0 and float(x[-1]) == 10.5
    assert _relmax(s, ref) < 3e-4                     # measured 1.1e-4


def test_fibonacci_sphere_strain_close_to_powder():
    # The powder strain branch (EasySpin summation: facet Gaussians with
    # Lambda smoothing) is itself an approximation; agreement is ~0.7 %.
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.05, 2.2]], HStrain=[60, 80, 120])
    expk = dict(mwFreq=9.5, Range=[290, 350], nPoints=1024, Harmonic=0)
    ref = _halfbin_powder(sys, expk)
    _, s = pepper(sys, Experiment(**expk, Orientations=fibonacci(20000)))
    assert _relmax(s, ref) < 1.5e-2
    assert abs(float(s.sum() / ref.sum()) - 1) < 1e-4


def test_theta_band_axial_closed_form():
    gpar, gper = 2.2, 2.0
    sys0 = SpinSystem(S=[0.5], g=[[gper, gper, gpar]], lw=[0.0, 0.0])
    expk = dict(mwFreq=9.5, Range=[300, 350], nPoints=2000, Harmonic=0)
    t1, t2 = math.radians(30), math.radians(60)
    n = 20000
    c = math.cos(t2) + (np.arange(n) + 0.5) * (math.cos(t1) - math.cos(t2)) / n     # midpoints in cos(theta)
    th = np.concatenate([np.arccos(c), math.pi - np.arccos(c)])
    ori = np.stack([np.zeros(th.size), th], axis=1)
    w = np.full(th.size, (math.cos(t1) - math.cos(t2)) / n / 2)                       # dΩ/4π (axial: φ trivial)

    def Bres(t):
        return PLANCK * 9.5e9 / BMAGN / math.sqrt(gpar ** 2 * math.cos(t) ** 2 + gper ** 2 * math.sin(t) ** 2) * 1e3
    B1, B2 = Bres(t1), Bres(t2)
    x, s = pepper(sys0, Experiment(**expk, Orientations=ori, OrientationWeights=w))
    xn, sn = x.numpy(), s.numpy()
    dx = xn[1] - xn[0]
    nz = np.nonzero(sn > 1e-12 * sn.max())[0]
    assert abs(xn[nz[0]] - B1) <= dx and abs(xn[nz[-1]] - B2) <= dx

    # with broadening: equals the powder inside the band (away from its edges),
    # and the band complement adds up to the full sphere
    sys = SpinSystem(S=[0.5], g=[[gper, gper, gpar]], lw=[0.5, 0.0])
    _, sb = pepper(sys, Experiment(**expk, Orientations=ori, OrientationWeights=w))
    ref = _halfbin_powder(sys, expk).numpy()
    inside = (xn > B1 + 1.5) & (xn < B2 - 1.5)
    assert np.abs(sb.numpy()[inside] - ref[inside]).max() / np.abs(ref[inside]).max() < 3e-4   # measured ~1e-5
    outside = (xn < B1 - 2.0) | (xn > B2 + 2.0)
    assert np.abs(sb.numpy()[outside]).max() < 1e-6 * np.abs(sb.numpy()).max()


# ---------------------------------------------------------------------------
# pepper_autograd
# ---------------------------------------------------------------------------

ORI5 = np.array([[0.3, 0.7], [1.9, 2.2], [4.0, 1.1], [5.5, 0.4], [2.7, 1.6]])
W5 = np.array([0.1, 0.3, 0.2, 0.25, 0.15])


@pytest.mark.parametrize('kw', [dict(lw=[1.0, 0.0]), dict(lw=[0.5, 0.8]), dict(lw=[0.0, 0.8]),
                                dict(lw=[0.0, 0.0], HStrain=[30, 40, 60]), dict(lw=[0.0, 0.0])])
@pytest.mark.parametrize('harmonic', [0, 1, 2])
def test_autograd_forward_equals_pepper(kw, harmonic):
    sys = SpinSystem(**NITRO, **kw)
    e = Experiment(mwFreq=9.5, Range=[290, 350], nPoints=500, Harmonic=harmonic, Orientations=ORI5, OrientationWeights=W5)
    _, a = pepper(sys, e)
    _, b = pepper_autograd(sys, e)
    assert _relmax(b.detach(), a) < 1e-9


def _scalar(spec, proj):
    return float(spec @ proj)


def test_autograd_gradient_g_and_weights():
    proj = torch.from_numpy(np.random.default_rng(1).standard_normal(500))
    base = dict(S=[0.5], Nucs=['14N'], A=[[20, 30, 90]], lw=[1.0, 0.0])
    g0 = np.array([2.0, 2.05, 2.2])
    expk = dict(mwFreq=9.5, Range=[290, 350], nPoints=500, Harmonic=1, Orientations=ORI5)

    g = torch.tensor(g0, requires_grad=True)
    w = torch.tensor(W5, requires_grad=True)
    _, s = pepper(SpinSystem(**base, g=[g]), Experiment(**expk, OrientationWeights=w))   # routed to autograd
    assert s.requires_grad
    (s @ proj).backward()

    def f(gv, wv):
        return _scalar(pepper(SpinSystem(**base, g=[gv.tolist()]), Experiment(**expk, OrientationWeights=wv))[1], proj)
    for i in range(3):
        best = math.inf
        for h in (1e-5, 3e-6, 1e-6):
            gp, gm = g0.copy(), g0.copy(); gp[i] += h; gm[i] -= h
            fd = (f(gp, W5) - f(gm, W5)) / (2 * h)
            best = min(best, abs(fd - float(g.grad[i])) / max(abs(fd), 1e-12))
        assert best < 1e-4, (i, best)
    for i in range(5):
        h = 1e-4
        wp, wm = W5.copy(), W5.copy(); wp[i] += h; wm[i] -= h
        fd = (f(g0, wp) - f(g0, wm)) / (2 * h)
        assert abs(fd - float(w.grad[i])) <= 1e-7 * max(abs(fd), 1.0), (i, fd, float(w.grad[i]))


def test_autograd_gradient_hstrain_lw():
    proj = torch.from_numpy(np.random.default_rng(2).standard_normal(500))
    expk = dict(mwFreq=9.5, Range=[290, 350], nPoints=500, Harmonic=0, Orientations=ORI5, OrientationWeights=W5)
    hs0 = np.array([30.0, 40.0, 60.0])
    hs = torch.tensor(hs0, requires_grad=True)
    lwg = torch.tensor(0.6, requires_grad=True)
    _, s = pepper_autograd(SpinSystem(**NITRO, HStrain=hs, lw=[lwg, 0.0]), Experiment(**expk))
    (s @ proj).backward()

    def f(hv, lv):
        return _scalar(pepper(SpinSystem(**NITRO, HStrain=hv.tolist(), lw=[lv, 0.0]), Experiment(**expk))[1], proj)
    for i in range(3):
        h = 1e-4 * hs0[i]
        hp, hm = hs0.copy(), hs0.copy(); hp[i] += h; hm[i] -= h
        fd = (f(hp, 0.6) - f(hm, 0.6)) / (2 * h)
        assert abs(fd - float(hs.grad[i])) / max(abs(fd), 1e-12) < 1e-5
    fd = (f(hs0, 0.6 + 1e-5) - f(hs0, 0.6 - 1e-5)) / 2e-5
    assert abs(fd - float(lwg.grad)) / abs(fd) < 1e-5


# ---------------------------------------------------------------------------
# EasySpin: crystal spectra summed over two rotations about B0
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def es_cases():
    if not REF_FILE.exists():
        pytest.skip(f"Reference data not found: {REF_FILE}")
    refs = loadmat(str(REF_FILE), squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(refs['cases'])}


ES = {
    'nitro_voigt': (dict(**NITRO, lw=[0.7, 0.4]), dict(mwFreq=9.5, Range=[290, 350], nPoints=600, Harmonic=1), 5e-4),
    'hstrain_single': (dict(S=[0.5], g=[[2.0, 2.1, 2.2]], HStrain=[30, 60, 90]),
                       dict(mwFreq=9.5, Range=[290, 350], nPoints=800, Harmonic=0), 5e-4),
    # Lorentzian template: EasySpin's finite template truncates the tails
    'triplet_T': (dict(S=[1.0], g=[[2.0, 2.0, 2.0]], D=[[900, 150]], lw=[0.0, 1.5]),
                  dict(mwFreq=9.5, Range=[250, 430], nPoints=900, Harmonic=1, Temperature=20), 4e-3),
    'freqsweep': (dict(S=[0.5], g=[[2.0, 2.05, 2.2]], lw=[20, 0]),
                  dict(Field=330, mwRange=[9.0, 9.9], nPoints=700, Harmonic=0), 5e-4),
    'axial_fibonacci': (dict(S=[0.5], g=[[2.0, 2.0, 2.2]], lw=[1, 0]),
                        dict(mwFreq=9.5, Range=[300, 350], nPoints=1024, Harmonic=0), 5e-4),
}


@pytest.mark.parametrize('name', list(ES))
def test_easyspin_chi_averaged_crystal_sum(es_cases, name):
    sysk, expk, tol = ES[name]
    c = es_cases[name]
    ori = np.atleast_2d(np.asarray(c.ori, dtype=float))
    x, s = pepper(SpinSystem(**sysk), Experiment(**expk, Orientations=ori))
    assert np.abs(x.numpy() - np.asarray(c.x, dtype=float)).max() < 1e-9
    assert _relmax(s, c.spc) < tol


def test_easyspin_powder_halfbin_convention(es_cases):
    """EasySpin's own powder and its chi-averaged Fibonacci crystal sum differ by
    the half-bin offset of the projection (3 %), not after aligning the bins."""
    fib = np.asarray(es_cases['axial_fibonacci'].spc, dtype=float)
    assert _relmax(es_cases['axial_powder'].spc, fib) > 0.02
    assert _relmax(es_cases['axial_powder_halfbin'].spc, fib) < 6e-3


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('bad', [dict(Orientations=[[0.1, 0.2, 0.3]]), dict(Orientations=[0.1, 0.2, 0.3]),
                                 dict(Orientations=[[0.1, float('nan')]]),
                                 dict(Orientations=[[0.1, 0.2]], OrientationWeights=[0.5, 0.5]),
                                 dict(Orientations=[[0.1, 0.2]], OrientationWeights=[-1.0]),
                                 dict(Orientations=[[0.1, 0.2], [0.3, 0.4]], OrientationWeights=[0.0, 0.0]),
                                 dict(OrientationWeights=[1.0])])
def test_experiment_validation(bad):
    with pytest.raises(ValueError):
        Experiment(mwFreq=9.5, Range=[290, 350], **bad)


@pytest.mark.parametrize('conflict', [dict(Ordering=1.0), dict(SampleFrame=[[0.1, 0.2, 0.3]]),
                                      dict(SampleRotation=('z', [0.0, 1.0])), dict(MolFrame=[0, 0, 0]),
                                      dict(CrystalSymmetry='D2h')])
def test_conflicting_fields_raise(conflict):
    sys = SpinSystem(**NITRO, lw=[1.0, 0.0])
    e = Experiment(mwFreq=9.5, Range=[290, 350], nPoints=200, Orientations=[[0.1, 0.2]], **conflict)
    with pytest.raises(ValueError, match='Orientations'):
        pepper(sys, e)
    g = torch.tensor([2.0, 2.05, 2.2], requires_grad=True)
    with pytest.raises(ValueError, match='Orientations'):
        pepper_autograd(SpinSystem(S=[0.5], g=[g], lw=[1.0, 0.0]), e)


@pytest.mark.parametrize('opt', [Options(separate='transitions'), Options(separate='sites'), Options(Sites=[1])])
def test_unsupported_options_raise(opt):
    sys = SpinSystem(**NITRO, lw=[1.0, 0.0])
    e = Experiment(mwFreq=9.5, Range=[290, 350], nPoints=200, Orientations=[[0.1, 0.2]])
    with pytest.raises(ValueError, match='Orientations'):
        pepper(sys, e, opt)


def test_orientations_at_zero_field_match_the_powder():
    """Exp.Orientations with Options.ZeroField (0.7.4): a 4000-point Fibonacci sphere reproduces the zero-field
    powder (S = 1, D, E; 300 MHz Lorentzian; narrower lines differ by bin discretization, 7 % at 20 MHz)."""
    n = 4000
    i = np.arange(n) + 0.5
    ori = np.stack([(np.pi * (1 + 5 ** 0.5) * i) % (2 * np.pi), np.arccos(1 - 2 * i / n)], 1)
    sy = SpinSystem(S=1.0, g=[2, 2, 2], D=[-700, -1300, 2000], lw=[0, 300.0])
    a = pepper(sy, Experiment(Field=0.0, mwRange=[0, 5], nPoints=501), Options(ZeroField=True, GridSize=20))[1]
    b = pepper(sy, Experiment(Field=0.0, mwRange=[0, 5], nPoints=501, Orientations=ori), Options(ZeroField=True))[1]
    assert float((a - b).abs().max() / a.abs().max()) < 3e-3
