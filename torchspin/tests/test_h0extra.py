"""SpinSystem.H0Extra: generic extra field-independent operator terms.

EasySpin has no generic extra-operator field, so there is no EasySpin
reference. The references are (1) native torchspin terms written out as
operators (zero-field splitting, hyperfine S.A.I), which must reproduce the
native levels, resonance fields and pepper spectra, and (2) a hand-built
Hamiltonian (sop + physical constants, no ham()) for a U-type term
U sum_k C_k(S) I_k that torchspin has no native field for.
"""
import dataclasses
import math

import pytest
import torch
import torch.autograd.forward_ad as fwAD

from torchspin import (SpinSystem, Experiment, Options, ham, levels, pepper, sop,
                       hamsymm, garlic, salt, nucspinrmv, nucspinkeep, nucspinadd)
from torchspin.constants import BMAGN, NMAGN, PLANCK
from torchspin.autograd import inputs_require_grad
from torchspin.pepper_autograd import pepper_autograd
from torchspin.resfields_batch import resfields_batch

C128 = torch.complex128


def _ops(spins, i):
    """Sx, Sy, Sz of spin i (0-based) in the full basis."""
    return [sop(spins, [[i + 1, c]]) for c in (1, 2, 3)]


def _zfs_op(spins, i, D, E):
    """D (Sz^2 - S(S+1)/3) + E (Sx^2 - Sy^2) for electron i (operator, MHz)."""
    Sx, Sy, Sz = _ops(spins, i)
    S = spins[i]
    eye = torch.eye(Sz.shape[0], dtype=C128)
    return D * (Sz @ Sz - S * (S + 1) / 3 * eye) + E * (Sx @ Sx - Sy @ Sy)


def _hf_op(spins, ie, iN, Amat):
    """S . A . I with a 3x3 A (MHz) between electron ie and spin iN."""
    S = _ops(spins, ie)
    I = _ops(spins, iN)
    return sum(float(Amat[a][b]) * (S[a] @ I[b]) for a in range(3) for b in range(3))


def _orientations():
    return [(0.0, 0.0), (0.3, 0.7), (1.1, 1.4), (2.5, math.pi / 2), (4.0, 2.2)]


# ---------------------------------------------------------------------------
# default / normalization
# ---------------------------------------------------------------------------

def test_default_none_and_ham_unchanged():
    kw = dict(S=[1.0], g=[[2.0, 2.01, 2.02]], D=[[900.0, 120.0]], Nucs='14N', A=[[5, 6, 9]])
    s0 = SpinSystem(**kw)
    s1 = SpinSystem(**kw, H0Extra=None)
    s2 = SpinSystem(**kw, H0Extra=[])
    assert s0.H0Extra is None and s1.H0Extra is None and s2.H0Extra is None
    for a, b in zip(ham(s0), ham(s2)):
        assert torch.equal(a, b)
    assert torch.equal(ham(s0, B0=[10.0, 20.0, 300.0]), ham(s2, B0=[10.0, 20.0, 300.0]))


def test_zero_coefficient_term_is_exact_noop_in_ham():
    kw = dict(S=[1.0], g=[[2.0, 2.01, 2.02]], D=[[900.0, 120.0]])
    s0 = SpinSystem(**kw)
    op = _zfs_op(s0.Spins, 0, 1.0, 0.3)
    s1 = SpinSystem(**kw, H0Extra=[(0.0, op)])
    for a, b in zip(ham(s0), ham(s1)):
        assert torch.equal(a, b)


def test_normalization_idempotent_and_graph_kept():
    c = torch.tensor(250.0, requires_grad=True)
    sysA = SpinSystem(S=[1.0], g=2.0)
    op = _zfs_op(sysA.Spins, 0, 1.0, 0.0).real.numpy()     # real numpy array input
    s = SpinSystem(S=[1.0], g=2.0, H0Extra=[(c, op), (7.5, torch.as_tensor(op))])
    (c0, o0), (c1, o1) = s.H0Extra
    assert c0.shape == () and c0.dtype == torch.float64 and c0.requires_grad
    assert o0.dtype == C128 and o0.shape == (3, 3)
    assert float(c1) == 7.5
    s2 = dataclasses.replace(s, weight=0.5)                 # re-runs __post_init__
    assert len(s2.H0Extra) == 2
    assert torch.equal(s2.H0Extra[0][1], o0) and float(s2.H0Extra[1][0]) == 7.5
    H0 = ham(s2)[0]
    E = torch.linalg.eigvalsh(H0)
    E[-1].backward()
    assert c.grad is not None and abs(float(c.grad)) > 0


# ---------------------------------------------------------------------------
# native-term equivalence: levels, resfields, pepper
# ---------------------------------------------------------------------------

D_E = (1800.0, 300.0)
A_NIT = [[18.0, 21.0, 92.0]]


def _pairs_triplet():
    native = SpinSystem(S=[1.0], g=[[2.002, 2.004, 2.006]], D=[list(D_E)], lw=[1.5, 0])
    op = _zfs_op(native.Spins, 0, *D_E)
    extra = SpinSystem(S=[1.0], g=[[2.002, 2.004, 2.006]], H0Extra=[(1.0, op)], lw=[1.5, 0])
    return native, extra


def _pairs_nitroxide():
    native = SpinSystem(S=[0.5], g=[[2.008, 2.006, 2.003]], Nucs='14N', A=A_NIT, lw=[0.3, 0.1])
    Amat = torch.diag(torch.tensor(A_NIT[0], dtype=torch.float64))
    op = _hf_op(native.Spins, 0, 1, Amat)
    # hyperfine written as operator with a gradient-carrying unit coefficient
    extra = SpinSystem(S=[0.5], g=[[2.008, 2.006, 2.003]], Nucs='14N', A=[[0.0, 0.0, 0.0]],
                       H0Extra=[(0.5, op), (0.5, op)], lw=[0.3, 0.1])
    return native, extra


PAIRS = {'zfs_triplet': (_pairs_triplet, [200.0, 450.0]),
         'hf_nitroxide': (_pairs_nitroxide, [325.0, 355.0])}


@pytest.mark.parametrize('name', list(PAIRS))
def test_levels_native_equivalence(name):
    native, extra = PAIRS[name][0]()
    B = torch.linspace(0.0, 600.0, 13)
    for ph, th in _orientations():
        _, En = levels(native, ph, th, B)
        _, Ex = levels(extra, ph, th, B)
        scale = float(En.abs().max())
        assert float((En - Ex).abs().max()) < 1e-12 * max(scale, 1.0), name
    # ham(sys, B0) path too
    Bv = [40.0, -70.0, 310.0]
    Hn, Hx = ham(native, B0=Bv), ham(extra, B0=Bv)
    assert float((Hn - Hx).abs().max()) < 1e-12 * float(Hn.abs().max())


@pytest.mark.parametrize('name', list(PAIRS))
def test_resfields_native_equivalence(name):
    build, rng = PAIRS[name]
    native, extra = build()
    exp = Experiment(mwFreq=9.5, Range=rng, nPoints=1024)
    opt = Options(Verbosity=0)
    phi = torch.tensor([o[0] for o in _orientations()], dtype=torch.float64)
    th = torch.tensor([o[1] for o in _orientations()], dtype=torch.float64)
    Bn, In, _ = resfields_batch(*ham(native), phi, th, exp, opt, sys=native)
    Bx, Ix, _ = resfields_batch(*ham(extra), phi, th, exp, opt, sys=extra)
    n_lines = 0
    for bn, bx, i_n, i_x in zip(Bn, Bx, In, Ix):
        assert bn.shape == bx.shape
        bn_s, kn = torch.sort(bn)
        bx_s, kx = torch.sort(bx)
        assert float((bn_s - bx_s).abs().max()) < 1e-9
        assert float((i_n[kn] - i_x[kx]).abs().max()) < 1e-9 * float(i_n.abs().max())
        n_lines += bn.numel()
    assert n_lines > 0


@pytest.mark.parametrize('name', list(PAIRS))
def test_pepper_native_equivalence(name):
    build, rng = PAIRS[name]
    native, extra = build()
    exp = Experiment(mwFreq=9.5, Range=rng, nPoints=1024)
    # Fixed grid symmetry: hamsymm sends H0Extra through the eigenvalue probe,
    # which may pick a different (lower) group than the geometric analysis.
    opt = Options(Verbosity=0, GridSize=[19, 4], GridSymmetry='D2h')
    _, yn = pepper(native, exp, opt)
    _, yx = pepper(extra, exp, opt)
    rel = float((yn - yx).abs().max() / yn.abs().max())
    assert rel < 1e-10, f'{name}: {rel:.2e}'


# ---------------------------------------------------------------------------
# U-type term vs a hand-built Hamiltonian (no ham())
# ---------------------------------------------------------------------------

def _u_operator(spins):
    S = spins[0]
    X = S * (S + 1)
    Sx, Sy, Sz = _ops(spins, 0)
    Ix, Iy, Iz = _ops(spins, 1)
    C = [Sk @ Sk @ Sk - (3 * X - 1) / 5 * Sk for Sk in (Sx, Sy, Sz)]
    return C[0] @ Ix + C[1] @ Iy + C[2] @ Iz


def test_u_term_matches_hand_built_hamiltonian():
    g, D, E, a = 2.003, 2500.0, 400.0, 30.0
    U = 12.0
    spins = [1.5, 0.5]
    sysU = SpinSystem(S=[1.5], g=g, D=[[D, E]], Nucs='1H', A=[[a, a, a]],
                      H0Extra=[(U, _u_operator(spins))])
    sys0 = dataclasses.replace(sysU, H0Extra=None)
    Sx, Sy, Sz = _ops(spins, 0)
    Ix, Iy, Iz = _ops(spins, 1)
    gn = sysU.gn[0]
    H0_hand = _zfs_op(spins, 0, D, E) + a * (Sx @ Ix + Sy @ Iy + Sz @ Iz) + U * _u_operator(spins)
    B = torch.linspace(0.0, 500.0, 11)
    shifted = False
    for ph, th in _orientations():
        n = (math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th))
        SL = n[0] * Sx + n[1] * Sy + n[2] * Sz
        IL = n[0] * Ix + n[1] * Iy + n[2] * Iz
        _, EU = levels(sysU, ph, th, B)
        _, E0 = levels(sys0, ph, th, B)
        for k, b in enumerate(B.tolist()):
            Hh = (H0_hand + BMAGN / PLANCK * 1e-9 * g * b * SL
                  - NMAGN / PLANCK * 1e-9 * gn * b * IL)
            Eh = torch.linalg.eigvalsh(Hh)
            assert float((EU[:, k] - Eh).abs().max()) < 1e-12 * float(Eh.abs().max())
        shifted = shifted or float((EU - E0).abs().max()) > 1.0
    assert shifted   # the U term visibly moves the levels


# ---------------------------------------------------------------------------
# gradients wrt coeff
# ---------------------------------------------------------------------------

def _u_levels(U, spins=(1.5, 0.5)):
    sysU = SpinSystem(S=[1.5], g=2.003, D=[[2500.0, 400.0]], Nucs='1H', A=[[30, 30, 30]],
                      H0Extra=[(U, _u_operator(list(spins)))])
    _, E = levels(sysU, 0.4, 0.9, torch.tensor([60.0, 120.0, 340.0]))
    return E


def _central(f, x0, steps=(1e-2, 3e-3, 1e-3, 3e-4)):
    return [(f(x0 + h) - f(x0 - h)) / (2 * h) for h in steps]


def test_levels_gradient_reverse_and_forward_vs_fd():
    U0 = 12.0
    w = torch.linspace(0.5, 1.5, 24).reshape(8, 3)

    def f(u):
        return float((w * _u_levels(torch.tensor(u, dtype=torch.float64))).sum())

    U = torch.tensor(U0, dtype=torch.float64, requires_grad=True)
    (w * _u_levels(U)).sum().backward()
    g_rev = float(U.grad)
    fd = _central(f, U0)
    err = min(abs(g_rev - d) for d in fd)
    assert err < 1e-6 * max(1.0, abs(g_rev)), (g_rev, fd)
    assert abs(g_rev) > 1e-3

    with fwAD.dual_level():
        Ud = fwAD.make_dual(torch.tensor(U0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
        E = _u_levels(Ud)
        g_fwd = float(fwAD.unpack_dual((w * E).sum()).tangent)
    assert abs(g_fwd - g_rev) < 1e-9 * max(1.0, abs(g_rev))


def test_pepper_routes_to_autograd_and_gradient_vs_fd():
    native, _ = _pairs_triplet()
    op = _zfs_op(native.Spins, 0, 1.0, 0.0)        # extra axial ZFS on top of native D
    exp = Experiment(mwFreq=9.5, Range=[150.0, 500.0], nPoints=512)
    opt = Options(Verbosity=0, GridSize=[19, 4], GridSymmetry='D2h')
    w = torch.linspace(-1.0, 1.0, 512) ** 2

    def build(c):
        return dataclasses.replace(native, H0Extra=[(c, op)])

    c0 = 150.0
    c = torch.tensor(c0, dtype=torch.float64, requires_grad=True)
    s = build(c)
    assert inputs_require_grad(s)
    assert not inputs_require_grad(build(c0))
    _, y = pepper(s, exp, opt)                   # routed to pepper_autograd
    assert y.requires_grad
    _, y2 = pepper_autograd(s, exp, opt)
    _, y_ng = pepper(build(c0), exp, opt)          # non-differentiable path
    assert float((y.detach() - y_ng).abs().max() / y_ng.abs().max()) < 1e-9
    (w * y2).sum().backward()
    g_rev = float(c.grad)

    def f(cc):
        return float((w * pepper(build(cc), exp, opt)[1]).sum())

    fd = _central(f, c0, steps=(1.0, 0.3, 0.1))
    err = min(abs(g_rev - d) for d in fd)
    assert err < 2e-3 * max(abs(g_rev), 1e-12), (g_rev, fd)


# ---------------------------------------------------------------------------
# hamsymm
# ---------------------------------------------------------------------------

def test_hamsymm_eig_probe_and_lower_symmetry():
    axial = SpinSystem(S=[1.0], g=2.0, D=[[1000.0, 0.0]])
    assert hamsymm(axial)[0] == 'Dinfh'
    spins = axial.Spins
    Sx, Sy, Sz = _ops(spins, 0)
    # rhombic extra term: D2h
    rh = dataclasses.replace(axial, H0Extra=[(150.0, Sx @ Sx - Sy @ Sy)])
    assert hamsymm(rh)[0] == 'D2h'
    # tilted term (Sx Sz + Sz Sx): monoclinic
    tilt = dataclasses.replace(axial, H0Extra=[(150.0, Sx @ Sz + Sz @ Sx)])
    assert hamsymm(tilt)[0] == 'C2h'
    # isotropic system: never the O3 shortcut once H0Extra is set
    iso = SpinSystem(S=[0.5], g=2.0)
    assert hamsymm(iso)[0] == 'O3'
    Sz12 = sop([0.5], [[1, 3]])
    iso_x = dataclasses.replace(iso, H0Extra=[(50.0, Sz12)])
    assert hamsymm(iso_x)[0] == 'Dinfh'
    # an isotropic extra term keeps O3, found by the eigenvalue probe
    iso_e = dataclasses.replace(iso, H0Extra=[(50.0, torch.eye(2, dtype=C128))])
    assert hamsymm(iso_e)[0] == 'O3'


# ---------------------------------------------------------------------------
# error cases
# ---------------------------------------------------------------------------

def _good_op():
    return _hf_op([0.5, 1.0], 0, 1, torch.eye(3, dtype=torch.float64))


def test_validation_errors():
    kw = dict(S=[0.5], Nucs='14N', A=[[1, 1, 1]])
    with pytest.raises(ValueError, match='shape'):
        SpinSystem(**kw, H0Extra=[(1.0, torch.eye(4, dtype=C128))])
    nh = torch.zeros(6, 6, dtype=C128); nh[0, 1] = 1.0
    with pytest.raises(ValueError, match='Hermitian'):
        SpinSystem(**kw, H0Extra=[(1.0, nh)])
    with pytest.raises(ValueError, match='real'):
        SpinSystem(**kw, H0Extra=[(1.0 + 1j, _good_op())])
    with pytest.raises(ValueError, match='scalar'):
        SpinSystem(**kw, H0Extra=[(torch.ones(2), _good_op())])
    with pytest.raises(TypeError):
        SpinSystem(**kw, H0Extra=(1.0, _good_op()))
    with pytest.raises(TypeError):
        SpinSystem(**kw, H0Extra=[(1.0, _good_op(), 3)])
    with pytest.raises(ValueError, match='isotope mixtures'):
        SpinSystem(S=[0.5], Nucs='Cu', A=[[1, 1, 1]], H0Extra=[(1.0, torch.eye(8, dtype=C128))])
    with pytest.raises(ValueError, match='equivalent'):
        SpinSystem(S=[0.5], Nucs='1H', n=[2], A=[[1, 1, 1]], H0Extra=[(1.0, torch.eye(4, dtype=C128))])
    # Hermitian complex operator is accepted
    Sy = sop([0.5, 1.0], [[1, 2]])
    SpinSystem(**kw, H0Extra=[(1.0, Sy)])


def test_paths_without_ham_raise():
    s = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.1]], Nucs='14N', A=[[10, 10, 40]], lw=[0.5, 0],
                   H0Extra=[(0.5, _good_op())])
    exp = Experiment(mwFreq=9.5, Range=[310.0, 360.0], nPoints=256)
    for m in ('perturb', 'perturb1', 'perturb2'):
        with pytest.raises(ValueError, match='H0Extra'):
            pepper(s, exp, Options(Verbosity=0, Method=m, GridSize=7))
    with pytest.raises(ValueError, match='H0Extra'):
        pepper(s, exp, Options(Verbosity=0, Method='hybrid', GridSize=7))
    for fn in (lambda: nucspinrmv(s, 0), lambda: nucspinkeep(s, 0),
               lambda: nucspinadd(s, '1H', A=[1.0, 1.0, 1.0])):
        with pytest.raises(ValueError, match='H0Extra'):
            fn()
    with pytest.raises(ValueError, match='H0Extra'):
        garlic(s, exp)
    with pytest.raises(ValueError, match='H0Extra'):
        salt(s, Experiment(mwFreq=9.5, Field=340.0), Options(Verbosity=0, GridSize=7))
    from torchspin.resfreqs_perturb import resfreqs_perturb
    from torchspin.endorfrq_perturb import endorfrq_perturb
    from torchspin.resfields_perturb import resfields_perturb
    with pytest.raises(ValueError, match='H0Extra'):
        resfields_perturb(s, 0.0, 0.0, exp)
    with pytest.raises(ValueError, match='H0Extra'):
        resfreqs_perturb(s, 0.0, 0.0, Experiment(Field=340.0, mwRange=[9.0, 10.0]))
    with pytest.raises(ValueError, match='H0Extra'):
        endorfrq_perturb(s, 340.0)
    from torchspin.fastmotion import fastmotion
    with pytest.raises(ValueError, match='H0Extra'):
        fastmotion(s, 340.0, 1e-10)
    from torchspin.chili import chili
    with pytest.raises(ValueError, match='H0Extra'):
        chili(dataclasses.replace(s, tcorr=1e-9), exp)
    # matrix method works
    _, y = pepper(s, exp, Options(Verbosity=0, GridSize=7))
    assert torch.all(torch.isfinite(y)) and float(y.abs().max()) > 0
