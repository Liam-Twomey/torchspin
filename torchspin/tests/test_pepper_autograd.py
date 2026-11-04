"""Tests for torchspin.pepper_autograd — pepper's forward path on the autograd graph.

Forward: identical to pepper() (matrix method, no strain) for several spin systems,
grid symmetries, GridSize forms and harmonics.
Gradients: autograd Jacobians agree with central finite differences of pepper()
(the reference model), per parameter (g, A, D, Gaussian and Lorentzian lw).
"""
import math

import pytest
import torch

from torchspin import SpinSystem, Experiment, Options, pepper
from torchspin.pepper_autograd import pepper_autograd, convspec_t
from torchspin.convspec import convspec

torch.set_default_dtype(torch.float64)

CASES = {
    'cu_axial':   (dict(S=[0.5], g=[[2.05, 2.05, 2.25]], Nucs='63Cu', A=[[30, 30, 617.3]], lw=[1.0, 0]), [260, 360]),
    'cu_rhombic': (dict(S=[0.5], g=[[2.04, 2.08, 2.25]], Nucs='63Cu', A=[[30, 30, 617.3]], lw=[1.0, 0]), [260, 360]),
    'nitroxide':  (dict(S=[0.5], g=[[2.008, 2.006, 2.003]], Nucs='14N', A=[[20, 20, 85]], lw=[0.3, 0.2]), [330, 350]),
    'triplet':    (dict(S=[1.0], g=[[2.0, 2.0, 2.0]], D=[[900.0, 150.0]], lw=[1.5, 0]), [250, 430]),
    'lwpp_ci':    (dict(S=[0.5], g=[[2.0, 2.1, 2.2]], gFrame=[[0.3, 0.5, 0.2]], Nucs='1H', A=[[2, 2, 8]], lwpp=[1.0]), [300, 380]),
}


def _cos(a, b):
    return float(a @ b / (a.norm() * b.norm()))


@pytest.mark.parametrize('name', list(CASES))
@pytest.mark.parametrize('grid', [[31, 4], 19])
@pytest.mark.parametrize('harmonic', [0, 1])
def test_forward_equals_pepper(name, grid, harmonic):
    sd, rng = CASES[name]
    sys = SpinSystem(**sd)
    exp = Experiment(mwFreq=9.5, Range=rng, nPoints=512, Harmonic=harmonic)
    _, yp = pepper(sys, exp, Options(Verbosity=0, GridSize=grid))
    x, ya = pepper_autograd(sys, exp, Options(Verbosity=0, GridSize=grid))
    assert x.shape == (512,) and torch.all(torch.isfinite(ya))
    rel = float((yp - ya).abs().max() / yp.abs().max())
    assert rel < 1e-9, f'{name} grid={grid} H={harmonic}: max rel deviation {rel:.2e}'
    assert _cos(yp, ya) > 1 - 1e-12


@pytest.mark.parametrize('fg,fl,deriv', [(1.0, 0.0, 0), (1.0, 0.0, 1), (0.7, 0.3, 1), (0.0, 0.5, 2), (0.0, 0.0, 1)])
def test_convspec_t_matches_numpy(fg, fl, deriv):
    s = torch.zeros(1024)
    s[300] = 1.0; s[500:520] = 0.3; s[900] = 2.0
    a = convspec(s, 0.1, fwhm_g=fg, fwhm_l=fl, deriv=deriv)
    b = convspec_t(s, 0.1, fg, fl, deriv)
    assert float((a - b).abs().max() / a.abs().max()) < 1e-12


def _fd_pepper(build, base, name, i, h, exp, opt):
    p = {k: [list(r) if isinstance(r, (list, tuple)) else r for r in v] if isinstance(v, list) else v for k, v in base.items()}
    def _set(delta):
        q = {k: (list(map(list, v)) if isinstance(v[0], list) else list(v)) for k, v in base.items()}
        row = q[name]
        if isinstance(row[0], list):
            row[0][i] += delta
        else:
            row[i] += delta
        return q
    yp = pepper(build(_set(+h)), exp, opt)[1]
    ym = pepper(build(_set(-h)), exp, opt)[1]
    return (yp - ym) / (2 * h)


def _jacobian_rows(y, params, step):
    rows = []
    for k in range(0, y.numel(), step):
        e = torch.zeros(y.numel()); e[k] = 1.0
        rows.append(torch.cat([g.reshape(-1) for g in torch.autograd.grad(y, params, grad_outputs=e, retain_graph=True)]))
    return torch.stack(rows)


def test_gradients_vs_finite_differences_of_pepper_cu():
    """Cu(II) rhombic, interpolated grid, first derivative: d spec/d(g, A, lw)."""
    exp = Experiment(mwFreq=9.5, Range=[260, 360], nPoints=512, Harmonic=1)
    opt = Options(Verbosity=0, GridSize=[19, 4])
    g = torch.tensor([2.04, 2.08, 2.25], requires_grad=True)
    A = torch.tensor([[30.0, 30.0, 617.3]], requires_grad=True)
    lw = torch.tensor(1.0, requires_grad=True)
    sys = SpinSystem(S=[0.5], g=[g], Nucs='63Cu', A=A, lw=[lw, 0.0])
    _, y = pepper_autograd(sys, exp, opt)
    step = 4
    J = _jacobian_rows(y, [g, A, lw], step)
    base = dict(g=[2.04, 2.08, 2.25], A=[[30.0, 30.0, 617.3]], lw=[1.0, 0.0])

    def build(q):
        return SpinSystem(S=[0.5], g=[q['g']], Nucs='63Cu', A=q['A'], lw=q['lw'])
    # A and lw: smooth → tight agreement.  g shifts every line across bin edges of
    # the piecewise-linear projection, so the finite difference needs a small step.
    cols = [('g', 0, 1e-7, 1e-5), ('g', 2, 1e-7, 1e-5), ('A', 0, 1e-3, 1e-6), ('A', 2, 1e-3, 1e-6), ('lw', 0, 1e-4, 1e-6)]
    idx = {('g', 0): 0, ('g', 2): 2, ('A', 0): 3, ('A', 2): 5, ('lw', 0): 6}
    for name, i, h, tol in cols:
        fd = _fd_pepper(build, base, name, i, h, exp, opt)[::step]
        ag = J[:, idx[(name, i)]]
        rel = float((ag - fd).norm() / fd.norm())
        assert rel < tol, f'd/d{name}[{i}] relative norm error {rel:.2e} (tol {tol})'
        assert _cos(ag, fd) > 1 - 1e-6


def test_gradients_vs_finite_differences_triplet_and_lorentzian():
    """S=1 zero-field splitting and a Gaussian+Lorentzian line width (Dinfh grid)."""
    exp = Experiment(mwFreq=9.5, Range=[250, 430], nPoints=512, Harmonic=1)
    opt = Options(Verbosity=0, GridSize=[19, 4])
    D = torch.tensor([[900.0, 150.0]], requires_grad=True)
    lw = torch.tensor([1.5, 0.4], requires_grad=True)
    sys = SpinSystem(S=[1.0], g=[[2.0, 2.0, 2.0]], D=D, lw=[lw[0], lw[1]])
    _, y = pepper_autograd(sys, exp, opt)
    J = _jacobian_rows(y, [D, lw], 4)
    base = dict(D=[[900.0, 150.0]], lw=[1.5, 0.4])

    def build(q):
        return SpinSystem(S=[1.0], g=[[2.0, 2.0, 2.0]], D=q['D'], lw=q['lw'])
    for name, i, col, h in (('D', 0, 0, 1e-3), ('D', 1, 1, 1e-3), ('lw', 0, 2, 1e-4), ('lw', 1, 3, 1e-4)):
        fd = _fd_pepper(build, base, name, i, h, exp, opt)[::4]
        rel = float((J[:, col] - fd).norm() / fd.norm())
        assert rel < 1e-6, f'd/d{name}[{i}] relative norm error {rel:.2e}'


def test_spinsystem_keeps_graph_tensors():
    g = torch.tensor([2.0, 2.1, 2.2], requires_grad=True)
    A = torch.tensor([[30.0, 30.0, 600.0]], requires_grad=True)
    sys = SpinSystem(S=[0.5], g=[g], Nucs='63Cu', A=[[A[0, 0], A[0, 1], A[0, 2]]])
    assert sys.g.requires_grad and sys.A.requires_grad
    assert torch.equal(sys.A.detach(), A.detach())


def test_differentiable_spectrum_routes_to_pepper():
    """differentiable_spectrum (default method) equals pepper for a hyperfine system."""
    from torchspin.autograd import differentiable_spectrum
    g = torch.tensor([2.0, 2.1, 2.2], requires_grad=True)
    A = torch.tensor([[2.0, 2.0, 8.0]], requires_grad=True)
    B, spec = differentiable_spectrum(g, lw_mT=1.0, mwFreq_GHz=9.5, B_range=(300, 380), nPoints=512,
                                      GridSize=[19, 4], Nucs=['1H'], A=A, Harmonic=1)
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.1, 2.2]], Nucs='1H', A=[[2.0, 2.0, 8.0]], lw=[1.0, 0])
    _, yp = pepper(sys, Experiment(mwFreq=9.5, Range=[300, 380], nPoints=512, Harmonic=1), Options(GridSize=[19, 4]))
    assert float((spec - yp).abs().max() / yp.abs().max()) < 1e-9
    gg, gA = torch.autograd.grad(spec.sum(), [g, A])
    assert torch.all(torch.isfinite(gg)) and torch.all(torch.isfinite(gA))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_cuda_matches_cpu():
    sd, rng = CASES['nitroxide']
    sys = SpinSystem(**sd)
    exp = Experiment(mwFreq=9.5, Range=rng, nPoints=512, Harmonic=1)
    _, yc = pepper_autograd(sys, exp, Options(Verbosity=0, GridSize=[19, 4]))
    _, yg = pepper_autograd(sys, exp, Options(Verbosity=0, GridSize=[19, 4], device='cuda'))
    assert float((yc - yg.cpu()).abs().max() / yc.abs().max()) < 1e-8


def test_gradient_at_exactly_axial_point():
    """A_x == A_y ties triangle vertices along φ (zero-width ramps); the gradient
    with respect to the tie-lifting parameters must still match finite differences."""
    exp = Experiment(mwFreq=9.5, Range=[260, 360], nPoints=512, Harmonic=1)
    opt = Options(Verbosity=0, GridSize=[19, 4], GridSymmetry='D2h')
    A0 = [30.0, 30.0, 617.3]
    A = torch.tensor([A0], requires_grad=True)
    _, y = pepper_autograd(SpinSystem(S=[0.5], g=[[2.05, 2.05, 2.25]], Nucs='63Cu', A=A, lw=[1.0, 0]), exp, opt)
    J = _jacobian_rows(y, [A], 4)

    def f(d, k):
        Av = [list(A0)]; Av[0][k] += d
        return pepper(SpinSystem(S=[0.5], g=[[2.05, 2.05, 2.25]], Nucs='63Cu', A=Av, lw=[1.0, 0]), exp, opt)[1][::4]
    h = 1e-4
    for k in (0, 1):
        fd = (f(h, k) - f(-h, k)) / (2 * h)
        rel = float((J[:, k] - fd).norm() / fd.norm())
        assert rel < 1e-5, f'd/dA[{k}] at A_x == A_y: relative norm error {rel:.2e}'


def test_eigh_degeneracy_safe_backward():
    """torchspin._linalg.eigh: gradient of a gauge-invariant function through a
    degenerate Hermitian matrix matches finite differences."""
    from torchspin._linalg import eigh
    torch.manual_seed(1)
    H0 = torch.diag(torch.tensor([1.0, 1.0, 1.0, 3.0], dtype=torch.complex128))
    P = torch.randn(4, 4, dtype=torch.complex128); Pm = P + P.conj().T

    def f(t):
        L, V = eigh(H0 + t * Pm)
        M = V.conj().T @ Pm @ V
        return (M.abs() ** 2).sum() + (L ** 2).sum()
    t = torch.tensor(0.0, requires_grad=True)
    (g,) = torch.autograd.grad(f(t), t)
    h = 1e-6
    fd = (f(torch.tensor(h)) - f(torch.tensor(-h))) / (2 * h)
    assert torch.isfinite(g) and abs(float(g - fd)) < 1e-6 * max(1.0, abs(float(fd)))


@pytest.mark.parametrize('name', ['nitrox_HgStrain', 'axial_gStrain_Dinfh', 'AStrain'])
def test_strain_forward_equals_pepper(name):
    sd, rng = {
        'nitrox_HgStrain': (dict(S=[0.5], g=[[2.008, 2.006, 2.003]], Nucs='14N', A=[[20, 20, 85]],
                                 gStrain=[[0.003, 0.002, 0.001]], HStrain=[10, 10, 30], lwpp=[0.3, 0]), [330, 350]),
        'axial_gStrain_Dinfh': (dict(S=[0.5], g=[[2.05, 2.05, 2.25]], Nucs='63Cu', A=[[30, 30, 500]],
                                     gStrain=[[0.01, 0.01, 0.02]], lw=[0.5, 0]), [260, 360]),
        'AStrain': (dict(S=[0.5], g=[[2.008, 2.006, 2.003]], Nucs='14N', A=[[20, 20, 85]], AStrain=[2, 2, 8], lw=[0.2, 0]), [330, 350]),
    }[name]
    for H in (0, 1):
        sys = SpinSystem(**sd)
        exp = Experiment(mwFreq=9.5, Range=rng, nPoints=512, Harmonic=H)
        _, yp = pepper(sys, exp, Options(Verbosity=0, GridSize=[19, 4]))
        _, ya = pepper_autograd(sys, exp, Options(Verbosity=0, GridSize=[19, 4]))
        assert float((yp - ya).abs().max() / yp.abs().max()) < 1e-10


def test_strain_gradients_vs_finite_differences_of_pepper():
    """Nitroxide with HStrain and gStrain (summation branch): d spec / d(HStrain, gStrain, A, g)."""
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=512, Harmonic=1)
    opt = Options(Verbosity=0, GridSize=[19, 4], GridSymmetry='D2h')
    g0, A0, gS0, HS0 = [2.008, 2.006, 2.003], [20.0, 20.0, 85.0], [0.003, 0.002, 0.001], [10.0, 12.0, 30.0]
    g = torch.tensor(g0, requires_grad=True); A = torch.tensor([A0], requires_grad=True)
    gS = torch.tensor([gS0], requires_grad=True); HS = torch.tensor(HS0, requires_grad=True)
    sys = SpinSystem(S=[0.5], g=[g], Nucs='14N', A=A, gStrain=gS, HStrain=HS, lw=[0.2, 0.0])
    _, y = pepper_autograd(sys, exp, opt)
    J = _jacobian_rows(y, [g, A, gS, HS], 4)

    def f(name, i, d):
        gv, Av, gv2, Hv = list(g0), list(A0), list(gS0), list(HS0)
        {'g': gv, 'A': Av, 'gS': gv2, 'HS': Hv}[name][i] += d
        return pepper(SpinSystem(S=[0.5], g=[gv], Nucs='14N', A=[Av], gStrain=[gv2], HStrain=Hv, lw=[0.2, 0.0]), exp, opt)[1][::4]
    for name, i, col, h, tol in (('g', 2, 2, 1e-7, 1e-5), ('A', 2, 5, 1e-4, 1e-6), ('gS', 2, 8, 1e-6, 1e-6), ('HS', 2, 11, 1e-4, 1e-6)):
        fd = (f(name, i, h) - f(name, i, -h)) / (2 * h)
        rel = float((J[:, col] - fd).norm() / fd.norm())
        assert rel < tol, f'd/d{name}[{i}]: relative norm error {rel:.2e}'


def test_perturbation_route_forward_and_gradients():
    """Opt.Method='perturb2' uses pepper's perturbation resonance route (index-matched slots)."""
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=512, Harmonic=1)
    opt = Options(Verbosity=0, GridSize=[19, 4], Method='perturb2', GridSymmetry='D2h')
    A0, g0 = [20.0, 21.0, 85.0], [2.008, 2.006, 2.003]
    _, yp = pepper(SpinSystem(S=[0.5], g=[g0], Nucs='14N', A=[A0], lw=[0.3, 0.2]), exp, opt)
    A = torch.tensor([A0], requires_grad=True); g = torch.tensor(g0, requires_grad=True)
    _, y = pepper_autograd(SpinSystem(S=[0.5], g=[g], Nucs='14N', A=A, lw=[0.3, 0.2]), exp, opt)
    assert float((yp - y).abs().max() / yp.abs().max()) < 1e-9
    J = _jacobian_rows(y, [A, g], 4)
    for name, i, col, h in (('A', 2, 2, 1e-4), ('A', 0, 0, 1e-4), ('g', 2, 5, 1e-7)):
        def f(d):
            Av, gv = list(A0), list(g0)
            {'A': Av, 'g': gv}[name][i] += d
            return pepper(SpinSystem(S=[0.5], g=[gv], Nucs='14N', A=[Av], lw=[0.3, 0.2]), exp, opt)[1][::4]
        fd = (f(h) - f(-h)) / (2 * h)
        rel = float((J[:, col] - fd).norm() / fd.norm())
        assert rel < 1e-6, f'perturb2 d/d{name}[{i}]: {rel:.2e}'


def test_field_modulation_forward_and_gradient():
    """Exp.ModAmp: pseudo-modulation in torch (fieldmod_t) equals pepper; gradient vs FD."""
    sd = dict(S=[0.5], g=[[2.008, 2.006, 2.003]], Nucs='14N', lw=[0.2, 0.1])
    opt = Options(GridSize=[19, 4], GridSymmetry='D2h')
    for H in (1, 2):
        exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=512, Harmonic=H, ModAmp=0.15)
        _, yp = pepper(SpinSystem(**sd, A=[[20.0, 21.0, 85.0]]), exp, opt)
        _, ya = pepper_autograd(SpinSystem(**sd, A=[[20.0, 21.0, 85.0]]), exp, opt)
        assert float((yp - ya).abs().max() / yp.abs().max()) < 1e-9
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=512, Harmonic=1, ModAmp=0.15)
    A = torch.tensor([[20.0, 21.0, 85.0]], requires_grad=True)
    _, y = pepper_autograd(SpinSystem(**sd, A=A), exp, opt)
    J = _jacobian_rows(y, [A], 4)
    h = 1e-4

    def f(d):
        return pepper(SpinSystem(**sd, A=[[20.0, 21.0, 85.0 + d]]), exp, opt)[1][::4]
    fd = (f(h) - f(-h)) / (2 * h)
    assert float((J[:, 2] - fd).norm() / fd.norm()) < 1e-5


# --- GitHub #7: pepper dispatch, isotope mixtures, Q shorthand ---------------

def _zr_case():
    from torchspin import Experiment
    from torchspin.experiment import Options
    return (Experiment(mwFreq=9.65, Range=[300.0, 400.0], nPoints=256, Harmonic=1),
            Options(GridSize=[7, 2]))


def test_pepper_dispatches_to_autograd_when_inputs_require_grad():
    import warnings
    from torchspin import SpinSystem, pepper
    from torchspin.pepper_autograd import pepper_autograd
    exp, opt = _zr_case()
    g = torch.tensor([[2.0, 1.9, 1.9]], dtype=torch.float64, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)     # no tensor-to-scalar warning
        _, y = pepper(SpinSystem(S=[0.5], g=g, lw=[1.0, 0.0]), exp, opt)
    assert y.grad_fn is not None
    _, y_ad = pepper_autograd(SpinSystem(S=[0.5], g=g, lw=[1.0, 0.0]), exp, opt)
    assert torch.equal(y.detach(), y_ad.detach())
    y.sum().backward()
    assert g.grad is not None and torch.isfinite(g.grad).all()


def test_pepper_on_graph_unsupported_config_names_pepper_autograd():
    from torchspin import SpinSystem, pepper
    from torchspin.experiment import Options
    exp, _ = _zr_case()
    g = torch.tensor([[2.0, 1.9, 1.9]], dtype=torch.float64, requires_grad=True)
    sys = SpinSystem(S=[0.5], g=g, Nucs=['1H'], A=[[10.0, 10.0, 10.0]], lw=[1.0, 0.0])
    with pytest.raises(NotImplementedError, match="pepper_autograd.*hybrid"):
        pepper(sys, exp, Options(GridSize=[7, 2], Method='hybrid'))


def test_pepper_autograd_expands_natural_abundance_and_weight():
    from torchspin import SpinSystem, pepper
    from torchspin.pepper_autograd import pepper_autograd
    exp, opt = _zr_case()
    kw = dict(S=[0.5], Nucs=['Zr'], A=[[780.0] * 3], lw=[1.0, 0.0], weight=0.3)
    g = torch.tensor([[2.0, 1.9, 1.9]], dtype=torch.float64, requires_grad=True)
    _, y = pepper_autograd(SpinSystem(g=g, **kw), exp, opt)
    with torch.no_grad():
        _, y_ref = pepper(SpinSystem(g=g.detach(), **kw), exp, opt)
    assert (y.detach() - y_ref).norm() / y_ref.norm() < 1e-9

    # gradient w.r.t. A through the isotopologue sum matches central differences
    target = y_ref * 1.1
    def loss(a):
        A = a * torch.ones(1, 3, dtype=torch.float64)
        sys = SpinSystem(S=[0.5], g=[[2.0, 1.9, 1.9]], Nucs=['Zr'], A=A, lw=[1.0, 0.0])
        return ((pepper_autograd(sys, exp, opt)[1] - target) ** 2).mean()
    a = torch.tensor(790.0, dtype=torch.float64, requires_grad=True)
    loss(a).backward()
    h = 0.01
    with torch.no_grad():
        fd = (loss(torch.tensor(790.0 + h, dtype=torch.float64))
              - loss(torch.tensor(790.0 - h, dtype=torch.float64))) / (2 * h)
    assert abs(a.grad.item() - fd.item()) <= 1e-5 * abs(fd.item())


def test_q_shorthand_keeps_the_graph():
    from torchspin import SpinSystem, Experiment, pepper
    from torchspin.experiment import Options
    exp = Experiment(mwFreq=9.65, Range=[250.0, 470.0], nPoints=400, Harmonic=1)
    opt = Options(GridSize=[7, 2])
    pref = 1 / (4 * 2.5 * (2 * 2.5 - 1))            # I = 5/2
    base = dict(S=[0.5], g=[[2.0, 1.9, 1.895]], Nucs=['91Zr'], A=[[810.0, 805.0, 815.0]], lw=[0.3, 0.4])
    target = pepper(SpinSystem(**base, Q=[[-30 * pref, -30 * pref, 60 * pref]]), exp, opt)[1]

    def loss(q):
        Q = torch.stack([q, torch.zeros_like(q)]).reshape(1, 2)     # [e2qQ/h, eta]
        return ((pepper(SpinSystem(**base, Q=Q), exp, opt)[1] - target) ** 2).mean()
    q = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
    loss(q).backward()
    with torch.no_grad():
        fd = (loss(torch.tensor(20.005, dtype=torch.float64))
              - loss(torch.tensor(19.995, dtype=torch.float64))) / 0.01
    assert abs(q.grad.item() - fd.item()) <= 1e-4 * abs(fd.item())
