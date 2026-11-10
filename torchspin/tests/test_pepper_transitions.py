"""Options.Transitions in pepper (EasySpin Opt.Transitions).

None (default) keeps the automatic transition pre-selection; 'all' takes every
level pair with no Threshold pruning; a list of 1-based (u, v) level pairs
(levels by ascending energy at the resonance field) takes exactly those.  The
EasySpin comparison is in test_pepper_transitions_matlab_validation.py.
"""
import numpy as np
import pytest
import torch

from torchspin import Experiment, Options, SpinSystem
from torchspin.pepper import pepper, _user_transitions
from torchspin.pepper_autograd import pepper_autograd


def _spec(sys, exp, **o):
    x, y = pepper(sys, exp, Options(Verbosity=0, **o))
    return y.detach().numpy()


def _rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _quartet():
    return SpinSystem(S=1.5, g=2.0, D=[1500.0, 0.0], lwpp=2.0)


EXP_Q = Experiment(mwFreq=9.5, Range=[0, 600], nPoints=1024)


# ---------------------------------------------------------------- resolution

def test_user_transitions_resolution():
    assert _user_transitions(None, 4) == (None, False)
    pairs, all_ = _user_transitions('all', 3)
    assert all_ and pairs == [(0, 1), (0, 2), (1, 2)]
    pairs, all_ = _user_transitions([(3, 2), (1, 2), (2, 3), [2, 1]], 4)
    assert not all_ and pairs == [(1, 2), (0, 1)]            # u<v, 0-based, duplicates dropped
    pairs, _ = _user_transitions(np.array([[1, 4], [2, 3]]), 4)
    assert pairs == [(0, 3), (1, 2)]


@pytest.mark.parametrize('bad', ['some', [], [(1, 1)], [(0, 2)], [(1, 5)], [(1, 2, 3)],
                                 [(1.5, 2)], [(True, 2)], [3], 7])
def test_invalid_transitions_raise(bad):
    with pytest.raises(ValueError):
        pepper(_quartet(), EXP_Q, Options(Verbosity=0, Transitions=bad))


@pytest.mark.parametrize('method', ['perturb', 'perturb1', 'perturb2'])
def test_perturbation_theory_raises(method):
    sys = SpinSystem(S=0.5, g=[2.0, 2.1, 2.2], lwpp=1.0)
    exp = Experiment(mwFreq=9.5, Range=[300, 350], nPoints=256)
    with pytest.raises(ValueError, match='perturbation'):
        pepper(sys, exp, Options(Verbosity=0, Method=method, Transitions=[(1, 2)]))


# ---------------------------------------------------------------- field sweeps

def test_default_unchanged_and_all_equals_default_without_pruning():
    """For S=3/2 with large D nothing is pruned by default: 'all' gives the same spectrum."""
    sys = _quartet()
    d = _spec(sys, EXP_Q)
    assert np.array_equal(d, _spec(sys, EXP_Q, Transitions=None))
    assert np.array_equal(d, _spec(sys, EXP_Q, Transitions='all'))


def test_all_recovers_weak_half_field_line():
    """Triplet, D = 30 MHz: the ΔmS=2 line (~(D/hν)² weaker) falls below the default
    Threshold and is dropped; 'all' keeps it, it is exactly the (1,3) transition, and
    its intensity scales as D²."""
    exp = Experiment(mwFreq=9.5, Range=[150, 190], nPoints=1024, Harmonic=0)
    amp = {}
    for D in (30.0, 60.0):
        sys = SpinSystem(S=1, g=2.0, D=[D, 0.0], lwpp=1.0)
        d = _spec(sys, exp)
        a = _spec(sys, exp, Transitions='all')
        h = _spec(sys, exp, Transitions=[(3, 1)])
        assert np.abs(d).max() == 0.0
        assert np.abs(a).max() > 0
        assert _rel(a, h) < 1e-12
        amp[D] = np.abs(a).max()
    assert amp[60.0] / amp[30.0] == pytest.approx(4.0, rel=0.02)


def test_subset_sums_and_central_transition():
    """Disjoint subsets add up to their union (Threshold=0, so no relative pruning);
    the central transition (2,3) of S=3/2 is a single line set."""
    sys = _quartet()
    sets = [[(2, 3)], [(1, 2), (3, 4)], [(1, 3), (2, 4), (1, 4)]]
    parts = [_spec(sys, EXP_Q, Transitions=s, Threshold=0.0) for s in sets]
    union = _spec(sys, EXP_Q, Transitions=sum(sets, []), Threshold=0.0)
    allp = _spec(sys, EXP_Q, Transitions='all')
    assert _rel(sum(parts), union) < 1e-7      # adaptive field knots differ per set
    assert _rel(union, allp) < 1e-7
    # the central transition alone is a strong but different spectrum
    c = parts[0]
    assert np.abs(c).max() > 0.1 * np.abs(allp).max()
    assert _rel(c, allp) > 1e-2


def test_O3_path():
    """O3 path (isotropic S=3/2 + 1H, no D): selected pairs on the per-line route."""
    sys = SpinSystem(S=1.5, g=2.0, Nucs='1H', A=20.0, lwpp=0.3)
    exp = Experiment(mwFreq=9.5, Range=[335, 344], nPoints=1024)
    allp = _spec(sys, exp, Transitions='all')
    n = 8
    pairs = [(u, v) for u in range(1, n + 1) for v in range(u + 1, n + 1)]
    half = len(pairs) // 2
    a = _spec(sys, exp, Transitions=pairs[:half], Threshold=0.0)
    b = _spec(sys, exp, Transitions=pairs[half:], Threshold=0.0)
    assert np.abs(allp).max() > 0
    assert _rel(a + b, allp) < 1e-9
    assert _rel(_spec(sys, exp), allp) < 1e-9


def test_strain_summation_path():
    sys = SpinSystem(S=1.5, g=2.0, D=[1500.0, 200.0], HStrain=[30.0, 30.0, 30.0])
    exp = Experiment(mwFreq=9.5, Range=[0, 600], nPoints=512)
    allp = _spec(sys, exp, Transitions='all')
    parts = [_spec(sys, exp, Transitions=s, Threshold=0.0) for s in ([(2, 3)], [(1, 2), (3, 4), (1, 3), (2, 4), (1, 4)])]
    assert _rel(sum(parts), allp) < 1e-7
    assert _rel(_spec(sys, exp), allp) < 1e-7


def test_hybrid_core_levels():
    """Method='hybrid': level numbers refer to the exactly treated core."""
    sys = SpinSystem(S=1, g=2.0, D=[900.0, 0.0], Nucs='14N', A=[10.0, 10.0, 15.0], lwpp=1.0)
    exp = Experiment(mwFreq=9.5, Range=[200, 480], nPoints=512)
    opt = dict(Method='hybrid', HybridIntThreshold=0.0)   # sub-line pruning is relative to each set's max
    allp = _spec(sys, exp, Transitions='all', **opt)
    a = _spec(sys, exp, Transitions=[(1, 2)], Threshold=0.0, **opt)
    b = _spec(sys, exp, Transitions=[(2, 3), (1, 3)], Threshold=0.0, **opt)
    assert _rel(a + b, allp) < 1e-7
    with pytest.raises(ValueError, match='out of range'):
        pepper(sys, exp, Options(Verbosity=0, Transitions=[(1, 4)], **opt))   # core has 3 levels


def test_crystal_path():
    sys = _quartet()
    exp = Experiment(mwFreq=9.5, Range=[0, 600], nPoints=1024, CrystalSymmetry='P1',
                     MolFrame=[0.3, 0.7, 0.1])
    d = _spec(sys, exp)
    allp = _spec(sys, exp, Transitions='all')
    c = _spec(sys, exp, Transitions=[(2, 3)], Threshold=0.0)
    o = _spec(sys, exp, Transitions=[(1, 2), (3, 4), (1, 3), (2, 4), (1, 4)], Threshold=0.0)
    assert _rel(d, allp) < 1e-9
    assert _rel(c + o, allp) < 1e-7
    assert np.abs(c).max() > 0 and _rel(c, allp) > 1e-2


def test_frequency_sweep():
    sys = SpinSystem(S=1, g=2.0, D=[3000.0, 300.0], lwpp=10.0)
    exp = Experiment(Field=340.0, mwRange=[5.0, 15.0], nPoints=1024)
    allp = _spec(sys, exp, Transitions='all')
    a = _spec(sys, exp, Transitions=[(1, 2)], Threshold=0.0)
    b = _spec(sys, exp, Transitions=[(2, 3), (1, 3)], Threshold=0.0)
    assert np.abs(a).max() > 0 and np.abs(b).max() > 0
    assert _rel(a + b, allp) < 1e-9


# ---------------------------------------------------------------- autograd

def test_pepper_autograd_subset_forward_and_gradient():
    exp = Experiment(mwFreq=9.5, Range=[0, 600], nPoints=512)
    tr = [(2, 3), (1, 2)]
    opt = Options(Verbosity=0, GridSize=[19, 4], Transitions=tr, Threshold=0.0)
    D0 = [1500.0, 200.0]
    D = torch.tensor([D0], requires_grad=True)
    _, y = pepper_autograd(SpinSystem(S=1.5, g=2.0, D=D, lwpp=4.0), exp, opt)
    ref = _spec(SpinSystem(S=1.5, g=2.0, D=[D0], lwpp=4.0), exp, GridSize=[19, 4], Transitions=tr, Threshold=0.0)
    assert _rel(y.detach().numpy(), ref) < 1e-9
    # ... and differs from the all-transitions spectrum
    assert _rel(ref, _spec(SpinSystem(S=1.5, g=2.0, D=[D0], lwpp=4.0), exp, GridSize=[19, 4])) > 1e-2
    w = torch.from_numpy(np.random.default_rng(0).standard_normal(exp.nPoints))
    (y * w).sum().backward()
    g = D.grad[0].numpy()

    def L(Dv):
        return float((torch.from_numpy(_spec(SpinSystem(S=1.5, g=2.0, D=[Dv], lwpp=4.0), exp, GridSize=[19, 4],
                                             Transitions=tr, Threshold=0.0)) * w).sum())
    for i in range(2):
        best = np.inf
        for h in (1e-1, 3e-2, 1e-2):
            dp = list(D0); dm = list(D0); dp[i] += h; dm[i] -= h
            fd = (L(dp) - L(dm)) / (2 * h)
            best = min(best, abs(fd - g[i]) / max(abs(fd), 1e-12))
        assert best < 1e-5, (i, g[i], best)


def test_pepper_autograd_perturb_raises():
    g = torch.tensor([2.0, 2.1, 2.2], requires_grad=True)
    exp = Experiment(mwFreq=9.5, Range=[300, 350], nPoints=256)
    with pytest.raises(ValueError, match='perturbation'):
        pepper_autograd(SpinSystem(S=0.5, g=g, lwpp=1.0), exp,
                        Options(Verbosity=0, Method='perturb', Transitions=[(1, 2)]))


def test_list_keeps_threshold_post_selection():
    """As in EasySpin, a user list skips the pre-selection but the Threshold intensity
    post-selection still applies within it ('all' switches it off)."""
    sys = SpinSystem(S=1, g=2.0, D=[30.0, 0.0], lwpp=1.0)
    exp = Experiment(mwFreq=9.5, Range=[100, 400], nPoints=2048, Harmonic=0)
    x = np.linspace(100, 400, 2048); half = (x > 150) & (x < 190)
    lst = [(1, 2), (2, 3), (1, 3)]
    hf = np.abs(_spec(sys, exp, Transitions=lst, Threshold=0.0)[half]).max()
    assert hf > 1e-4
    assert np.abs(_spec(sys, exp, Transitions=lst)[half]).max() < 1e-6 * hf      # pruned (FFT floor only)
    assert np.abs(_spec(sys, exp, Transitions='all')[half]).max() == pytest.approx(hf, rel=1e-6)
    assert np.abs(_spec(sys, exp, Transitions=[(1, 3)])[half]).max() == pytest.approx(hf, rel=1e-3)  # alone it is its own maximum


def test_frequency_sweep_auto_range():
    sys = SpinSystem(S=1, g=2.0, D=[3000.0, 300.0], lwpp=10.0)
    exp = Experiment(Field=340.0, nPoints=512)
    x, y = pepper(sys, exp, Options(Verbosity=0, Transitions=[(1, 2)]))
    xa, ya = pepper(sys, exp, Options(Verbosity=0))
    assert np.abs(y.numpy()).max() > 0
    assert float(x[-1] - x[0]) < float(xa[-1] - xa[0])       # range spans the selected line only
