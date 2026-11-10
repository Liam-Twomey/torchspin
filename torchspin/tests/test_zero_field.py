"""Zero-field frequency sweeps and fixed-field ENDOR (Options.ZeroField, 0.7.4).

* the default (ZeroField=False) still rejects Experiment.Field = 0 with the
  same ValueError, now raised by the simulator instead of the constructor;
* pepper at Field = 0 against closed forms (S = 1: lines at D-E, D+E, 2E, each
  with the powder rate Σ_k|<i|μ_k|j>|²/3; S = 3/2 Kramers: one line at
  2·sqrt(D²+3E²) carrying the rate summed over the 2×2 degenerate pairs),
  against an independent numpy brute force, against EasySpin (pepper accepts
  Exp.Field = 0 for frequency sweeps), and against the B → 0 limit;
* gradients of the zero-field resonances (resfreqs_batch) and of the
  zero-field salt spectrum against central differences (pepper itself has no
  differentiable frequency-sweep path: pepper_autograd raises for them).
"""
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

from torchspin import SpinSystem
from torchspin.constants import BMAGN, NMAGN, PLANCK
from torchspin.experiment import Experiment, Options
from torchspin.garlic import garlic
from torchspin.ham import ham
from torchspin.ham_nz import ham_nz
from torchspin.pepper import pepper
from torchspin.resfreqs_matrix import resfreqs_batch
from torchspin.salt import salt
from torchspin.sphgrid import sphgrid

ZF = Options(ZeroField=True)
MSG = r"Experiment\.Field must be positive \(mT\)\."
REF_FILE = Path(__file__).parent.parent.parent / "tests" / "data" / "ref_pepper_zerofield.mat"


def _area(x, y, center, half):
    x = np.asarray(x); y = np.asarray(y)
    m = (x > center - half) & (x < center + half)
    return float(y[m].sum() * (x[1] - x[0]))


def _spin_ops(S):
    """Spin matrices (Sx, Sy, Sz) in the |S, m> basis, m = S..-S (numpy, independent of torchspin)."""
    m = np.arange(S, -S - 1, -1)
    sp = np.diag(np.sqrt(S * (S + 1) - m[1:] * (m[1:] + 1)), 1)
    return (sp + sp.T) / 2, (sp - sp.T) / 2j, np.diag(m)


def _zfs_lines(S, D, E, g=2.0):
    """Brute-force zero-field lines of a single spin S with D, E (MHz): positions
    (MHz) and the isotropic powder rate Σ_{u∈a, v∈b} Σ_k |<u|g μB S_k|v>|² / 3
    summed over the level pairs of each pair of degenerate clusters."""
    Sx, Sy, Sz = _spin_ops(S)
    H = D * (Sz @ Sz - S * (S + 1) / 3 * np.eye(Sz.shape[0])) + E * (Sx @ Sx - Sy @ Sy)
    Ev, V = np.linalg.eigh(H)
    labels = np.concatenate([[0], np.cumsum(np.diff(Ev) > 1e-6)])
    gmu = g * BMAGN / PLANCK * 1e-9          # MHz/mT
    Ops = [V.conj().T @ (gmu * O) @ V for O in (Sx, Sy, Sz)]
    lines = []
    K = labels.max() + 1
    for a in range(K):
        for b in range(a + 1, K):
            ua, ub = labels == a, labels == b
            r = sum((np.abs(O[np.ix_(ua, ub)]) ** 2).sum() for O in Ops) / 3
            lines.append((Ev[ub].mean() - Ev[ua].mean(), r))
    return lines


@pytest.fixture(scope="module")
def half_line_area():
    """Area of the S = 1/2 (g = 2) line of a high-field frequency sweep, whose
    rate is (gμB)²/4 at every orientation: the pepper normalization constant."""
    x, y = pepper(SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], lw=[5.0, 0.0]),
                  Experiment(Field=340, mwRange=[9.3, 9.75], nPoints=2000, Harmonic=0))
    gmu = 2.0 * BMAGN / PLANCK * 1e-9
    return _area(x, y, float(x[y.argmax()]), 0.05) / (gmu ** 2 / 4)


# ── default behavior unchanged ────────────────────────────────────────────────

class TestDefaultRejectsZeroField:
    sys = SpinSystem(S=[1.0], g=[[2.0, 2.0, 2.0]], D=[3000.0, 400.0], lw=[0.0, 5.0])

    def test_negative_field_still_raises_at_construction(self):
        with pytest.raises(ValueError, match=MSG):
            Experiment(Field=-1.0, mwRange=[0.1, 4.0])
        with pytest.raises(ValueError, match=MSG):
            Experiment(Field=-1.0)

    @pytest.mark.parametrize("opt", [None, Options(), Options(ZeroField=False)])
    def test_pepper_freq_sweep(self, opt):
        exp = Experiment(Field=0, mwRange=[0.1, 4.0])
        with pytest.raises(ValueError, match=MSG):
            pepper(self.sys, exp, opt)

    def test_pepper_auto_range(self):
        with pytest.raises(ValueError, match=MSG):
            pepper(self.sys, Experiment(Field=0.0))

    def test_salt_fixed_field(self):
        s = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], Nucs='1H', A=[[10.0, 10.0, 10.0]])
        with pytest.raises(ValueError, match=MSG):
            salt(s, Experiment(Field=0), freq_range=(5, 15))

    def test_garlic_rejects_even_with_zero_field_option(self):
        s = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], lw=[0.0, 1.0])
        with pytest.raises(ValueError, match=MSG):
            garlic(s, Experiment(Field=0, mwRange=[0.1, 1.0]), ZF)

    def test_positive_field_unaffected(self):
        x, y = pepper(self.sys, Experiment(Field=1e-4, mwRange=[0.1, 4.0], nPoints=500))
        assert torch.isfinite(y).all() and float(y.abs().max()) > 0


# ── pepper at Field = 0 ──────────────────────────────────────────────────────

class TestPepperZeroField:
    def test_s1_closed_form(self, half_line_area):
        """S = 1: zero-field levels Tz, Tx, Ty; lines at D-E, D+E, 2E, each with
        |<Ti|S_k|Tj>|² = 1 for one k → powder rate (gμB)²/3."""
        D, E = 3000.0, 400.0
        x, y = pepper(SpinSystem(S=[1.0], g=[[2.0, 2.0, 2.0]], D=[D, E], lw=[5.0, 0.0]),
                      Experiment(Field=0, mwRange=[0.1, 4.0], nPoints=4000, Harmonic=0), ZF)
        dx = float(x[1] - x[0])
        gmu = 2.0 * BMAGN / PLANCK * 1e-9
        for nu in (D - E, D + E, 2 * E):
            c = nu / 1e3
            m = (x > c - 0.05) & (x < c + 0.05)
            assert abs(float(x[m][y[m].argmax()]) - c) <= dx
            assert _area(x, y, c, 0.05) / half_line_area == pytest.approx(gmu ** 2 / 3, rel=2e-3)

    def test_kramers_axial_closed_form(self, half_line_area):
        """S = 3/2, E = 0: doublets ±3/2, ±1/2 at 2D apart; the summed rate over
        the 4 inter-doublet pairs is Σ|<3/2|S±|1/2>|²·... = 3·(gμB)²/3."""
        D = 5000.0
        x, y = pepper(SpinSystem(S=[1.5], g=[[2.0, 2.0, 2.0]], D=[D, 0.0], lw=[5.0, 0.0]),
                      Experiment(Field=0, mwRange=[9.0, 11.0], nPoints=4000, Harmonic=0), ZF)
        gmu = 2.0 * BMAGN / PLANCK * 1e-9
        assert abs(float(x[y.argmax()]) - 2 * D / 1e3) <= float(x[1] - x[0])
        assert _area(x, y, 2 * D / 1e3, 0.05) / half_line_area == pytest.approx(gmu ** 2, rel=2e-3)

    def test_kramers_rhombic_vs_brute_force(self, half_line_area):
        """S = 3/2, D, E: one line at 2·sqrt(D² + 3E²), intensity summed over the
        degenerate pairs, against an independent numpy diagonalization."""
        D, E = 5000.0, 1000.0
        lines = _zfs_lines(1.5, D, E)
        assert len(lines) == 1
        nu, r = lines[0]
        assert nu == pytest.approx(2 * math.sqrt(D ** 2 + 3 * E ** 2), rel=1e-12)
        x, y = pepper(SpinSystem(S=[1.5], g=[[2.0, 2.0, 2.0]], D=[D, E], lw=[5.0, 0.0]),
                      Experiment(Field=0, mwRange=[9.5, 11.5], nPoints=4000, Harmonic=0), ZF)
        assert abs(float(x[y.argmax()]) - nu / 1e3) <= float(x[1] - x[0])
        assert _area(x, y, nu / 1e3, 0.05) / half_line_area == pytest.approx(r, rel=2e-3)

    def test_resfreqs_batch_sums_degenerate_pairs(self):
        """resfreqs_batch(zero_field=True) returns one transition per cluster pair
        (here: one for the two Kramers doublets) whose powder average equals the
        brute-force summed rate; a plain per-pair run would return 4 lines."""
        D, E = 5000.0, 1000.0
        s = SpinSystem(S=[1.5], g=[[2.0, 2.0, 2.0]], D=[D, E])
        ph, th, w, _ = sphgrid('D2h', 19)
        exp = Experiment(Field=0, mwRange=[1.0, 20.0])
        P, I, _ = resfreqs_batch(*ham(s, None), ph, th, exp, ZF, s, zero_field=True)
        assert all(p.numel() == 1 for p in P)
        avg = float(sum(float(i[0]) * float(wk) for i, wk in zip(I, w)) / float(w.sum()))
        nu, r = _zfs_lines(1.5, D, E)[0]
        assert float(P[0][0]) * 1e3 == pytest.approx(nu, rel=1e-12)
        assert avg == pytest.approx(r, rel=5e-4)          # SOPHE quadrature of a quadratic form in n

    def test_o3_system_uses_grid(self, capsys):
        """Isotropic S = 1/2, I = 1/2 (O3): F = 0 ↔ F = 1 line at A, computed on a
        D2h grid instead of the one-orientation O3 shortcut."""
        A = 1420.0
        s = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], Nucs='1H', A=[[A, A, A]], lw=[2.0, 0.0])
        x, y = pepper(s, Experiment(Field=0, mwRange=[1.3, 1.5], nPoints=2001, Harmonic=0),
                      Options(ZeroField=True, Verbosity=1))
        assert "O3 grid replaced by D2h" in capsys.readouterr().out
        assert abs(float(x[y.argmax()]) - A / 1e3) <= float(x[1] - x[0])

    @pytest.mark.parametrize("S, D, E, rng", [(1.0, 3000.0, 400.0, [0.1, 4.0]),
                                              (1.5, 5000.0, 1000.0, [9.5, 11.5])])
    def test_continuity_small_field(self, S, D, E, rng):
        """Field = 0 (ZeroField) equals the default path at Field → 0+."""
        s = SpinSystem(S=[S], g=[[2.0, 2.0, 2.0]], D=[D, E], lw=[0.0, 5.0])
        _, y0 = pepper(s, Experiment(Field=0, mwRange=rng, nPoints=3000), ZF)
        _, y1 = pepper(s, Experiment(Field=1e-4, mwRange=rng, nPoints=3000))
        assert float((y0 - y1).abs().max() / y0.abs().max()) < 1e-6

    def test_strain_and_temperature_run(self):
        s = SpinSystem(S=[1.0], g=[[2.0, 2.0, 2.0]], D=[3000.0, 400.0], HStrain=[10.0, 20.0, 30.0])
        x, y = pepper(s, Experiment(Field=0, mwRange=[0.1, 4.0], nPoints=1000, Temperature=0.1), ZF)
        assert torch.isfinite(y).all() and float(y.abs().max()) > 0

    def test_initstate_raises(self):
        s = SpinSystem(S=[1.0], g=[[2.0, 2.0, 2.0]], D=[3000.0, 400.0], lw=[0.0, 5.0],
                       initState=([0.2, 0.3, 0.5], 'zerofield'))
        with pytest.raises(ValueError, match="ZeroField"):
            pepper(s, Experiment(Field=0, mwRange=[0.1, 4.0], nPoints=500), ZF)


def test_resfreqs_zero_field_gradient():
    """d(positions, summed intensities)/dD at zero field (Kramers S = 3/2,
    anisotropic g) through the degeneracy-safe eigh, vs central differences."""
    ph, th, _, _ = sphgrid('D2h', 7)
    exp = Experiment(Field=0, mwRange=[1.0, 20.0])

    def loss(Dv):
        s = SpinSystem(S=[1.5], g=[[2.0, 2.1, 2.2]],
                       D=torch.stack([Dv, torch.tensor(700.0, dtype=torch.float64)]))
        P, I, _ = resfreqs_batch(*ham(s, None), ph, th, exp, ZF, s, zero_field=True)
        P, I = torch.cat(P), torch.cat(I)
        return (P * I).sum() + 1e3 * (I ** 2).sum()

    D0 = 5000.0
    Dv = torch.tensor(D0, dtype=torch.float64, requires_grad=True)
    g, = torch.autograd.grad(loss(Dv), Dv)
    errs = []
    for h in (1.0, 0.1, 0.01):
        fd = (loss(torch.tensor(D0 + h, dtype=torch.float64))
              - loss(torch.tensor(D0 - h, dtype=torch.float64))) / (2 * h)
        errs.append(abs(float(g) / float(fd) - 1))
    assert min(errs) < 1e-6


# ── EasySpin ────────────────────────────────────────────────────────────────

def _es_case(name):
    if name.startswith('s1'):
        kw = dict(S=[1.0], g=[[2.0, 2.0, 2.0]], D=[3000.0, 400.0])
        if name == 's1_DE_hstrain':
            kw['HStrain'] = [10.0, 20.0, 30.0]
        else:
            kw['lw'] = [5.0, 0.0]
        T = 0.1 if name == 's1_DE_temp' else None
        return SpinSystem(**kw), Experiment(Field=0, mwRange=[0.1, 4.0], nPoints=4000, Harmonic=0, Temperature=T)
    if name == 's32_kramers':
        return (SpinSystem(S=[1.5], g=[[2.0, 2.0, 2.0]], D=[5000.0, 1000.0], lw=[20.0, 0.0]),
                Experiment(Field=0, mwRange=[9.0, 12.0], nPoints=3000, Harmonic=0))
    if name == 'hyd_iso':
        return (SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], Nucs='1H', A=[[1420.0, 1420.0, 1420.0]], lw=[2.0, 0.0]),
                Experiment(Field=0, mwRange=[1.3, 1.5], nPoints=2001, Harmonic=0))
    raise KeyError(name)


# hyd_iso: EasySpin keeps its single-orientation O3 shortcut (exact for an
# isotropic system once the degenerate pairs are summed); torchspin uses a D2h
# grid, whose binning differs at the 1e-3 level.
_ES_TOL = {'hyd_iso': (0.99999, 2e-3)}
_ES_NAMES = ['s1_DE', 's1_DE_temp', 's1_DE_hstrain', 's32_kramers', 'hyd_iso']


@pytest.fixture(scope="module")
def es_cases():
    if not REF_FILE.exists():
        pytest.skip(f"Reference data not found: {REF_FILE}")
    refs = loadmat(str(REF_FILE), squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(refs['cases'])}


@pytest.mark.parametrize("name", _ES_NAMES)
def test_pepper_zero_field_vs_easyspin(es_cases, name):
    c = es_cases[name]
    s, e = _es_case(name)
    x, y = pepper(s, e, ZF)
    ref = np.asarray(c.spc, dtype=float); y = y.numpy()
    np.testing.assert_allclose(x.numpy(), np.asarray(c.nu, dtype=float), atol=1e-12)
    cos_min, amp_tol = _ES_TOL.get(name, (0.999999, 1e-4))
    cos = float(y @ ref / np.linalg.norm(y) / np.linalg.norm(ref))
    assert cos > cos_min
    assert y.max() / ref.max() == pytest.approx(1.0, abs=amp_tol)


# ── salt at Field = 0 ────────────────────────────────────────────────────────

class TestSaltZeroField:
    def test_hydrogen_like_closed_form(self):
        """S = 1/2, I = 1/2, isotropic A: one line at A; Σ_k Σ_m |<F=0|I_k|F=1,m>|² = 3/4,
        so the powder-averaged ENDOR rate is (gn μN)²·(3/4)/3."""
        A = 10.0
        s = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], Nucs='1H', A=[[A, A, A]])
        f, y = salt(s, Experiment(Field=0), ZF, freq_range=(5, 15), n_points=1001, lw_mhz=0.2)
        gn_mu = NMAGN / PLANCK * float(s.gn[0]) * 1e-9               # MHz/mT
        assert float(f[y.argmax()]) == pytest.approx(A, abs=0.011)
        assert float(y.sum()) == pytest.approx(gn_mu ** 2 / 4, rel=1e-9)

    def test_lines_vs_brute_force_and_continuity(self):
        """S = 1/2 + 14N (A, Q): three Kramers doublets; line areas against the
        cluster-summed rates from ham/ham_nz in an independent loop, and the
        spectrum against the default (matrix) path at Field → 0+."""
        s = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], Nucs='14N', A=[[5.0, 7.0, 30.0]], Q=[[-0.3, -0.5, 0.8]])
        fr = (1.0, 40.0)
        f, y = salt(s, Experiment(Field=0), ZF, freq_range=fr, n_points=2001, lw_mhz=0.3)
        H0 = ham(s, None)[0]
        E, V = np.linalg.eigh(H0.numpy())
        mu = [V.conj().T @ m.numpy() @ V for m in ham_nz(s, None)]
        lab = np.concatenate([[0], np.cumsum(np.diff(E) > 1e-6)])
        assert lab.max() == 2
        fn, yn = f.numpy(), y.numpy()
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            ua, ub = lab == a, lab == b
            r = sum((np.abs(m[np.ix_(ua, ub)]) ** 2).sum() for m in mu) / 3
            nu = E[ub].mean() - E[ua].mean()
            assert _area(fn, yn, nu, 0.55) / (fn[1] - fn[0]) == pytest.approx(r, rel=2e-3)
        _, y1 = salt(s, Experiment(Field=1e-5), Options(), freq_range=fr, n_points=2001, lw_mhz=0.3)
        assert float(y.sum()) == pytest.approx(float(y1.sum()), rel=1e-6)
        # line positions sit inside bins here; nearer a bin edge the B → 0+
        # path splits a line over two bins (binning, not physics)
        assert float((y - y1).abs().max() / y.abs().max()) < 0.05

    def test_gradient_wrt_Q(self):
        """Zero-field salt on the autograd graph: d/dQz vs central differences."""
        wts = torch.linspace(0, 1, 801, dtype=torch.float64) ** 2

        def loss(Qz):
            Q = torch.stack([torch.tensor(-0.3, dtype=torch.float64), torch.tensor(-0.5, dtype=torch.float64), Qz])
            s = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], Nucs='14N', A=[[5.0, 7.0, 30.0]], Q=Q.unsqueeze(0))
            _, y = salt(s, Experiment(Field=0), Options(ZeroField=True, GridSize=[10, 2]),
                        freq_range=(1, 40), n_points=801, lw_mhz=0.5)
            return (y * wts).sum() * 1e5

        Q0 = 0.8
        Qz = torch.tensor(Q0, dtype=torch.float64, requires_grad=True)
        g, = torch.autograd.grad(loss(Qz), Qz)
        assert abs(float(g)) > 0
        errs = []
        for h in (1e-3, 1e-4):
            fd = (loss(torch.tensor(Q0 + h, dtype=torch.float64))
                  - loss(torch.tensor(Q0 - h, dtype=torch.float64))) / (2 * h)
            errs.append(abs(float(g) / float(fd) - 1))
        assert min(errs) < 1e-6

    def test_orientation_selection_runs(self):
        s = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], Nucs='14N', A=[[5.0, 7.0, 30.0]], Q=[[-0.3, -0.5, 0.8]])
        f, y = salt(s, Experiment(Field=0, mwFreq=0.0312, ExciteWidth=5.0), ZF,
                    freq_range=(1, 40), n_points=401, lw_mhz=0.3)
        assert torch.isfinite(y).all() and float(y.abs().max()) > 0
