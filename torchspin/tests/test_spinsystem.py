"""Tests for nuclear spin manipulation: nucspinrmv, nucspinkeep, nucspinadd, isotopologues."""
import numpy as np
import pytest
import torch

from torchspin.spinsystem import SpinSystem, nucspinrmv, nucspinkeep, nucspinadd, isotopologues


def _make_sys():
    """Three-nucleus test system: 14N, 1H, 14N."""
    return SpinSystem(
        S=[0.5],
        g=[[2.0, 2.0, 2.0]],
        Nucs=['14N', '1H', '14N'],
        A=[[10.0, 10.0, 50.0],
           [5.0, 5.0, 5.0],
           [8.0, 8.0, 30.0]],
    )


# ---------------------------------------------------------------------------
# nucspinrmv
# ---------------------------------------------------------------------------

class TestNucspinrmv:
    def test_remove_single(self):
        """Remove the middle nucleus."""
        sys = _make_sys()
        sys2 = nucspinrmv(sys, 1)
        assert sys2.Nucs == ['14N', '14N']
        assert sys2.nNuclei == 2

    def test_remove_first(self):
        sys = _make_sys()
        sys2 = nucspinrmv(sys, 0)
        assert sys2.Nucs == ['1H', '14N']

    def test_remove_last(self):
        sys = _make_sys()
        sys2 = nucspinrmv(sys, 2)
        assert sys2.Nucs == ['14N', '1H']

    def test_remove_multiple(self):
        sys = _make_sys()
        sys2 = nucspinrmv(sys, [0, 2])
        assert sys2.Nucs == ['1H']
        assert sys2.nNuclei == 1

    def test_A_shape_correct(self):
        """A tensor shape matches new nucleus count."""
        sys = _make_sys()
        sys2 = nucspinrmv(sys, 1)
        assert sys2.A.shape[0] == 2

    def test_electrons_preserved(self):
        sys = _make_sys()
        sys2 = nucspinrmv(sys, 1)
        assert sys2.S == sys.S

    def test_out_of_range_raises(self):
        sys = _make_sys()
        with pytest.raises(ValueError, match="out of range"):
            nucspinrmv(sys, 5)

    def test_remove_all(self):
        sys = _make_sys()
        sys2 = nucspinrmv(sys, [0, 1, 2])
        assert sys2.Nucs == []
        assert sys2.nNuclei == 0


# ---------------------------------------------------------------------------
# nucspinkeep
# ---------------------------------------------------------------------------

class TestNucspinkeep:
    def test_keep_first(self):
        sys = _make_sys()
        sys2 = nucspinkeep(sys, 0)
        assert sys2.Nucs == ['14N']

    def test_keep_two(self):
        sys = _make_sys()
        sys2 = nucspinkeep(sys, [0, 2])
        assert sys2.Nucs == ['14N', '14N']

    def test_keep_all(self):
        sys = _make_sys()
        sys2 = nucspinkeep(sys, [0, 1, 2])
        assert sys2.Nucs == sys.Nucs

    def test_A_values_preserved(self):
        """A tensor rows should match the kept nuclei."""
        sys = _make_sys()
        sys2 = nucspinkeep(sys, [0])
        assert torch.allclose(sys2.A[0], sys.A[0])


# ---------------------------------------------------------------------------
# nucspinadd
# ---------------------------------------------------------------------------

class TestNucspinadd:
    def test_add_to_empty(self):
        """Add a nucleus to a bare electron system."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        sys2 = nucspinadd(sys, '1H', A=[5.0, 5.0, 10.0])
        assert sys2.Nucs == ['1H']
        assert sys2.nNuclei == 1

    def test_add_to_existing(self):
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]], Nucs=['14N'], A=[[10, 10, 50]])
        sys2 = nucspinadd(sys, '1H', A=[5.0, 5.0, 5.0])
        assert sys2.Nucs == ['14N', '1H']
        assert sys2.nNuclei == 2

    def test_A_shape(self):
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        sys2 = nucspinadd(sys, '31P', A=[3.0, 3.0, 12.0])
        assert sys2.A.shape == (1, 3)  # 1 nucleus, 1 electron

    def test_no_A(self):
        """Adding without A gives zero coupling."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        sys2 = nucspinadd(sys, '1H')
        assert sys2.nNuclei == 1
        assert torch.all(sys2.A == 0)

    def test_electrons_unchanged(self):
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        sys2 = nucspinadd(sys, '1H')
        assert sys2.S == sys.S


# ---------------------------------------------------------------------------
# isotopologues
# ---------------------------------------------------------------------------

class TestIsotopologues:
    def test_no_nuclei(self):
        """System with no nuclei → weight=1."""
        sys = SpinSystem(S=[0.5])
        result = isotopologues(sys)
        assert len(result) == 1
        assert result[0][0] == pytest.approx(1.0)

    def test_single_nucleus_abundance(self):
        """14N has abundance ~0.9963."""
        sys = SpinSystem(S=[0.5], Nucs=['14N'], A=[[10, 10, 50]])
        result = isotopologues(sys)
        w, s = result[0]
        assert 0.99 < w < 1.0
        assert s.Nucs == ['14N']

    def test_two_nuclei_weight(self):
        """Two 14N: weight ≈ 0.9963^2 ≈ 0.9926."""
        sys = SpinSystem(S=[0.5], Nucs=['14N', '14N'], A=[[10, 10, 50], [8, 8, 30]])
        result = isotopologues(sys)
        w = result[0][0]
        assert w == pytest.approx(0.9963 ** 2, abs=0.01)

    def test_weight_set_on_copy(self):
        """Returned SpinSystem has the abundance as its weight."""
        sys = SpinSystem(S=[0.5], Nucs=['14N'], A=[[10, 10, 50]])
        result = isotopologues(sys)
        w, s = result[0]
        assert s.weight == pytest.approx(w)


# ---------------------------------------------------------------------------
# validate() tests
# ---------------------------------------------------------------------------

class TestValidateBasic:
    """Basic validation tests (already-existing checks)."""

    def test_valid_system_passes(self):
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        sys.validate()  # Should not raise

    def test_valid_nitroxide_passes(self):
        sys = SpinSystem(
            S=[0.5], g=[[2.009, 2.006, 2.002]],
            Nucs=['14N'], A=[[10.0, 10.0, 95.0]]
        )
        sys.validate()

    def test_negative_S_raises(self):
        sys = SpinSystem(S=[-0.5])
        with pytest.raises(ValueError, match="positive"):
            sys.validate()

    def test_non_halfinteger_S_raises(self):
        sys = SpinSystem(S=[0.3])
        with pytest.raises(ValueError, match="multiple of 1/2"):
            sys.validate()


class TestValidateMutualExclusivity:
    """Mutual exclusivity checks."""

    def test_tcorr_and_logtcorr_raises(self):
        sys = SpinSystem(S=[0.5], tcorr=1e-9, logtcorr=-9.0)
        with pytest.raises(ValueError, match="tcorr or logtcorr"):
            sys.validate()

    def test_Diff_and_logDiff_raises(self):
        sys = SpinSystem(S=[0.5], Diff=1e8, logDiff=8.0)
        with pytest.raises(ValueError, match="Diff or logDiff"):
            sys.validate()

    def test_D_and_B2_raises(self):
        import torch
        sys = SpinSystem(
            S=[1.0],
            D=[[1000.0, 0.0, 0.0]],
            B=[[100.0, 0.0, 0.0, 0.0, 0.0]],  # B2 rank-2 Stevens
        )
        with pytest.raises(ValueError, match="D and B2"):
            sys.validate()

    def test_lw_and_lwpp_raises(self):
        # lw/lwpp mutual exclusivity is enforced in __post_init__
        with pytest.raises(ValueError, match="lw or lwpp"):
            SpinSystem(S=[0.5], lw=[0.5, 0.1], lwpp=[0.3, 0.05])

    def test_tcorr_only_passes(self):
        sys = SpinSystem(S=[0.5], tcorr=1e-9)
        sys.validate()

    def test_logtcorr_only_passes(self):
        sys = SpinSystem(S=[0.5], logtcorr=-9.0)
        sys.validate()


class TestValidateStrainBounds:
    """Strain parameter validation."""

    def test_DStrainCorr_above_1_raises(self):
        sys = SpinSystem(S=[0.5], DStrainCorr=1.5)
        with pytest.raises(ValueError, match="DStrainCorr"):
            sys.validate()

    def test_DStrainCorr_below_neg1_raises(self):
        sys = SpinSystem(S=[0.5], DStrainCorr=-1.5)
        with pytest.raises(ValueError, match="DStrainCorr"):
            sys.validate()

    def test_DStrainCorr_at_boundary_passes(self):
        for val in [-1.0, 0.0, 0.5, 1.0]:
            sys = SpinSystem(S=[0.5], DStrainCorr=val)
            sys.validate()

    def test_gAStrainCorr_invalid_raises(self):
        sys = SpinSystem(S=[0.5], gAStrainCorr=0.5)
        with pytest.raises(ValueError, match="gAStrainCorr"):
            sys.validate()

    def test_gAStrainCorr_valid_passes(self):
        for val in [1.0, -1.0]:
            sys = SpinSystem(S=[0.5], gAStrainCorr=val)
            sys.validate()

    def test_negative_HStrain_raises(self):
        sys = SpinSystem(S=[0.5], HStrain=[-1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="HStrain.*non-negative"):
            sys.validate()

    def test_negative_gStrain_raises(self):
        sys = SpinSystem(S=[0.5], gStrain=[[-0.001, 0.0, 0.0]])
        with pytest.raises(ValueError, match="gStrain.*non-negative"):
            sys.validate()

    def test_negative_DStrain_raises(self):
        sys = SpinSystem(S=[1.0], DStrain=[-10.0, 5.0])
        with pytest.raises(ValueError, match="DStrain.*non-negative"):
            sys.validate()

    def test_positive_strain_passes(self):
        sys = SpinSystem(
            S=[0.5],
            HStrain=[10.0, 10.0, 10.0],
            gStrain=[[0.001, 0.001, 0.001]],
        )
        sys.validate()


class TestValidateQSymmetry:
    """Q matrix symmetry validation."""

    def test_symmetric_fullQ_passes(self):
        import torch
        Q = torch.tensor([
            [1.0, 0.1, 0.0],
            [0.1, -0.5, 0.0],
            [0.0, 0.0, -0.5],
        ])
        sys = SpinSystem(
            S=[0.5], Nucs=['14N'],
            A=[[10.0, 10.0, 95.0]], Q=Q
        )
        sys.validate()

    def test_asymmetric_fullQ_raises(self):
        import torch
        Q = torch.tensor([
            [1.0, 0.1, 0.0],
            [0.2, -0.5, 0.0],  # Q[1,0] != Q[0,1]
            [0.0, 0.0, -0.5],
        ])
        sys = SpinSystem(
            S=[0.5], Nucs=['14N'],
            A=[[10.0, 10.0, 95.0]], Q=Q
        )
        with pytest.raises(ValueError, match="symmetric"):
            sys.validate()


class TestValidateCFCoherence:
    """Crystal field / orbital angular momentum coherence."""

    def test_CF_without_L_raises(self):
        sys = SpinSystem(S=[0.5], CF2=[[1.0, 0.0, 0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="CF2 requires.*L"):
            sys.validate()

    def test_CF_rank_exceeds_2L_raises(self):
        sys = SpinSystem(S=[0.5], L=[1.0], soc=[[100.0]],
                         CF4=[[1.0, 0, 0, 0, 0, 0, 0, 0, 0]])
        with pytest.raises(ValueError, match="CF4 rank 4 exceeds 2\\*L=2"):
            sys.validate()

    def test_CF_within_2L_passes(self):
        sys = SpinSystem(S=[0.5], L=[2.0], soc=[[100.0]],
                         CF2=[[1.0, 0.0, 0.0, 0.0, 0.0]])
        sys.validate()


def test_q_shorthand_principal_values_and_graph():
    """[e2qQ/h, eta] -> e2qQ/h/(4I(2I-1))·[-1+eta, -1-eta, 2], built with torch
    ops so a Q that requires grad stays on the graph (GitHub #7)."""
    import torch
    from torchspin import SpinSystem
    q = torch.tensor([[12.0, 0.3]], dtype=torch.float64, requires_grad=True)
    s = SpinSystem(S=[0.5], g=2.0, Nucs=['14N'], A=[[1.0, 1.0, 1.0]], Q=q)
    k = 12.0 / (4 * 1 * (2 * 1 - 1))
    torch.testing.assert_close(s.Q.detach(), torch.tensor([[k * -0.7, k * -1.3, k * 2.0]], dtype=torch.float64))
    assert s.Q.requires_grad
    s.Q.sum().backward()
    assert q.grad is not None
