"""Tests for torchspin.utils — signal-processing and EPR utilities."""
import math

import numpy as np
import pytest

from torchspin.utils import (
    datasmooth,
    equivcouple,
    equivsplit,
    hilberttrans,
    larmorfrq,
    rcfilt,
    unitconvert,
)
from torchspin.constants import NMAGN, PLANCK, CLIGHT


# ---------------------------------------------------------------------------
# larmorfrq
# ---------------------------------------------------------------------------

class TestLarmorfrq:
    def test_1h_scalar(self):
        """1H at 340 mT should give ~14.47 MHz (proton Larmor frequency)."""
        freq = larmorfrq('1H', 340.0)
        # 1H: gn ≈ 5.5857, NMAGN/PLANCK = 7.6226 MHz/T → /1e3 → MHz/mT → *5.5857 ≈ 0.04256 MHz/mT
        # 0.04256 * 340 ≈ 14.47 MHz
        assert abs(freq[0] - 14.47) < 0.1

    def test_returns_array(self):
        freqs = larmorfrq('1H', [300.0, 340.0, 380.0])
        assert freqs.shape == (3, 1) or freqs.shape == (3,)

    def test_two_nuclei(self):
        freqs = larmorfrq('1H,14N', 340.0)
        assert freqs.shape == (2,)
        assert freqs[0] > freqs[1]   # proton Larmor > 14N Larmor at same field

    def test_frequency_proportional_to_field(self):
        """Larmor frequency is proportional to field."""
        f1 = larmorfrq('1H', 100.0)
        f2 = larmorfrq('1H', 200.0)
        assert abs(f2[0] / f1[0] - 2.0) < 1e-10

    def test_list_input(self):
        freqs = larmorfrq(['1H', '14N'], 340.0)
        assert freqs.shape == (2,)

    def test_positive_frequency(self):
        """Larmor frequencies should be positive (abs gn convention)."""
        for nuc in ['1H', '14N', '31P', '13C']:
            f = larmorfrq(nuc, 340.0)
            assert f[0] > 0


# ---------------------------------------------------------------------------
# equivsplit
# ---------------------------------------------------------------------------

class TestEquivsplit:
    def test_single_spin_half(self):
        """1 spin-1/2 → [1, 1]."""
        p = equivsplit(0.5, 1)
        assert np.allclose(p, [1.0, 1.0])

    def test_two_spins_half(self):
        """2 spins-1/2 → [1, 2, 1]."""
        p = equivsplit(0.5, 2)
        assert np.allclose(p, [1.0, 2.0, 1.0])

    def test_three_spins_half(self):
        """3 spins-1/2 → [1, 3, 3, 1]."""
        p = equivsplit(0.5, 3)
        assert np.allclose(p, [1.0, 3.0, 3.0, 1.0])

    def test_five_spins_half(self):
        """5 spins-1/2 → [1, 5, 10, 10, 5, 1]."""
        p = equivsplit(0.5, 5)
        assert np.allclose(p, [1.0, 5.0, 10.0, 10.0, 5.0, 1.0])

    def test_one_spin_1(self):
        """1 spin-1 → [1, 1, 1]."""
        p = equivsplit(1.0, 1)
        assert np.allclose(p, [1.0, 1.0, 1.0])

    def test_two_spins_1(self):
        """2 spins-1 → [1, 2, 3, 2, 1]."""
        p = equivsplit(1.0, 2)
        assert np.allclose(p, [1.0, 2.0, 3.0, 2.0, 1.0])

    def test_length(self):
        """Length of pattern is 2*n*I + 1."""
        for I in [0.5, 1.0, 1.5]:
            for n in [1, 2, 3]:
                p = equivsplit(I, n)
                assert len(p) == int(round(2 * n * I)) + 1

    def test_pattern_symmetric(self):
        """Pattern is symmetric."""
        for I in [0.5, 1.0]:
            for n in [2, 3, 4]:
                p = equivsplit(I, n)
                assert np.allclose(p, p[::-1])

    def test_bad_n_raises(self):
        with pytest.raises(ValueError):
            equivsplit(0.5, 0)

    def test_bad_I_raises(self):
        with pytest.raises(ValueError):
            equivsplit(0.3, 2)


# ---------------------------------------------------------------------------
# equivcouple
# ---------------------------------------------------------------------------

class TestEquivcouple:
    def test_single_nucleus(self):
        """n=1: returns (I,), (1,)."""
        F, N = equivcouple(0.5, 1)
        assert np.allclose(F, [0.5])
        assert np.allclose(N, [1])

    def test_two_spins_half(self):
        """2 × 1/2: decompose into [1, 0]."""
        F, N = equivcouple(0.5, 2)
        # 2 spins-1/2 = one spin-1 + one spin-0
        assert 1.0 in F
        assert 0.0 in F

    def test_three_spins_half(self):
        """3 × 1/2: [3/2, 1/2, 1/2]."""
        F, N = equivcouple(0.5, 3)
        assert 1.5 in F
        assert 0.5 in F

    def test_five_spins_half(self):
        """5 × 1/2 → F=[5/2, 3/2, 1/2], N=[1, 4, 5]."""
        F, N = equivcouple(0.5, 5)
        assert np.isclose(F[0], 2.5)
        assert np.isclose(F[1], 1.5)
        assert np.isclose(F[2], 0.5)
        assert np.allclose(N, [1, 4, 5])

    def test_intensity_sum_matches_equivsplit(self):
        """Sum over (2F+1)*N should match sum(equivsplit)."""
        for I in [0.5, 1.0]:
            for n in [2, 3, 4]:
                F, N = equivcouple(I, n)
                total_from_couple = np.sum((2 * F + 1) * N)
                total_from_split = np.sum(equivsplit(I, n))
                assert abs(total_from_couple - total_from_split) < 1e-10

    def test_f_decreasing(self):
        """F values should be in decreasing order."""
        F, _ = equivcouple(0.5, 5)
        assert np.all(np.diff(F) < 0)


# ---------------------------------------------------------------------------
# datasmooth
# ---------------------------------------------------------------------------

class TestDatasmooth:
    def test_m0_passthrough(self):
        """m=0 returns input unchanged."""
        y = np.random.default_rng(0).standard_normal(100)
        assert np.allclose(datasmooth(y, 0), y)

    def test_constant_signal(self):
        """Constant signal should be unchanged."""
        y = np.ones(100) * 5.0
        assert np.allclose(datasmooth(y, 3), y, atol=1e-12)

    def test_flat_weights_sum_to_one(self):
        """Flat smoothing of constant-1 signal → constant-1."""
        y = np.ones(200)
        y_s = datasmooth(y, 10, method='flat')
        assert np.allclose(y_s, y, atol=1e-10)

    def test_binom_reduces_noise(self):
        """Binomial smoothing reduces RMS noise."""
        rng = np.random.default_rng(42)
        y = np.sin(np.linspace(0, 4 * np.pi, 512)) + rng.standard_normal(512) * 0.5
        y_s = datasmooth(y, 10)
        assert y_s.std() < y.std()

    def test_savgol_preserves_low_freq(self):
        """Savitzky-Golay should preserve a linear signal."""
        x = np.linspace(0, 1, 200)
        y = 3.0 * x + 1.0   # perfectly linear
        y_s = datasmooth(y, 5, method='savgol', poly_order=2)
        assert np.allclose(y_s[10:-10], y[10:-10], atol=1e-8)

    def test_output_same_length(self):
        y = np.random.default_rng(1).standard_normal(137)
        for m in [1, 5, 10]:
            y_s = datasmooth(y, m)
            assert len(y_s) == len(y)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown method"):
            datasmooth(np.ones(50), 3, method='bad_method')


# ---------------------------------------------------------------------------
# rcfilt
# ---------------------------------------------------------------------------

class TestRcfilt:
    def test_zero_time_constant_passthrough(self):
        """tau=0 → identity (no filtering)."""
        y = np.random.default_rng(5).standard_normal(200)
        y_f = rcfilt(y, sample_time=1e-4, time_constant=0.0)
        assert np.allclose(y_f, y)

    def test_output_same_shape(self):
        y = np.ones(150)
        y_f = rcfilt(y, 1e-4, 1e-3)
        assert y_f.shape == y.shape

    def test_constant_signal_preserved(self):
        """Constant signal passes through RC filter unchanged (steady state)."""
        y = np.ones(500) * 3.0
        y_f = rcfilt(y, 1e-4, 5e-4)
        # After transient, output should reach 3.0
        assert abs(y_f[-1] - 3.0) < 1e-6

    def test_reduces_high_freq_noise(self):
        """RC filter reduces high-frequency noise."""
        rng = np.random.default_rng(7)
        y = rng.standard_normal(1000)
        y_f = rcfilt(y, sample_time=1e-4, time_constant=1e-3)
        assert y_f.std() < y.std()

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            rcfilt(np.ones(10), 1e-4, 1e-3, direction='sideways')

    def test_down_vs_up_different(self):
        """Up and down directions give different filtered results."""
        rng = np.random.default_rng(9)
        y = rng.standard_normal(200)
        y_up = rcfilt(y, 1e-4, 2e-3, direction='up')
        y_dn = rcfilt(y, 1e-4, 2e-3, direction='down')
        assert not np.allclose(y_up, y_dn)


# ---------------------------------------------------------------------------
# hilberttrans
# ---------------------------------------------------------------------------

class TestHilberttrans:
    def test_real_part_unchanged(self):
        """Real part of output equals input."""
        y = np.sin(np.linspace(0, 4 * np.pi, 512))
        y_h = hilberttrans(y)
        assert np.allclose(y_h.real, y, atol=1e-12)

    def test_imaginary_part_cos(self):
        """Hilbert transform of sin is -cos (quadrature shift)."""
        x = np.linspace(0, 10 * np.pi, 1024, endpoint=False)
        y = np.sin(x)
        y_h = hilberttrans(y)
        # Hilbert(sin) = -cos; ignoring edge transients
        center = slice(50, 974)
        assert np.allclose(y_h.imag[center], -np.cos(x[center]), atol=1e-4)

    def test_absorption_to_dispersion(self):
        """Hilbert transform of Gaussian gives Gaussian-shaped dispersion."""
        x = np.linspace(-5, 5, 1024)
        dx = x[1] - x[0]
        y = np.exp(-x ** 2)
        y_h = hilberttrans(y)
        disp = y_h.imag
        # Dispersion is antisymmetric
        assert np.allclose(disp + disp[::-1], 0.0, atol=1e-10)

    def test_output_complex(self):
        y = np.ones(64)
        y_h = hilberttrans(y)
        assert np.iscomplexobj(y_h)

    def test_2d_raises(self):
        y = np.ones((10, 10))
        with pytest.raises(ValueError, match="1D"):
            hilberttrans(y)

    def test_even_and_odd_length(self):
        """Both even and odd length arrays should work."""
        for n in [63, 64, 127, 128]:
            y = np.random.default_rng(n).standard_normal(n)
            y_h = hilberttrans(y)
            assert len(y_h) == n


# ---------------------------------------------------------------------------
# unitconvert
# ---------------------------------------------------------------------------

class TestUnitconvert:
    def test_cm1_to_MHz(self):
        """1 cm^-1 = c / 1 cm = 29 979.2458 MHz (exact)."""
        assert abs(float(unitconvert(1.0, 'cm^-1->MHz')) - 29979.2458) < 1e-6

    def test_cm1_against_codata(self):
        """cm^-1 conversions against exact SI values, not against the module's own constants.

        1 cm^-1 = 1.438 776 877 K (second radiation constant c2 = hc/k, exact)
                = 1.239 841 984e-4 eV (hc/e, exact)
        and back again.
        """
        assert abs(float(unitconvert(1.0, 'cm^-1->K')) - 1.438776877) < 1e-9
        assert abs(float(unitconvert(1.0, 'cm^-1->eV')) - 1.239841984e-4) < 1e-13
        assert abs(float(unitconvert(1.0, 'K->cm^-1')) - 1 / 1.438776877) < 1e-9
        assert abs(float(unitconvert(1.0, 'eV->cm^-1')) - 8065.543937) < 1e-5
        assert abs(float(unitconvert(29979.2458, 'MHz->cm^-1')) - 1.0) < 1e-12
        # cm^-1 -> mT at g = 2 against MHz -> mT of the same energy
        via_mhz = float(unitconvert(29979.2458, 'MHz->mT', g=2.0))
        assert abs(float(unitconvert(1.0, 'cm^-1->mT', g=2.0)) - via_mhz) < 1e-9 * via_mhz
        assert abs(float(unitconvert(via_mhz, 'mT->cm^-1', g=2.0)) - 1.0) < 1e-12

    def test_MHz_to_cm1_roundtrip(self):
        """MHz → cm^-1 → MHz should give identity."""
        v0 = 9500.0  # MHz
        v1 = unitconvert(v0, 'MHz->cm^-1')
        v2 = unitconvert(v1, 'cm^-1->MHz')
        assert abs(float(v2) - v0) < 1e-8

    def test_mT_to_MHz_gfree(self):
        """340 mT at g=2.0023 → ≈9520 MHz."""
        from torchspin.constants import GFREE, BMAGN, PLANCK
        v = unitconvert(340.0, 'mT->MHz', g=GFREE)
        expected = 340.0 * GFREE * (1e-3 * BMAGN / PLANCK / 1e6)
        assert abs(float(v) - expected) < 1e-6

    def test_K_to_MHz(self):
        """1 K → BOLTZMANN/PLANCK/1e6 MHz."""
        from torchspin.constants import BOLTZMANN, PLANCK
        v = unitconvert(1.0, 'K->MHz')
        expected = BOLTZMANN / PLANCK / 1e6
        assert abs(float(v) - expected) < 1e-4

    def test_mT_MHz_mT_roundtrip(self):
        """mT → MHz → mT should round-trip."""
        B0 = 340.0
        f = unitconvert(B0, 'mT->MHz', g=2.0)
        B1 = unitconvert(f, 'MHz->mT', g=2.0)
        assert abs(float(B1) - B0) < 1e-10

    def test_eV_K_eV_roundtrip(self):
        """eV → K → eV should round-trip."""
        e0 = 0.025  # eV (thermal energy at 290 K)
        T = unitconvert(e0, 'eV->K')
        e1 = unitconvert(T, 'K->eV')
        assert abs(float(e1) - e0) < 1e-12

    def test_array_input(self):
        """unitconvert accepts array inputs."""
        vals = np.array([300.0, 340.0, 380.0])
        out = unitconvert(vals, 'mT->MHz', g=2.0)
        assert out.shape == (3,)
        assert np.all(out > 0)

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="unknown conversion"):
            unitconvert(1.0, 'GHz->mT')

    def test_case_sensitive_suggestion(self):
        """Wrong case gives a helpful error suggestion."""
        with pytest.raises(ValueError, match="Did you mean"):
            unitconvert(1.0, 'mhz->mt')


# ---------------------------------------------------------------------------
# mhz2mt / mt2mhz
# ---------------------------------------------------------------------------

class TestMhz2mt:
    def test_roundtrip(self):
        """mt2mhz(mhz2mt(f)) == f."""
        from torchspin.utils import mhz2mt, mt2mhz
        f = 9500.0
        g = 2.0
        B = mhz2mt(f, g)
        f2 = mt2mhz(float(B), g)
        assert abs(float(f2) - f) < 1e-6

    def test_free_electron(self):
        """9.5 GHz ≈ 338.8 mT for g=2."""
        from torchspin.utils import mhz2mt
        from torchspin.constants import GFREE
        B = mhz2mt(9500.0, GFREE)
        assert abs(float(B) - 338.8) < 0.5

    def test_array_input(self):
        from torchspin.utils import mhz2mt
        import numpy as np
        freqs = np.array([9000.0, 9500.0, 10000.0])
        B = mhz2mt(freqs, 2.0)
        assert B.shape == (3,)
        assert np.all(B > 0)
        assert B[0] < B[1] < B[2]

    def test_default_g_is_gfree(self):
        from torchspin.utils import mhz2mt, mt2mhz
        from torchspin.constants import GFREE
        B1 = mhz2mt(9500.0)
        B2 = mhz2mt(9500.0, GFREE)
        assert abs(float(B1) - float(B2)) < 1e-10

    def test_mt2mhz_array(self):
        from torchspin.utils import mt2mhz
        import numpy as np
        B = np.array([300.0, 340.0, 380.0])
        f = mt2mhz(B, 2.0)
        assert f.shape == (3,)
        assert np.all(f > 0)


# ---------------------------------------------------------------------------
# hsdim
# ---------------------------------------------------------------------------

class TestHsdim:
    def test_spin_half(self):
        """S=1/2: dim = 2."""
        from torchspin.utils import hsdim
        assert hsdim([0.5]) == 2

    def test_spin_one(self):
        from torchspin.utils import hsdim
        assert hsdim([1.0]) == 3

    def test_two_spins(self):
        """S=1/2 + S=1/2: dim = 4."""
        from torchspin.utils import hsdim
        assert hsdim([0.5, 0.5]) == 4

    def test_mixed_system(self):
        """S=1/2 electron + I=1 nitrogen: dim = 2*3 = 6."""
        from torchspin.utils import hsdim
        assert hsdim([0.5, 1.0]) == 6

    def test_spinsystem_input(self):
        """hsdim accepts a SpinSystem."""
        from torchspin.utils import hsdim
        from torchspin.spinsystem import SpinSystem
        sys = SpinSystem(S=[0.5], Nucs=['14N'], A=[[10, 10, 10]])
        # S=1/2 + I=1 → 2*3 = 6
        assert hsdim(sys) == 6

    def test_scalar_input(self):
        """hsdim accepts a bare float."""
        from torchspin.utils import hsdim
        assert hsdim(0.5) == 2
        assert hsdim(1.0) == 3
        assert hsdim(1.5) == 4


def test_levelsplot_cm1_axis():
    """levelsplot(Units='cm^-1') puts 29 979.2458 MHz at 1 cm^-1 (was 0.01 cm^-1)."""
    import importlib
    import numpy as np
    lp = importlib.import_module('torchspin.levelsplot')
    E = np.array([0.0, 29979.2458, 2 * 29979.2458])
    np.testing.assert_allclose(lp._convert_energy_values(E, 'cm^-1'), [0.0, 1.0, 2.0], rtol=1e-12)
    E2, _, _ = lp._convert_energy(E, 'cm^-1')
    np.testing.assert_allclose(E2, [0.0, 1.0, 2.0], rtol=1e-12)
