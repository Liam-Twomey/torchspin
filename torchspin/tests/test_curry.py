"""Tests for torchspin/curry.py — magnetic susceptibility and magnetization."""
import math

import pytest
import torch

from torchspin.spinsystem import SpinSystem
from torchspin.curry import curry
from torchspin.constants import BMAGN, BOLTZMANN, PLANCK


# Physical constants used in tests
_MU0 = 4.0 * math.pi * 1e-7   # T·m/A
_N_A = 6.02214076e23           # mol⁻¹


# ---------------------------------------------------------------------------
# Helper: Curie law for reference values
# ---------------------------------------------------------------------------

def curie_law_chi_mol_SI(g, S, T):
    """Curie law molar susceptibility [m³/mol] for isotropic spin in SI."""
    return _MU0 * _N_A * g**2 * BMAGN**2 * S * (S + 1) / (3 * BOLTZMANN * T)


def curie_law_mu_BM(g, S, B, T):
    """
    Brillouin function magnetization in Bohr magnetons.

    At high T: mu_BM ≈ g² * S(S+1) * μ_B * B / (3 * k_B * T)
    """
    x = g * BMAGN * B * 1e-3 / (BOLTZMANN * T)   # B in mT → T
    # Brillouin function B_S(x)
    J = S
    return g * J * _brillouin(J, g * J * x / J)


def _brillouin(J, x):
    """Brillouin function for spin J and reduced field x = g*J*mu_B*B/(k_B*T)."""
    if abs(x) < 1e-10:
        return 0.0
    a = (2 * J + 1) / (2 * J)
    b = 1 / (2 * J)
    return a / math.tanh(a * x) - b / math.tanh(b * x)


# ---------------------------------------------------------------------------
# Basic output shape and type
# ---------------------------------------------------------------------------

class TestBasicOutput:
    def setup_method(self):
        self.sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])

    def test_returns_two_tensors(self):
        B = torch.tensor([340.0])
        T = torch.tensor([300.0])
        mu, chi = curry(self.sys, B, T, grid_size=5)
        assert isinstance(mu, torch.Tensor)
        assert isinstance(chi, torch.Tensor)

    def test_output_shape(self):
        B = torch.tensor([0.0, 100.0, 200.0, 340.0])
        T = torch.tensor([10.0, 100.0, 300.0])
        mu, chi = curry(self.sys, B, T, grid_size=5)
        assert mu.shape == (4, 3)
        assert chi.shape == (4, 3)

    def test_dtype_float64(self):
        B = torch.tensor([340.0])
        T = torch.tensor([300.0])
        mu, chi = curry(self.sys, B, T, grid_size=5)
        assert mu.dtype == torch.float64
        assert chi.dtype == torch.float64

    def test_scalar_inputs(self):
        """Scalar B and T are accepted and return shape (1, 1)."""
        mu, chi = curry(self.sys, 340.0, 300.0, grid_size=5)
        assert mu.shape == (1, 1)
        assert chi.shape == (1, 1)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError, match="positive"):
            curry(self.sys, 340.0, -10.0, grid_size=5)


# ---------------------------------------------------------------------------
# Physics: Curie law at high temperature
# ---------------------------------------------------------------------------

class TestCurieLaw:
    """At high T, chi_mol should follow the Curie law χ = C/T."""

    def test_chi_mol_curie_law_S_half(self):
        """
        For S=1/2, g=2.0, T=300 K (high T limit):
            χ_mol ≈ μ₀ N_A g² μ_B² S(S+1) / (3 k_B T) ≈ 1.57 × 10⁻⁸ m³/mol
        """
        g = 2.0
        S = 0.5
        T = 300.0
        sys = SpinSystem(S=[S], g=[[g, g, g]])
        B = torch.tensor([340.0])
        T_arr = torch.tensor([T])
        _, chi = curry(sys, B, T_arr, grid_size=15)
        chi_val = chi[0, 0].item()
        expected = curie_law_chi_mol_SI(g, S, T)
        rel_err = abs(chi_val - expected) / expected
        assert rel_err < 0.05, (
            f"chi_mol={chi_val:.3e} m³/mol, expected (Curie) {expected:.3e}, "
            f"relative error {rel_err*100:.1f}%"
        )

    def test_chi_T_matches_textbook_cgs_value(self):
        """χT = 0.375 cm³·K/mol for S = 1/2, g = 2: the familiar CGS Curie constant.

        curry returns SI (m³/mol); 1 cm³/mol (CGS) = 4π × 10⁻⁶ m³/mol. Checks the
        conversion stated in the docstring against a value independent of torchspin.
        """
        import math
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        _, chi = curry(sys, torch.tensor([10.0]), torch.tensor([300.0]), grid_size=15)
        chiT_cgs = chi[0, 0].item() * 1e6 / (4 * math.pi) * 300.0
        assert abs(chiT_cgs - 0.375) < 0.002, chiT_cgs

    def test_chi_mol_curie_law_S1(self):
        """Curie law for S=1, g=2.0 at T=300 K."""
        g = 2.0
        S = 1.0
        T = 300.0
        sys = SpinSystem(S=[S], g=[[g, g, g]])
        B = torch.tensor([340.0])
        T_arr = torch.tensor([T])
        _, chi = curry(sys, B, T_arr, grid_size=15)
        chi_val = chi[0, 0].item()
        expected = curie_law_chi_mol_SI(g, S, T)
        rel_err = abs(chi_val - expected) / expected
        assert rel_err < 0.05, (
            f"S=1: chi_mol={chi_val:.3e}, expected {expected:.3e}, "
            f"rel_err {rel_err*100:.1f}%"
        )

    def test_chi_mol_curie_law_S_3half(self):
        """Curie law for S=3/2, g=2.0 at T=300 K."""
        g = 2.0
        S = 1.5
        T = 300.0
        sys = SpinSystem(S=[S], g=[[g, g, g]])
        B = torch.tensor([340.0])
        T_arr = torch.tensor([T])
        _, chi = curry(sys, B, T_arr, grid_size=15)
        chi_val = chi[0, 0].item()
        expected = curie_law_chi_mol_SI(g, S, T)
        rel_err = abs(chi_val - expected) / expected
        assert rel_err < 0.05, (
            f"S=3/2: chi_mol={chi_val:.3e}, expected {expected:.3e}, "
            f"rel_err {rel_err*100:.1f}%"
        )

    def test_chi_times_T_is_constant(self):
        """χ · T should be constant (Curie law) for high T."""
        g = 2.0
        sys = SpinSystem(S=[0.5], g=[[g, g, g]])
        B = torch.tensor([340.0])
        T_arr = torch.tensor([100.0, 200.0, 300.0, 400.0])
        _, chi = curry(sys, B, T_arr, grid_size=15)
        # chi_mol * T should be ~constant (Curie constant C = χ*T)
        chi_T = chi[0, :] * T_arr
        # Variation should be < 5%
        cv = (chi_T.max() - chi_T.min()) / chi_T.mean()
        assert cv.item() < 0.05, (
            f"χ·T varies by {cv.item()*100:.1f}% over T range (should be ~0)"
        )


# ---------------------------------------------------------------------------
# Physics: magnetization
# ---------------------------------------------------------------------------

class TestMagnetization:
    def test_mu_zero_at_zero_field_high_T(self):
        """At B=0, the thermal average magnetic moment should be zero."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        B = torch.tensor([0.0])
        T = torch.tensor([300.0])
        mu, _ = curry(sys, B, T, grid_size=10)
        assert abs(mu[0, 0].item()) < 1e-10, (
            f"mu at B=0 is {mu[0,0].item():.2e} (expected 0)"
        )

    def test_mu_positive_at_positive_field(self):
        """Magnetic moment should be positive (paramagnetic) for positive B, low T."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        B = torch.tensor([340.0])
        T = torch.tensor([1.0])    # very low T → large polarization
        mu, _ = curry(sys, B, T, grid_size=10)
        assert mu[0, 0].item() > 0, "Magnetic moment should be positive for B>0"

    def test_mu_increases_with_field(self):
        """Magnetic moment should increase with increasing field."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        B = torch.linspace(0.0, 500.0, 6)
        T = torch.tensor([10.0])
        mu, _ = curry(sys, B, T, grid_size=10)
        mu_vals = mu[:, 0]
        assert (mu_vals[1:] > mu_vals[:-1]).all(), "mu should increase monotonically with B"

    def test_mu_decreases_with_temperature(self):
        """Magnetic moment should decrease with increasing temperature."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        B = torch.tensor([340.0])
        T = torch.tensor([1.0, 10.0, 100.0, 300.0])
        mu, _ = curry(sys, B, T, grid_size=10)
        mu_vals = mu[0, :]
        assert (mu_vals[:-1] > mu_vals[1:]).all(), "mu should decrease with T"

    def test_mu_saturation_low_T_S_half(self):
        """
        At very low T and high B, S=1/2 saturates to 1 Bohr magneton.
        (Moment = g·S = 2×0.5 = 1 μ_B at full polarization)
        """
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        B = torch.tensor([5000.0])   # very high field (5 T)
        T = torch.tensor([0.1])      # very low T
        mu, _ = curry(sys, B, T, grid_size=10)
        # Should approach 1 Bohr magneton
        assert abs(mu[0, 0].item() - 1.0) < 0.1, (
            f"Saturation moment: {mu[0,0].item():.3f} μ_B (expected ~1.0)"
        )


# ---------------------------------------------------------------------------
# Physics: susceptibility properties
# ---------------------------------------------------------------------------

class TestSusceptibility:
    def test_chi_positive(self):
        """Susceptibility must be positive for a paramagnetic system."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        B = torch.tensor([340.0])
        T = torch.tensor([300.0])
        _, chi = curry(sys, B, T, grid_size=10)
        assert chi[0, 0].item() > 0

    def test_chi_increases_at_low_T(self):
        """Susceptibility increases as temperature decreases (Curie law)."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        B = torch.tensor([340.0])
        T = torch.tensor([10.0, 100.0, 300.0])
        _, chi = curry(sys, B, T, grid_size=10)
        assert chi[0, 0] > chi[0, 1] > chi[0, 2], "χ should increase as T decreases"

    def test_chi_g_anisotropy(self):
        """
        For an anisotropic g-tensor, the powder-averaged susceptibility
        should differ from the isotropic case with g=gxx.
        """
        sys_aniso = SpinSystem(S=[0.5], g=[[2.00, 2.05, 2.15]])
        sys_iso = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
        B = torch.tensor([340.0])
        T = torch.tensor([300.0])
        _, chi_aniso = curry(sys_aniso, B, T, grid_size=15)
        _, chi_iso = curry(sys_iso, B, T, grid_size=15)
        # Anisotropic system has larger effective g → larger susceptibility
        assert chi_aniso[0, 0] > chi_iso[0, 0], (
            "Anisotropic g-tensor should give larger average susceptibility"
        )


# ---------------------------------------------------------------------------
# ZFS effects on susceptibility
# ---------------------------------------------------------------------------

class TestZFSEffects:
    def test_zfs_reduces_chi_at_low_T(self):
        """
        For S=1 with large ZFS (D ≫ k_B·T), the ZFS splits the ground state
        population and dramatically changes the susceptibility.

        ZFS D=10,000 MHz ≈ 480 mK in temperature units.
        At T=0.1 K (k_B·T ≈ 2,084 MHz) and B=5 mT (Zeeman ≈ 140 MHz ≪ D):
        - The m_s=0 state is the ground state (with our D sign convention)
        - The m_s=±1 states are ≈D above the ground state → nearly empty
        - This suppresses the parallel susceptibility and reduces chi_mol
          by >20% compared to the Curie (no-ZFS) value

        Note: the ZFS effect is strongest at low B (Zeeman ≪ D) and low T.
        At T=1 K, B=340 mT the Zeeman ≈ D ≈ k_B·T so the effect is small.
        """
        D_val = 10000.0   # MHz → temperature equivalent ≈ 480 mK
        sys_zfs = SpinSystem(
            S=[1],
            g=[[2.0, 2.0, 2.0]],
            D=[[-D_val / 3, -D_val / 3, 2 * D_val / 3]],  # axial D>0 → m_s=0 ground
        )
        sys_no = SpinSystem(S=[1], g=[[2.0, 2.0, 2.0]])

        # Conditions: D ≫ k_B·T ≫ Zeeman → ZFS dominates
        B = torch.tensor([5.0])     # mT → Zeeman ≈ 140 MHz ≪ D=10,000 MHz
        T = torch.tensor([0.1])     # K → k_B·T ≈ 2,084 MHz (D/k_B·T ≈ 4.8)

        _, chi_zfs = curry(sys_zfs, B, T, grid_size=15)
        _, chi_no = curry(sys_no, B, T, grid_size=15)

        # ZFS significantly reduces chi compared to Curie (no-ZFS) at low T
        assert chi_zfs[0, 0] < chi_no[0, 0], \
            "ZFS (D>0) ground state m_s=0 should reduce chi vs no-ZFS"
        rel_diff = (chi_no[0, 0] - chi_zfs[0, 0]) / chi_no[0, 0]
        assert rel_diff.item() > 0.20, (
            f"ZFS should reduce chi by >20% at T=0.1K, D=10GHz; got {rel_diff.item()*100:.1f}%"
        )
