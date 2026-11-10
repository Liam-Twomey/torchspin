"""Tests for Tier 3c: ham_ezho (Higher-Order Zeeman Hamiltonian).

Tests cover:
- SpinSystem with Ham dict
- ham_ezho direct Hamiltonian computation
- ham_ezho tensor decomposition (G0, G1, G2, G3)
- Hermiticity
- Consistency between H(B) and tensor decomposition
"""
import math

import numpy as np
import pytest
import torch

from torchspin import SpinSystem
from torchspin.ham_ezho import ham_ezho


# ============================================================================
# SpinSystem Ham Extension
# ============================================================================

class TestSpinSystemHam:
    """SpinSystem with Ham dict."""

    def test_ham_none_by_default(self):
        """Ham is None when not provided."""
        sys = SpinSystem(S=[0.5])
        assert sys.Ham is None

    def test_ham_dict_preserved(self):
        """Ham dict is preserved and tensors are normalized."""
        sys = SpinSystem(S=[0.5], Ham={'112': [[1.0, 0, 0, 0, 0.5]]})
        assert '112' in sys.Ham
        assert sys.Ham['112'].shape == (1, 5)
        assert sys.Ham['112'].dtype == torch.float64

    def test_ham_1d_expanded(self):
        """1-D Ham value is expanded to 2-D."""
        sys = SpinSystem(S=[0.5], Ham={'110': [1.0]})
        assert sys.Ham['110'].ndim == 2
        assert sys.Ham['110'].shape == (1, 1)

    def test_ham_multiple_keys(self):
        """Multiple Ham keys stored correctly."""
        sys = SpinSystem(S=[0.5], Ham={
            '110': [[2.0]],
            '112': [[0.1, 0, 0, 0, 0.05]],
        })
        assert '110' in sys.Ham
        assert '112' in sys.Ham


# ============================================================================
# ham_ezho — Direct Hamiltonian
# ============================================================================

class TestHamEzho:
    """Higher-order Zeeman Hamiltonian tests."""

    def test_no_ham_returns_zero(self):
        """Returns zero when no Ham parameters are present."""
        sys = SpinSystem(S=[0.5])
        H = ham_ezho(sys, B0=[0, 0, 340])
        assert torch.allclose(H, torch.zeros_like(H))

    def test_hermitian(self):
        """Output is Hermitian."""
        sys = SpinSystem(S=[0.5], Ham={'112': [[1.0, 0.5, 0, 0.2, 0.1]]})
        H = ham_ezho(sys, B0=[100, 200, 340])
        assert torch.allclose(H, H.conj().T, atol=1e-10)

    def test_hermitian_multiple_orders(self):
        """Hermitian with multiple field orders."""
        sys = SpinSystem(S=[1.0], Ham={
            '110': [[2.0]],
            '112': [[0.1, 0, 0, 0, 0.05]],
            '220': [[0.01]],
        })
        H = ham_ezho(sys, B0=[50, 100, 340])
        assert torch.allclose(H, H.conj().T, atol=1e-10)

    def test_zero_field_lB1_gives_zero(self):
        """lB=1 terms vanish at zero field."""
        sys = SpinSystem(S=[0.5], Ham={'112': [[1.0, 0, 0, 0, 0.5]]})
        H = ham_ezho(sys, B0=[0, 0, 0])
        assert torch.allclose(H, torch.zeros_like(H), atol=1e-14)

    def test_linear_in_field_for_lB1(self):
        """lB=1 terms scale linearly with field magnitude."""
        sys = SpinSystem(S=[0.5], Ham={'112': [[1.0, 0, 0, 0, 0.5]]})
        H1 = ham_ezho(sys, B0=[0, 0, 100])
        H2 = ham_ezho(sys, B0=[0, 0, 200])
        # lB=1 only, so H should scale linearly
        assert torch.allclose(H2, 2 * H1, atol=1e-10)

    def test_quadratic_in_field_for_lB2(self):
        """lB=2 terms scale quadratically with field magnitude."""
        sys = SpinSystem(S=[1.0], Ham={'220': [[1.0]]})
        H1 = ham_ezho(sys, B0=[0, 0, 100])
        H2 = ham_ezho(sys, B0=[0, 0, 200])
        # lB=2 only, so H should scale as B^2
        assert torch.allclose(H2, 4 * H1, atol=1e-10)

    def test_lB_select(self):
        """lB_select filters specific orders."""
        sys = SpinSystem(S=[0.5], Ham={
            '110': [[2.0]],
            '112': [[0.1, 0, 0, 0, 0]],
        })
        B0 = [0, 0, 340]
        H_all = ham_ezho(sys, B0=B0)
        H_0 = ham_ezho(sys, B0=B0, lB_select=0)
        H_1 = ham_ezho(sys, B0=B0, lB_select=1)
        # lB=0 terms should give zero at any field (no Ham0XX defined)
        # H_all should equal H_1 since we only have lB=1 terms
        assert torch.allclose(H_all, H_0 + H_1, atol=1e-10)

    def test_correct_shape(self):
        """Output has correct shape."""
        sys = SpinSystem(S=[1.0], Ham={'112': [[1.0, 0, 0, 0, 0]]})
        H = ham_ezho(sys, B0=[0, 0, 340])
        assert H.shape == (3, 3)

    def test_eSpins_selection(self):
        """eSpins selects specific electron spins."""
        sys = SpinSystem(S=[0.5, 1.0], Ham={
            '112': [[1.0, 0, 0, 0, 0],
                     [0.5, 0, 0, 0, 0]],
        })
        B0 = [0, 0, 340]
        H_all = ham_ezho(sys, B0=B0)
        H_1 = ham_ezho(sys, B0=B0, eSpins=[1])
        H_2 = ham_ezho(sys, B0=B0, eSpins=[2])
        assert torch.allclose(H_all, H_1 + H_2, atol=1e-10)

    def test_field_direction_dependence(self):
        """Anisotropic terms give different results for different field directions."""
        sys = SpinSystem(S=[0.5], Ham={'112': [[1.0, 0, 0, 0, 0.5]]})
        H_z = ham_ezho(sys, B0=[0, 0, 340])
        H_x = ham_ezho(sys, B0=[340, 0, 0])
        # With non-trivial CF2 parameters, x and z should differ
        assert not torch.allclose(H_z, H_x, atol=1e-6)


# ============================================================================
# ham_ezho — Tensor Decomposition
# ============================================================================

class TestHamEzhoTensor:
    """Tensor decomposition (G0, G1, G2, G3) tests."""

    def test_tensor_output_shapes(self):
        """Tensor output has correct shapes."""
        sys = SpinSystem(S=[0.5], Ham={'112': [[1.0, 0, 0, 0, 0]]})
        G0, G1, G2, G3 = ham_ezho(sys)
        assert G0.shape == (2, 2)
        assert len(G1) == 3
        assert all(g.shape == (2, 2) for g in G1)
        assert len(G2) == 3 and all(len(row) == 3 for row in G2)
        assert len(G3) == 3

    def test_tensor_G0_zero_for_lB1(self):
        """G0 is zero when only lB=1 terms exist."""
        sys = SpinSystem(S=[0.5], Ham={'112': [[1.0, 0, 0, 0, 0]]})
        G0, G1, G2, G3 = ham_ezho(sys)
        assert torch.allclose(G0, torch.zeros_like(G0), atol=1e-14)

    def test_tensor_G1_nonzero_for_lB1(self):
        """G1 is non-zero when lB=1 terms exist."""
        sys = SpinSystem(S=[0.5], Ham={'112': [[1.0, 0, 0, 0, 0.5]]})
        G0, G1, G2, G3 = ham_ezho(sys)
        has_nonzero = any(not torch.allclose(g, torch.zeros_like(g), atol=1e-10)
                         for g in G1)
        assert has_nonzero

    def test_tensor_G1_hermitian(self):
        """G1 components are Hermitian."""
        sys = SpinSystem(S=[0.5], Ham={'112': [[1.0, 0.5, 0, 0.2, 0.1]]})
        G0, G1, G2, G3 = ham_ezho(sys)
        for g in G1:
            assert torch.allclose(g, g.conj().T, atol=1e-10)

    def test_tensor_consistency_lB1(self):
        """H(B) ≈ G0 + sum_i G1_i * B_i for lB=1 system."""
        sys = SpinSystem(S=[0.5], Ham={'112': [[1.0, 0, 0, 0, 0.5]]})
        G0, G1, G2, G3 = ham_ezho(sys)

        B = [50.0, 100.0, 340.0]
        H_direct = ham_ezho(sys, B0=B)
        H_tensor = G0 + G1[0] * B[0] + G1[1] * B[1] + G1[2] * B[2]
        assert torch.allclose(H_direct, H_tensor, atol=1e-8)

    def test_tensor_consistency_lB2(self):
        """H(B) includes G2 terms for lB=2 system."""
        sys = SpinSystem(S=[1.0], Ham={
            '112': [[1.0, 0, 0, 0, 0]],
            '220': [[0.01]],
        })
        G0, G1, G2, G3 = ham_ezho(sys)

        B = [0.0, 0.0, 340.0]
        H_direct = ham_ezho(sys, B0=B)

        # Reconstruct: H = G0 + G1·B + G2:B⊗B
        H_tensor = G0.clone()
        for i in range(3):
            H_tensor = H_tensor + G1[i] * B[i]
        for i in range(3):
            for j in range(3):
                H_tensor = H_tensor + G2[i][j] * B[i] * B[j]

        assert torch.allclose(H_direct, H_tensor, atol=1e-6)

    def test_no_ham_returns_zeros(self):
        """Tensor output is all zeros when no Ham parameters."""
        sys = SpinSystem(S=[0.5])
        G0, G1, G2, G3 = ham_ezho(sys)
        assert torch.allclose(G0, torch.zeros_like(G0))
        assert all(torch.allclose(g, torch.zeros_like(g)) for g in G1)


def test_linear_ham_terms_same_sign_with_and_without_field():
    """ham(sys, B0) == H0 - mu.B when Sys.Ham has linear-in-B terms (B11).

    The moment operators used to carry +dH/dB, so the linear Sys.Ham terms entered
    the field-free decomposition (used by pepper, levels, salt, ...) with the
    opposite sign to ham(sys, B0) and to the nuclear Zeeman term.
    """
    from torchspin.ham import ham
    B = [100.0, 200.0, 340.0]
    for Ham in ({'110': [1.0]}, {'112': [[1.0, 0.5, 0.0, 0.2, 0.1]]}, {'110': [0.4], '112': [[0.3, 0.1, 0.2, 0.0, 0.5]]}):
        for nuc in ({}, dict(Nucs='1H', A=[5.0, 6.0, 9.0])):
            s = SpinSystem(S=[1.0], Ham=Ham, **nuc)
            H0, mx, my, mz = ham(s)
            Hd = H0 - (mx * B[0] + my * B[1] + mz * B[2])
            assert torch.allclose(ham(s, B0=B), Hd, atol=1e-9 * float(Hd.abs().max())), (Ham, nuc)


def test_isotropic_g_given_as_one_element_list():
    """g=[2.0] means isotropic g = 2 (as g=2.0 does), not an IndexError."""
    from torchspin.ham import ham
    for S in ([0.5], [1.0]):
        a, b = SpinSystem(S=S, g=[2.0]), SpinSystem(S=S, g=2.0)
        assert torch.equal(a.g, b.g)
        assert torch.allclose(ham(a, B0=[0, 0, 340]), ham(b, B0=[0, 0, 340]))

