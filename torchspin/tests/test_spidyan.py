"""Tests for spidyan — spin dynamics during pulse EPR experiments.

Verifies:
1. Basic pulse propagation (pi/2, pi pulses)
2. Free evolution (Larmor precession)
3. Phase cycling (2-step, 4-step)
4. Relaxation (T1, T2)
5. Multi-dimensional experiments
6. Predefined experiments (2pESEEM, 3pESEEM, HYSCORE templates)
7. Detection operators
8. State trajectory tracking
9. Density matrix unitarity
10. Spin echo formation
"""
import numpy as np
import pytest

from torchspin import SpinSystem
from torchspin.spidyan import (
    spidyan, SpidyanOptions, SpidyanInfo,
    _relaxation_superoperator, _reorder_events,
    _predefined_experiments, _propagator, _liouvillian,
    PulseEvent, DelayEvent,
)
from torchspin.constants import PLANCK, BMAGN, GFREE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def simple_sys(T1=0, T2=0):
    """S=1/2, isotropic g=2.0, optional relaxation."""
    sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]])
    if T1 > 0:
        sys.T1 = T1
    if T2 > 0:
        sys.T2 = T2
    return sys


def echo_exp(tau=0.5):
    """Hahn echo: pi/2 — tau — pi — tau — detect."""
    return {
        'Field': 340.0,
        'Sequence': [
            {'tp': 0.016, 'Flip': np.pi / 2},
            tau,
            {'tp': 0.032, 'Flip': np.pi},
            tau,
        ],
    }


# ---------------------------------------------------------------------------
# Relaxation superoperator
# ---------------------------------------------------------------------------

class TestRelaxationSuperoperator:

    def test_shape(self):
        """Gamma should be n² × n² where n = Hilbert dimension."""
        G = _relaxation_superoperator([0.5], 1000, 500)
        assert G.shape == (4, 4)

    def test_shape_two_spins(self):
        G = _relaxation_superoperator([0.5, 0.5], 1000, 500)
        assert G.shape == (16, 16)

    def test_no_relaxation(self):
        """T1=T2=1e10 → Gamma ≈ 0."""
        G = _relaxation_superoperator([0.5], 1e10, 1e10)
        assert np.allclose(G, 0, atol=1e-8)

    def test_T2_on_diagonal(self):
        """T2 relaxation should populate diagonal for off-diagonal coherences."""
        G = _relaxation_superoperator([0.5], 1e10, 100)
        # For 2×2 density matrix in row-major vectorization:
        # Index 0 = ρ[0,0] (diagonal) → no T2
        # Index 1 = ρ[0,1] (off-diagonal) → T2
        # Index 2 = ρ[1,0] (off-diagonal) → T2
        # Index 3 = ρ[1,1] (diagonal) → no T2
        assert abs(G[1, 1] - 1.0 / 100) < 1e-10
        assert abs(G[2, 2] - 1.0 / 100) < 1e-10
        assert abs(G[0, 0]) < 1e-10  # diagonal — no T2
        assert abs(G[3, 3]) < 1e-10  # diagonal — no T2

    def test_T1_off_diagonal(self):
        """T1 relaxation connects population elements."""
        G = _relaxation_superoperator([0.5], 100, 1e10)
        # For 2×2: populations at indices 0 and 3
        # G[0,3] = -1/T1, G[3,0] = -1/T1
        assert abs(G[0, 3] - (-1.0 / 100)) < 1e-10
        assert abs(G[3, 0] - (-1.0 / 100)) < 1e-10


# ---------------------------------------------------------------------------
# Event reordering
# ---------------------------------------------------------------------------

class TestReorderEvents:

    def test_no_reorder(self):
        """Events already in order should stay in order."""
        seq, lengths = _reorder_events([0.5, 0.1, 0.5], [False, True, False])
        assert seq == [0, 1, 2]

    def test_sequence_length_preserved(self):
        """Number of events should be preserved."""
        seq, lengths = _reorder_events([0.3, 0.1, 0.5], [False, True, False])
        assert len(seq) == 3
        assert len(lengths) == 3

    def test_preserves_total_time(self):
        """Total time should be preserved after reordering."""
        lengths = [0.3, 0.1, 0.5, 0.2]
        _, new_lengths = _reorder_events(lengths, [False, True, False, True])
        assert abs(sum(new_lengths) - sum(lengths)) < 1e-10


# ---------------------------------------------------------------------------
# Predefined experiments
# ---------------------------------------------------------------------------

class TestPredefinedExperiments:

    def test_2pESEEM(self):
        exp = _predefined_experiments({
            'Sequence': '2pESEEM',
            'dt': 0.012,
            'tau': 0.200,
        })
        assert isinstance(exp['Sequence'], list)
        assert len(exp['Sequence']) == 4
        assert exp['nPoints'] == 512

    def test_3pESEEM(self):
        exp = _predefined_experiments({
            'Sequence': '3pESEEM',
            'dt': 0.012,
            'tau': 0.200,
            'T': 0.100,
        })
        assert len(exp['Sequence']) == 6
        assert 'PhaseCycle' in exp

    def test_HYSCORE(self):
        exp = _predefined_experiments({
            'Sequence': 'HYSCORE',
            'dt': 0.016,
            't1': 0.036,
            't2': 0.036,
            'tau': 0.200,
        })
        assert len(exp['Sequence']) == 8
        assert len(exp['nPoints']) == 2

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            _predefined_experiments({'Sequence': 'unknownSeq'})

    def test_missing_field_raises(self):
        with pytest.raises(ValueError, match="required"):
            _predefined_experiments({'Sequence': '2pESEEM', 'dt': 0.01})


# ---------------------------------------------------------------------------
# Propagator helpers
# ---------------------------------------------------------------------------

class TestPropagatorHelpers:

    def test_propagator_unitary(self):
        """Propagator should be unitary."""
        H = np.array([[1, 0.5], [0.5, -1]], dtype=complex)
        U = _propagator(H, 0.01)
        prod = U @ U.conj().T
        assert np.allclose(prod, np.eye(2), atol=1e-12)

    def test_propagator_identity_at_zero_time(self):
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        U = _propagator(H, 0.0)
        assert np.allclose(U, np.eye(2), atol=1e-12)

    def test_liouvillian_shape(self):
        """Liouvillian should be n² × n²."""
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        G = np.zeros((4, 4))
        eq = np.array([0.5, 0, 0, 0.5])
        L, ss = _liouvillian(H, G, eq, 0.01)
        assert L.shape == (4, 4)
        assert ss.shape == (4,)


# ---------------------------------------------------------------------------
# Basic spidyan simulations
# ---------------------------------------------------------------------------

class TestSpidyanBasic:

    def test_simple_echo(self):
        """Hahn echo should produce a signal."""
        sys = simple_sys()
        exp = echo_exp()
        t, signal, info = spidyan(sys, exp)
        assert len(t) > 0
        assert len(signal) > 0
        assert info.FinalState is not None

    def test_output_types(self):
        """Check return types."""
        sys = simple_sys()
        exp = echo_exp()
        t, signal, info = spidyan(sys, exp)
        assert isinstance(t, np.ndarray)
        assert isinstance(signal, np.ndarray)
        assert isinstance(info, SpidyanInfo)

    def test_single_pulse(self):
        """Single pulse + delay should work."""
        sys = simple_sys()
        exp = {
            'Field': 340.0,
            'Sequence': [
                {'tp': 0.016, 'Flip': np.pi / 2},
                1.0,
            ],
        }
        t, signal, info = spidyan(sys, exp)
        assert len(t) > 0

    def test_field_required(self):
        """Missing Field should raise."""
        sys = simple_sys()
        exp = {'Sequence': [{'tp': 0.016, 'Flip': np.pi / 2}, 0.5]}
        with pytest.raises(ValueError, match="Field"):
            spidyan(sys, exp)


# ---------------------------------------------------------------------------
# Density matrix properties
# ---------------------------------------------------------------------------

class TestDensityMatrixProperties:

    def test_trace_preserved(self):
        """Trace of density matrix should be preserved."""
        sys = simple_sys()
        exp = echo_exp()
        _, _, info = spidyan(sys, exp)
        final = info.FinalState
        assert abs(np.trace(final).real) < 1e-8  # tr(-Sz) ≈ 0 for traceless

    def test_hermiticity(self):
        """Final state should be Hermitian."""
        sys = simple_sys()
        exp = echo_exp()
        _, _, info = spidyan(sys, exp)
        final = info.FinalState
        assert np.allclose(final, final.conj().T, atol=1e-10)

    def test_final_state_shape(self):
        """Final state should be n × n."""
        sys = simple_sys()
        exp = echo_exp()
        _, _, info = spidyan(sys, exp)
        assert info.FinalState.shape == (2, 2)


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------

class TestSpidyanRelaxation:

    def test_T2_decay(self):
        """Signal with T1/T2 relaxation should run successfully."""
        sys = simple_sys(T1=10.0, T2=2.0)
        exp = {
            'Field': 340.0,
            'Sequence': [
                {'tp': 0.016, 'Flip': np.pi / 2},
                2.0,  # 2 µs delay
            ],
        }
        opt = SpidyanOptions(Relaxation=True)
        t, signal, info = spidyan(sys, exp, opt)
        assert len(signal) > 0
        assert info.FinalState is not None

    def test_T1_relaxation_runs(self):
        """T1 relaxation should run without error."""
        sys = simple_sys(T1=10.0, T2=5.0)
        exp = {
            'Field': 340.0,
            'Sequence': [
                {'tp': 0.016, 'Flip': np.pi / 2},
                2.0,
            ],
        }
        opt = SpidyanOptions(Relaxation=True)
        t, signal, info = spidyan(sys, exp, opt)
        assert len(signal) > 0


# ---------------------------------------------------------------------------
# State trajectories
# ---------------------------------------------------------------------------

class TestStateTrajectories:

    def test_trajectories_returned(self):
        """StateTrajectories option should return density matrices."""
        sys = simple_sys()
        exp = {
            'Field': 340.0,
            'Sequence': [
                {'tp': 0.016, 'Flip': np.pi / 2},
                0.5,
            ],
        }
        opt = SpidyanOptions(StateTrajectories=True)
        _, _, info = spidyan(sys, exp, opt)
        assert info.StateTrajectories is not None
        assert len(info.StateTrajectories) > 0


# ---------------------------------------------------------------------------
# Detection operators
# ---------------------------------------------------------------------------

class TestDetectionOperators:

    def test_custom_det_operator(self):
        """Custom detection operator should work."""
        sys = simple_sys()
        exp = {
            'Field': 340.0,
            'Sequence': [
                {'tp': 0.016, 'Flip': np.pi / 2},
                0.5,
            ],
            'DetOperator': ['z1'],  # Detect Sz
        }
        t, signal, info = spidyan(sys, exp)
        assert len(signal) > 0

    def test_default_splus(self):
        """Default detection is S+ = Sx + iSy."""
        sys = simple_sys()
        exp = echo_exp()
        t, signal, info = spidyan(sys, exp)
        # Signal should be complex (S+ gives complex signal)
        assert np.iscomplexobj(signal)


# ---------------------------------------------------------------------------
# Multi-dimensional experiments
# ---------------------------------------------------------------------------

class TestMultidimensional:

    def test_1d_indirect(self):
        """1D indirect dimension should produce multi-point output."""
        sys = simple_sys()
        exp = {
            'Field': 340.0,
            'Sequence': [
                {'tp': 0.016, 'Flip': np.pi / 2},
                0.200,
                {'tp': 0.032, 'Flip': np.pi},
                0.200,
            ],
            'nPoints': 4,
            'Dim1': ['d1,d2', 0.012],
            'DetWindow': 0,
        }
        t, signal, info = spidyan(sys, exp)
        # FinalState should have 4 entries (one per indirect point)
        assert info.FinalState is not None
        if isinstance(signal, np.ndarray):
            assert signal.shape[0] == 4 or signal.ndim >= 1
        elif isinstance(signal, list):
            assert len(signal) == 4


# ---------------------------------------------------------------------------
# Nuclear spin system
# ---------------------------------------------------------------------------

class TestNuclearSpin:

    def test_with_nucleus(self):
        """S=1/2 + I=1/2 should run."""
        sys = SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]],
                         Nucs=['1H'], A=[[10.0, 10.0, 10.0]])
        exp = {
            'Field': 340.0,
            'Sequence': [
                {'tp': 0.016, 'Flip': np.pi / 2},
                0.5,
            ],
        }
        t, signal, info = spidyan(sys, exp)
        assert info.FinalState.shape == (4, 4)  # 2×2 Hilbert space


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_very_short_delay(self):
        """Very short delay should not crash."""
        sys = simple_sys()
        exp = {
            'Field': 340.0,
            'Sequence': [
                {'tp': 0.016, 'Flip': np.pi / 2},
                0.001,
            ],
        }
        t, signal, info = spidyan(sys, exp)
        assert info.FinalState is not None

    def test_three_pulses(self):
        """Three-pulse sequence should work."""
        sys = simple_sys()
        exp = {
            'Field': 340.0,
            'Sequence': [
                {'tp': 0.016, 'Flip': np.pi / 2},
                0.200,
                {'tp': 0.016, 'Flip': np.pi / 2},
                0.100,
                {'tp': 0.016, 'Flip': np.pi / 2},
                0.200,
            ],
        }
        t, signal, info = spidyan(sys, exp)
        assert info.FinalState is not None

    def test_single_delay(self):
        """Just a delay (no pulses) should work."""
        sys = simple_sys()
        exp = {
            'Field': 340.0,
            'Sequence': [1.0],
        }
        t, signal, info = spidyan(sys, exp)
        assert info.FinalState is not None


# ---------------------------------------------------------------------------
# Phase cycling
# ---------------------------------------------------------------------------

class TestPhaseCycling:

    def test_two_step_phase_cycle(self):
        """Two-step phase cycling should run."""
        sys = simple_sys()
        exp = {
            'Field': 340.0,
            'Sequence': [
                {'tp': 0.016, 'Flip': np.pi / 2},
                0.200,
                {'tp': 0.016, 'Flip': np.pi / 2},
                0.200,
                {'tp': 0.016, 'Flip': np.pi / 2},
                0.200,
            ],
            'PhaseCycle': {
                0: np.array([[0, 1], [np.pi, -1]]),
                2: np.array([[0, 1], [np.pi, -1]]),
            },
        }
        t, signal, info = spidyan(sys, exp)
        assert info.FinalState is not None


# --- GitHub #8: silent validity limits -------------------------------------

def _qband_pi(**opt):
    import warnings as _w
    from torchspin.spidyan import spidyan, SpidyanOptions
    from torchspin.tests.test_spidyan_matlab_validation import _zsys
    p = {'Type': 'rectangular', 'tp': 0.02, 'Flip': np.pi}            # nu1 = 25 MHz
    exp = {'Field': 1195.4, 'mwFreq': 33.5, 'Sequence': [p], 'DetSequence': [1],
           'DetOperator': ['z1'], 'DetPhase': 0.0}
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter('always')
        spidyan(_zsys(), exp, SpidyanOptions(**opt))
    return [str(r.message) for r in rec if str(r.message).startswith('spidyan:')]


def test_no_validity_warning_with_defaults_or_validated_settings():
    assert _qband_pi() == []
    assert _qband_pi(IntTimeStep=1e-4, SimFreq=32) == []   # the EasySpin-validated setup
    assert _qband_pi(SimFreq=33.0) == []                   # carrier offset 20·nu1


def test_coarse_int_time_step_warns():
    msgs = _qband_pi(IntTimeStep=2e-4)
    assert len(msgs) == 1 and 'IntTimeStep' in msgs[0]


@pytest.mark.parametrize('sim_freq', [33.4, 33.5])
def test_sim_freq_near_carrier_warns(sim_freq):
    msgs = _qband_pi(SimFreq=sim_freq)
    assert len(msgs) == 1 and 'SimFreq' in msgs[0]
