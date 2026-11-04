"""Tests for saffron's thyme engine — real-pulse time-domain propagation.

These are the internal-consistency and API tests. The MATLAB cross-validation
lives in ``test_saffron_thyme_matlab_validation.py``; nothing here should be
read as evidence of EasySpin agreement, which is why the checks below are
relations between torchspin runs (shape contracts, limits, monotonicity)
rather than absolute values.

Verifies:
1. Engine selection — Opt.SimulationMode and the Exp.tp / Exp.DetWindow
   inference, and the errors the fast algorithm raises instead
2. The output contract: single-point vs transient detection
3. Custom sequences, both EasySpin's list syntax and saffron's Flip/tp arrays
4. Relaxation (T1/T2), phase cycling, multi-dimensional experiments
5. Exp.mwFreq is required, as it is in EasySpin's s_sequencer
"""

import numpy as np
import pytest

from torchspin import SpinSystem
from torchspin.constants import BMAGN, PLANCK
from torchspin.saffron import saffron, PulseExperiment, SaffronOptions
from torchspin.saffron_thyme import saffron_thyme, _saffron_exp_to_spidyan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Rectangular pulses in EasySpin's thyme syntax (Flip in radians).
P90 = {'tp': 0.016, 'Flip': np.pi / 2}
P180 = {'tp': 0.032, 'Flip': np.pi}

# Opt.IntTimeStep is stated everywhere: EasySpin's default is derived from the
# Nyquist criterion and rounded, so leaving it implicit makes these tests
# sensitive to a formula change that has nothing to do with what they check.
OPT = dict(SimulationMode='thyme', IntTimeStep=1e-4, GridSize=5)


def opt(**kw):
    return SaffronOptions(**{**OPT, **kw})


def _simple_sys():
    """S=1/2 with one 1H nucleus."""
    return SpinSystem(
        S=[0.5],
        g=[[2.006, 2.006, 2.002]],
        Nucs=['1H'],
        A=[[3.0, 3.0, 8.0]],
    )


def _bare_sys():
    return SpinSystem(S=[0.5], g=[[2.006, 2.006, 2.002]])


def _nitroxide_sys():
    """Nitroxide: S=1/2, 14N."""
    return SpinSystem(
        S=[0.5],
        g=[[2.009, 2.006, 2.002]],
        Nucs=['14N'],
        A=[[16.0, 16.0, 95.0]],
    )


def _eseem_exp(**kw):
    """Real-pulse two-pulse ESEEM, single-point detected."""
    base = dict(Field=340.0, mwFreq=9.5, Sequence=[P90, 0.2, P180, 0.2],
                DetWindow=0.0, DetPhase=0.0, nPoints=24,
                Dim1=['d1,d2', 0.02])
    base.update(kw)
    return PulseExperiment(**base)


# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------

class TestEngineSelection:
    """Opt.SimulationMode, and what the fast algorithm rejects."""

    def test_custom_with_real_pulses_dispatches(self):
        """saffron() routes nonzero Exp.tp to the real-pulse engine."""
        sys = _simple_sys()
        exp = PulseExperiment(
            Field=340.0, mwFreq=9.5,
            Sequence='custom',
            Flip=[1, 2, 1],            # π/2, π, π/2 (units of π/2)
            Phase=[1, 1, 1],
            tp=[0.016, 0.032, 0.016],  # 16 ns, 32 ns, 16 ns
            t=[0.200, 0.200, 0.200],
            Inc=[1, 1, 0],
            dt=0.012,
            nPoints=[8],
            DetWindow=0.0,
        )
        x, signal, info = saffron(sys, exp, SaffronOptions(**OPT, ))
        assert info['SinglePointDetection'] is True
        assert signal.shape == (8,)
        assert np.all(np.isfinite(signal))

    def test_list_sequence_selects_thyme(self):
        """A list-valued Exp.Sequence selects the real-pulse engine."""
        _, signal, info = saffron(_simple_sys(), _eseem_exp(), opt())
        assert signal.shape == (24,)
        assert info['SinglePointDetection'] is True

    def test_explicit_simulation_mode_fast_rejects_list_sequence(self):
        """Opt.SimulationMode='fast' with a list Exp.Sequence is an error."""
        with pytest.raises(ValueError, match="real-pulse syntax"):
            saffron(_simple_sys(), _eseem_exp(),
                    SaffronOptions(SimulationMode='fast'))

    def test_unknown_simulation_mode_raises(self):
        with pytest.raises(ValueError, match="'fast' or 'thyme'"):
            saffron(_simple_sys(), _eseem_exp(),
                    SaffronOptions(SimulationMode='banana'))

    def test_predefined_with_real_pulses_rejected_by_fast(self):
        """EasySpin saffron.m:351 — predefined sequence + Exp.tp in fast mode."""
        exp = PulseExperiment(Sequence='2pESEEM', Field=340.0, dt=0.02,
                              tau=0.2, nPoints=[32], tp=[0.016, 0.032])
        with pytest.raises(ValueError, match="predefined sequences"):
            saffron(_simple_sys(), exp, SaffronOptions(SimulationMode='fast'))

    @pytest.mark.parametrize("mode", [None, 'thyme'])
    def test_predefined_with_real_pulses_rejected_in_every_mode(self, mode):
        """GitHub #9: with Exp.tp the engine used to be inferred as thyme, whose
        predefined templates use fixed 16/32 ns pulses, so Exp.tp was silently
        ignored. EasySpin raises in this case (saffron.m:351), and so do we."""
        exp = PulseExperiment(Sequence='2pESEEM', Field=340.0, dt=0.02,
                              tau=0.2, nPoints=[32], tp=[0.001, 0.002])
        with pytest.raises(ValueError, match="predefined sequences.*real"):
            saffron(_simple_sys(), exp, SaffronOptions(SimulationMode=mode))

    def test_predefined_with_zero_tp_runs_fast(self):
        """All-zero Exp.tp means ideal pulses, which EasySpin accepts."""
        exp = PulseExperiment(Sequence='2pESEEM', Field=340.0, dt=0.02,
                              tau=0.2, nPoints=[32], tp=[0.0, 0.0])
        y = saffron(_simple_sys(), exp, SaffronOptions(GridSize=7))[1]
        assert np.all(np.isfinite(np.asarray(y)))

    def test_det_fields_rejected_by_fast(self):
        """Exp.DetSequence / Exp.DetIntegrate do not apply to the fast path."""
        exp = PulseExperiment(Sequence='2pESEEM', Field=340.0, dt=0.02,
                              tau=0.2, nPoints=[32], DetSequence='last')
        with pytest.raises(ValueError, match="spidyan"):
            saffron(_simple_sys(), exp, SaffronOptions(SimulationMode='fast'))

    def test_multielectron_error_names_a_real_option(self):
        """The fast-path error must point at an option that exists.

        It used to tell the user to set Opt.SimulationMode='thyme' when
        SaffronOptions had no such field.
        """
        sys = SpinSystem(S=[0.5, 0.5], g=[[2.0, 2.0, 2.0], [2.01, 2.01, 2.01]])
        exp = PulseExperiment(Sequence='2pESEEM', Field=340.0, dt=0.02,
                              tau=0.2, nPoints=[16])
        with pytest.raises(NotImplementedError) as err:
            saffron(sys, exp, SaffronOptions())
        assert 'SimulationMode' in str(err.value)
        assert hasattr(SaffronOptions(), 'SimulationMode')

    def test_mwfreq_required(self):
        """EasySpin s_sequencer: Exp.mwFreq is required for the thyme method."""
        exp = _eseem_exp()
        exp.mwFreq = None
        with pytest.raises(ValueError, match="Exp.mwFreq"):
            saffron(_simple_sys(), exp, opt())


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

class TestOutputContract:
    """Single-point vs transient detection, and the returned axes."""

    def test_single_point_one_value_per_increment(self):
        x, signal, info = saffron(_simple_sys(), _eseem_exp(), opt())
        assert info['SinglePointDetection'] is True
        assert signal.shape == (24,)
        assert np.asarray(x).shape == (24,)
        np.testing.assert_allclose(np.diff(np.asarray(x)), 0.02)

    def test_transient_appends_the_time_axis(self):
        exp = _eseem_exp(DetWindow=[-0.02, 0.02], nPoints=3)
        x, signal, info = saffron(_simple_sys(), exp, opt())
        assert info['SinglePointDetection'] is False
        assert isinstance(x, list) and len(x) == 2
        assert signal.shape == (3, len(x[1]))
        # The transient axis spans the requested window.
        assert x[1][0] == pytest.approx(-0.02)
        assert x[1][-1] == pytest.approx(0.02, abs=1e-9)

    def test_det_integrate_collapses_the_transient(self):
        exp = _eseem_exp(DetWindow=[-0.02, 0.02], nPoints=3,
                         DetIntegrate=True)
        _, signal, info = saffron(_simple_sys(), exp, opt())
        assert info['SinglePointDetection'] is True
        assert signal.shape == (3,)

    def test_two_indirect_dimensions(self):
        exp = PulseExperiment(
            Field=340.0, mwFreq=9.5,
            Sequence=[P90, 0.2, P90, 0.05, P180, 0.05, P90, 0.2],
            DetWindow=0.0, DetPhase=0.0, nPoints=[3, 4],
            Dim1=['d2', 0.02], Dim2=['d3', 0.02])
        x, signal, _ = saffron(_simple_sys(), exp, opt(GridSize=3))
        assert signal.shape == (3, 4)
        assert isinstance(x, list) and len(x) == 2

    def test_detection_phase_rotates_the_signal(self):
        exp0 = _eseem_exp(DetPhase=0.0)
        expq = _eseem_exp(DetPhase=np.pi / 3)
        _, y0, _ = saffron(_simple_sys(), exp0, opt())
        _, yq, _ = saffron(_simple_sys(), expq, opt())
        np.testing.assert_allclose(yq, y0 * np.exp(-1j * np.pi / 3),
                                   rtol=1e-12)


# ---------------------------------------------------------------------------
# Predefined experiments through the real-pulse engine
# ---------------------------------------------------------------------------

class TestThymePredefined:
    """Predefined experiments via thyme."""

    @pytest.mark.parametrize('name,extra', [
        ('2pESEEM', {}),
        ('3pESEEM', {'T': 0.1}),
        ('HYSCORE', {'t1': 0.02, 't2': 0.03, 'nPoints': [4, 4]}),
    ])
    def test_predefined_runs_and_modulates(self, name, extra):
        """Each thyme template runs and shows nuclear modulation."""
        kw = dict(Sequence=name, Field=340.0, mwFreq=9.5, dt=0.02, tau=0.2,
                  nPoints=[8], DetWindow=0.0, DetPhase=0.0)
        kw.update(extra)
        _, signal, info = saffron_thyme(_simple_sys(), PulseExperiment(**kw),
                                        opt(GridSize=3))
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0
        assert info['SinglePointDetection'] is True

    def test_4peseem_has_no_thyme_template(self):
        """EasySpin's s_predefinedexperiments has no 4pESEEM/MimsENDOR case."""
        exp = PulseExperiment(Sequence='4pESEEM', Field=340.0, mwFreq=9.5,
                              dt=0.02, tau=0.2, T=0.1, nPoints=[8],
                              DetWindow=0.0)
        with pytest.raises(ValueError, match="no real-pulse template"):
            saffron_thyme(_simple_sys(), exp, opt())

    def test_predefined_requires_dt(self):
        exp = PulseExperiment(Sequence='2pESEEM', Field=340.0, mwFreq=9.5,
                              tau=0.2, nPoints=[8], DetWindow=0.0)
        with pytest.raises(ValueError, match="Exp.dt"):
            saffron_thyme(_simple_sys(), exp, opt())


# ---------------------------------------------------------------------------
# Custom sequences
# ---------------------------------------------------------------------------

class TestThymeCustom:
    """Custom sequences with thyme."""

    def test_two_pulse_echo(self):
        _, signal, _ = saffron(_simple_sys(), _eseem_exp(), opt())
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0

    def test_on_resonance_fid_is_a_constant_phasor(self):
        """An isotropic spin driven exactly on resonance gives a flat FID.

        With Exp.DetFrequency defaulting to Exp.mwFreq, the down-conversion
        removes the precession exactly, so both the magnitude and the phase of
        the transient must be constant. That pins the down-conversion stage
        (frame shift, FreqTranslation, rfmixer) on its own, with no reference
        needed: a wrong translation frequency shows up as a phase ramp and a
        missing factor of two as a magnitude step.
        """
        g = 2.002
        field = 340.0
        mw = g * BMAGN * field * 1e-3 / PLANCK / 1e9   # GHz, on resonance
        sys = SpinSystem(S=[0.5], g=[[g, g, g]])
        exp = PulseExperiment(Field=field, mwFreq=mw, Sequence=[P90, 0.5],
                              DetWindow=[0.0, 0.5], DetPhase=0.0)
        _, signal, _ = saffron(sys, exp, opt(GridSize=7))
        amp = np.abs(signal)
        assert np.ptp(amp) < 1e-10 * amp.mean()
        assert np.ptp(np.unwrap(np.angle(signal))) < 1e-8

    def test_g_anisotropy_dephases_the_powder_fid(self):
        """Spread in g must make the late FID smaller than the early FID."""
        g = 2.002
        field = 340.0
        mw = g * BMAGN * field * 1e-3 / PLANCK / 1e9
        sys = SpinSystem(S=[0.5], g=[[2.012, 2.012, 2.002]])
        exp = PulseExperiment(Field=field, mwFreq=mw, Sequence=[P90, 0.5],
                              DetWindow=[0.0, 0.5], DetPhase=0.0)
        _, signal, _ = saffron(sys, exp, opt(GridSize=19))
        amp = np.abs(signal)
        assert amp[int(0.8 * len(amp)):].mean() < 0.9 * amp[0]

    def test_flip_array_syntax_matches_list_syntax(self):
        """saffron's Flip/Phase/tp arrays and EasySpin's list agree.

        Exp.Flip and Exp.Phase are in units of pi/2, as elsewhere in saffron,
        so Flip=[1,2] with Phase=[0,0] is the list form with Flip=pi/2, pi.
        """
        common = dict(Field=340.0, mwFreq=9.5, DetWindow=0.0, DetPhase=0.0,
                      nPoints=[8])
        exp_arr = PulseExperiment(
            Sequence='custom', Flip=[1, 2], Phase=[0, 0],
            tp=[0.016, 0.032], t=[0.2, 0.2], Inc=[1, 1], dt=0.02, **common)
        exp_list = PulseExperiment(
            Sequence=[{'tp': 0.016, 'Flip': np.pi / 2, 'Phase': 0.0}, 0.2,
                      {'tp': 0.032, 'Flip': np.pi, 'Phase': 0.0}, 0.2],
            Dim1=['d1,d2', 0.02], **common)
        _, y_arr, _ = saffron(_simple_sys(), exp_arr, opt(GridSize=3))
        _, y_list, _ = saffron(_simple_sys(), exp_list, opt(GridSize=3))
        np.testing.assert_allclose(y_arr, y_list, rtol=1e-12)

    def test_zero_pulse_length_rejected(self):
        exp = PulseExperiment(Field=340.0, mwFreq=9.5, Sequence='custom',
                              Flip=[1, 2], tp=[0.016, 0.0], t=[0.2, 0.2],
                              Inc=[1, 1], dt=0.02, nPoints=[8],
                              DetWindow=0.0)
        with pytest.raises(ValueError, match="finite length"):
            saffron_thyme(_simple_sys(), exp, opt())

    def test_phase_cycle_applied(self):
        """An explicit two-step phase cycle changes the signal."""
        cycle = {0: np.array([[0.0, 1.0], [np.pi, -1.0]])}
        _, y_plain, _ = saffron(_simple_sys(), _eseem_exp(), opt(GridSize=3))
        _, y_cycled, _ = saffron(_simple_sys(),
                                 _eseem_exp(PhaseCycle=cycle),
                                 opt(GridSize=3))
        assert not np.allclose(y_plain, y_cycled)


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------

class TestThymeRelaxation:
    """Relaxation (T1/T2) effects."""

    def test_t2_shortens_the_trace(self):
        """A shorter T2 must decay the ESEEM trace faster."""
        def run(T2):
            kw = dict(S=[0.5], g=[[2.006, 2.006, 2.002]], Nucs=['1H'],
                      A=[[3.0, 3.0, 8.0]])
            if T2 is not None:
                kw.update(T1=100.0, T2=T2)
            _, y, _ = saffron(SpinSystem(**kw), _eseem_exp(nPoints=24),
                              opt(GridSize=3))
            return np.abs(np.asarray(y))

        slow = run(None)
        fast = run(0.3)
        assert np.all(np.isfinite(fast))
        # Both start at the same tau; the relaxed one must fall off faster.
        assert fast[-1] / fast[0] < slow[-1] / slow[0]

    def test_t2_decay_rate_matches_the_relaxation_time(self):
        """The relaxed/unrelaxed echo ratio decays at 1/T2.

        A two-pulse echo carries transverse coherence for 2*tau, and the Dim1
        increment advances tau in both delays, so the ratio of the relaxed to
        the unrelaxed trace falls as exp(-2*n*dt/T2). The ratio is the right
        quantity: the unrelaxed echo amplitude itself varies by a few percent
        across the increments because a real pulse pair does not refocus
        perfectly, and dividing removes that.
        """
        T2 = 0.5
        dt = 0.05
        n_points = 10
        g = 2.002
        field = 340.0
        mw = g * BMAGN * field * 1e-3 / PLANCK / 1e9
        kw = dict(S=[0.5], g=[[g, g, g]])
        exp = PulseExperiment(Field=field, mwFreq=mw,
                              Sequence=[P90, 0.2, P180, 0.2],
                              DetWindow=0.0, DetPhase=0.0, nPoints=n_points,
                              Dim1=['d1,d2', dt])
        _, y_relax, _ = saffron(SpinSystem(T1=1e6, T2=T2, **kw), exp,
                                opt(GridSize=3))
        _, y_plain, _ = saffron(SpinSystem(**kw), exp, opt(GridSize=3))
        ratio = np.abs(y_relax) / np.abs(y_plain)
        t = 2 * dt * np.arange(n_points)
        slope = np.polyfit(t, np.log(ratio), 1)[0]
        assert slope == pytest.approx(-1.0 / T2, rel=0.05)


# ---------------------------------------------------------------------------
# Frequency domain (torchspin convenience, not part of EasySpin's thyme)
# ---------------------------------------------------------------------------

class TestThymeFFT:
    """info['fd'] is a convenience transform of a single-point trace."""

    def test_fd_offered_for_single_point_traces(self):
        _, _, info = saffron(_simple_sys(), _eseem_exp(),
                             opt(GridSize=3, TimeDomain=False))
        assert info['fd'] is not None
        assert info['fd']['signal'].size == 24
        assert info['fd']['x'].size == 24

    def test_fd_absent_for_transients(self):
        """A transient is not a one-point-per-increment trace, so no FFT."""
        exp = _eseem_exp(DetWindow=[-0.02, 0.02], nPoints=3)
        _, _, info = saffron(_simple_sys(), exp,
                             opt(GridSize=3, TimeDomain=False))
        assert info['fd'] is None

    def test_time_domain_option_suppresses_fd(self):
        _, _, info = saffron(_simple_sys(), _eseem_exp(),
                             opt(GridSize=3, TimeDomain=True))
        assert info['fd'] is None


# ---------------------------------------------------------------------------
# Experiment conversion
# ---------------------------------------------------------------------------

class TestExpConversion:
    """Conversion of a saffron PulseExperiment to the spidyan dict."""

    def test_predefined_conversion(self):
        exp = PulseExperiment(Sequence='2pESEEM', Field=340.0, mwFreq=9.5,
                              dt=0.020, tau=0.200, nPoints=[128],
                              DetWindow=0.0)
        d, exp_id = _saffron_exp_to_spidyan(exp)
        assert exp_id == '2pESEEM'
        assert d['Sequence'] == '2pESEEM'
        assert d['Field'] == 340.0
        assert d['mwFreq'] == 9.5
        assert d['tau'] == 0.200
        assert d['DetWindow'] == 0.0

    def test_flip_array_conversion(self):
        exp = PulseExperiment(
            Field=340.0, mwFreq=9.5, Sequence='custom',
            Flip=[1, 2, 1], Phase=[1, 1, 1], tp=[0.016, 0.032, 0.016],
            t=[0.200, 0.200, 0.200], Inc=[1, 1, 0], dt=0.012, nPoints=[64])
        d, exp_id = _saffron_exp_to_spidyan(exp)
        assert exp_id is None
        seq = d['Sequence']
        assert isinstance(seq[0], dict) and 'tp' in seq[0]
        # Flip/Phase in units of pi/2 become radians
        assert seq[0]['Flip'] == pytest.approx(np.pi / 2)
        assert seq[2]['Flip'] == pytest.approx(np.pi)
        assert seq[0]['Phase'] == pytest.approx(np.pi / 2)
        # Exp.Inc = [1 1 0] increments the first two delays along dimension 1
        assert d['Dim1'] == ['d1,d2', 0.012]

    def test_list_sequence_passthrough(self):
        exp = PulseExperiment(Field=340.0, mwFreq=9.5,
                              Sequence=[P90, 0.2, P180, 0.2],
                              Dim1=['d1,d2', 0.02], nPoints=[8],
                              DetWindow=0.0)
        d, exp_id = _saffron_exp_to_spidyan(exp)
        assert exp_id is None
        assert d['Sequence'][1] == 0.2
        assert d['Dim1'] == ['d1,d2', 0.02]

    def test_malformed_dim_rejected(self):
        exp = PulseExperiment(Field=340.0, mwFreq=9.5,
                              Sequence=[P90, 0.2], Dim1=['d1', 0.02, 3],
                              nPoints=[8], DetWindow=0.0)
        with pytest.raises(ValueError, match="Exp.Dim1"):
            _saffron_exp_to_spidyan(exp)


# ---------------------------------------------------------------------------
# Physical sanity
# ---------------------------------------------------------------------------

class TestThymeNumerical:
    """Relations between runs that must hold whatever the absolute scale."""

    def test_eseem_modulation_depth(self):
        """2pESEEM with 1H must show nuclear modulation, not a flat trace."""
        _, signal, _ = saffron(_simple_sys(), _eseem_exp(), opt(GridSize=5))
        amp = np.abs(np.asarray(signal))
        variation = np.std(amp) / np.mean(amp)
        assert variation > 0.01, (
            f"signal appears flat (variation={variation:.4f}), expected "
            "nuclear ESEEM modulation")

    def test_signal_depends_on_hyperfine(self):
        _, y_bare, _ = saffron(_bare_sys(), _eseem_exp(), opt(GridSize=3))
        _, y_hf, _ = saffron(_simple_sys(), _eseem_exp(), opt(GridSize=3))
        assert not np.allclose(y_bare, y_hf)

    def test_nitroxide_runs(self):
        _, signal, _ = saffron(_nitroxide_sys(), _eseem_exp(), opt(GridSize=3))
        assert np.all(np.isfinite(signal))

    def test_powder_sum_scales_with_grid_weights(self):
        """The orientation sum is not renormalized, so it keeps the 4*pi scale.

        A single-orientation powder grid (GridSize 7 on a Dinfh system still
        has several points, so compare against an explicit crystal instead):
        the powder result is the weighted sum over the grid, and the weights
        sum to 4*pi, so a powder of an isotropic system equals 4*pi times one
        crystal orientation.
        """
        iso = SpinSystem(S=[0.5], g=[[2.002, 2.002, 2.002]])
        exp_powder = PulseExperiment(Field=340.0, mwFreq=9.5,
                                     Sequence=[P90, 0.3],
                                     DetWindow=[0.0, 0.2], DetPhase=0.0)
        exp_crystal = PulseExperiment(Field=340.0, mwFreq=9.5,
                                      Sequence=[P90, 0.3],
                                      DetWindow=[0.0, 0.2], DetPhase=0.0,
                                      CrystalSymmetry='P1',
                                      MolFrame=[0, 0, 0],
                                      SampleFrame=[[0, 0, 0]])
        _, y_p, _ = saffron(iso, exp_powder, opt(GridSize=7))
        _, y_c, _ = saffron(iso, exp_crystal, opt())
        # Each crystal orientation carries weight 4*pi too, so for an
        # isotropic system the powder and the crystal agree exactly.
        np.testing.assert_allclose(y_p, y_c, rtol=1e-10)


# ---------------------------------------------------------------------------
# Cross-checks against independently validated code paths
# ---------------------------------------------------------------------------

class TestThymeAgainstSpidyan:
    """Relations that need no MATLAB reference but pin the normalization.

    ``spidyan`` is cross-validated against EasySpin on its own (10 cases,
    1e-14..1e-12), so comparing the real-pulse saffron route against it checks
    everything saffron adds on top -- the orientation grid, the ``4*pi``
    weights, the detection stage and the down-conversion -- against a code
    path whose agreement is established separately.
    """

    def test_single_crystal_equals_spidyan_times_grid_weight(self):
        """One crystal site = 4*pi x spidyan for the same sequence.

        EasySpin gives every crystal orientation the weight 4*pi and sums
        without renormalizing, so a single-site P1 crystal is exactly 4*pi
        times the single-orientation spidyan result. This is the cheapest
        guard there is against a lost or doubled normalization, and it does
        not depend on any stored reference.
        """
        from torchspin.spidyan import spidyan, SpidyanOptions
        sys = _simple_sys()
        seq = [P90, 0.15, P180, 0.15]
        field, mw, its = 340.0, 9.5, 1e-4

        exp_saffron = PulseExperiment(
            Field=field, mwFreq=mw, Sequence=seq, DetWindow=[-0.02, 0.02],
            DetPhase=0.0, CrystalSymmetry='P1', MolFrame=[0, 0, 0],
            SampleFrame=[[0, 0, 0]])
        _, y_saffron, _ = saffron(
            sys, exp_saffron,
            SaffronOptions(SimulationMode='thyme', IntTimeStep=its))

        exp_spidyan = {'Field': field, 'mwFreq': mw, 'Sequence': seq,
                       'DetWindow': [-0.02, 0.02], 'DetPhase': 0.0,
                       'DetFreq': mw}
        _, y_spidyan, _ = spidyan(sys, exp_spidyan,
                                  SpidyanOptions(IntTimeStep=its))
        y_spidyan = np.asarray(y_spidyan).ravel()
        y_saffron = np.asarray(y_saffron).ravel()
        assert y_saffron.shape == y_spidyan.shape
        np.testing.assert_allclose(y_saffron, 4 * np.pi * y_spidyan,
                                   rtol=1e-9)

    def test_nominal_flip_angle_is_honored(self):
        """A nominal pi/2 gives full transverse magnetization, a pi none.

        ``pulse()`` sets the amplitude so that Pulse.Flip is the flip angle
        actually delivered, which already accounts for the linearly polarized
        excitation. Detecting S+ on resonance for a bare electron therefore
        peaks at a nominal pi/2 and nulls at a nominal pi -- the sharpest
        cheap check that the pulse amplitude convention has not slipped by a
        factor of two.
        """
        g = 2.0023
        field = 340.0
        mw = g * BMAGN * field * 1e-3 / PLANCK / 1e9
        sys = SpinSystem(S=[0.5], g=[[g, g, g]])

        def transverse(flip):
            exp = PulseExperiment(
                Field=field, mwFreq=mw,
                Sequence=[{'tp': 0.02, 'Flip': flip}, 0.001],
                DetWindow=[0.0, 0.001], DetPhase=0.0,
                CrystalSymmetry='P1', MolFrame=[0, 0, 0],
                SampleFrame=[[0, 0, 0]])
            _, y, _ = saffron(
                sys, exp,
                SaffronOptions(SimulationMode='thyme', IntTimeStep=1e-5))
            return abs(np.asarray(y).ravel()[0]) / (4 * np.pi)

        assert transverse(np.pi / 2) == pytest.approx(1.0, abs=1e-4)
        assert transverse(np.pi) < 1e-2
        assert transverse(np.pi / 4) == pytest.approx(
            np.sin(np.pi / 4), abs=2e-3)


# ---------------------------------------------------------------------------
# Warnings for fields the real-pulse engine ignores
# ---------------------------------------------------------------------------

class TestThymeIgnoredFields:
    """EasySpin reports the fast-only fields it ignores; so does torchspin."""

    def test_excitewidth_warns(self):
        exp = _eseem_exp(ExciteWidth=100.0)
        with pytest.warns(UserWarning, match="fast algorithm"):
            saffron(_simple_sys(), exp, opt(GridSize=3))

    def test_detoperator_warns(self):
        exp = _eseem_exp(DetOperator='z1')
        with pytest.warns(UserWarning, match="DetOperator"):
            saffron(_simple_sys(), exp, opt(GridSize=3))

    def test_sampleframe_without_crystal_warns(self):
        exp = _eseem_exp(SampleFrame=[[0, 0.3, 0]])
        with pytest.warns(UserWarning, match="SampleFrame"):
            saffron(_simple_sys(), exp, opt(GridSize=3))
