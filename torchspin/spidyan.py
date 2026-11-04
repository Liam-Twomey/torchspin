"""
spidyan — Simulate spin dynamics during pulse EPR experiments.

Density matrix propagation engine for arbitrary pulse sequences with
shaped pulses, relaxation, phase cycling, and multi-dimensional experiments.

Based on EasySpin's spidyan.m (Pribitzer et al.).

All times in **microseconds**, frequencies in **GHz**, fields in **mT**.
"""

import math
import warnings

import numpy as np
import scipy.linalg as la
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any, Tuple

from torchspin.spinsystem import SpinSystem
from torchspin.ham import ham
from torchspin.spinops import sop
from torchspin.constants import PLANCK, BMAGN, GFREE
from torchspin.pulse import pulse as generate_pulse
from torchspin.rfmixer import rfmixer
from torchspin.signalprocessing import signalprocessing, trace_kind


def _round_significant(x: float, digits: int) -> float:
    """MATLAB's ``round(x, digits, 'significant')``."""
    if x == 0 or not np.isfinite(x):
        return float(x)
    mag = math.floor(math.log10(abs(x)))
    factor = 10.0 ** (digits - 1 - mag)
    return float(round(x * factor) / factor)


def _parse_sop_string(spins, s):
    """Parse an EasySpin-style operator string like 'z1', 'x2', '+1' into sop args."""
    # Map component letters to indices
    comp_map = {'e': 0, 'x': 1, 'y': 2, 'z': 3, '+': 4, '-': 5}
    if len(s) == 2 and s[0] in comp_map and s[1].isdigit():
        comp = comp_map[s[0]]
        spin_idx = int(s[1])
        return sop(spins, [spin_idx, comp]).detach().cpu().numpy()
    else:
        # Try passing directly to sop
        return sop(spins, s).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class SpidyanOptions:
    """Options for spidyan simulation."""
    Verbosity: int = 0
    Relaxation: Optional[bool] = None
    IntTimeStep: Optional[float] = None
    DetOperator: Optional[list] = None
    ExcOperator: Optional[list] = None
    StateTrajectories: bool = False
    SimFreq: Optional[float] = None   # GHz — simulation frame frequency
                                      # (None = auto, 0 = lab frame)


@dataclass
class SpidyanInfo:
    """Information returned by spidyan."""
    FinalState: np.ndarray = None
    StateTrajectories: Any = None
    Events: list = None


# ---------------------------------------------------------------------------
# Internal data classes for the event structure
# ---------------------------------------------------------------------------

@dataclass
class PulseEvent:
    """A pulse event in the sequence."""
    type: str = 'pulse'
    t: np.ndarray = None          # time axis (µs)
    IQ: np.ndarray = None         # complex IQ waveform
    PhaseCycle: np.ndarray = None  # (nCycles, 2) array: [phase, weight]
    xOp: np.ndarray = None        # excitation operator
    TimeStep: float = 0.0         # integration time step (µs)
    Detection: bool = False
    Relaxation: bool = False
    StateTrajectories: bool = False
    ComplexExcitation: bool = False
    Propagation: dict = field(default_factory=dict)
    vertRes: int = 1024
    FrameShift: float = 0.0


@dataclass
class DelayEvent:
    """A free evolution (delay) event."""
    type: str = 'free evolution'
    t: float = 0.0                # duration (µs)
    TimeStep: float = 0.0         # propagation time step (µs)
    Detection: bool = False
    Relaxation: bool = False
    StateTrajectories: bool = False
    Propagation: dict = field(default_factory=dict)
    FrameShift: float = 0.0


# ---------------------------------------------------------------------------
# Internal data class for Vary structure (indirect dimensions)
# ---------------------------------------------------------------------------

@dataclass
class VaryStructure:
    """Structure for multi-dimensional experiments."""
    Points: list = field(default_factory=list)
    IncrementationTable: list = field(default_factory=list)
    Pulses: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal data class for Relaxation
# ---------------------------------------------------------------------------

@dataclass
class RelaxationData:
    """Relaxation superoperator and equilibrium state."""
    Gamma: np.ndarray = None
    equilibriumState: np.ndarray = None


# ---------------------------------------------------------------------------
# Relaxation superoperator (s_relaxationsuperoperator.m)
# ---------------------------------------------------------------------------

def _relaxation_superoperator(spins, T1, T2):
    """Build relaxation superoperator in Liouville space.

    Parameters
    ----------
    spins : list
        Spin quantum numbers for all particles.
    T1, T2 : float or ndarray
        Longitudinal/transverse relaxation times (µs).
        Scalar: same value for all pathways.
        Matrix: element-wise specification.

    Returns
    -------
    Gamma : ndarray, shape (n², n²)
        Relaxation superoperator.
    """
    n = int(np.prod([2 * s + 1 for s in spins]))

    # Expand scalar T1/T2 to full matrix
    T1 = np.atleast_2d(T1)
    T2 = np.atleast_2d(T2)

    T1_ud = False
    if T1.shape == (1, 1):
        T1 = float(T1.item()) * np.ones((n, n))
    else:
        if np.any(np.tril(T1)):
            T1_ud = True

    if T2.shape == (1, 1):
        T2 = float(T2.item()) * np.ones((n, n))
    else:
        # Mirror upper triangle to lower triangle
        T2 = np.triu(T2) + np.triu(T2, 1).T

    # Replace zeros with large value (no relaxation)
    T1[T1 == 0] = 1e10
    T2[T2 == 0] = 1e10

    Gamma = np.zeros((n * n, n * n))

    # Longitudinal (T1) relaxation — connects diagonal elements (populations)
    kk = 0
    jj = 1
    for k1 in range(0, n * n, n + 1):
        for k2 in range(k1 + n + 1, n * n, n + 1):
            Gamma[k1, k2] = -1.0 / T1[kk, jj]
            if T1_ud:
                Gamma[k2, k1] = -1.0 / T1[jj, kk]
            else:
                Gamma[k2, k1] = -1.0 / T1[kk, jj]
            jj += 1
        kk += 1
        jj = kk + 1

    # Transverse (T2) relaxation — dephasing of off-diagonal coherences
    T2vec = T2.reshape(-1)
    diag_indices = set(range(0, n * n, n + 1))
    for k in range(1, n * n - 1):
        if k not in diag_indices:
            Gamma[k, k] = 1.0 / T2vec[k]

    return Gamma


# ---------------------------------------------------------------------------
# Predefined experiments (s_predefinedexperiments.m)
# ---------------------------------------------------------------------------

def _predefined_experiments(exp):
    """Expand predefined experiment templates.

    Parameters
    ----------
    exp : dict
        Experiment specification with 'Sequence' as a string name.

    Returns
    -------
    exp : dict
        Updated with full Sequence, Dim1/Dim2, PhaseCycle, etc.
    """
    P90 = {'tp': 0.016, 'Flip': np.pi / 2}
    P180 = {'tp': 0.032, 'Flip': np.pi}
    default_points = 512

    seq_name = exp['Sequence']

    if seq_name == '2pESEEM':
        for f in ['dt', 'tau']:
            if f not in exp:
                raise ValueError(f"Exp.{f} is required for {seq_name}.")
        if 'DetWindow' not in exp:
            exp.pop('DetSequence', None)
            exp['DetWindow'] = 0
        exp['Sequence'] = [P90, exp['tau'], P180, exp['tau']]
        exp.setdefault('nPoints', default_points)
        exp['Dim1'] = ['d1,d2', exp['dt']]

    elif seq_name == '3pESEEM':
        for f in ['dt', 'tau', 'T']:
            if f not in exp:
                raise ValueError(f"Exp.{f} is required for {seq_name}.")
        if 'DetWindow' not in exp:
            exp.pop('DetSequence', None)
            exp['DetWindow'] = 0
        exp['Sequence'] = [P90, exp['tau'], P90, exp['T'], P90, exp['tau']]
        exp.setdefault('nPoints', default_points)
        exp['Dim1'] = ['d2', exp['dt']]
        exp['PhaseCycle'] = {
            0: np.array([[0, 1], [np.pi, -1]]),
            2: np.array([[0, 1], [np.pi, -1]]),
        }

    elif seq_name == 'HYSCORE':
        for f in ['t1', 't2', 'tau']:
            if f not in exp:
                raise ValueError(f"Exp.{f} is required for {seq_name}.")
        if 'DetWindow' not in exp:
            exp.pop('DetSequence', None)
            exp['DetWindow'] = 0
        exp['Sequence'] = [P90, exp['tau'], P90, exp['t1'],
                           P180, exp['t2'], P90, exp['tau']]
        if 'nPoints' not in exp:
            exp['nPoints'] = [default_points // 4, default_points // 4]
        elif isinstance(exp['nPoints'], (int, float)):
            exp['nPoints'] = [int(exp['nPoints']), int(exp['nPoints'])]
        exp['Dim1'] = ['d2', exp['dt']]
        exp['Dim2'] = ['d3', exp['dt']]

    else:
        raise ValueError(f"Unknown predefined experiment: {seq_name}")

    return exp


# ---------------------------------------------------------------------------
# Event reordering (s_reorder_events.m)
# ---------------------------------------------------------------------------

def _reorder_events(event_lengths, pulse_list):
    """Reorder events chronologically, detect pulse overlaps.

    Parameters
    ----------
    event_lengths : list of float
        Duration of each event.
    pulse_list : list of bool
        True if event is a pulse.

    Returns
    -------
    new_sequence : list of int
        Permutation of event indices in chronological order.
    new_event_lengths : list of float
        Durations after reordering.
    """
    n_events = len(event_lengths)
    t = np.concatenate(([0.0], np.cumsum(event_lengths)))

    # Sort by start time
    sorted_idx = np.argsort(t)
    time_intervals = np.sort(t)
    new_sequence = list(sorted_idx[:n_events])

    # Recalculate durations
    new_event_lengths = np.zeros(n_events)
    for k in range(n_events):
        new_event_lengths[k] = time_intervals[k + 1] - time_intervals[k]

    # Round to avoid floating point precision issues
    new_event_lengths = np.round(new_event_lengths, 10)

    # Build event matrix for overlap detection
    event_matrix = np.zeros((n_events, n_events), dtype=int)
    for i_event in range(n_events):
        t_start = t[i_event]
        t_end = t[i_event] + event_lengths[i_event]
        t_start, t_end = min(t_start, t_end), max(t_start, t_end)
        idx = (time_intervals >= t_start) & (time_intervals < t_end)
        idx = idx[:n_events]
        if pulse_list[i_event]:
            event_matrix[i_event, idx] = 2
        else:
            event_matrix[i_event, idx] = 1

    # Check for pulse overlap
    pulse_cols = np.sum(event_matrix == 2, axis=0)
    if np.any(pulse_cols > 1):
        overlaps = np.where(pulse_cols > 1)[0]
        overlap_events = np.where(np.sum(event_matrix[:, overlaps] == 2, axis=1) > 0)[0]
        raise ValueError(
            f"Pulse overlap detected between events: {list(overlap_events)}"
        )

    # Handle zero-duration pulses
    for i in range(n_events - 1):
        if pulse_list[new_sequence[i]] and new_event_lengths[i] == 0:
            new_sequence[i], new_sequence[i + 1] = new_sequence[i + 1], new_sequence[i]

    return new_sequence, list(new_event_lengths)


# ---------------------------------------------------------------------------
# Sequencer (s_sequencer.m — simplified)
# ---------------------------------------------------------------------------

def _build_pulse_iq(par, pc, freq_shift, int_time_step):
    """Build the phase-cycled, carrier-shifted IQ matrix for a pulse.

    EasySpin s_sequencer: the envelope is generated with pulse() at
    frequencies relative to mwFreq, then the carrier is shifted up by
    freqShift = mwFreq - FrameShift into the simulation frame.

    Returns (iq_all, dt) with iq_all of shape (nPhaseSteps, nSamples).
    """
    par = dict(par)
    if 'Type' not in par:
        par['Type'] = 'rectangular'
    if 'Frequency' not in par:
        par['Frequency'] = [0.0, 0.0]
    par['TimeStep'] = int_time_step
    base_phase = float(par.get('Phase', 0.0))

    rows = []
    t_p = None
    for j in range(pc.shape[0]):
        par_j = dict(par)
        par_j['Phase'] = base_phase + float(pc[j, 0])
        t_p, iq_j, _ = generate_pulse(par_j)
        t_p = np.asarray(t_p, dtype=float)
        iq_j = np.asarray(iq_j, dtype=complex)
        if freq_shift != 0:
            t_p, iq_j = rfmixer(t_p, iq_j, freq_shift, 'IQshift',
                                dt=int_time_step)
        rows.append(iq_j)
    iq_all = np.atleast_2d(np.array(rows))
    dt = t_p[1] - t_p[0] if t_p is not None and len(t_p) > 1 else int_time_step
    return iq_all, dt


def _sequencer(exp, opt):
    """Parse pulse sequence into Events and Vary structures.

    Parameters
    ----------
    exp : dict
        Experiment specification.
    opt : SpidyanOptions
        Simulation options.

    Returns
    -------
    events : list
        List of PulseEvent / DelayEvent objects.
    vary : VaryStructure or None
        Structure for indirect dimensions.
    frame_shift : float
        Frequency shift to simulation frame (GHz).
    int_time_step : float
        Integration time step (µs).
    single_point_detection : bool
        If True, single-point detection at end of sequence.
    """
    sequence = exp['Sequence']

    # Handle predefined experiments
    if isinstance(sequence, str):
        exp = _predefined_experiments(exp)
        sequence = exp['Sequence']

    # Frame shift and pulse carrier (EasySpin s_sequencer semantics):
    # - Pulse.Frequency (MHz) is relative to Exp.mwFreq (GHz).
    # - The simulation runs in a frame at FrameShift (GHz); pulse IQs are
    #   up-shifted by freqShift = mwFreq - FrameShift and H0 is down-shifted
    #   by FrameShift via an electron g-shift (done in spidyan()).
    mw_freq = float(exp.get('mwFreq', 0.0) or 0.0)
    freq_shift = mw_freq  # GHz

    max_freq = 0.0  # GHz, for Nyquist time step
    for elem in sequence:
        if isinstance(elem, dict) and 'tp' in elem and 'Frequency' in elem:
            freqs = np.atleast_1d(elem['Frequency']).astype(float) / 1e3
            max_freq = max(max_freq, np.max(np.abs(freqs + mw_freq)))

    if opt.SimFreq is not None:
        if opt.SimFreq == 0:
            frame_shift = 0.0
        else:
            frame_shift = float(opt.SimFreq)
            freq_shift = freq_shift - frame_shift
    else:
        # Auto: at least 2 GHz below the lowest frequency in the sequence
        min_pulse_freq = mw_freq if mw_freq != 0 else None
        for elem in sequence:
            if isinstance(elem, dict) and 'tp' in elem and 'Frequency' in elem:
                fmin = np.min(np.atleast_1d(elem['Frequency']).astype(float)) / 1e3
                if min_pulse_freq is None:
                    min_pulse_freq = fmin
                else:
                    min_pulse_freq = min(min_pulse_freq, fmin + mw_freq)
        frame_shift = 0.0
        if min_pulse_freq is not None:
            fs = math.floor(min_pulse_freq - 2)
            if fs > 0:
                frame_shift = float(fs)
                freq_shift = freq_shift - frame_shift

    # Time step from the Nyquist criterion of the highest sim-frame frequency.
    # EasySpin (s_sequencer.m 435-451): NyquistTime = 1/(2*maxPulseFreq)/1000,
    # and the default integration step is that divided by IntStepFactor = 50
    # and rounded to two significant digits. torchspin <= 0.5.0 divided by 10
    # and did not round, which gave a different step from EasySpin's whenever
    # Opt.IntTimeStep was not stated explicitly.
    sim_max_freq = max(max_freq - frame_shift, freq_shift, 0.0)
    if sim_max_freq > 0:
        nyquist_time = 1e-3 / (2 * sim_max_freq)
        int_time_step = _round_significant(nyquist_time / 50, 2)
    else:
        int_time_step = 0.001  # 1 ns default

    if opt.IntTimeStep is not None:
        int_time_step = float(opt.IntTimeStep)

    # Determine detection (EasySpin s_sequencer.m 255-325)
    det_window = exp.get('DetWindow', None)
    det_sequence = exp.get('DetSequence', None)
    single_point_detection = False

    if det_window is not None:
        # Exp.DetWindow is applied by *shifting the last sequence element* by
        # DetWindow(1) and appending the detected event, not by slicing the
        # transient afterwards.
        dw = np.atleast_1d(np.asarray(det_window, dtype=float))
        last_is_pulse = isinstance(sequence[-1], dict) and 'tp' in sequence[-1]
        if det_sequence is not None:
            warnings.warn(
                "Both Exp.DetWindow and Exp.DetSequence were given; "
                "Exp.DetSequence is ignored.", UserWarning, stacklevel=3)
        if np.any(dw < 0) and last_is_pulse:
            raise ValueError(
                "Exp.DetWindow has a negative value but the last element of "
                "Exp.Sequence is a pulse. Detection into a pulse is not "
                "possible; append a free evolution event.")
        if not last_is_pulse and dw.min() < 0 and \
                abs(dw.min()) > float(sequence[-1]):
            raise ValueError(
                "Exp.DetWindow extends beyond the last free evolution into a "
                "pulse. Shorten the window or lengthen the free evolution.")
        sequence = list(sequence)
        if last_is_pulse:
            if dw[0] > 0:
                sequence.append(float(dw[0]))
        else:
            sequence[-1] = float(sequence[-1]) + float(dw[0])
        if dw.size == 1 or dw[0] == dw[1]:
            single_point_detection = True
            sequence.append(0.0)
        else:
            sequence.append(float(dw[1] - dw[0]))
        det_sequence = [False] * (len(sequence) - 1) + [True]
    elif det_sequence is not None:
        if isinstance(det_sequence, str):
            if det_sequence == 'last':
                det_sequence = [False] * (len(sequence) - 1) + [True]
            elif det_sequence == 'all':
                det_sequence = [True] * len(sequence)
            else:
                raise ValueError(
                    f"Exp.DetSequence string {det_sequence!r} not recognized; "
                    "use 'last' or 'all'.")
        else:
            # EasySpin: a scalar DetSequence applies to every event
            ds = np.atleast_1d(det_sequence)
            if ds.size == 1:
                det_sequence = [bool(ds[0])] * len(sequence)
            elif ds.size == len(sequence):
                det_sequence = [bool(v) for v in ds]
            else:
                raise ValueError(
                    "The lengths of Exp.Sequence and Exp.DetSequence do not "
                    "match. Exp.DetSequence must have length 1 or "
                    f"{len(sequence)}, got {ds.size}.")
        # A single detected zero-length delay is single-point detection
        if sum(det_sequence) == 1:
            i_det = det_sequence.index(True)
            elem = sequence[i_det]
            if not (isinstance(elem, dict) and 'tp' in elem) \
                    and float(elem) == 0.0:
                single_point_detection = True
    else:
        # EasySpin default: detection is active during the entire sequence
        det_sequence = [True] * len(sequence)

    # Determine relaxation flag per event
    global_relaxation = opt.Relaxation if opt.Relaxation is not None else False

    # Build Events
    events = []
    phase_cycles = exp.get('PhaseCycle', {})
    pulse_idx = 0

    for i, elem in enumerate(sequence):
        if isinstance(elem, dict) and 'tp' in elem:
            # Pulse
            evt = PulseEvent()
            tp = elem['tp']

            # Phase cycle for this pulse
            pc = None
            if isinstance(phase_cycles, dict) and pulse_idx in phase_cycles:
                pc = np.atleast_2d(phase_cycles[pulse_idx])
            elif (isinstance(phase_cycles, list)
                  and pulse_idx < len(phase_cycles)
                  and phase_cycles[pulse_idx] is not None):
                pc = np.atleast_2d(phase_cycles[pulse_idx])
            if pc is None:
                pc = np.array([[0.0, 1.0]])
            evt.PhaseCycle = pc
            n_pc = pc.shape[0]

            if 'IQ' in elem:
                # User-defined IQ: shift down into the simulation frame
                iq_user = np.atleast_2d(np.asarray(elem['IQ'], dtype=complex))
                t_user = np.asarray(elem.get('t',
                    np.arange(iq_user.shape[1]) * int_time_step), dtype=float)
                rows = []
                for j in range(iq_user.shape[0]):
                    if frame_shift != 0:
                        t_p, row = rfmixer(t_user, iq_user[j], -frame_shift,
                                           'IQshift', dt=int_time_step)
                    else:
                        t_p, row = t_user, iq_user[j]
                    rows.append(row)
                iq_all = np.atleast_2d(np.array(rows))
                if iq_all.shape[0] == 1 and n_pc > 1:
                    iq_all = np.vstack([iq_all[0] * np.exp(1j * pc[j, 0])
                                        for j in range(n_pc)])
                dt = t_p[1] - t_p[0] if len(t_p) > 1 else int_time_step
            else:
                iq_all, dt = _build_pulse_iq(elem, pc, freq_shift,
                                             int_time_step)
                evt._base_par = dict(elem)   # kept for Dim pulse variation

            evt.TimeStep = dt
            evt.t = np.arange(1, iq_all.shape[1] + 1) * dt
            evt.IQ = iq_all
            # Default: linearly polarized (real-part) excitation, like
            # EasySpin. Circular (complex) excitation is opt-in.
            evt.ComplexExcitation = bool(
                str(exp.get('mwPolarization', '')).lower() == 'circular')

            evt.Relaxation = global_relaxation
            evt.FrameShift = frame_shift

            evt.Detection = bool(det_sequence[i])

            evt.StateTrajectories = opt.StateTrajectories

            events.append(evt)
            pulse_idx += 1
        else:
            # Free evolution (delay)
            duration = float(elem)
            evt = DelayEvent()
            evt.t = duration
            # EasySpin (s_sequencer.m 785) stores Opt.IntTimeStep on every
            # event, pulses and delays alike; a detected delay shorter than
            # one step is then simply not propagated, and Exp.DetWindow's
            # output axis uses this step.
            evt.TimeStep = int_time_step
            evt.Relaxation = global_relaxation
            evt.FrameShift = frame_shift

            evt.Detection = bool(det_sequence[i])

            evt.StateTrajectories = opt.StateTrajectories

            events.append(evt)

    # Build Vary structure for indirect dimensions
    vary = None
    if 'nPoints' in exp:
        n_points = np.atleast_1d(exp['nPoints']).astype(int)
        vary = VaryStructure()
        vary.Points = list(n_points)

        n_dim = len(n_points)
        vary.IncrementationTable = []
        vary.Pulses = [None] * pulse_idx

        for d in range(n_dim):
            dim_key = f'Dim{d + 1}'
            if dim_key not in exp:
                vary.IncrementationTable.append(
                    np.zeros((len(events), int(n_points[d])))
                )
                continue

            dim_spec = exp[dim_key]
            event_spec = dim_spec[0]  # e.g. 'd1', 'd2', 'd1,d2'
            increment = dim_spec[1]   # step size in µs

            inc_table = np.zeros((len(events), int(n_points[d])))

            # Parse event specification
            delay_indices = []
            for part in event_spec.split(','):
                part = part.strip()
                if part.startswith('d'):
                    idx = int(part[1:]) - 1  # Convert to 0-indexed
                    # Find the delay at this position in the sequence
                    delay_count = 0
                    for ei, e in enumerate(events):
                        if e.type == 'free evolution':
                            if delay_count == idx:
                                delay_indices.append(ei)
                                break
                            delay_count += 1
                elif part.startswith('p'):
                    # Pulse parameter variation, e.g. 'p1.Flip'
                    sub = part[1:]
                    if '.' not in sub:
                        raise ValueError(
                            f"Dim{d + 1}: pulse spec '{part}' must be of the "
                            "form 'p<n>.<Parameter>' (e.g. 'p1.Flip').")
                    num_s, param = sub.split('.', 1)
                    p_num = int(num_s) - 1  # 0-based pulse number
                    p_count = 0
                    target = None
                    for e in events:
                        if e.type == 'pulse':
                            if p_count == p_num:
                                target = e
                                break
                            p_count += 1
                    if target is None:
                        raise ValueError(
                            f"Dim{d + 1}: pulse p{p_num + 1} not found.")
                    base_par = getattr(target, '_base_par', None)
                    if base_par is None:
                        raise ValueError(
                            f"Dim{d + 1}: cannot vary parameters of a "
                            "user-defined-IQ pulse.")
                    variants = []
                    for pt in range(int(n_points[d])):
                        par_v = dict(base_par)
                        par_v[param] = (float(base_par.get(param, 0.0))
                                        + pt * increment)
                        iq_v, _ = _build_pulse_iq(par_v, target.PhaseCycle,
                                                  freq_shift, int_time_step)
                        variants.append(iq_v)
                    target.VariedIQ = (d, variants)

            for ei in delay_indices:
                for pt in range(int(n_points[d])):
                    inc_table[ei, pt] = pt * increment

            vary.IncrementationTable.append(inc_table)

    _warn_validity_limits(sequence, events, opt, freq_shift, sim_max_freq, int_time_step)
    return events, vary, frame_shift, int_time_step, single_point_detection


def _warn_validity_limits(sequence, events, opt, freq_shift, sim_max_freq, int_time_step):
    """Warn about user settings that silently degrade the pulses (GitHub #8).

    Calibrated on a rectangular pi pulse at Q-band: an integration step of
    Nyquist/5 leaves an error of 5e-4 in <Sz>, Nyquist/2.5 1.4 %, Nyquist/2
    about 3 % and Nyquist 40 %. A simulation frame closer to the carrier than
    a few nutation frequencies breaks the separation of the co- and
    counter-rotating components: carrier offset 4·nu1 gives 0.8 %, 2·nu1 3.6 %.
    """
    if opt.IntTimeStep is not None and sim_max_freq > 0:
        nyquist_time = 1e-3 / (2 * sim_max_freq)          # µs
        if int_time_step > nyquist_time / 2:
            warnings.warn(
                f"spidyan: Opt.IntTimeStep = {int_time_step:.3g} µs is coarser than "
                f"{nyquist_time / 2:.3g} µs, half the Nyquist step of the highest "
                f"simulation-frame frequency ({sim_max_freq:.4g} GHz), so pulses are "
                f"inaccurate (a pi pulse at Nyquist/2 is off by ~3 %). Leave "
                f"IntTimeStep unset for the automatic step "
                f"({_round_significant(nyquist_time / 50, 2):.2g} µs), or move "
                f"Opt.SimFreq closer to the carrier.", UserWarning, stacklevel=4)
    if opt.SimFreq is not None and opt.SimFreq != 0:
        pulse_elems = [e for e in sequence if isinstance(e, dict) and 'tp' in e]
        pulse_events = [e for e in events if isinstance(e, PulseEvent)]
        worst = None
        for elem, evt in zip(pulse_elems, pulse_events):
            if 'IQ' in elem:
                continue
            nu1 = float(np.max(np.abs(evt.IQ))) if np.size(evt.IQ) else 0.0   # MHz
            if nu1 <= 0:
                continue
            f_rel = np.atleast_1d(elem.get('Frequency', 0.0)).astype(float)   # MHz
            carrier = freq_shift * 1e3 + float(np.min(f_rel))                  # MHz, sim frame
            ratio = carrier / nu1
            if worst is None or ratio < worst[0]:
                worst = (ratio, carrier, nu1)
        if worst is not None and worst[0] < 5:
            _, carrier, nu1 = worst
            warnings.warn(
                f"spidyan: Opt.SimFreq = {opt.SimFreq} GHz puts the lowest pulse carrier "
                f"{carrier:.4g} MHz above the simulation frame, less than 5 times the "
                f"nutation frequency ({nu1:.3g} MHz). The co- and counter-rotating "
                f"components are no longer separated and pulses are inaccurate (or "
                f"absent at offset 0). Lower Opt.SimFreq, or leave it unset.",
                UserWarning, stacklevel=4)


# ---------------------------------------------------------------------------
# Propagation setup (s_propagationsetup.m)
# ---------------------------------------------------------------------------

def _propagation_setup(sys, events, opt):
    """Set up initial state, operators, and relaxation.

    Parameters
    ----------
    sys : SpinSystem
        Spin system.
    events : list
        Event list.
    opt : SpidyanOptions
        Options.

    Returns
    -------
    sigma0 : ndarray
        Initial density matrix.
    det_ops : list of ndarray
        Detection operators.
    events : list
        Updated events with excitation operators.
    relaxation : RelaxationData or None
        Relaxation data.
    """
    import torch

    # Get spin quantum numbers
    spins = list(sys.Spins)

    # Total electron spin operators
    n_electrons = len(sys.S) if hasattr(sys, 'S') and sys.S is not None else 1
    Sx = None
    Sy = None
    Sz = None

    for i_spin in range(n_electrons):
        sx = sop(spins, [i_spin + 1, 1]).detach().cpu().numpy()
        sy = sop(spins, [i_spin + 1, 2]).detach().cpu().numpy()
        sz = sop(spins, [i_spin + 1, 3]).detach().cpu().numpy()
        if Sx is None:
            Sx, Sy, Sz = sx, sy, sz
        else:
            Sx = Sx + sx
            Sy = Sy + sy
            Sz = Sz + sz

    # Initial state: -Sz (all electrons)
    if hasattr(sys, 'initState') and sys.initState is not None:
        sigma0 = np.array(sys.initState, dtype=complex)
    else:
        sigma0 = -Sz.astype(complex)

    # Relaxation
    relaxation = None
    T1 = getattr(sys, 'T1', 0)
    T2 = getattr(sys, 'T2', 0)
    T1 = np.atleast_1d(np.array(T1, dtype=float))
    T2 = np.atleast_1d(np.array(T2, dtype=float))

    if np.any(T1 > 0) or np.any(T2 > 0):
        Gamma = _relaxation_superoperator(spins, T1, T2)
        relaxation = RelaxationData()
        relaxation.Gamma = Gamma

        if hasattr(sys, 'eqState') and sys.eqState is not None:
            relaxation.equilibriumState = np.array(sys.eqState, dtype=complex)
        else:
            relaxation.equilibriumState = sigma0.copy()

    # Excitation operators
    pulse_idx = 0
    for i, evt in enumerate(events):
        if evt.type == 'pulse':
            if (opt.ExcOperator is not None and
                    pulse_idx < len(opt.ExcOperator) and
                    opt.ExcOperator[pulse_idx] is not None):
                exc_op = opt.ExcOperator[pulse_idx]
                if isinstance(exc_op, str):
                    evt.xOp = _parse_sop_string(spins, exc_op)
                else:
                    evt.xOp = np.array(exc_op, dtype=complex)
            else:
                evt.xOp = Sx.copy()
                if evt.ComplexExcitation:
                    evt.xOp = evt.xOp + Sy
            pulse_idx += 1

    # Detection operators
    det_ops = []
    if opt.DetOperator is not None:
        for d_op in opt.DetOperator:
            if isinstance(d_op, str):
                det_ops.append(_parse_sop_string(spins, d_op))
            else:
                det_ops.append(np.array(d_op, dtype=complex))
    else:
        # Default: S+ = Sx + i*Sy
        det_ops.append(Sx + 1j * Sy)

    return sigma0, det_ops, events, relaxation


# ---------------------------------------------------------------------------
# Propagator and Liouvillian helpers
# ---------------------------------------------------------------------------

def _propagator(Ham, dt):
    """Compute unitary propagator U = expm(-i*Ham*dt)."""
    return la.expm(-1j * Ham * dt)


def _gamma_to_eigenbasis(gamma, U):
    """Transform a relaxation superoperator with the Hamiltonian eigenbasis.

    EasySpin (``saffron.m`` 1872-1876, ``spidyan.m``) uses
    ``R = kron(transpose(U), U')`` and ``Gamma -> R'*Gamma*R``, written for
    MATLAB's column-major ``vec()``. The row-major equivalent swaps the two
    Kronecker factors, for the same reason as in :func:`_liouvillian`.
    """
    R = np.kron(U.conj().T, U.T)
    return R.conj().T @ gamma @ R


def _liouvillian(Ham, Gamma, eq_state_vec, dt):
    """Compute Liouville space propagator and steady state.

    Returns
    -------
    L : ndarray
        Liouville propagator expm(L_full * dt).
    sigma_ss : ndarray
        Steady-state density vector.
    """
    n = Ham.shape[0]
    I_n = np.eye(n)
    # EasySpin writes the commutator superoperator as kron(I,H) - kron(H.',I),
    # which is correct for MATLAB's column-major vec(). numpy flattens
    # row-major, and vec_r(A@rho@B) = kron(A, B.T) @ vec_r(rho), so the two
    # Kronecker factors swap. Using EasySpin's order with a row-major flatten
    # propagates rho.T = conj(rho), which conjugates every complex (S+)
    # detection trace while leaving a real (Sz) one untouched.
    ham_su = np.kron(Ham, I_n) - np.kron(I_n, Ham.T)
    L_full = -1j * ham_su - Gamma
    sigma_ss = Gamma @ eq_state_vec
    # Solve for steady state. Use pinv as fallback for singular matrices.
    neg_L = -L_full
    if np.any(~np.isfinite(neg_L)) or np.any(~np.isfinite(sigma_ss)):
        sigma_ss = np.zeros_like(sigma_ss)
    else:
        try:
            sigma_ss = la.solve(neg_L, sigma_ss)
        except (la.LinAlgError, ValueError):
            try:
                sigma_ss = np.linalg.pinv(neg_L) @ sigma_ss
            except Exception:
                sigma_ss = np.zeros_like(sigma_ss)
    L = la.expm(L_full * dt)
    return L, sigma_ss


def _build_propagator_table(Ham0, xOp, dt, vert_res, scale):
    """Pre-compute propagators for all discrete amplitude levels."""
    table = [None] * (vert_res + 1)
    for i_res in range(vert_res + 1):
        Ham1 = scale * (i_res - vert_res / 2) * np.real(xOp)
        table[i_res] = _propagator(Ham0 + Ham1, dt)
    return table


def _build_liouvillian_table(Ham0, Gamma, eq_state_vec, xOp, dt, vert_res, scale):
    """Pre-compute Liouvillians for all discrete amplitude levels."""
    L_table = [None] * (vert_res + 1)
    ss_table = [None] * (vert_res + 1)
    for i_res in range(vert_res + 1):
        Ham1 = scale * (i_res - vert_res / 2) * np.real(xOp)
        L, ss = _liouvillian(Ham0 + Ham1, Gamma, eq_state_vec, dt)
        L_table[i_res] = L
        ss_table[i_res] = ss
    return L_table, ss_table


# ---------------------------------------------------------------------------
# Thyme — time-domain propagation engine (s_thyme.m)
# ---------------------------------------------------------------------------

def _thyme(sigma, Ham0, det_ops, events, relaxation, vary):
    """Core density matrix propagation engine.

    Parameters
    ----------
    sigma : ndarray, (n, n)
        Initial density matrix.
    Ham0 : ndarray, (n, n)
        Lab-frame Hamiltonian (MHz, will be multiplied by 2π).
    det_ops : list of ndarray
        Detection operators.
    events : list
        Event structures.
    relaxation : RelaxationData or None
    vary : VaryStructure or None

    Returns
    -------
    time_array : ndarray or list
    signal_array : ndarray or list
    final_states : ndarray
    state_trajectories : list or None
    events : list
    """
    n_events = len(events)
    n_det = len(det_ops)
    n = sigma.shape[0]

    # Convert Ham to angular frequency (rad/µs): multiply by 2π
    # Ham0 is in MHz, 2π * MHz = rad/µs
    Ham0_rad = Ham0 * 2 * np.pi

    # Bookkeeping for dimensions
    if vary is not None and vary.Points:
        n_dimensions = len(vary.Points)
        dim_indices = [0] * n_dimensions
        n_points = int(np.prod(vary.Points))
    else:
        n_dimensions = 0
        dim_indices = [0]
        n_points = 1

    initial_sigma = sigma.copy()
    final_states_list = []
    all_signals = []
    all_times = []
    all_state_traj = []

    # Get initial event lengths
    initial_event_lengths = []
    for evt in events:
        if evt.type == 'pulse':
            initial_event_lengths.append(evt.t[-1] if len(evt.t) > 0 else 0.0)
        else:
            initial_event_lengths.append(float(evt.t))

    # Pulse list for reordering
    is_pulse = [evt.type == 'pulse' for evt in events]

    # ---------- Main loop over acquisition points ----------
    for i_point in range(n_points):
        sigma = initial_sigma.copy()
        event_lengths = list(initial_event_lengths)

        # Apply incrementation tables
        if vary is not None and vary.Points:
            for d in range(n_dimensions):
                if (vary.IncrementationTable and
                        d < len(vary.IncrementationTable)):
                    inc_table = vary.IncrementationTable[d]
                    if inc_table is not None and np.any(inc_table):
                        col = dim_indices[d]
                        for ei in range(len(events)):
                            if ei < inc_table.shape[0] and col < inc_table.shape[1]:
                                event_lengths[ei] += inc_table[ei, col]

        # Select varied pulse waveforms for this acquisition point
        if vary is not None and vary.Points:
            for ei_v, evt in enumerate(events):
                varied = getattr(evt, 'VariedIQ', None)
                if varied is not None:
                    d, variants = varied
                    new_iq = variants[dim_indices[d]]
                    if (not isinstance(evt.IQ, np.ndarray)
                            or evt.IQ.shape != new_iq.shape
                            or not np.array_equal(evt.IQ, new_iq)):
                        evt.IQ = new_iq
                        evt.t = np.arange(1, new_iq.shape[1] + 1) * evt.TimeStep
                        evt.Propagation = {}
                    # pulse-length (tp) sweeps change the event duration
                    event_lengths[ei_v] = evt.t[-1] if len(evt.t) > 0 else 0.0

        # Determine event order
        if n_events > 1:
            try:
                sequence, new_event_lengths = _reorder_events(
                    event_lengths, is_pulse
                )
            except ValueError:
                sequence = list(range(n_events))
                new_event_lengths = event_lengths
        else:
            sequence = [0]
            new_event_lengths = event_lengths

        # Update free evolution event durations
        for i_evt in range(n_events):
            new_pos = sequence.index(i_evt) if i_evt in sequence else None
            if new_pos is not None and events[i_evt].type == 'free evolution':
                old_t = events[i_evt].t
                new_t = new_event_lengths[new_pos]
                if abs(new_t - old_t) > 1e-12:
                    events[i_evt].t = new_t
                    events[i_evt].Propagation = {}

        # ---------- Build propagators ----------
        for i_evt in range(n_events):
            evt = events[i_evt]

            if evt.type == 'pulse' and not evt.Propagation:
                _build_pulse_propagators(
                    evt, Ham0_rad, relaxation, sigma.shape[0]
                )

        # ---------- Propagate through events ----------
        t_total = 0.0
        current_signal_parts = []
        current_time_parts = []
        current_state_traj = []

        for i_seq_pos, i_evt in enumerate(sequence):
            evt = events[i_evt]
            evt_length = new_event_lengths[i_seq_pos]

            if evt.type == 'pulse':
                sigma, sig_part, t_part, dm_part = _propagate_pulse(
                    sigma, evt, det_ops, n_det, t_total
                )
            else:
                sigma, sig_part, t_part, dm_part = _propagate_delay(
                    sigma, evt, evt_length, Ham0_rad, det_ops, n_det,
                    relaxation, t_total
                )

            if sig_part is not None:
                current_signal_parts.append(sig_part)
                current_time_parts.append(t_part)

            if dm_part:
                current_state_traj.extend(dm_part)

            t_total += evt_length

        # Combine signal parts
        if current_signal_parts:
            # Concatenate, avoiding double-counting endpoints
            combined_sig = current_signal_parts[0]
            combined_t = current_time_parts[0]
            for k in range(1, len(current_signal_parts)):
                combined_sig = np.concatenate(
                    [combined_sig, current_signal_parts[k][:, 1:]], axis=1
                )
                combined_t = np.concatenate(
                    [combined_t, current_time_parts[k][1:]]
                )
            all_signals.append(combined_sig)
            all_times.append(combined_t)
        else:
            all_signals.append(None)
            all_times.append(None)

        final_states_list.append(sigma.copy())
        all_state_traj.append(current_state_traj if current_state_traj else None)

        # Increment dimension indices
        if vary is not None and vary.Points:
            for d in range(n_dimensions - 1, -1, -1):
                if dim_indices[d] < vary.Points[d] - 1:
                    dim_indices[d] += 1
                    break
                else:
                    dim_indices[d] = 0

    # ---------- Format output ----------
    # Single acquisition point
    if n_points == 1:
        time_array = all_times[0] if all_times[0] is not None else np.array([])
        signal_array = all_signals[0] if all_signals[0] is not None else np.array([])
        if signal_array.ndim == 2 and signal_array.shape[0] == 1:
            signal_array = signal_array[0]
        final_states = final_states_list[0]
        state_trajectories = all_state_traj[0]
    else:
        # Multi-point: stack signals
        non_none_sigs = [s for s in all_signals if s is not None]
        if non_none_sigs:
            # Check if all have same shape
            shapes = [s.shape for s in non_none_sigs]
            if len(set(shapes)) == 1:
                signal_array = np.stack(non_none_sigs, axis=0)
                time_array = np.stack(
                    [t for t in all_times if t is not None], axis=0
                )
                # If all time axes are the same, reduce to single vector
                if time_array.ndim == 2:
                    if np.allclose(time_array, time_array[0]):
                        time_array = time_array[0]
                # Reshape to match dimension structure
                if vary is not None and vary.Points:
                    shape = tuple(vary.Points)
                    sig_shape = signal_array.shape[1:]
                    signal_array = signal_array.reshape(shape + sig_shape)
            else:
                # Variable-length traces: use list
                signal_array = all_signals
                time_array = all_times
        else:
            signal_array = np.array([])
            time_array = np.array([])

        final_states = np.stack(final_states_list)
        if vary is not None and vary.Points:
            fs_shape = final_states.shape[1:]
            final_states = final_states.reshape(
                tuple(vary.Points) + fs_shape
            )

        state_trajectories = all_state_traj

    return time_array, signal_array, final_states, state_trajectories, events


def _build_pulse_propagators(evt, Ham0_rad, relaxation, n):
    """Build and cache propagators for a pulse event."""
    iq = evt.IQ
    n_phase_cycle = iq.shape[0]
    n_wave = iq.shape[1]
    vert_res = evt.vertRes

    # Scale factor
    r_max = np.max(np.abs(np.real(iq)))
    i_max = np.max(np.abs(np.imag(iq)))
    max_wave = max(r_max, i_max)

    if max_wave == 0:
        max_wave = 1.0

    scale = 2 * 2 * 2 * np.pi * max_wave / vert_res

    # Digitize waveform to binary levels
    real_arb = np.real(iq) / max_wave
    real_binary = np.round(vert_res * (real_arb + 1) / 2).astype(int)
    real_binary = np.clip(real_binary, 0, vert_res)

    has_imag = evt.ComplexExcitation
    if has_imag:
        imag_arb = np.imag(iq) / max_wave
        imag_binary = np.round(vert_res * (imag_arb + 1) / 2).astype(int)
        imag_binary = np.clip(imag_binary, 0, vert_res)

    use_relaxation = evt.Relaxation and relaxation is not None

    if not has_imag and not use_relaxation:
        # Build lookup table
        U_table = _build_propagator_table(
            Ham0_rad, evt.xOp, evt.TimeStep, vert_res, scale
        )

        if evt.Detection:
            # Store per-step propagators
            U_total = [[None] * n_wave for _ in range(n_phase_cycle)]
            for ipc in range(n_phase_cycle):
                for iw in range(n_wave):
                    U_total[ipc][iw] = U_table[real_binary[ipc, iw]]
        else:
            # Combine into single propagator per phase cycle
            U_total = [[None] for _ in range(n_phase_cycle)]
            for ipc in range(n_phase_cycle):
                U_combined = np.eye(n, dtype=complex)
                for iw in range(n_wave):
                    U_combined = U_table[real_binary[ipc, iw]] @ U_combined
                U_total[ipc][0] = U_combined

        evt.Propagation = {'Utotal': U_total}

    elif not has_imag and use_relaxation:
        eq_vec = relaxation.equilibriumState.reshape(-1)
        L_table, ss_table = _build_liouvillian_table(
            Ham0_rad, relaxation.Gamma, eq_vec, evt.xOp,
            evt.TimeStep, vert_res, scale
        )

        L_total = [[None] * n_wave for _ in range(n_phase_cycle)]
        ss_total = [[None] * n_wave for _ in range(n_phase_cycle)]
        for ipc in range(n_phase_cycle):
            for iw in range(n_wave):
                L_total[ipc][iw] = L_table[real_binary[ipc, iw]]
                ss_total[ipc][iw] = ss_table[real_binary[ipc, iw]]

        evt.Propagation = {'Ltotal': L_total, 'SigmaSStotal': ss_total}

    else:
        # Complex excitation — no lookup table
        if use_relaxation:
            eq_vec = relaxation.equilibriumState.reshape(-1)
            L_total = [[None] * n_wave for _ in range(n_phase_cycle)]
            ss_total = [[None] * n_wave for _ in range(n_phase_cycle)]
            for ipc in range(n_phase_cycle):
                for iw in range(n_wave):
                    Ham1 = (scale / 2 * (real_binary[ipc, iw] - vert_res / 2) *
                            np.real(evt.xOp) +
                            1j * scale / 2 * (imag_binary[ipc, iw] - vert_res / 2) *
                            np.imag(evt.xOp))
                    L, ss = _liouvillian(
                        Ham0_rad + Ham1, relaxation.Gamma, eq_vec, evt.TimeStep
                    )
                    L_total[ipc][iw] = L
                    ss_total[ipc][iw] = ss
            evt.Propagation = {'Ltotal': L_total, 'SigmaSStotal': ss_total}
        else:
            if evt.Detection:
                U_total = [[None] * n_wave for _ in range(n_phase_cycle)]
                for ipc in range(n_phase_cycle):
                    for iw in range(n_wave):
                        Ham1 = (scale / 2 * (real_binary[ipc, iw] - vert_res / 2) *
                                np.real(evt.xOp) +
                                1j * scale / 2 * (imag_binary[ipc, iw] - vert_res / 2) *
                                np.imag(evt.xOp))
                        U_total[ipc][iw] = _propagator(
                            Ham0_rad + Ham1, evt.TimeStep
                        )
            else:
                U_total = [[None] for _ in range(n_phase_cycle)]
                for ipc in range(n_phase_cycle):
                    U_combined = np.eye(n, dtype=complex)
                    for iw in range(n_wave):
                        Ham1 = (scale / 2 * (real_binary[ipc, iw] - vert_res / 2) *
                                np.real(evt.xOp) +
                                1j * scale / 2 * (imag_binary[ipc, iw] - vert_res / 2) *
                                np.imag(evt.xOp))
                        U = _propagator(Ham0_rad + Ham1, evt.TimeStep)
                        U_combined = U @ U_combined
                    U_total[ipc][0] = U_combined
            evt.Propagation = {'Utotal': U_total}


def _detection_array(det_ops, n_det, n):
    """Normalized, flattened detection operators (EasySpin s_thyme.m 479-487)."""
    det_array = np.zeros((n_det, n * n), dtype=complex)
    for i_det in range(n_det):
        d_vec = det_ops[i_det].T.reshape(-1)
        det_array[i_det] = d_vec / np.vdot(d_vec, d_vec)
    return det_array


def _propagate_pulse(sigma, evt, det_ops, n_det, t_offset):
    """Propagate through a pulse event.

    Returns (sigma, signal, time, density_matrices).
    """
    n = sigma.shape[0]
    n_phase_cycle = evt.PhaseCycle.shape[0]
    use_relaxation = 'Ltotal' in evt.Propagation

    # Detection setup
    signal = None
    t_vec = None
    if evt.Detection:
        n_wave = len(evt.Propagation.get('Utotal', evt.Propagation.get('Ltotal', [[]]))[0])
        signal = np.zeros((n_det, n_wave + 1), dtype=complex)
        t_vec = np.zeros(n_wave + 1)
        t_vec[0] = t_offset
        for i in range(n_wave):
            t_vec[i + 1] = t_offset + evt.t[i] if i < len(evt.t) else t_vec[i]

        # Initial detection
        det_array = _detection_array(det_ops, n_det, n)
        signal[:, 0] = det_array @ sigma.reshape(-1)

    density_matrices = []
    if evt.StateTrajectories:
        density_matrices.append(sigma.copy())

    # Phase cycling
    if n_phase_cycle > 1:
        state_before_pc = sigma.copy()
        loop_state = np.zeros_like(sigma)
        pc_norm = np.sum(np.abs(evt.PhaseCycle[:, 2] if evt.PhaseCycle.shape[1] > 2 else evt.PhaseCycle[:, 1]))
        pc_signal = None

    for i_pc in range(n_phase_cycle):
        if n_phase_cycle > 1 and i_pc > 0:
            sigma = state_before_pc.copy()

        if use_relaxation:
            L_total = evt.Propagation['Ltotal']
            ss_total = evt.Propagation['SigmaSStotal']
            n_steps = len(L_total[i_pc])
            sigma_vec = sigma.reshape(-1)

            for i_w in range(n_steps):
                L = L_total[i_pc][i_w]
                ss = ss_total[i_pc][i_w]
                sigma_vec = ss + L @ (sigma_vec - ss)
                sigma = sigma_vec.reshape(n, n)

                if evt.Detection:
                    signal[:, i_w + 1] = det_array @ sigma_vec

                if evt.StateTrajectories:
                    density_matrices.append(sigma.copy())
        else:
            U_total = evt.Propagation['Utotal']
            n_steps = len(U_total[i_pc])

            for i_w in range(n_steps):
                U = U_total[i_pc][i_w]
                sigma = U @ sigma @ U.conj().T

                if evt.Detection:
                    signal[:, i_w + 1] = det_array @ sigma.reshape(-1)

                if evt.StateTrajectories:
                    density_matrices.append(sigma.copy())

        if n_phase_cycle > 1:
            pc_weight = evt.PhaseCycle[i_pc, 1] / pc_norm
            loop_state = loop_state + pc_weight * sigma
            if evt.Detection:
                if pc_signal is None:
                    pc_signal = pc_weight * signal.copy()
                else:
                    pc_signal = pc_signal + pc_weight * signal

    if n_phase_cycle > 1:
        sigma = loop_state
        if evt.Detection:
            signal = pc_signal

    return sigma, signal, t_vec, density_matrices if density_matrices else None


def _propagate_delay(sigma, evt, duration, Ham0_rad, det_ops, n_det,
                     relaxation, t_offset):
    """Propagate through a free evolution event.

    Returns (sigma, signal, time, density_matrices).
    """
    n = sigma.shape[0]
    use_relaxation = evt.Relaxation and relaxation is not None

    if duration <= 0:
        # EasySpin (s_thyme.m 474-489): a detected zero-length delay yields a
        # single point, the expectation value of the state entering it. This
        # is how Exp.DetWindow expresses single-point detection.
        if not evt.Detection:
            return sigma, None, None, None
        det_array = _detection_array(det_ops, n_det, n)
        signal = np.zeros((n_det, 1), dtype=complex)
        signal[:, 0] = det_array @ sigma.reshape(-1)
        return (sigma, signal, np.array([t_offset]),
                [sigma.copy()] if evt.StateTrajectories else None)

    # Determine time steps
    if evt.Detection:
        # EasySpin builds tvector = 0:TimeStep:t, so a duration that is not a
        # multiple of the time step is truncated (floor), not rounded, and a
        # delay shorter than one step is not propagated at all.
        time_step = evt.TimeStep
        if time_step <= 0:
            time_step = duration
        n_steps = int(math.floor(duration / time_step + 1e-9))
        t_vec = np.arange(n_steps + 1) * time_step + t_offset
    else:
        n_steps = 1
        time_step = duration
        t_vec = np.array([t_offset, t_offset + duration])

    # Build propagator
    if not use_relaxation:
        if 'Utotal' in evt.Propagation and evt.Propagation['Utotal'] is not None:
            U = evt.Propagation['Utotal']
        else:
            U = _propagator(Ham0_rad, time_step)
            evt.Propagation['Utotal'] = U
    else:
        eq_vec = relaxation.equilibriumState.reshape(-1)
        if 'Ltotal' in evt.Propagation and evt.Propagation['Ltotal'] is not None:
            L = evt.Propagation['Ltotal']
            ss = evt.Propagation['SigmaSStotal']
        else:
            L, ss = _liouvillian(
                Ham0_rad, relaxation.Gamma, eq_vec, time_step
            )
            evt.Propagation['Ltotal'] = L
            evt.Propagation['SigmaSStotal'] = ss

    # Detection setup
    signal = None
    if evt.Detection:
        signal = np.zeros((n_det, len(t_vec)), dtype=complex)
        det_array = _detection_array(det_ops, n_det, n)
        signal[:, 0] = det_array @ sigma.reshape(-1)

    density_matrices = []
    if evt.StateTrajectories:
        density_matrices.append(sigma.copy())

    # Propagate
    if use_relaxation:
        sigma_vec = sigma.reshape(-1)
        for i_t in range(n_steps):
            sigma_vec = ss + L @ (sigma_vec - ss)
            sigma = sigma_vec.reshape(n, n)
            if evt.Detection:
                signal[:, i_t + 1] = det_array @ sigma_vec
            if evt.StateTrajectories:
                density_matrices.append(sigma.copy())
    else:
        for i_t in range(n_steps):
            sigma = U @ sigma @ U.conj().T
            if evt.Detection:
                signal[:, i_t + 1] = det_array @ sigma.reshape(-1)
            if evt.StateTrajectories:
                density_matrices.append(sigma.copy())

    return sigma, signal, t_vec, density_matrices if density_matrices else None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _complex_trace_needs_doubling(trace: np.ndarray) -> bool:
    """Replicate EasySpin signalprocessing.m's purely-real/imaginary test.

    For a trace that is neither essentially purely real nor essentially purely
    imaginary -- i.e. a genuinely complex S+/S- signal -- EasySpin corrects the
    normalization by a factor of two before demodulating it
    (``signalprocessing.m``, "the normalization is wrong by a factor of two").
    Purely real or purely imaginary traces (e.g. Sz) are not scaled.
    """
    return trace_kind(trace) == 'complex'


def spidyan(
    sys: SpinSystem,
    exp: dict,
    opt: Optional[SpidyanOptions] = None,
) -> tuple:
    """Simulate spin dynamics during pulse EPR experiments.

    Uses density matrix propagation in Hilbert or Liouville space to
    simulate arbitrary pulse sequences with shaped pulses, relaxation,
    phase cycling, and multi-dimensional experiments.

    Parameters
    ----------
    sys : SpinSystem
        Spin system with electron spins and optional nuclei.
        Additional fields: T1, T2 (relaxation times, µs),
        initState, eqState (density matrices).
    exp : dict
        Experimental parameters:
        - Field : float — magnetic field (mT)
        - Sequence : list or str — pulse sequence
        - DetOperator : str or list — detection operator(s)
        - DetPhase : float or list — detection phase(s)
        - DetFreq : float — detection frequency (GHz)
        - mwFreq : float — microwave frequency (GHz)
        - nPoints : int or list — indirect dimension sizes
        - Dim1, Dim2 : list — dimension specifications
        - PhaseCycle : dict or list — phase cycling
        - DetWindow : float — detection window
        - DetSequence : list of bool — per-event detection flags
    opt : SpidyanOptions, optional
        Simulation options.

    Returns
    -------
    t : ndarray
        Time axis (µs).
    signal : ndarray
        Detected signal.
    info : SpidyanInfo
        Additional information (FinalState, StateTrajectories, Events).
    """
    if opt is None:
        opt = SpidyanOptions()

    # Validate inputs
    if 'Field' not in exp:
        if hasattr(sys, 'ZeemanFreq') and sys.ZeemanFreq is not None:
            zf = np.atleast_1d(sys.ZeemanFreq)
            exp['Field'] = (zf[0] * 1e6) * PLANCK / BMAGN / (GFREE * 1e-6)
        elif hasattr(sys, 'g') and sys.g is not None:
            raise ValueError("Exp['Field'] is required when Sys.g is provided.")
        else:
            raise ValueError("Exp['Field'] is required.")

    # If relaxation times exist but no explicit Relaxation flag, enable
    has_T1 = hasattr(sys, 'T1') and sys.T1 is not None and np.any(np.array(sys.T1) > 0)
    has_T2 = hasattr(sys, 'T2') and sys.T2 is not None and np.any(np.array(sys.T2) > 0)
    if (has_T1 or has_T2) and opt.Relaxation is None:
        opt.Relaxation = True

    # Transfer DetOperator from exp to opt
    if 'DetOperator' in exp:
        det_op = exp['DetOperator']
        if not isinstance(det_op, list):
            det_op = [det_op]
        opt.DetOperator = det_op

    # Parse the sequence
    events, vary, frame_shift, int_time_step, single_point_det = _sequencer(
        exp, opt
    )

    # --- Build the simulation-frame spin system (EasySpin spidyan.m) ---
    # 1. Sys.ZeemanFreq (GHz) overrides the electron g values:
    #    g = h*nu / (mu_B * B)
    # 2. The simulation frame at FrameShift (GHz) is entered by shifting
    #    all electron g values: g -= h*FrameShift / (mu_B * B)
    import copy
    import torch
    field_T = float(exp['Field']) * 1e-3
    sys_sim = copy.deepcopy(sys)
    if getattr(sys_sim, 'ZeemanFreq', None) is not None:
        zf = np.atleast_1d(np.asarray(sys_sim.ZeemanFreq, dtype=float))  # GHz
        n_e = sys_sim.nElectrons
        g_rows = []
        for i_e in range(n_e):
            nu = zf[i_e] if i_e < len(zf) else zf[-1]
            g_iso = (nu * 1e9) * PLANCK / (BMAGN * field_T)
            g_rows.append([g_iso, g_iso, g_iso])
        sys_sim.g = torch.tensor(g_rows, dtype=torch.float64)
    if frame_shift != 0:
        gshift = (frame_shift * 1e9) * PLANCK / (BMAGN * field_T)
        sys_sim.g = sys_sim.g - gshift

    # Set up spin system, operators, relaxation
    sigma0, det_ops, events, relaxation = _propagation_setup(
        sys_sim, events, opt
    )

    # Compute simulation-frame Hamiltonian
    B_vec = [0, 0, exp['Field']]
    H = ham(sys_sim, B_vec)
    Ham0 = H.detach().cpu().numpy().astype(complex)

    # Transform relaxation superoperator to system frame
    if relaxation is not None:
        _, U_vecs = np.linalg.eigh(Ham0)
        relaxation.Gamma = _gamma_to_eigenbasis(relaxation.Gamma, U_vecs)

    # Run propagation
    time_array, signal_array, final_states, state_traj, events = _thyme(
        sigma0, Ham0, det_ops, events, relaxation, vary
    )

    # Signal processing
    has_signal = (signal_array is not None and
                   not (isinstance(signal_array, np.ndarray) and signal_array.size == 0))
    if has_signal:
        # Apply detection phase
        if 'DetPhase' in exp:
            det_phase = np.exp(-1j * np.atleast_1d(exp['DetPhase']))
            if isinstance(signal_array, np.ndarray):
                if signal_array.ndim == 1:
                    signal_array = signal_array * det_phase[0]
                else:
                    for i in range(len(det_phase)):
                        if i < signal_array.shape[-2] if signal_array.ndim > 1 else 1:
                            signal_array[..., i, :] *= det_phase[i]

        # Down-conversion
        if not single_point_det and 'DetFreq' in exp:
            det_freq = np.atleast_1d(np.asarray(exp['DetFreq'], dtype=float))
            if isinstance(time_array, np.ndarray) and time_array.ndim == 1:
                # EasySpin: adapt DetFreq to the simulation frame
                # (>0: subtract FrameShift; <0: add; ==0: no mixing),
                # then translate each channel by -DetFreq (down-conversion).
                freq = det_freq.copy()
                freq[freq > 0] -= frame_shift
                freq[freq < 0] += frame_shift
                translation = -freq
                if signal_array.ndim == 1:
                    if translation[0] != 0:
                        if _complex_trace_needs_doubling(signal_array):
                            signal_array = signal_array * 2.0
                        signal_array = signal_array * np.exp(
                            1j * 2 * np.pi * translation[0] * 1e3 * time_array)
                else:
                    signal_array = np.array(signal_array, copy=True)
                    n_ch = signal_array.shape[-2]
                    for i in range(n_ch):
                        f_t = (translation[i] if i < len(translation)
                               else 0.0)
                        if f_t != 0:
                            tr = signal_array[..., i, :]
                            if _complex_trace_needs_doubling(tr):
                                tr = tr * 2.0
                            signal_array[..., i, :] = tr * \
                                np.exp(1j * 2 * np.pi * f_t * 1e3 * time_array)

    # Build info
    info = SpidyanInfo()
    info.FinalState = final_states
    info.StateTrajectories = state_traj
    info.Events = events

    return time_array, signal_array, info
