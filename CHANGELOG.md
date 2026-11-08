# Changelog

All notable changes to torchspin will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.2] — 2026-11-08

Bug fixes found while running the GPU tests on a CUDA host.

### Fixed

- **pepper on CUDA failed in every public release (0.3.0–0.6.1)** with
  "Expected all tensors to be on the same device". The transition pre-selection
  (`_preselect_pairs`) built its rate and mask tensors on the CPU, so every
  `Options(device='cuda')` call aborted before simulating. The CPU-vs-CUDA
  consistency tests catch this, but they only run on a GPU host.
- **pepper on CUDA ran out of memory on dense strain grids.** Batched CUDA
  eigendecompositions went to cuSOLVER in a single call. The solver reserves
  about 1–2 MB of workspace per matrix whatever its size, so strain broadening
  (one 6×6 diagonalization per resonance) asked for tens of GB. CUDA batches
  are now split into chunks that fit in half the free GPU memory.
- **pepper on CUDA was up to 25× slower than on CPU.** The resonance search
  finds cubic roots as eigenvalues of 3×3 companion matrices. On CUDA, torch
  computes general (non-Hermitian) eigenvalues one matrix at a time with a
  host sync each, hundreds of thousands per call: 34.7 s for the Mn(II)
  matrix workload against 1.4 s on 32 CPU threads. These small solves now run
  as one batched CPU call, the same LAPACK routine the CPU path uses.
- **pepper was not thread-safe.** The slot keys of the matrix method were kept
  in a function attribute shared by the whole process. With concurrent pepper
  calls, such as `esfit(..., FitOptions(workers='threads'))` or any user thread
  pool, one call could read another's keys: `_filter_slots` raised
  `IndexError: list index out of range`, or `separate='transitions'` mislabeled
  transitions. The keys are now per thread. This was found when a benchmark
  fit with the thread backend crashed on a single-core allocation.
- `sigeq(..., polarization=True)` built its identity matrix on the CPU, so it
  failed with a device mismatch for a CUDA Hamiltonian.

### Changed — behavior differs from 0.6.1

- **Seeded cardamom results no longer depend on the thread count.** The fast
  path groups orientations, and the group size scales with
  `torch.get_num_threads()`. Every group used to draw its trajectories from the
  same `CardamomPar.seed`, so a seeded spectrum changed with the number of
  threads, and orientations in different groups shared identical trajectories.
  Each orientation now draws from its own stream, derived from the seed with
  `numpy.random.SeedSequence`. The global-dynamics (`DiffGlobal`) trajectories
  of a group also started from a spiral grid over the whole group rather than
  over `nTraj`, which made the result depend on the group size too; they now
  start from the same per-orientation grid as the serial path. Seeded cardamom
  spectra differ from 0.6.1. Unseeded runs are unaffected.

### Added

- `DiffusionPar.seed` and `JumpPar.seed` accept a sequence of per-block seeds:
  the trajectories split into equal blocks with independent random streams.

### Tests

- The garlic CPU-vs-CUDA test compared the CPU with itself (garlic has no device
  path) and is removed. salt and `sigeq` have real CPU-vs-CUDA tests.

## [0.6.1] — 2026-11-04

Fixes for GitHub issues #5–#9. Each one was a case where torchspin returned a
plausible result without any warning that it was wrong. Two more defects found
while verifying them are fixed too: dict-style esfit ignored all but the last
element of each array field, and pepper_autograd returned NaN gradients for
an exactly isotropic A.

### Changed — behavior differs from 0.6.0

- **esfit `autoscale='lsq'` with a negative unconstrained scale** (#6). The
  scale is now the sign-constrained least-squares optimum: 0, with the
  baseline refitted alone. EasySpin, and torchspin up to 0.6.0, took
  `abs(scale)` but kept the baseline solved for the opposite sign. That
  objective is not a least-squares fit, so optimizers walked away from the
  true minimum (an inversion-recovery fit with `exp(-t/T)` ran to its bound).
  esfit warns when the final fit has a negative scale. The old behavior is
  `FitOptions(lsq_negative_scale='abs')`.
- **esfit uncertainties** (#5). A parameter that has no effect on the
  residuals at the finite-difference step now gets `pstd = nan` and a
  warning, not `pstd = 0`, which read as a perfectly determined parameter.
- **pepper with inputs that require grad** (#7) now returns a spectrum on the
  autograd graph by dispatching to `pepper_autograd`. It used to return a
  detached spectrum without any warning. Unsupported configurations raise an
  error that names `pepper_autograd`.
- **saffron with a predefined `Sequence` and nonzero `Exp.tp`** (#9) now raises,
  as `saffron.m` 351 does, in every simulation mode. 0.6.0 meant to do this,
  but the check sat in the fast path, and nonzero `tp` routed to the
  real-pulse engine first. Its predefined templates use fixed 16/32 ns pulses
  and never read `Exp.tp`, so the requested pulse lengths were silently
  ignored.

### Fixed

- **Dict-style esfit fitted only the last element of each array field.**
  `TorchSpinParameterHandler.to_dict` rebuilt each element from the original
  `p0`, so varying `g[0,0]`, `A[0,0]` or `lw[0]` never reached the simulator.
  Present since 0.3.0.
- **esfit was blind to line positions with garlic's default
  `AccumMethod='binning'`** (#5). The binned spectrum is piecewise-constant in
  g and A, so every finite-difference Jacobian was zero for them and levmar
  never moved A. Dict-style esfit now runs garlic with `'linear'` when
  `AccumMethod` is unset. levmar, trf and the uncertainty estimate warn when
  a Jacobian column is dead, which also covers custom models that wrap garlic.
- **esfit baselines with `autoscale='none'` and `'maxabs'`.** `'none'` fitted
  the baseline jointly with a free scale and then applied it at scale 1.
  `'maxabs'` built `scale * x**sum(arange)` and not a polynomial. Both now fit
  the polynomial to the data minus the scaled model.
- **pepper_autograd and natural-abundance nuclei** (#7). `Nucs=['Zr']` failed
  in `nucdata`. Components are now expanded into isotopologues and summed with
  `Sys.weight` × abundance, as in pepper. `Sys.weight` was previously ignored,
  and lists of components are accepted.
- **Quadrupole shorthand `[e2qQ/h, eta]` dropped the graph** (#7). It was
  converted with `float()`, so dL/dQ was silently zero in a fit. The principal
  values are now built with torch ops.
- **NaN gradients in pepper_autograd's axial-grid interpolation.** When a
  transition slot was missing at some orientations (e.g. exactly isotropic A),
  `torch.where`'s backward computed 0·NaN. The forward output is unchanged.
- The preselection cutoff no longer triggers PyTorch's "converting a tensor
  with requires_grad=True to a scalar" warning.

### Added

- spidyan warnings (#8) for a user-set `IntTimeStep` coarser than Nyquist/2
  (a π pulse is ~3 % off there and 40 % at Nyquist), and for `SimFreq` less
  than 5 nutation frequencies below the lowest pulse carrier.
- `docs/pitfalls.md` (#8): perturbation errors at large A/ν (measured on ⁹¹Zr,
  A/ν = 0.08), binned garlic spectra in fits, the lsq sign, satellite-assignment
  minima, Q fits started at zero, and the pulse-simulation limits.
- `examples/fitting/large_hyperfine_fit.py`: a staged fit of a synthetic
  natural-abundance Zr spectrum.
- `torchspin.autograd.inputs_require_grad`.

### Fixed — a flaky tolerance in the real-pulse frame-rotation test

`test_frame_tilts_match_an_equivalent_sample_tilt` asserted that a tensor tilt
and an equal sample tilt agree to `1e-12` of the peak, copying EasySpin's own
`areequal(y1, y2, 1e-12, 'abs')`. That bound sits below the round-off floor of
torchspin's route, which conjugates the Hamiltonian into the lab-frame basis
rather than rotating the interaction tensors: the two Hamiltonians agree to
2e-16 relative, but integrating that over the 0.3 µs detection window makes a
phase error growing linearly in time, 2e-11 absolute on a signal of magnitude
12. The test passed on two CI runners and failed on a third with identical
code. It is now scored as a relative L2 error (measured 1.8e-12 against a
1e-9 bound) plus a per-point bound at 1e-9 of the peak, and the mechanism is
written down in the docstring and in KNOWN_LIMITATIONS.md. Library code is
unchanged; both sides remain validated against MATLAB at ~1e-12.

## [0.6.0] — 2026-10-19

Follow-up to 0.5.0: the pulse-EPR test suites now assert the relative L2 error
against MATLAB rather than a cosine similarity, the one physics issue 0.5.0
left open turned out not to be a physics issue, and the real-pulse
(`saffron_thyme`) route — the last pulse-EPR route with no cross-validation at
all — is now EasySpin-equivalent and MATLAB-validated on 28 cases. Every
`saffron` route, ideal-pulse and real-pulse, rectangular and shaped, is now
cross-validated against MATLAB EasySpin.

Minor rather than patch: the real-pulse route's output changes (it was not
reproducing EasySpin before), and one input that used to be accepted now
raises. A predefined sequence with a nonzero `Exp.tp` was answered with a
silent *ideal*-pulse result; it now raises, as in `saffron.m` 351. Scripts
relying on that fallback were getting a wrong answer, and need either
`Exp.tp = 0` to keep the ideal-pulse result or a list-valued `Exp.Sequence`
to get the real-pulse one.

### Fixed — the real-pulse `saffron` route now reproduces EasySpin

Two new oracles, 28 MATLAB cases in all (MATLAB R2024b), agreeing on both the
raw and the mean-centered signal at 3e-13..8e-12:

- `tests/data/generate_saffron_thyme_refs.m` →
  `torchspin/tests/test_saffron_thyme_matlab_validation.py` (18 cases,
  rectangular pulses): powder and single-crystal samples, transient and
  single-point detection, echo integration, detection phase, explicit
  simulation frame, relaxation, one and two indirect dimensions, and
  predefined 2pESEEM / 3pESEEM / HYSCORE run through the real-pulse engine.
- `tests/data/generate_saffron_thyme_shaped_refs.m` →
  `torchspin/tests/test_saffron_thyme_shaped_matlab_validation.py` (10 cases,
  transcribed from EasySpin's own `tests/saffron_thyme_*.m`): `quartersin/
  linear` chirps with `Pulse.trise` and a swept `Pulse.Frequency`, relaxation
  during a chirp, `Opt.SimFreq = 15` against the automatic 32 GHz, a delay
  sweep crossed with a pulse-parameter sweep (`Exp.Dim2 = {'p1.Flip' pi/3}`,
  which regenerates the chirp at every point), `Sys.gFrame`/`Sys.AFrame`
  tensor tilts against an equivalent sample tilt, and a predefined sequence
  detected over a transient window.

Shaped pulses needed no new code: `torchspin/pulse.py` already built them for
`spidyan`, and a pulse dict in a list-valued `Exp.Sequence` passes through to
it unchanged. What had kept EasySpin's own thyme cases from running was the
propagation and detection, below.

0.5.0 reported that this route "does not reproduce EasySpin and has a different
output contract", and that no reduction of its per-increment transients
recovered EasySpin's trace. That was correct, and the cause was not only the
missing detection stage: the propagation was wrong too. Six separate problems,
each of which alone makes the comparison meaningless:

- **The pulses were never given a carrier.** `Exp.mwFreq` was not passed to the
  sequencer, so `FrameShift` was 0 and every pulse was built as a DC envelope
  and applied in the lab frame, against an electron Zeeman term of ~9.5 GHz.
  Nothing was excited. `Exp.mwFreq` is now required, as in EasySpin's
  `s_sequencer`, and the simulation frame is entered by the same isotropic
  g shift EasySpin uses (`saffron.m` 1838-1846).
- **The Hamiltonian and the operators were in different frames.** The
  Hamiltonian was built in the molecular frame with the field rotated into it,
  while the initial state (`-Sz`), excitation (`Sx`) and detection (`S+`)
  operators stayed lab-frame. Any orientation with a nonzero phi angle was
  therefore wrong, and rotating the operators instead is not an option because
  `s_thyme` builds its pulse propagators from `real(xOp)` and
  `1i*imag(xOp)`, which presumes `xOp = Sx`. The Hamiltonian is now
  transformed into the lab-frame basis with the spin-rotation operator
  `exp(-i*phi*Jz) exp(-i*theta*Jy) exp(-i*chi*Jz)` over all spins, which is
  the same operator EasySpin's `rotatesystem` + `ham(Sys_L, [0 0 B])`
  produces (agreement 5e-13) without needing per-interaction tensor rotation.
- **The orientation sum was renormalized.** EasySpin sums over orientations
  weighted by `Exp.OriWeights` and does not divide by the total weight, which
  is `4*pi`. Dividing by it scaled a powder down by ~12.6 and turned a
  multi-site crystal stack into a mean instead of a sum.
- **The whole detection stage was missing** (`saffron.m` 1884-1940):
  `Exp.DetPhase`, `Exp.DetIntegrate` echo integration, down-conversion through
  `signalprocessing.m` and the `Exp.DetWindow` time axis. `Exp.DetWindow` is
  applied the way EasySpin applies it, by shifting the last sequence element
  by `DetWindow(1)` and appending the detected event — a zero-length one for
  single-point detection — rather than by slicing the transient afterwards,
  which would move the echo rather than the window. The output contract is now
  EasySpin's: one detected point per increment for single-point detection,
  and the transient axis appended to the indirect axes otherwise.
- **`Opt.IntTimeStep`'s default did not match.** EasySpin takes the Nyquist
  time of the highest simulation-frame frequency, divides by 50 and rounds to
  two significant digits; torchspin divided by 10 and did not round. The two
  codes therefore integrated with different steps whenever the step was left
  implicit.
- **A zero-length detected delay produced no signal**, so EasySpin's way of
  expressing single-point detection yielded an empty trace.

### Fixed — relaxation propagated `conj(rho)` in Liouville space

`spidyan`'s Liouville-space commutator superoperator was written as EasySpin
writes it, `kron(I,H) - kron(H.',I)`, which is correct for MATLAB's
column-major `vec()`. numpy flattens row-major, where
`vec_r(A@rho@B) = kron(A, B.T) @ vec_r(rho)`, so the two Kronecker factors
swap. With EasySpin's order the relaxation path propagated `rho.T`, which for
a Hermitian density matrix is `conj(rho)`. The same applies to the transform
of the relaxation superoperator into the Hamiltonian eigenbasis.

This affects `spidyan` and the real-pulse `saffron` route whenever relaxation
is active. A real detection trace (`Sz`) is untouched, which is why the
existing `spidyan` relaxation reference — `Exp.DetOperator = {'z1'}`, compared
on its real part only — could not see it; a demodulated `S+` trace was wrong
by a factor of several. `generate_spidyan_refs.m` gains case 8,
`spidyan_relaxationcomplexdetection.mat`, which detects `S+` under relaxation
and is compared on its imaginary part as well as its real part. Regenerating
the other seven spidyan references reproduced the committed files bit for bit.

### Added — `Exp.Det*` fields and `Opt.SimulationMode` on saffron

`PulseExperiment` gains `DetWindow`, `DetSequence`, `DetPhase`,
`DetFrequency`, `DetIntegrate` and `DetOperator`, plus `Dim1`, `Dim2` and
`PhaseCycle` and support for a list-valued `Exp.Sequence` (EasySpin's general
pulse syntax: pulse dicts and delays). `SaffronOptions` gains
`SimulationMode` (`'fast'` | `'thyme'`), `IntTimeStep` and `SimFreq`.

Engine selection now follows EasySpin: `Opt.SimulationMode` chooses
explicitly, and leaving it unset keeps torchspin's inference from nonzero
`Exp.tp`, a list-valued `Exp.Sequence` or an `Exp.DetWindow`. The fast
algorithm raises EasySpin's errors rather than ignoring the input: a
predefined sequence with nonzero `Exp.tp` is rejected (`saffron.m` 351) where
torchspin used to ignore `Exp.tp` and silently return an ideal-pulse result,
and `Exp.DetSequence`/`Exp.DetOperator` are rejected (`saffron.m` 303). The
real-pulse engine warns about the fast-only fields it ignores, as EasySpin's
`s_sequencer` reports them.

### Fixed — misleading `NotImplementedError` for multi-electron saffron

`saffron.py` told users to set `Opt.SimulationMode='thyme'` for multi-electron
systems, an option `SaffronOptions` did not have. The option now exists, and
the message says what else is needed (a finite length for every pulse).

### Not changed — the ideal-pulse and real-pulse routes do not converge

The plan this work followed expected that as `Exp.tp -> 0` the real-pulse
result would approach the ideal-pulse result for the same sequence, as a
reference-free check on the normalization. It does not, and that is EasySpin's
behavior too: at matched settings torchspin reproduces EasySpin's own
ideal-vs-real mismatch to 1e-7 in scale (measured at four pulse lengths from
20 ns down to 2.5 ns), so there is nothing here for torchspin to fix. Three
reasons, none of them a defect:

- With single-point detection EasySpin does not down-convert, so the detected
  point carries the simulation-frame carrier phase, which advances ~50 cycles
  per 20 ns increment.
- A single detected point contains coherence pathways that the ideal route
  excludes analytically; the two-step cycles EasySpin's own templates use do
  not remove them for this observable.
- As `tp -> 0` at fixed flip angle the required B1 grows without bound and
  eventually exceeds the carrier frequency, so the rotating-wave picture
  breaks down and a rectangular real pulse does not approach an ideal one.

Putting the simulation frame on the carrier to remove the first effect is not
a neutral change: it removes the carrier from the pulse too, so the linearly
polarized field becomes a rotating-frame field of twice the strength.

What that check was reaching for is covered instead by three relations that
need no reference and do hold: a single-site P1 crystal equals `4*pi` times
the independently validated `spidyan` result (1e-12); an isotropic powder
equals a single crystal; and a nominal `Pulse.Flip` of `pi/2` delivers full
transverse magnetization while `pi` delivers none, which is the sharpest cheap
guard against the pulse amplitude slipping by a factor of two.

### Changed — `spidyan`'s default detection, and `Exp.DetSequence` strings

With neither `Exp.DetWindow` nor `Exp.DetSequence` given, `_sequencer` used to
detect only the last free evolution event. EasySpin's documented default is
`Exp.DetSequence = ones(1, length(Exp.Sequence))` — detection active during
the entire sequence — and that is now what happens. Calls that state their
detection explicitly, which is every cross-validated case, are unaffected.
`Exp.DetSequence` also accepts EasySpin's `'last'` and `'all'` strings, and a
length that is neither 1 nor `length(Exp.Sequence)` now raises instead of
being silently truncated.

### Fixed — `rfmixer` could drop the last sample

The output time axis was built with `np.arange(t[0], t[-1] + dt/2, dt)` and
then clipped to `t <= t[-1]`, so floating-point round-off in the last step
could discard the final point and return a signal one sample shorter than its
input. MATLAB's `tIn(1):Opt.dt:tIn(end)` does not, and the axis is now built
from a step count.

### Fixed — the "spidyan multi-spin phase drift" was a test transcription error

0.5.0's `KNOWN_LIMITATIONS` reported an undiagnosed residual in `spidyan` that
grew linearly through free evolution for multi-spin systems (4.7e-3 for two
coupled electrons, 6.4e-5 electron-nucleus, 8.8e-6 for S=3/2), and warned that
it would be worse for longer sequences.  **There is no such defect.**

Fitting the accumulated phase showed torchspin reproducing the analytic Larmor
frequency exactly (1489.828541 MHz at 1195 mT, g = 2.00232, in the 32 GHz
simulation frame) while the MATLAB reference sat 11.6 kHz below it.  That
offset is `(2.00232 - gfree)/2.00232 = 3.47e-7` of the full Larmor frequency:
`generate_spidyan_refs.m` sets `Sys.g = gfree` (2.0023193043...), and the Python
tests had transcribed it as a rounded `2.00232`.  A 3.5e-7 error in g shifts the
precession frequency enough to accumulate visibly as phase over a 0.5 us delay.

The three affected tests are exactly the three whose generator uses
`Sys.g = gfree`; the cases that set `Sys.ZeemanFreq` were unaffected and had
always agreed to 1e-13.  With `GFREE` used in place of the rounded value, every
`spidyan` case agrees to 1e-14..1e-12.  No library code changed.

### Changed — pulse-EPR tests assert relative error, not cosine

`test_spidyan_matlab_validation.py`, `test_spidyan_tp_matlab_validation.py` and
`test_saffron_matlab_validation.py` now assert the relative L2 error against the
MATLAB reference (`STRICT_TOL = 1e-8` and `1e-6` respectively) instead of
`cosine > 0.85..0.99`.  A cosine is scale-invariant and dominated by a pulse-EPR
trace's DC offset, so it could see neither the eight defects fixed in 0.5.0 nor
the `gfree` rounding above.

Making them strict required fixing three kinds of harness fault that a cosine
had been hiding.  **None was a library fault** -- checked against matched
references, agreement is 1e-16..1e-7 in every case:

- **Accumulation path.** No saffron reference generator sets `Opt.TimeDomain`,
  so every reference comes from the frequency-binned path, but twelve tests ran
  with `TimeDomain=True`.  That compares direct time evolution against
  EasySpin's own binned approximation -- a 7-60% difference in the modulation
  that is real, expected, and nothing to do with correctness.
- **Normalization.** Six generators write `y = y/max(abs(y))`; the tests
  compared torchspin's unnormalized output against them, leaving fixed scale
  factors of ~4*pi and similar.
- **Orientation grid.** `saffron_hyscore14n` and `saffron_hyscore_ganiso` are
  generated at `Opt.GridSize = 181` but were tested at 91, and
  `saffron_2psimple` is generated at EasySpin's default 31 but was tested at 20.
  `GridSize` is now stated explicitly in every case rather than inherited from a
  default that could drift apart between the two codes.

`test_orientation_selective` documents a measured case where the **committed
fixture** is stale: regenerating it from the local EasySpin gives a reference
torchspin matches to 4.4e-15, and that fresh reference differs from the stored
one by 1.4e-6 in overall scale.  Its amplitude is therefore checked at 1e-5 with
the peak-normalized shape checked strictly, rather than silently loosening
everything.

The dead cosine helpers are removed, and the tests no longer carry a `_cosine`
suffix that would misdescribe what they assert.

## [0.5.0] — 2026-10-19

Found while cross-validating `saffron` against MATLAB EasySpin on a 13C
HYSCORE model, where the modulation came out wrong although the raw cosine
stayed at 0.99999.  The existing `saffron` suite scored
agreement with a cosine similarity, which for a pulse-EPR time trace is
dominated by the DC offset: a HYSCORE trace can have a ~100% wrong modulation
and still score >0.999.  Every defect below lived in that blind spot.

New oracle: `tests/data/generate_saffron_phase_refs.m` →
`torchspin/tests/test_saffron_phase_matlab_validation.py` (25 cases, MATLAB
R2024b), plus a strict `spidyan` amplitude case.  It asserts the **mean-centered** relative L2 error — the modulation
— as well as the raw error, at tolerances near double-precision roundoff.  All
26 strict cases, 25 of them fail on 0.4.0 and pass here (the 26th, two-nucleus
Mims ENDOR, is the one case 0.4.0's hardcoded `EndorMethod=1` got right).  Agreement across the predefined
2p/3p/4pESEEM, HYSCORE and Mims-ENDOR routes, direct and frequency-binned,
powder and explicit single crystal, S=1/2 and S=3/2, with and without
orientation selection and nuclear quadrupole coupling, is now 1e-16 to 1e-14
rather than "cosine ≥ 0.98".

### Fixed — `saffron` tau phase factor in all four predefined ESEEM/HYSCORE routes

`saffron.m` builds the tau-dependent phase factor as
`Q_ = exp(-2i*pi*Ea*tau)*exp(-2i*pi*Eb*tau)'`, and MATLAB's `'` is a
**conjugate** transpose, so element *(i,j)* is `exp(-2i*pi*(Ea_i - Eb_j)*tau)`
— a *difference* of the two manifold energies.  torchspin formed the outer
product without conjugating the second factor, giving their *sum*.  The error
vanishes at `tau = 0`, which is why it survived.

Fixed for `exp_id` 1, 2, 3 and 4 (2pESEEM, 3pESEEM, 4pESEEM, HYSCORE); the
Mims-ENDOR branch already conjugated correctly.  On the in-repo
`saffron_HYSCORE.mat` case (tau = 0.08 µs) the mean-centered error against
MATLAB drops from **112% to roundoff**, while the raw cosine moves only from
0.999991 to 1.000000.

### Fixed — `saffron` alpha/beta manifold assignment for S=1/2

EasySpin's `Transitions = [1 2]` makes `Ea` the **upper** manifold
(`<S> = +1/2·n`) and `Eb` the lower.  The S=1/2 fast path had them swapped,
while torchspin's own S>1/2 path assigned them correctly — an internal
inconsistency.  Invisible while the nuclear Hamiltonian is linear in **I**
(the two HYSCORE coherence-transfer pathways then symmetrize the swap away),
it becomes a 7–33% error as soon as a nuclear quadrupole makes the manifold
eigenvectors non-trivial.

### Fixed — `saffron` dimension order of the 2D frequency-bin accumulation

`sf_peaks.c` bins a 2D spectrum at `idx = nu1 + nu2*nPoints1`, i.e. dimension
1 carries `nu1`.  `saffron_peaks.py` computed `idx2 * n_points1 + idx1` into a
row-major `(n_points1, n_points2)` buffer, which **transposes** the spectrum
and mis-strides it when the two dimensions differ in length.  Fixed at all
five 2D increment schemes (11–15, 17).  The binned and direct paths of
torchspin itself disagreed by 2% before this; a new test asserts they agree,
which catches this class of bug without any MATLAB reference.

### Fixed — `saffron` `Opt.OriThreshold` reference level under orientation selection

EasySpin takes the pruning threshold from the **unweighted**
`|<i|SyLab|j>|` maximum and only then multiplies in the excitation amplitude,
so a narrow `Exp.ExciteWidth` discards more transitions.  torchspin took the
maximum after excitation weighting and so kept transitions EasySpin drops,
giving a 75% modulation error for S=3/2 + 14N at `ExciteWidth = 100` MHz.
Affects S>1/2 with `Exp.ExciteWidth` only.

### Fixed — `saffron` `Opt.Expand` and `Opt.EndorMethod` defaults are conditional in EasySpin

Both were hardcoded to a single value, so the default-argument path silently
ran a different algorithm than EasySpin:

- `Opt.Expand` defaults to **4 for 1D and 2 for 2D** sequences
  (`saffron.m:751`); torchspin always used 4, changing the zero-fill of every
  binned HYSCORE.
- `Opt.EndorMethod` defaults to **0 for a single SHF nucleus or under
  `ProductRule`, else 1** (`saffron.m:729`); torchspin always used 1, which
  selects a different RF-pulse model.  For one nucleus this put the ENDOR
  amplitude 40% out and changed the lineshape, at cosine 0.996.

Both now accept `None` (the default) to mean EasySpin's rule; an explicit
integer still overrides it.

### Fixed — `saffron`/`saffron_thyme` treated `Exp.SampleFrame` alone as a single crystal

EasySpin decides sample type from `CrystalSymmetry` or `MolFrame` only
(`p_sampletype.m:26`): `Exp.SampleFrame` rotates the sample, but a disordered
powder stays a powder.  `saffron` and `saffron_thyme` also keyed on
`SampleFrame`, so `SampleFrame` alone silently simulated a **single crystal
where EasySpin simulates a powder** — a different physical sample, not a
numerical discrepancy.  `pepper` already had this right; `saffron` now matches
it.  Code that relied on `SampleFrame` alone to select a crystal must now pass
`CrystalSymmetry` (e.g. `'P1'`) or `MolFrame`, as EasySpin requires.

### Fixed — `saffron` custom sequences were not divided by their pathway prefactor

For a user-defined sequence (`Exp.Sequence` given as pulses and delays, not a
predefined name) `saffron.m:1669` divides the signal by the largest surviving
pathway prefactor.  torchspin omitted this, leaving a custom HYSCORE-style
sequence a factor of **8** too small.  Shape was unaffected, so the cosine was
1.000000 and the existing `cosine > 0.95` test passed.  Predefined sequences
are deliberately *not* renormalized this way, and are unchanged.

### Fixed — `spidyan` complex detection operators were half amplitude

`signalprocessing.m` multiplies a **complex** detection trace (`S+`/`S-`) by
two when it is demodulated, because the analytic-signal normalization is
otherwise off by that factor; a purely real or purely imaginary trace (`Sz`) is
left alone.  torchspin applied the demodulation phase but not the factor, so a
demodulated `<S+>` channel came out at exactly half EasySpin's amplitude — at
cosine 1.000000, under a `cosine >= 0.99` assertion.  The `<Sz>` channel in the
same simulation was already correct, which is why it went unnoticed.

### Audit of the remaining pulse-EPR routes

Prompted by the above, the routes that were scored only by cosine were
re-checked against fresh MATLAB references with strict metrics.  Now at
roundoff and covered by tests: custom user sequences (direct and binned),
`ProductRule` x `TimeDomain` for two nuclei (all four combinations),
multi-component input with `Sys.weight`, and S=1 systems.

Several apparent discrepancies turned out to be **test-harness** faults rather
than library faults, and are worth knowing about before trusting an old
number: `test_3psimple`, `test_4psimple`, `test_hyscore14n`,
`test_hyscore_ganiso`, `test_mimssimple` and `test_twocomponents` compare
torchspin's unnormalized output against a MATLAB reference its generator had
normalized (`y = y/max(abs(y))`), so they sat at a fixed scale factor of 4*pi
or similar; and `test_saffron_2pESEEM`, `test_saffron_3pESEEM`,
`test_saffron_HYSCORE_cosine_1d`, `test_S1_*`, `test_S32_HYSCORE`,
`test_orientation_selective` and `test_customsequence` run with
`TimeDomain=True` against references generated *without* `Opt.TimeDomain`,
i.e. they compare direct time evolution against EasySpin's own
frequency-binned approximation.  Both faults are invisible to a cosine.
Checked against matched references, the library agrees to roundoff in every
one of these cases.

Still open, and recorded in `KNOWN_LIMITATIONS.md`:

- ~~`spidyan` multi-spin systems show a small residual that grows linearly
  through free evolution — a phase drift of order 1e-5 relative — for two
  coupled electrons (4.7e-3 overall), an electron-nucleus pair (6.4e-5) and
  S=3/2 (8.8e-6).  Single-electron cases are exact to 1e-13.  Not diagnosed.~~
  **Retracted — there is no such defect.** This was a test transcription error
  (the tests rounded EasySpin's `gfree` to 2.00232); torchspin reproduces the
  analytic Larmor frequency exactly. See [Unreleased]. The claim is struck
  through rather than deleted because 0.5.0 shipped with it.
- `saffron` with real pulses returns per-increment transients
  (`nPoints x nDet x nTime`) where EasySpin returns one detected point per
  increment, and no simple reduction of those transients recovers EasySpin's
  trace — the closest is ~3% off after fitting an arbitrary scale.  Treat the
  real-pulse route as unvalidated; `saffron_thyme` remains without MATLAB
  cross-validation.
- `saffron` accepts a predefined `Exp.Sequence` together with real pulses
  (`Exp.tp`) and silently routes it to the thyme engine; EasySpin rejects that
  combination outright.  The output is therefore unvalidated.
- EasySpin's shaped/chirped pulses (`Type`, `trise`, `Frequency`),
  `Exp.DetWindow`/`DetPhase` and `Opt.SimulationMode` are not implemented in
  `saffron_thyme`, so EasySpin's own thyme test cases cannot be run against it.

### Note

`saffron` gradients are unaffected in form: a finite-difference check against
the corrected simulator at nonsymmetric hyperfine values agrees to 1e-10.
Gradients taken before this release were consistent with the *old*, incorrect
forward model.

## [0.4.0] — 2026-10-09

Found by fitting a Cu(II) phthalocyanine spectrum (CuPc in ZnPc: S=1/2,
A∥(Cu) ≈ 647 MHz, four equivalent 14N) — a system that hit every limit of
`pepper`'s resonance solvers at once.  Every change below is validated against
EasySpin on MATLAB R2024b through a new 34-case oracle,
`tests/data/generate_pepper_cupc_refs.m` →
`torchspin/tests/test_pepper_cupc_matlab_validation.py`, covering all three
solvers on the same systems plus natural-abundance isotopologues, tilted and
anisotropic ligand A tensors, nuclear quadrupole coupling, an enlarged exact
core, and a two-component model.

### Added — `Options.Method='hybrid'` (EasySpin `Opt.Method='hybrid'`)

Exact diagonalization of a core — all electron spins plus the nuclei named in
the new `Options.HybridCoreNuclei` (1-based, as in EasySpin) — with the
remaining nuclei shifting and splitting the core lines.  This is the method for
a large central hyperfine coupling surrounded by small ligand couplings, where
the exact treatment is unaffordable and perturbation theory is wrong about the
central coupling.  Cu + 4×14N needs a 648-dimensional Hilbert space exactly,
which neither code can put in a fit loop; as hybrid the core is 8-dimensional
and each nitrogen is a 3×3 problem, and the spectrum takes **0.27 s against
EasySpin's 0.28 s**.

The implementation follows `resfields.m`: the electron spin operators in the
hyperfine term are replaced by their expectation values in the two core
eigenstates at the resonance field, and the resulting nuclear sub-Hamiltonian
is then diagonalized *exactly*, so the quadrupole and nuclear Zeeman terms and
the nuclear state mixing are not approximated.  Line amplitudes come from the
Mims overlap matrix |⟨u_r|v_c⟩|², sub-lines below
`Options.HybridIntThreshold` (0.005, as EasySpin) are dropped, and the nuclei
are combined by an outer sum of shifts and an outer product of amplitudes.
Matches EasySpin's hybrid to cosine 0.9998 on every case.

Against the exact `matrix` result hybrid reaches cosine 0.9805 (Cu + 1×14N) and
0.9740 (Cu + 2×14N).  **EasySpin's own hybrid gives 0.982 and 0.974 on the same
comparison** — that gap is the method's first-order decoupling, not a port
defect.  Second-order perturbation theory gives 0.896 and 0.918 on the same
systems.  See § Pepper in `KNOWN_LIMITATIONS.md` for when to use which.

### Added — `pepper` supports sets of equivalent nuclei

`SpinSystem.n > 1` now works on the `hybrid` and `perturb` paths, which treat
nuclei one at a time and combine them combinatorially, so a multiplicity is a
pure saving rather than an obstacle: four equivalent 14N written as `n=[1,4]`
take 0.30 s instead of 2.0 s and agree with four explicit nitrogens to cosine
0.99999999.  A set of `n` equivalent nuclei shares one sub-Hamiltonian, so its
copies are combined once by multiset enumeration — for four 14N that is 495
combinations instead of 9⁴ = 6561.

The `matrix` path still rejects `n > 1`, since there the nuclei multiply the
Hilbert space and cannot be collapsed.  EasySpin's `pepper` rejects `n > 1` for
every method, so this goes beyond it.

### Fixed — `Method='matrix'` crashed on highly degenerate systems

`pepper(Sys, Exp)` on S=1/2 Cu + 4 equivalent 14N aborted with
`torch._C._LinAlgError: linalg.eigh: (Batch element 5): The algorithm failed to
converge … too many repeated eigenvalues (error code: 394590)` after 13 s.  One
matrix out of a batch of 59 complex 648×648 Hermitians: the input is Hermitian
to 0.0 and finite, but LAPACK's divide-and-conquer driver stalls on the
hundreds of exactly equal eigenvalue gaps that four identical nuclei produce.

`_linalg.eigh` and `_linalg.eigvalsh` had no error handling, so one bad matrix
killed the whole orientation batch.  They now retry a rejected batch matrix by
matrix — the torch routine alone, then NumPy (a different LAPACK build, and
`zheevr`/`zheev` rather than divide-and-conquer), then a 1e-14 relative diagonal
perturbation that breaks the exact ties.  Spectra are invariant under unitary
rotations inside a degenerate subspace, so none of these changes the result
beyond round-off.  On the autograd path the NumPy detour is skipped, because it
would silently drop gradients.

### Fixed — `Options.IsoCutoff` had no effect

`expand_components` passed the cutoff as the second positional argument of
`isotopologues`, which is `n`, so it was silently discarded and pruning always
used the 1e-4 default.  Affected `pepper` and `garlic`.  No test caught it
because every test passed `rel_threshold` by keyword.

### Fixed — the perturbation path built a Hamiltonian it never used

`pepper` built the full `H0`/`mux`/`muy`/`muz` operators before dispatching on
`Options.Method`, but `resfields_perturb_batch` works from the spin system and
never receives them.  For Cu + 4×14N that is a 648×648 Kronecker product
costing about 80 % of the run time, repeated for every isotopologue — which is
why the cost barely depended on `GridSize`.  They are now built on first use.
Natural-abundance Cu + 4×N at `[91,4]` goes from **2.70 s to 0.41 s**, now
faster than EasySpin's 0.71 s, with bit-identical output (verified on 14 systems
spanning both solvers, strain broadening and multi-component input).

### Fixed — `Options.Method` accepted perturbation orders `pepper` cannot run

`'perturb3'`, `'perturb4'` and `'perturb5'` were accepted and silently run as
second order, and the requested order was never passed to the solver at all, so
`'perturb1'` also ran second order.  `pepper` now rejects the orders it does not
implement and forwards the order it does, and the `Options.Method` docstring
describes the methods per simulator instead of claiming `'perturb'` means fifth
order everywhere (that is garlic's convention; pepper's `'perturb'` is second
order, as in EasySpin).

### Fixed — two examples did not compute what they claimed

`examples/solidstate/copper_nitrogens.py` was titled "Four Equivalent 14N
Ligands" and labelled its plot "4×14N" while simulating **two**, and had dropped
the `AFrame` tilts that make the four nitrogens inequivalent in the first place.
It is now the system from the EasySpin original (Buchanan & Dismukes 1987) and
shows `perturb` against `hybrid`.  `examples/solidstate/matrixperturb.py` ran
the matrix method twice, labelled one curve "perturb", and carried a comment
claiming perturbation theory "is not yet exposed as a pepper option in
torchspin" — untrue since 0.2.x.  It now calls `Method='perturb'`.

### Fixed — concurrent eigendecompositions could corrupt the thread count

`_linalg._pooled` and `pool_map` save, set and restore `torch.set_num_threads`,
which is process-global.  `esfit` evaluates objective functions in a thread
pool, so two concurrent batches could interleave and leave the process pinned
at one intra-op thread for the rest of the session.  Both now hold a lock
across the critical section.

### Added — `grid_convergence`, to answer whether `Options.GridSize` is enough

The default `GridSize=[19,4]` is badly unconverged for a narrow line on a large
hyperfine coupling — for the reported Cu(II) case the derivative peak is 3 % out
at `[19,4]` and still moving 4 % between `[91,4]` and `[361,4]`, in EasySpin
equally — and a single spectrum gives no hint of it.

`torchspin.grid_convergence(sys, exp, opt)` simulates again on a `refine`×
finer coarse grid and reports the cosine and amplitude change, with
`.converged` against tolerances a tenth of the repository's parity bar, and
`.report()` giving a sentence that names a grid to use.

A heuristic was tried first and deliberately abandoned: comparing the field step
between knots against the linewidth sounds right, but the SOPHE projection
integrates analytically over each grid segment, so a large step per knot is
normal and harmless. That heuristic flagged ~60 simulations in this repository's
own suite which agree with EasySpin to cosine 0.9999. A check that cries wolf on
validated-correct cases is worse than none, so the honest version costs a second
simulation and is opt-in rather than a warning inside `pepper`.

### Fixed — `torchspin.fitgui` now says which extra to install

`FitPanel(...)` failed with a bare `ModuleNotFoundError: No module named
'ipywidgets'` and no hint that an optional dependency group provides it.  The
imports now raise an `ImportError` naming the module, what it is needed for, and
`pip install "torchspin[gui]"`.

The `gui` extra was also incomplete: it omitted `ipython`, which
`FitPanel.show()` needs, and `cloudpickle`, which the panel's own `n_workers`
control needs.  Both are now in it, and `ipywidgets` is in `test` and `dev` so
the four `test_fitgui.py` tests run instead of skipping.

### Fixed — `vary` could put a linewidth or a weight below zero

`vary` becomes `p0 ± vary`, so `Vary.lw = 1` around `lw = 0.54`, or
`Vary.weight = 0.2` around `weight = 0.001` — both taken from a real fit script
— handed the optimizer a negative lower bound for a quantity that cannot be
negative.  In the dict style the field name is known, so the lower bound is now
clipped at zero for `lw`, `lwpp`, `lwEndor`, `HStrain`, `gStrain`, `AStrain`,
`DStrain`, `weight`, `tcorr`, `Diff`, `T1` and `T2`, which is what EasySpin's
`esfit` does (`nonnegFieldNames`).  Explicit `lb`/`ub` are still used exactly as
given — clipping a bound the user stated would be worse than the problem.

A plain parameter vector carries no field names, so neither `esfit` nor
`FitSession` can clip there, as in EasySpin; both docstrings now say so and
point at `lb`/`ub`.

### Fixed — `eprload` units are now documented where they bite

`eprload` returns the abscissa in the file's own units, so a Bruker field axis
is in **gauss** while the rest of torchspin works in mT.  That is deliberate and
matches EasySpin, and it was recorded in `KNOWN_LIMITATIONS.md` — but the
`eprload` docstring said only "Abscissa (magnetic field, time, frequency,
etc.)", which is where a user actually looks before feeding the axis straight
into `Experiment.Range`.  The docstring now states the units per format, shows
the `/ 10` conversion in a worked example, and notes that the file's own unit
string survives in `params`.

`eprload_info` now prints the unit and, for a gauss axis, that it needs dividing
by 10 (it also no longer crashes on multi-dimensional data, where `x` is a list
of axes and `y` can be a list of datasets).

### Documented — derivative amplitude at a turning point is grid-limited

Investigated because it first looked like a parity defect of the exact solver,
and turned out not to be one.  Comparing Cu + 1×14N by `matrix` against EasySpin
at `GridSize=[91,4]` gives cosine 0.99948 but amplitude 0.990, and 0.978 at two
nitrogens — outside the 2 % amplitude convention.  Refining the grid removes it:
0.9903 → 0.9978 → **1.0010** and cosine 0.99948 → 0.99998 → **1.00000** over
`[91,4]` → `[181,4]` → `[361,4]`.

Neither code is converged at `[91,4]`: EasySpin's own derivative peak moves
53.70 → 51.62 → 51.44 across the same refinement, a 4 % change in its own
answer.  The two are simply unconverged by slightly different amounts.  The
derivative extremum at a turning point is the most grid-sensitive quantity in a
powder spectrum, much more so than the integrated intensity.

Absolute intensity agrees throughout, including with nuclei: integrated
absorption against EasySpin is 578.38/578.38, 559.77/559.90 and 535.49/534.99
for 0, 1 and 2 nitrogens (0.02–0.09 %), and absorption peak heights agree to
0.2–0.8 %.  That the integral falls as nuclei are added is real but **EasySpin
does the same to 0.1 %** — it is the matrix method's transition pre-selection
dropping weak transitions, and both codes select exactly the same number of
level pairs (60 and 215).  `hybrid` and `perturb` hold at 578 because they put
all the nuclear intensity on the core lines.

The two `cupc_*_2N_matrix` cases are kept at `[91,4]` with the amplitude
tolerance set to the measured grid error, as the regression anchor for this.

### Added — `benchmarks/python/cupc_claims.py`

Reproduces the five reported defects and records the measured before/after
state, so the performance claims have a ledger rather than an anecdote.

## [0.3.0] — 2026-10-04

First public release.

### Fixed — NumPy 2.0 and case-sensitive filesystems (2026-10-04)

Found by installing the release into an empty environment on a machine that
had never run torchspin (NumPy 2.5, SciPy 1.18, PyTorch 2.14, Linux).  All
three were silent on the development hosts and broke 26 tests on the clean
one; none changes any numerical result.

- `pepper` partial ordering (`Exp.Ordering`) raised
  `AttributeError: module 'numpy' has no attribute 'trapz'` on **every** call
  under NumPy >= 2.0, which removed `np.trapz`.  `torchspin/ordering.py` now
  uses `np.trapezoid` through the same fallback shim already used in
  `torchspin/pulse.py` and `torchspin/exponfit.py`.  The 10 MATLAB-validation
  cases for ordering (`test_pepper_features2_matlab_validation.py`,
  `test_pepper_round3_matlab_validation.py`) were failing on this alone and
  now pass — the ordering physics was never in question.
- `eprload` could not open Bruker ESP/WinEPR data on a case-sensitive
  filesystem: a `FOO.SPC` file was paired with a hard-coded `FOO.par`, so the
  real `FOO.PAR` was reported missing.  The companion-file probe that the
  BES3T loader already had is now a module-level `_find_companion` used by
  both loaders.
- Three saffron tests called `float()` on the shape-`(nNuclei,)` array
  returned by `larmorfrq`, which NumPy 2.0 rejects; they now take the element
  explicitly.


### Added — esfit `method='trf'` (2026-09-09)
- `esfit`: `FitOptions(method='trf')`, SciPy's bounded trust-region reflective
  least squares on esfit's target-transformed residual vector. Parameter
  mapping (dict/array `p0`, fixed parameters, `lb`/`ub`/`vary`), amplitude and
  baseline fitting (`autoscale`/`baseline`), `target`, progress reporting and
  `max_time`/`stop_when`/interrupt handling are shared with the other methods.
  New options: `trf_diff_step` (relative finite-difference step, SciPy
  `diff_step`), `trf_x_scale` (`'jac'`, a float, or one entry per active
  parameter) and `trf_gradient_tol` (`gtol`); `tol_fun`/`tol_x` are passed as
  `ftol`/`xtol`. For this method `max_iter` is SciPy's `max_nfev`, the number
  of step evaluations; the finite-difference Jacobian probes are not charged
  against it but are counted in `FitResult.n_evaluations` and drive the
  progress/stop checks. An exhausted budget returns `success=False,
  interrupted=False` with SciPy's message; a stop or interrupt returns the
  best point so far with `interrupted=True`; `n_iterations` is the
  Jacobian-evaluation count (one per trust-region iteration). TRF is an
  optimization method, not a CPU-acceleration feature: it runs sequentially;
  simulation threading and the population-worker `n_workers` parallelism are
  separate. The default method is unchanged (`simplex`). Parity with a direct
  `scipy.optimize.least_squares` call is tested (`tests/test_esfit_trf.py`).

### Fixed — fitting panel (2026-09-09)
- `fitgui`: `'trf'` is offered in the method selector. Sessions built with
  explicit `lb`/`ub` keep those absolute bounds across restarts and slider
  moves (they were rebuilt as p0 ± vary around each new start). A parameter
  fixed for one run (the "fix" box, or `start(vary=...)` with a zero) can be
  released again on the next run (the zeroed `vary` used to overwrite the
  session's). A start outside the bounds raises before any session state
  changes, and the panel re-enables its controls. The residual trace is now
  data − fit (it was fit − data).
- `FitResult.residuals` docstring states the actual sign (fit − data).

### Added — ML layer, cardamom seed (2026-09-06)
- `torchspin.ml`: `ParamSpec` (free spin-system parameters with ranges and
  linear/log/softplus transforms; `pack`/`unpack` between a flat tensor and a
  `SpinSystem` built from a template, on the autograd graph when the vector
  requires grad), `simulate`/`simulate_batch` (any differentiable simulator,
  `(N, nPoints)` output, optional spawned pool for no-grad generation),
  `SpectrumDataset` (seeded on-the-fly samples with resampling, normalization,
  noise and baseline), and torch losses (`cosine_loss`, `rmsd`, `normalize`,
  `resample`) so network predictions can be refined through the simulators.
- `convspec` accepts tensor widths: differentiable ones are routed to the
  torch implementation, others are used as floats.
- `CardamomPar.seed`: reproducible trajectory ensembles (`DiffusionPar`/
  `JumpPar` seeds were not reachable from `cardamom`).

### Changed — differentiable field modulation (2026-09-06)
- `dataproc.fieldmod_t`: the pseudo-modulation (Jacobi–Anger Bessel kernel,
  Kaelin & Schweiger) in torch (orders 0–2 closed form, higher by recurrence),
  equal to `fieldmod` to 1e-12; `pepper_autograd` and `garlic` use it, so
  `Exp.ModAmp` spectra are differentiable (pepper: forward = `pepper()` to
  5e-10, gradient vs FD 2e-6).

### Changed — pepper_autograd: perturbation methods (2026-09-06)
- `Opt.Method='perturb*'` now uses pepper's perturbation resonance route in
  `pepper_autograd` (forward equals `pepper()` to 1e-10 for the dense-grid
  perturb2 workload, nitroxide perturb1 and Cu perturb2). Second-order trace
  and determinant scalars in `resfields_perturb_batch` stay tensors (they were
  `.item()`-detached, which left 1e-2 errors in the A gradient); gradients vs
  finite differences of `pepper()` are 1e-8 for A and g.

### Changed — pepper_autograd: strain broadening (2026-09-06)
- `pepper_autograd` / `differentiable_spectrum` handle strains (HStrain,
  gStrain, AStrain, DStrain): the per-transition widths from
  `compute_strain_widths_batch` are already torch, and the EasySpin summation
  branch (facet centers, spread-smoothed widths, bin-integrated Gaussians —
  `gaussian_bins_t`) is now torch too. Forward equals `pepper()` to 2e-12 on
  nitroxide HStrain+gStrain (D2h), axial gStrain (Dinfh), AStrain and S=1
  DStrain cases; gradients vs finite differences of `pepper()` 1e-6–1e-8 for
  g, A, gStrain, HStrain. DStrain magnitudes are constants (not differentiable).

### Changed — differentiable saffron (2026-09-06)
- `saffron` (ESEEM/HYSCORE, predefined and custom ideal-pulse sequences): the
  relaxation decay, apodisation and FFT tail and the S=1/2 orientation-selection
  weights are torch, so with grad tensors in the spin system the time-domain
  signal and `info['td']`/`info['fd']` are tensors on the autograd graph; without
  grad tensors the NumPy outputs are unchanged (forward identical to rounding).
  The default frequency-domain accumulation bins peaks (EasySpin `sf_peaks`) and
  is not differentiable in the peak frequencies, so grad tensors switch the
  call to `SaffronOptions(TimeDomain=True)` (exact evolution; forward differs
  from the binned default by the bin quantisation). Gradients vs finite
  differences of the time-domain path: 1e-5–1e-7 for A, Q, g (2pESEEM,
  HYSCORE; `test_saffron_autograd.py`). Mims-ENDOR raises for grad tensors.

### Changed — differentiable garlic (2026-09-06)
- `garlic` runs in torch end to end (same algorithm: isotropic g/A, Breit–Rabi
  fixed point and perturbation Newton for the line positions, Kivelson–Freed
  fast-motion widths in `fastmotion_t`, explicit Lorentzian accumulation with
  torch line shapes and convolution, torch resampling), so spectra are on the
  autograd graph for g, A, Q, tcorr/logtcorr and lw/lwpp. Forward unchanged to
  1e-9 (Fremy's salt, nitroxide fast motion, methyl perturb2, frequency sweep,
  Temperature, Harmonic 2, ModAmp); gradients vs finite differences 1e-8
  (`test_garlic_autograd.py`; the fine accumulation grid is re-discretized with
  the smallest width, so tcorr finite differences need a step within one
  discretization).
- `Options.AccumMethod='linear'` (EasySpin `makespec`: a line is split between
  its two neighboring bins) is the differentiable form of the default
  nearest-bin stick spectrum and is selected automatically when the spin
  system carries grad tensors; it differs from nearest-bin binning by the
  sub-bin quantisation only (≈(Δx/2)/σ relative, 1 % for a 0.3 mT Gaussian at
  Δx = 6 µT; cosine 0.99998). `ModAmp` with grad tensors raises.
- `fastmotion()` keeps its NumPy signature and wraps `fastmotion_t`.

### Changed — differentiable salt; projection and eigh gradient fixes (2026-09-06)
- `salt` (fixed-field ENDOR) is differentiable: the transition arrays stay on
  the graph and the powder average uses the torch interpolation / projection /
  convolution shared with `pepper_autograd` (forward unchanged to 1e-12 on the
  8-case MATLAB suite; Jacobians vs finite differences 1e-6–1e-8 for g, A, Q,
  lw in both the perturbative and the matrix path). The field-swept ENDOR mode
  still bins sticks and raises `NotImplementedError` for grad tensors.
- `pepper_autograd` projection: a tent ramp of zero width (tied vertices —
  generic for axial systems, where the resonance field does not depend on φ)
  contributes no value but a first-order term; it was masked away, giving
  9 % errors in d spec/dA_x at A_x = A_y. Now kept on the graph (1e-7).
- `torchspin._linalg.eigh` uses a degeneracy-safe backward on the graph:
  within-subspace terms 1/(λ_j − λ_i) of coincident eigenvalues are dropped,
  the correct gradient for gauge-invariant outputs such as spectra.
- `torchspin.pepper_autograd.projecttriangles_t/projectzones_t/convspec_t`
  are reused by `salt`; `salt(..., lw_mhz=tensor)` is allowed.

### Changed — differentiable pepper shares pepper's forward path (2026-09-05)
- New `torchspin.pepper_autograd.pepper_autograd(sys, exp, opt)`: pepper's
  rigid-limit powder spectrum (matrix method, no strain) on the autograd graph.
  It reuses pepper's resonance-field search and transition tracking
  (`resfields_batch`, whose converged Newton step yields the implicit gradient
  of the resonance fields), and adds torch ports of the transition-slot
  bookkeeping, EasySpin grid interpolation (as linear maps, `L3` in torch),
  SOPHE triangle/zone projection (`projecttriangles_t`, `projectzones_t`) and
  EasySpin's sampled-kernel convolution/harmonic (`convspec_t`). Forward output
  equals `pepper()` to 1e-11 relative; autograd Jacobians match central finite
  differences of `pepper()` for g, A, D, Gaussian/Lorentzian lw and
  Temperature (1e-8 relative for A/D/lw; g to the finite-difference kink limit
  of the piecewise-linear projection).
- `differentiable_spectrum` now builds a `SpinSystem` from the parameter
  tensors and calls `pepper_autograd` (`method='pepper'`, default); `GridSize`
  accepts pepper's `[N, Ninterp]` form, `GridSymmetry` defaults to `'auto'`,
  a tensor `Temperature` is differentiable. The earlier stand-alone models are
  kept as deprecated `method='broadband'` / `'analytical'`: they sum discrete
  orientations without interpolation or projection and carry orientation-grid
  ripple at GridSizes where pepper is converged (Cu(II) hyperfine, GridSize 31:
  cosine to EasySpin 0.985–0.999 in absorption, 0.73–0.88 in first derivative;
  pepper 0.9998–1.0000 / 0.999). Training a CNN on their output was training on
  a model that differs from pepper/EasySpin by more than the features to learn.
- `SpinSystem` keeps tensors that `requires_grad` (no copy) and accepts lists of
  tensors (`g=[g_tensor]`, `A=[[ax, ay, az]]`), so parameters can be optimized
  through the regular simulators; `resfields_batch` Boltzmann normalization is
  out of place (autograd through `exp`); `hamsymm` detaches before comparing.
- Not yet differentiable: strains, `ModAmp`, `mwPhase`, ordering,
  photoselection, crystals, frequency sweeps (`pepper_autograd` raises
  `NotImplementedError`; `pepper()` is unchanged).

## [0.2.2] — 2026-09-03

### Fixed — salt powder average
- `salt` (fixed-field ENDOR): the powder spectrum was accumulated as a stick
  spectrum (one nearest-bin hit per orientation), which left orientation-grid
  ripple on top of every powder pattern (EasySpin parity cosine 0.9947 for the
  rhombic-g 1H reference at GridSize 31). It now follows EasySpin `salt.m`:
  transition positions/intensities are tracked per transition across the
  grid, interpolated when `GridSize=[N, Ninterp]` is given (G3 positions,
  linear intensities; off when any orientation is missing), and projected with
  `projecttriangles` / `projectzones`, reusing the pepper ports. Parity on the
  8-case MATLAB suite rises to 0.9917–0.99999 (rhombic-g 0.99996 at GridSize
  31, 0.999998 at the reference's `[20 5]`; S=3/2 0.942 → 0.992).
- `salt` accepted only an integer `GridSize` and crashed with the default
  `Options()` (`GridSize=[19, 4]`); the `[N, Ninterp]` form is now honoured.

## [0.2.1] — 2026-09-03

### Added — interactive fitting panel
- `torchspin.fitgui`: `FitSession` (esfit in a background thread with live
  progress; Stop returns the best fit so far) and `FitPanel` (ipywidgets
  sliders with bounds and fix boxes, method/budget controls, Start/Stop/Reset,
  live figure with data, best fit, residual and RMSD trace; ipympl canvas or
  PNG refresh). Optional extra `torchspin[gui]`; tutorial notebook
  `examples/notebooks/09_fit_gui.ipynb`.

## [0.2.0] — 2026-09-03

EasySpin parity rounds 1–3, performance rounds 1–2, esfit progress/early stop,
notebooks 01–08, benchmark report against MATLAB/EasySpin. Details below.

### Added — esfit progress, resource summary, early stop (2026-09-02)
- `FitOptions.progress` (`'text'`, `'bar'` with tqdm, or a callable), `progress_every`,
  `progress_interval`: rate-limited reports of iteration, evaluations, current and
  best RMSD, elapsed time, evaluations/s and ETA for every method (population
  methods report per evaluation chunk when worker processes are used); a
  resource summary (torch threads, worker processes, CPUs, CUDA/MPS, RSS) at
  the start and a totals line at the end.
- Early stop: `max_time`, `stop_when(info)`, a progress callable returning
  `False`, or a keyboard/kernel interrupt now return the best-so-far
  `FitResult` (`success=False`, `interrupted=True`, message with the reason)
  instead of losing the fit; an interrupted worker pool is terminated cleanly.
- `FitResult.n_evaluations`, `elapsed_s`, `interrupted`.

### Fixed — chili default basis (2026-09-02)
- `chili`: for one S = 1/2 electron with ≤ 2 nuclei, no potential and a
  rhombic basis (Mmax > 0, Kmax > 0) — where EasySpin auto-selects its `fast`
  builder and that builder equals general+MpSymm at half amplitude — the
  M–pS–pI symmetry basis is now the default (`MpSymm=None` = automatic) with
  the fast-method amplitude convention (verified in MATLAB on the stored cases).  The plain general
  basis was 6 % off EasySpin's default in the slow-motion limit
  (τc = 100 ns nitroxide: cosine 0.937 → 1.000).
- Examples: `fitting/fit_bruker_cw.py` (load a Bruker file with `eprload`,
  read the acquisition parameters, baseline, fit with `esfit`);
  `slowmotion/nitroxide_tcorr.py` plot fix; `liquids/fremysalt.py` uses the
  reference's A conversion; `fitting/multicomponents.py` built its target from separately weighted
  spectra (minor component weighted twice) — now uses the multi-component call,
  seeded noise and `method='global'`; `examples/tests/compare_with_matlab_refs.py` path fixed.

### Changed — performance round 2 (2026-09-02, after the reference-workstation baseline)
- `pepper` strain summation: Gaussian accumulation in cache-sized chunks run in
  a thread pool, one spline evaluation per fine grid, cached spherical grids.
- `pepper` interpolation: the D2h G3 coarse→fine interpolation (row splines +
  bicubic) is applied as one cached matrix per grid size (75× faster per slot;
  agreement 1e-12), which halves a small interpolated `pepper` call.
- `cardamom`: orientations propagated in groups (trajectories were unseeded per
  orientation, so results are statistically unchanged), torch propagator and
  step loop, torch quaternion→rotation conversion.
- `esfit`: population methods can evaluate the population in a spawned process
  pool (`FitOptions.workers='auto'|'processes'|'threads'`, one torch thread per
  worker); the objective is a picklable partial.
- Test: `test_esfit_vs_matlab[field_g3lw_simplex]` accepts the gx↔gy mirror
  solution — the case starts from gx0 == gy0 without hyperfine coupling, so the
  two minima are the same powder spectrum and which one a simplex reaches is
  decided by rounding noise.

### Changed — performance round 1 (2026-09-02)
Results are unchanged (MATLAB parity suites and the full test suite green after
every step; resonance positions within 1e-11 mT of the previous search).
- `resfields_batch`: one exact diagonalization per resonance candidate at the
  cubic-model root (EasySpin's scheme) with a free Hellmann–Feynman Newton
  correction, instead of three polish evaluations plus a final pass; the
  transition eigenvectors are gathered there and reused for intensities;
  knot and candidate diagonalizations chunked to ~400 MB; per-orientation
  reassembly and strain widths as tensor ops (one host sync per call).
- `torchspin/_linalg.py`: batched `eigh`/`eigvalsh` split over a thread pool of
  single-threaded LAPACK calls (torch loops serially over CPU batches; 21–23×
  on 32 cores); CUDA batches of matrices larger than cuSOLVER's batched limit
  (32) are routed to the CPU pool.  Used by resfields, resfreqs, salt, curry
  and the strain widths.
- `Options.BatchSize` default is now automatic (all orientations per call,
  memory-capped) instead of 10.
- `pepper`: vectorised SOPHE projection (`_projecttriangles`/`_projectzones`,
  loop ports kept as `*_loop`), windowed Gaussian bin accumulation, no
  per-resonance Python in the strain path.
- `cardamom`: batched density-matrix propagation and tensor rotation.
- `chili`: vectorised Liouvillian assembly (`liouvhamiltonian`; loop port kept
  as `_liouvhamiltonian_loop`).
- Benchmarks: `benchmarks/cluster/run_workstation.sh` (serial campaign incl. MATLAB),
  `benchmarks/analysis/workstation_report.py`;
  results in `benchmarks/results/workstation_20260902*/` and
  `benchmarks/results/BENCHMARK_REPORT.md`.

### Added — EasySpin parity round 2 (2026-09-01)
- `chili`: full port of EasySpin's general stochastic-Liouville method
  (`torchspin/chili_sle.py`): arbitrary spin systems, several nuclei, S>1/2,
  orienting potentials with powder integration, LjKKM basis switches,
  Lanczos/direct/eigen solvers, field-sweep methods, post-convolution nuclei,
  multi-component input; 60-case MATLAB suite (59 at cosine ≥0.999).
- `pepper`: first-principles absolute intensity (dBdE, nuclear-sublevel
  sharing, density normalization) — the empirical scale factors are gone;
  symmetry-frame fix for tilted axial tensors; level-pair transition
  bookkeeping and EasySpin's strain summation on the interpolated grid;
  Delaunay triangulation, rectified interpolation and projection for the
  open-φ grids (Ci, C2h, C1, …); exact templates for isotropic systems;
  `Options.Method` honoured (matrix default, perturbation opt-in);
  `Experiment.SampleFrame`; 102+12+11-case MATLAB suites.
- `sphgrid.grid_triangulation` for all grid symmetries.
- `pepper` features (all MATLAB-validated): single crystals and site
  transforms, parallel mode, isotopologues (`torchspin/isotopologues.py`,
  shared with garlic and chili), non-equilibrium populations
  (`SpinSystem.initState`, `torchspin/initstate.py`), partial ordering
  (`Experiment.Ordering`, `torchspin/ordering.py`), photoselection
  (`lightBeam`/`lightScatter`/`tdm`), field modulation, dispersion,
  `Options.separate`, and frequency sweeps through the unified
  interpolation/projection/summation path (`resfreqs_batch`).
- `garlic`: component × isotopologue loop with weights,
  `Options.separate='components'`, automatic sweep ranges
  (`Options.Stretch`), frequency sweeps with `Experiment.Field` alone.
- `SpinSystem.ee2` (biquadratic exchange), `SpinSystem.D_`, `SpinSystem.Abund`,
  `SpinSystem.Q` shorthand forms (`[e2qQ/h eta]`), `Options.IsoCutoff`.
- `spidyan`: `Dim` sweeps of the pulse length `tp`.
- `eprload` validated on all 75 EasySpin-readable example files; multi-D data,
  companion-file axes, multi-value BES3T, JEOL and specman fixes.
- Ports of EasySpin's `isotopologues_*` tests; cardamom MATLAB references.

### Added — EasySpin parity round 3 (2026-09-02)
- `pepper`: `Experiment.mwMode` excitation modes (`torchspin/excitation.py`),
  photoselection on the perturbation paths, `Options.separate='transitions'`,
  automatic sweep ranges, EasySpin transition pre-/post-selection, paired
  resonances of looping transitions, EasySpin interpolation modes for
  intensities and widths; 32-case MATLAB suite.
- `esfit`: `levmar` is a port of `esfit_levmar.m`; 6-case EasySpin oracle.
- `orca2torchspin`: 33-file `orca2easyspin` oracle; binary `.prop` frames and
  coordinates read column-major.
- `cardamom`: three EasySpin 6 references (diffusion/fast, jump/fast, ISTOs)
  at cosine ≥0.97–0.9998.

### Changed — EasySpin parity round 3
- `cardamom`: field-swept spectra no longer apply `Experiment.Harmonic`
  (EasySpin returns the imaginary FFT of the time-weighted FID); the output
  scale follows EasySpin's double orientation normalization; the jump model
  no longer requires a correlation time.
- `hamsymm`: principal-value equality at 1e-12 relative (was `torch.allclose`).
- `FitOptions.lm_delta` default 1e-3 and `lm_gradient_tol` 1e-5 (EasySpin).

### Changed — EasySpin parity round 2
- `eprload` returns abscissae in the file's native units (Bruker: Gauss) and
  one abscissa per dimension for multi-dimensional data, as EasySpin does.
- `Experiment.Harmonic` defaults to 1 for field sweeps and 0 for frequency
  sweeps; `Experiment.Range` may be omitted for automatic ranging (garlic).
- `resfields`/`resfreqs` intensity thresholds use absolute values so emissive
  lines of non-equilibrium states survive.

### Fixed — EasySpin parity round 2
- `SpinSystem` silently defaulted `lw` to 0.5 mT; the default is now no
  broadening (EasySpin). pepper/garlic fall back to Harmonic 0 when nothing
  broadens.
- `convspec` samples the line-shape kernel on the grid like EasySpin
  (sub-increment widths differed by up to 3.7× from the analytic kernel).
- pepper: resonances were matched across orientations by list index;
  strain widths were evaluated at one field per orientation; slots with
  missing resonances were filled from neighbors; isotropic systems and
  frequency sweeps had wrong normalization/units; single components ignored
  `Sys.weight`.

### Added — EasySpin port completion (2026-09-01)
- `saffron`: S>1/2 systems, MimsENDOR blind-spot fix, `ProductRule` with
  `TimeDomain`, multi-component input with `Sys.weight` and
  `opt.separate='components'`, single-crystal simulations via
  `PulseExperiment.SampleFrame`/`CrystalSymmetry`/`MolFrame` backed by a full
  `sitetransforms` port (230 space groups, vendored `data/spacegroups.txt`);
  `saffron_thyme` crystal orientations. MATLAB validation suite: 25 cases,
  zero skips.
- `salt`: EasySpin fixed-field ENDOR mode (`Experiment.Field` + RF `Range`,
  `ExciteWidth` orientation selection) and an 8-case MATLAB validation suite.
- `spidyan`: simulation-frame machinery (ZeemanFreq→g, `SimFreq`, pulse
  carriers through `pulse()`/`rfmixer`, phase cycling, `DetFreq`
  down-conversion, `Dim` pulse-parameter sweeps); 7-case MATLAB validation.
- `garlic`: EasySpin's line-position engine — exact Breit–Rabi fixed-point
  solver (default) and `perturb1`–`perturb5`, equivalent nuclei
  (`SpinSystem.n`, `equivcouple`), frequency sweeps, `CenterSweep`, field
  modulation (`ModAmp`), Boltzmann polarization, full g/A matrices; 18-case
  MATLAB suite (positions <1e-6 mT).
- `hamsymm`: eigenvalue-based point-group analysis (`hamsymm_eigs` port) for
  Stevens/`Ham*`/crystal-field/full-tensor systems; sigma and nn tensors in
  the geometric pass; 22-case MATLAB suite plus the 20 EasySpin `hamsymm_*`
  tests.
- Strain: multi-electron g strain and per-electron D strain with tilted
  `DFrame` and `DStrainCorr`; `SpinSystem.DStrain` is `(nElectrons, 2)`.
  Transition-level MATLAB validation of strain widths.
- `ham_zf`: `BFrame` rotation of high-order Stevens terms (Wigner D).
- `curry`/`blochsteady` MATLAB validation suite (5 cases).
- Hamiltonian MATLAB validation extended from 10 to 92 cases mirroring
  EasySpin's `ham_*` tests.
- `esfit`: dict-style `lb`/`ub` bounds.
- `Options.Method` accepts `exact`/`perturb`/`perturbN`; new
  `Options.AccumMethod`, `Accuracy`, `MaxIterations`;
  `Experiment.CenterSweep`, `mwCenterSweep`, `ModAmp`.

### Fixed — EasySpin port completion
- `ham()` omitted higher-order Zeeman terms (`Sys.Ham*`); now includes
  `ham_ezho` like EasySpin `ham.m`.
- `SpinSystem`: scalar `D` meant `[D D D]` (isotropic); now `[D, E=0]` as in
  EasySpin. EasySpin input shorthands that crashed (per-electron scalar
  g/D, 1-D `Q`/`sigma`/frames for one nucleus, scalar/per-pair `nn`, scalar
  `gnscale`) are accepted; `ee=[J]` per-pair isotropic input no longer
  raises in `ham_ee`; `g` must be zero with `Ham110`/`Ham112`.
- `saffron`: double orientation weighting under `ProductRule`; `t2`
  UnboundLocalError for 2D custom sequences.
- `spidyan`: pulses were propagated in the lab frame with DC envelopes
  (off-resonance) and `Sys.ZeemanFreq` was ignored.
- `salt`: `Experiment.Range` was treated as an EPR field window instead of
  the RF window at fixed field.
- `pepper`/`chili` reject sets of equivalent nuclei (`SpinSystem.n > 1`)
  explicitly, as EasySpin does.

### Fixed
- `esfit` Nelder-Mead simplex: initial simplex edge length increased from
  0.05 to 0.2 in the transformed [-1, 1] parameter space, matching MATLAB
  EasySpin's `esfit_simplex.m` (`delta = 0.1 * (ub - lb)` in original space).
  The undersized initial simplex caused premature convergence to local
  minima on rugged loss surfaces (alpha-test report: 2-parameter EPR fits
  converging to wrong values even with reasonable bounds). Added
  `TestSimplexExploration` regression tests in `tests/test_esfit.py`.
- `esfit` autoscale: scale factor is now forced positive (matches EasySpin
  `esfit.m:1049` `coeffs(1) = abs(coeffs(1))`). A negative scale lets the
  optimizer "fit" by sign-flipping the model, which created spurious local
  minima on derivative spectra.
- `salt` CUDA device wiring: frequency-swept path kept `g_mat`, `A_i`,
  frame rotations, flattened arrays, and spectrum bins on the active
  compute device. `Options(device='cuda')` now flows end-to-end through
  the freq-swept salt path; `freq_axis` and `spec` are returned on CPU
  to preserve the public API contract.
- `pepper`, `resfields_batch`, `resfields_perturb`, and all
  `ham_{ee,ez,hf,nn,nq,nz,zf}.py`: tightened device propagation so
  `Options(device='cuda')` no longer leaks intermediate CPU tensors.
- `eprload` Bruker BES3T: case-insensitive companion-file lookup
  (probes both `.DSC/.DTA` and `.dsc/.dta`) so Linux checkouts with
  lowercase fixtures load correctly.
- MATLAB validation tests (`test_chili_matlab_validation.py`,
  `test_endorfrq_matlab_validation.py`): `float(np.asarray(x).squeeze())`
  replaces `float(x)` on `.mat` fields to work with numpy ≥ 1.25 where
  0-dim-array→scalar conversion is a hard error rather than a warning.

### Added
- `esfit` new option `target='auto'` (now the default): inspects the data
  and picks `'int'` for derivative-like spectra (mean ≪ std) and `'fcn'`
  otherwise. Matches EasySpin's behavior of auto-selecting the integral
  target for pepper/garlic with `Harmonic > 0`. Smooths the loss landscape
  for derivative EPR.
- `esfit` new method `method='global'`: particle-swarm (global) followed by
  Nelder-Mead (local polish). Recommended for realistic EPR fits with
  rugged loss landscapes where plain `simplex` gets trapped. Recovers
  `gx=2.001 (true 2.000)` on the canonical Harmonic=1 powder fit where
  plain `simplex` returned `gx=2.080`. Brings the optimizer count to 9.
- `esfit` new option `n_workers: int = 1`: enables ThreadPoolExecutor
  evaluation of the population for `swarm`, `genetic`, `montecarlo`,
  `grid`, and `global` methods. Effective for forward models that release
  the GIL during torch tensor ops (pepper/garlic/chili/saffron).
- `TestAutoTargetAndGlobal` and `TestSimplexExploration` regression tests
  in `tests/test_esfit.py` (6 new tests total).
- `benchmarks/PERFORMANCE_SUMMARY.md` and `benchmarks/RUNLOG_2026-04-20.md`
  (development records, not distributed; their results are summarized in
  `benchmarks/results/BENCHMARK_REPORT.md` and embedded in
  `benchmarks/analysis/make_manuscript_figures.py`)
  — performance and accuracy summary (MATLAB parity
  table, pepper CPU vs GPU scaling, `differentiable_spectrum` 2.5× GPU
  speedup, esfit method comparison, 5-parameter stress test, MSE vs
  integral-MSE finding for gradient-based fitting).

## [0.1.0] — 2026-04-18

### Initial PyPI release

Full EasySpin parity achieved — all scientifically relevant MATLAB functions
ported to Python/PyTorch with 1820+ passing tests.

**CW EPR simulators**
- `pepper` — powder CW EPR (MATLAB cosine > 0.999)
- `garlic` — solution / fast-motion EPR (MATLAB cosine 1.0000)
- `chili` — slow-motion SLE (MATLAB cosine 0.92–0.997)
- `salt` — ENDOR powder (MATLAB cosine > 0.92)
- `curry` — magnetometry (susceptibility, magnetization)
- `blochsteady` — Bloch steady-state
- `levels` — energy level diagrams
- `fastmotion` — Kivelson/Freed linewidths

**Pulse EPR**
- `saffron` — predefined (2p/3p/4p ESEEM, HYSCORE, MimsENDOR) + custom sequences
- `saffron_thyme` — real-pulse orientation-averaged propagation
- `spidyan` — arbitrary pulse sequences with relaxation and phase cycling
- `pulse`, `exciteprofile`, `resonator`, `rfmixer`, `transmitter` — pulse primitives

**Trajectory-based**
- `cardamom` — diffusion, jump, MD-direct, ISTOs methods
- `stochtraj_diffusion`, `stochtraj_jump`
- `mdload`, `mdhmm`, `mdtraj2oripot` — MD trajectory analysis

**Differentiable**
- `differentiable_spectrum` — S=1/2 analytical + N-spin broadband via batch `eigh`
- Gradients validated vs finite differences for g, A, D tensors

**Fitting**
- `esfit` — 8 optimizers: Nelder-Mead, L-BFGS-B, Powell, grid, Monte Carlo,
  genetic, particle swarm, Levenberg-Marquardt
- `autoguess` — starting-parameter estimation

**Data I/O**
- `eprload` — 12 vendor formats (BES3T, ESP, Bruker, Varian, JEOL, Magnettech, …)
- `eprsave` — BES3T writer
- `orca2torchspin` — ORCA QC output import (.out, .prop, _property.txt)

**GPU + compilation**
- `Options.device='cuda'` support throughout simulator chain
- Opt-in `torch.compile` via `TORCHSPIN_COMPILE=1`
- Batched simulation via `batch_pepper`, `batch_simulate`

### Conventions
- Energy units: MHz throughout
- Field units: mT
- Euler angles: radians, z-y'-z'' passive rotation
- Default dtype: `torch.complex128`

<!-- 0.3.0 was the first public release and the history before it was squashed,
     so v0.1.0-v0.2.2 have no tags in this repository to link to. -->
[0.6.0]: https://github.com/follmerlab/torchspin/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/follmerlab/torchspin/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/follmerlab/torchspin/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/follmerlab/torchspin/releases/tag/v0.3.0
