# Known Limitations

torchspin was developed and audited rigorously, but a few simulators
have documented caveats users should understand before relying on them for
publication-quality work. This document is the public record of where the
package is fully MATLAB-validated, where there are quantified caveats, and
where validation is still pending.

If you find behavior that contradicts this document, please open an issue.

---

## Trust matrix at a glance

| Module | Shape parity vs MATLAB | Absolute intensity | Caveats |
|---|---|---|---|
| `pepper` (CW powder, crystals, frequency sweeps) | ≥0.999 on 102+12+11 general cases plus 22 crystal, 4 parallel-mode, 10 feature (isotopologues, ModAmp, dispersion, separate) and 26 feature-2 cases (non-equilibrium populations, ordering, photoselection, frequency sweeps), plus 34 Cu(II)+ligand cases covering `matrix`/`perturb`/`hybrid` on the same systems | matches EasySpin from first principles (isotropic closed form to 1e-4, amplitude ratio 1.00±0.02) | See § Pepper |
| `garlic` (solution) | line positions exact vs EasySpin Breit–Rabi (<1e-6 mT, 18-case suite); spectra cosine >0.9999; components/isotopologues/auto-range 11-case suite identical | matches EasySpin's TransitionRate·dBdE convention to 1e-6 | See § Garlic |
| `chili` (slow-motion) | 59/60 MATLAB cases at cosine ≥0.999 (general Liouvillian port; intermediates exact) | matches EasySpin's scaling chain | See § Chili |
| `salt` (ENDOR powder) | 8-case EasySpin-mirroring suite passes (fixed-field mode) | matches in regime | See § ENDOR |
| `saffron` (pulse EPR) | 25 strict cases (raw + mean-centered rel. error 1e-16..1e-14) over 2p/3p/4p ESEEM, HYSCORE, MimsENDOR, direct + binned, S=1/2 and S=3/2, quadrupole, orientation selection, P1 crystal, custom sequences, ProductRule, multi-component; plus 26 EasySpin-mirroring cases, also now strict (rel. error 1e-16..1e-7) | matches to roundoff | See § Pulse |
| `spidyan` (arbitrary pulse) | 10 MATLAB cases (incl. `tp` sweeps and relaxation with complex detection), strict rel. error 1e-14..1e-12 | matches to roundoff | See § Pulse |
| `saffron_thyme` (real-pulse) | 28 strict cases (raw + mean-centered rel. error 3e-13..8e-12): 18 rectangular-pulse cases over powder and P1 crystal, transient and single-point detection, `DetIntegrate`, `DetPhase`, explicit `SimFreq`, relaxation, 1 and 2 indirect dimensions, predefined 2p/3pESEEM and HYSCORE; plus EasySpin's own 10 `saffron_thyme_*` cases with `quartersin/linear` chirps, `trise`, swept `Frequency`, pulse-parameter sweeps and `gFrame`/`AFrame` tilts | matches to roundoff | See § Pulse |
| `curry`, `blochsteady` | 5 MATLAB cases within 0.5–1 % | n/a | None |
| `hamsymm` | 22 MATLAB cases exact (groups and frames) | n/a | See § Pepper (tilted axial frames) |
| `cardamom` (trajectory) | 3 EasySpin 6 references (diffusion/fast, jump/fast, diffusion/ISTOs): cosine 0.9998 / ≥0.97 / 0.9992, amplitude 1.00 | matches EasySpin's scale | See § Cardamom |
| `esfit` (fitting) | 6 EasySpin esfit references (levmar/simplex from offset starts): torchspin reaches EasySpin's residual or better; `levmar` is a port of `esfit_levmar.m` | n/a | EasySpin's own levmar stalls at symmetric starts in 2 of 6 cases |
| `orca2torchspin` | 33 ORCA files identical to `orca2easyspin` (tensors compared in the molecular frame) | n/a | atoms without data are not listed (EasySpin lists zero rows) |
| `autograd` (differentiable) | `differentiable_spectrum` = `pepper_autograd`: forward equals `pepper` to 1e-11; Jacobians vs finite differences of `pepper` for g, A, D, lw, Temperature (see § Autograd) | same as `pepper` | strains, ModAmp, mwPhase, ordering, crystals, frequency sweeps not differentiable |
| Hamiltonians, utilities, constants | exact — 92 cases mirroring EasySpin's `ham_*` test suite (rtol 1e-10) plus `ee2`/`D_` cases | exact | None |
| `eprload` | identical to EasySpin on all 75 readable files in `tests/eprfiles` (Bruker, JEOL, specman, Magnettech, Adani, CIQTEK, ETH formats) | n/a | See § eprload (native units) |
| `isotopologues` | EasySpin's 15 `isotopologues_*` tests ported (incl. the `A_` spherical form); pepper/garlic/chili mixtures validated in MATLAB | exact | None |

---

## Pepper — choosing a method, and what each costs (2026-10-07)

Added after a Cu(II) phthalocyanine fit (CuPc diluted in ZnPc: S=1/2, g =
[2.049 2.181], A(Cu) = [15.3 646.6] MHz, four effectively isotropic 14N at
45 MHz, lw 0.54 mT, X-band) ran into every limit of the resonance solvers at
once.  All timings below are the same machine, `GridSize=[91,4]`, 2667 points,
with EasySpin on MATLAB R2024b for comparison; the generator is
`tests/data/generate_pepper_cupc_refs.m` and the comparison
`torchspin/tests/test_pepper_cupc_matlab_validation.py` (34 cases).

**Accuracy.** Against the exact `matrix` result for the same system:

| Nuclei | `hybrid` | `perturb` (2nd order) |
|---|---|---|
| Cu + 1×14N | 0.9805 | 0.896 |
| Cu + 2×14N | 0.9740 | 0.918 |

These are cosine similarities, and **EasySpin's own hybrid gives 0.982 and 0.974
on the same comparison** — the deviation is the first-order electron–nuclear
decoupling that defines the method, not a port defect.  It shrinks as the
perturbational coupling shrinks (0.9805 → 0.9999 as A(N) goes 45 → 0.1 MHz) and
as the exact core grows (0.974 → 0.984 when the first nitrogen joins the core).
Second-order perturbation theory does **not** converge the same way: with
A∥(Cu) ≈ 650 MHz it is wrong about the copper itself, so it plateaus near 0.95
however small the nitrogen coupling is.

Pick `matrix` when it is affordable, `hybrid` when a large central hyperfine
coupling is surrounded by small ligand couplings, and `perturb` only when all
couplings are small compared with the microwave quantum.

**Cost.** Exact diagonalization grows with the Hilbert space, and the
natural-abundance isotopologue expansion multiplies whatever that costs
(Cu + 4N natural abundance is 10 separate simulations):

| System | method | torchspin | EasySpin |
|---|---|---|---|
| 63Cu + 2×14N (72 states) | `matrix` | 5.9 s | 16.3 s |
| 63Cu + 2×14N | `hybrid` | 0.06 s | 0.14 s |
| 63Cu + 2×14N, tilted `AFrame` | `matrix` | 590 s | 630 s |
| 63Cu + 2×14N, tilted `AFrame` | `hybrid` | 5.1 s | 143 s |
| 63Cu + 4×14N (648 states) | `matrix` | not measured to completion | not measured to completion |
| 63Cu + 4×14N | `hybrid` | 0.27 s | 0.28 s |
| 63Cu + 4×14N | `perturb` | 0.06 s | 0.04 s |
| Cu + 4×N, natural abundance | `hybrid` | 2.0 s | 5.6 s |
| Cu + 4×N, natural abundance | `perturb` | 0.41 s | 0.71 s |
| Cu + N with `n=[1,4]` | `hybrid` | 0.30 s | not supported |

Two things follow.  `matrix` on Cu + 4×14N is not a fit-loop operation in either
code — that is what `hybrid` is for; neither code was run to completion on it
here, and the tilted-frame row is the largest exact comparison that was.  Note
that torchspin's exact solver is **not** the slower of the two: it is 2.8×
faster at two nitrogens and comparable with a full A tensor.  And if the isotope matters less than the
run time, name the isotopes (`'63Cu'`, `'14N'`) or raise `Opt.IsoCutoff`:
natural-abundance copper alone doubles the work for a 0.3 % spectral change.

**Sets of equivalent nuclei.** `pepper` accepts `SpinSystem.n > 1` on the
`hybrid` and `perturb` paths, where nuclei are treated one at a time and
combined combinatorially, so a multiplicity is a pure saving — four equivalent
14N written as `n=[1,4]` cost 0.30 s instead of 2.0 s and agree with four
explicit nitrogens to cosine 0.99999999.  The `matrix` path still rejects
`n > 1`, because there the nuclei multiply the Hilbert space and cannot be
collapsed; EasySpin's `pepper` rejects `n > 1` for every method.  Equivalence
means *identical couplings and identical orientations*: four nitrogens with
different `AFrame` tilts are not a set of equivalent nuclei, whatever their
principal values.

**The derivative peak at a turning point needs a finer grid than the spectrum
as a whole.** Worth knowing because it looks like a parity defect and is not.
Comparing Cu + 1x14N by `matrix` against EasySpin at `GridSize=[91,4]` gives
cosine 0.99948 but an amplitude ratio of 0.990, and at two nitrogens 0.978 —
outside the 2 % amplitude convention.  Refining the grid removes it entirely:

| `GridSize` | torchspin peak | EasySpin peak | ratio | cosine |
|---|---|---|---|---|
| `[91,4]` | 53.18 | 53.70 | 0.9903 | 0.99948 |
| `[181,4]` | 51.51 | 51.62 | 0.9978 | 0.99998 |
| `[361,4]` | 51.49 | 51.44 | 1.0010 | 1.00000 |

**Neither code is converged at `[91,4]`** — EasySpin's own peak moves 53.70 →
51.62 → 51.44 over the same refinement, a 4 % change in its own answer — and the
two are unconverged by slightly different amounts, which is the whole of the
apparent disagreement.  At `[361,4]` they agree to 0.1 % in amplitude and to
cosine 1.00000.  The derivative extremum at a turning point is the most
grid-sensitive number in a powder spectrum, far more so than the integrated
intensity, so it is the first thing to check when an amplitude looks wrong.

**Absolute intensity does agree, including with nuclei.** Integrated absorption
(`Harmonic=0`, `[91,4]`) against EasySpin: 578.38/578.38 with no nuclei,
559.77/559.90 with one nitrogen, 535.49/534.99 with two — 0.02-0.09 %.
Absorption peak heights agree to 0.2-0.8 %.

That the integral *falls* as nuclei are added (578 → 560 → 535) is real but is
not a torchspin effect: **EasySpin does the same, to 0.1 %.** It is the matrix
method's transition pre-selection dropping weak transitions — both codes select
exactly the same number of level pairs (60 with one nitrogen, 215 with two) —
and `Options.Threshold=0` does not recover it, because without pre-selection the
pure nuclear transitions come back and put intensity in the wrong places.
`hybrid` and `perturb` hold at 578 because they place all the nuclear intensity
on the core lines instead.

**Default `GridSize=[19,4]` is unconverged for narrow lines on a large
hyperfine.** For this system the default grid leaves the powder average
dominated by ripple, and the two codes place that ripple differently: the same
Cu + 4×N spectrum agrees with EasySpin to cosine 0.9998 at `[91,4]` but only
0.96–0.97 at `[19,4]`.  Neither result is right — both are unconverged.  There is no
automatic warning, because there is no reliable shortcut: a heuristic on the
field step between knots flags dozens of simulations that agree with EasySpin to
cosine 0.9999, since the SOPHE projection integrates analytically over each grid
segment.  Use `torchspin.grid_convergence(sys, exp, opt)`, which simulates again
on a finer grid and reports what changed — the only trustworthy test, and the
one used to produce the table above.  The two `cupc_grid19_4N_*` cases are kept in the validation suite as
the record of this.

## Higher-order Zeeman terms (`Sys.Ham*`): deliberate deviation from EasySpin (0.6.7)

`ham(Sys)` returns `(H0, mux, muy, muz)` with H(B) = H0 − μ·B. The simulators that
work from this decomposition (pepper's resonance search, `levels`, `endorfrq`, `salt`,
`curry`, `nucfrq2d`) rely on it agreeing with `ham(Sys, B0)`.

- **Linear terms (`Ham1xx`).** EasySpin's `ham.m` adds the linear-in-B part dH/dB to the
  moment operators instead of subtracting it. Its own H0 − μ·B therefore has the opposite
  sign of these terms to its `ham(Sys, B0)`; the EasySpin reference data show this to
  machine precision. torchspin up to 0.6.6 copied this. Since 0.6.7 the term enters as
  −dH/dB, so H0 − μ·B equals `ham(Sys, B0)`. torchspin's `mux/muy/muz` differ from
  EasySpin's for these systems by design; H0 and `ham(Sys, B0)` still match EasySpin.
- **Effect on results.** Without nuclei, flipping the sign of the entire linear term
  leaves the energies and transition probabilities unchanged (time reversal), so
  field-swept EPR spectra were not affected. With nuclei, the term had the wrong sign
  relative to the nuclear Zeeman interaction, which changes ENDOR and nuclear frequencies
  and the intensities of forbidden transitions.
- **Quadratic and cubic terms (`Ham2xx`, `Ham3xx`)** cannot be written as H0 − μ·B. As in
  EasySpin, they enter only `ham(Sys, B0)`. Simulators that use the decomposition ignore
  them unless `Options.QuadraticZeeman=True` (0.7.1, next section).

## Simulator — field-quadratic Zeeman terms, `Options.QuadraticZeeman` (0.7.1)

By default (`QuadraticZeeman=False`) the simulators use H = H0 − B n·μ and drop
the `Sys.Ham` terms of order lB = 2 and 3, as described above. With
`Options.QuadraticZeeman=True` (or `levels(..., quadratic_zeeman=True)`) these
terms are included as H(B) = H0 − B n·μ + B² n·G2·n + B³ G3[n,n,n], which equals
`ham(Sys, B·n)`.

- **Supported:** `pepper` and `pepper_autograd` with `Method='matrix'` (powder
  grids including O3, crystals, field and frequency sweeps, strains), `levels`,
  `resfreqs_matrix`, and `resfields_batch`/`resfreqs_batch` (keyword
  `zeeman_ho=`).
- **Raise `NotImplementedError`** when the option is on and a nonzero lB ≥ 2 term
  is present: `Method='perturb*'` (also where pepper would otherwise fall back to
  the matrix solver), `Method='hybrid'`, `resfields` (single orientation),
  `resfields_perturb(_batch)`, `garlic`, `salt`; `pepper_autograd` with
  `'perturb*'`. Terms of order lB ≥ 4 raise in every path.
- **Not covered by the option:** `resfields_eig` (configured by `EigOptions`),
  `endorfrq`, `curry`, `nucfrq2d` and `levelsplot` still drop lB ≥ 2 terms.
- **Transition moments** are those of the linear moment operators μ, as in
  EasySpin. The B1-linear part of the quadratic term, 2 (B0·G2·B1), is not
  included in the transition operator; its relative size is about 2|n·G2·n|B/(gμB).
- **dB/dE:** torchspin uses the full slope ⟨v|dH/dB|v⟩ − ⟨u|dH/dB|u⟩. EasySpin's
  `resfields` builds its level model from the full `ham(Sys, B)` but takes dB/dE
  from the linear moment only. Line intensities therefore differ by the ratio of the
  two slopes (0.03–0.3 % for the reference cases).
- **EasySpin frequency sweeps** with `Sys.Ham2xx` fail: `resfreqs_matrix.m` looks
  for fields named `ZB*` to detect higher-order terms and then calls `ham` with
  four outputs, which errors. torchspin's frequency-sweep path is therefore
  checked against the full Hamiltonian instead of EasySpin.
- `Sys.Ham` coefficients are constants (`ham_ezho` detaches them): gradients
  flow to g, D, A, … but not to the Ham terms themselves.
- In the McGavin–Tennant–Weil formalism the field enters as a rank-lB spherical
  tensor, so a pure B²·identity term cannot be written (`Ham200` is ignored by
  `ham_ezho`, as in EasySpin). This does not change transition energies.

## Pepper — absolute intensity (fixed 2026-09-01)

Until 2026-09-01 `pepper()` multiplied the spectrum by one of four empirical
factors (8.48, 6.29, 4.27, 3.806) chosen by accumulation path. These are gone.
The intensity chain now follows EasySpin exactly: resonance-field intensities
carry the Aasa–Vänngård factor dBdE = 1/|⟨v|μ_z|v⟩−⟨u|μ_z|u⟩| and the
1/∏(2I+1) share per hyperfine line, orientation weights sum to 4π, the χ
integral contributes 2π, and stick/line accumulations are converted to a
spectral density by 1/ΔB. Verified against EasySpin's own checks: the
isotropic-powder closed form ∫spec dB = 8π²·TransitionRate·dBdE (matrix and
perturbation methods, 1e-4), nuclei-independent double integrals (1e-6),
per-line intensities from `resfields` (7 digits), and amplitude ratios of
1.00 ± 0.02 against every MATLAB pepper reference. The only path still off is
the `Ci`/`C1` scattered-interpolation path (≈5 % low), tracked in
the round-2 entry in `CHANGELOG.md`.

## Pepper — tilted axial symmetry frames (fixed 2026-09-01)

`hamsymm` reproduces EasySpin's point-group and symmetry-frame detection
exactly. Until 2026-09-01 pepper interpolated the rotated grid in the
molecular frame, which broke every **Dinfh** system whose unique axis is not
the molecular z (tilted `gFrame`/`DFrame`, x- or y-axial tensors: cosine 0.34
or an all-zero spectrum). Interpolation and zone weights now use the
symmetry-frame grid; these cases match MATLAB to cosine 0.99996–1.0
(`test_pepper_matlab_validation_ext.py`).

## Pepper — strain broadening (validated 2026-09-01)

Strain widths are validated per transition and orientation against EasySpin
`resfields` (≤1.3e-3 relative; the residual is EasySpin's linear eigenvector
interpolation between field knots). Powder spectra with H/g/A/D strain,
including several electron spins, tilted D frames and correlated D/E strain,
match EasySpin to cosine ≥0.999 and amplitude within 3 % on 24 MATLAB cases
(`test_strain_*_matlab_validation.py`, `test_pepper_dstrain_matlab_validation.py`).
pepper follows EasySpin's summation scheme for anisotropic widths (per-facet
Gaussians on the interpolated grid with Lambda smoothing) and its level-pair
transition bookkeeping.

## Chili — slow-motion SLE (rewritten 2026-09-01)

`chili()` is now a function-by-function port of EasySpin's *general*
Liouvillian method (`torchspin/chili_sle.py`): LjKKM basis with the same
pruning switches (`jKmin`, `evenK`, `highField`, `pImax`, `MpSymm`), ISTO
interaction coefficients, rotational-basis operators, the Liouville
Hamiltonian, the diffusion superoperator with orienting potentials of any
(L, M, K), the equilibrium starting vector, the Lanczos/Lentz continued
fraction (and direct / eigenvalue solvers), EasySpin's field-sweep methods
and output scaling, post-convolution nuclei and multi-component input.
Every intermediate was checked against EasySpin's `Opt.Diagnostics` dump
(basis identical, H and Γ to machine precision). On 60 MATLAB cases
mirroring EasySpin's chili tests, 59 match to cosine ≥0.999 and amplitude
within 2 %; the one at 0.998 is a case whose reference used EasySpin's
compiled `fast` builder, which differs from EasySpin's own general method by
0.2 % in the slow-motion limit.

Default basis symmetry (fixed 2026-09-02): EasySpin auto-selects its compiled
`fast` builder for one S = 1/2 electron with ≤ 2 nuclei and no potential.  In
MATLAB that builder equals the general method *with* the M–pS–pI (Meirovitch)
symmetry basis at half the amplitude whenever the basis is rhombic (Mmax > 0
and Kmax > 0), and equals the plain general method when the basis is axial
(Mmax = 0 or Kmax = 0); the plain general basis differs from it in the
slow-motion limit (cosine 0.937 at τc = 100 ns for the nitroxide example,
1.000 with the symmetry basis).  torchspin applies that empirical rule when
`ChiliOptions.MpSymm` is left at `None`, so its default output reproduces
EasySpin's default output on every stored case; an explicit `MpSymm=True/False`
behaves like EasySpin's general method with that option.  The `fast` builder
itself (`chili_lm.c`) is not ported, so systems outside the stored cases
(e.g. unusual LLMK choices) should be checked against EasySpin.

Not ported: the `fast` C builder (`chili_lm.c`), `Sys.Exchange`,
`Exp.Ordering`, `Opt.useLMKbasis`, and the automatic sweep range. The basis
size (`Opt.LLMK`) must be chosen by the user, exactly as in EasySpin (there is
no adaptive basis selection in EasySpin either).

## ENDOR (`salt`)

Since 2026-09-01 `salt()` implements EasySpin's fixed-field, rf-swept powder
ENDOR (`Experiment.Field` + RF `Range`, optional `ExciteWidth` orientation
selection) and is validated on an 8-case suite mirroring EasySpin's salt
tests (`test_salt_matlab_validation.py`). Since 2026-09-03 the powder
average uses EasySpin's scheme (grid interpolation with `GridSize=[N, Ninterp]`
followed by triangle/zone projection) instead of stick binning, so the
reference `Opt.GridSize = [20 5]` case reproduces MATLAB to cosine 0.999998.
The three orientation-selected (`ExciteWidth`) cases remain at cosine
0.993–0.998; the residual there is in the orientation weights, not the grid.
Since 2026-09-06 the fixed-field path is differentiable (torch interpolation /
projection / convolution shared with `pepper_autograd`; gradients vs finite
differences 1e-6–1e-8 for g, A, Q, lw, `test_salt_autograd.py`); the field-swept
(multi-field) mode is not and raises for grad tensors.

## ENDOR — hyperfine enhancement (`Options.Enhancement`, 0.7.3)

`Options.Enhancement=True` (default `False`, as in EasySpin) makes the rf operator of
`endorfrq` and of `salt`'s matrix paths the total moment (electron + nuclear,
`ham()`'s `mux/muy/muz`) perpendicular to the field, chi-averaged over lab x and y.
Agreement with EasySpin `endorfrq`/`salt` (`Opt.Enhancement='on'`) is 1e-13 of the
strongest line per level pair (endorfrq) and cosine > 0.99995 (salt).

- **Selection.** As in EasySpin, `threshold` / `Opt.Threshold` selects on the nuclear-only
  rates and electronic level pairs are excluded. "Electronic" is decided by EasySpin's
  rule: energy-sorted levels grouped into blocks of prod(2I+1) and only pairs in the same
  block kept. This is right when the electron Zeeman (or zero-field) splitting exceeds
  all nuclear splittings. Near level crossings or at very low field the block assignment,
  and with it the line list, can be wrong.
- **Moment operator.** torchspin uses `ham()`'s moments. These also contain orbital
  Zeeman (`Sys.L`) and linear higher-order Zeeman (`Sys.Ham1xx`) terms. EasySpin uses
  `ham_ez + ham_nz` only. The two agree for ordinary spin systems. There is no
  `Opt.Nuclei` subset: all nuclei contribute.
- **Perturbation theory.** `endorfrq_perturb(..., opt=Options(Enhancement=True))` raises.
  EasySpin skips enhancement there with a log message. `salt` normally picks first-order
  perturbation theory by itself for one electron without Q. With `Enhancement=True` it
  uses the matrix path instead, which is slower (one eigh per orientation).
- **Intensity scale.** For those one-electron, no-Q systems, the default first-order path
  uses unit line intensities, while the matrix path uses |⟨i|μ|j⟩|² in (MHz/mT)². So
  switching `Enhancement` on changes the overall scale by orders of magnitude in addition
  to the enhancement. Normalize before you compare the two. Systems with Q, or more than
  one electron, are on the matrix path either way, and their on/off ratio matches
  EasySpin to 1e-3.

## Garlic — what is and is not ported

Line positions use EasySpin's algorithm exactly (per-nuclear-group
Breit–Rabi fixed-point solver by default, or `Options.Method='perturbN'`),
including sets of equivalent nuclei (`SpinSystem.n`), frequency sweeps
(`Field` + `mwRange`), `CenterSweep`, field modulation (`ModAmp`), Boltzmann
polarization and full g/A matrices. Not ported: automatic sweep ranging
(`Experiment.Range` or `CenterSweep` is required), natural-abundance isotope
mixtures and multi-component input (`compisoloop`), and `Opt.separate`.
Equivalent nuclei are not allowed in the fast-motion regime (same as
EasySpin). When no broadening is given, a requested derivative harmonic
falls back to the absorption stick spectrum (EasySpin's auto-harmonic).

## Pulse EPR — saffron, spidyan and saffron_thyme are MATLAB-validated

`saffron()` is cross-validated on 25 MATLAB reference cases: predefined
2p/3p/4p ESEEM, HYSCORE and MimsENDOR, S>1/2 systems, custom sequences,
`ProductRule` (with and without `TimeDomain`), single crystals
(`CrystalSymmetry`/`MolFrame`, all 230 space groups via `sitetransforms`) and
two-component input. Cosine ≥0.98 throughout (crystal HYSCORE 0.989, two
components >0.98, everything else ≥0.999). HYSCORE remains much slower than
EasySpin.

**A cosine similarity is a weak oracle for pulse EPR, and these 25 cases
should not be read as tight agreement.** A predefined ESEEM/HYSCORE trace is
dominated by its DC offset, so the modulation — the entire physical content of
the experiment — can be ~100% wrong at cosine >0.999. Six defects were found
in exactly that blind spot after 0.4.0 (tau phase conjugation, S=1/2 manifold
assignment, 2D bin dimension order, `OriThreshold` reference level, and the
`Expand`/`EndorMethod` conditional defaults); see the CHANGELOG. Agreement on
the routes covered by
`torchspin/tests/test_saffron_phase_matlab_validation.py` is now 1e-16 to
1e-14 on both the raw and the mean-centered signal. **New pulse-EPR
comparisons should assert the mean-centered relative error, not a cosine.**
Custom user sequences, `ProductRule` x `TimeDomain`, multi-component input and
S=1 systems have since been re-checked at roundoff as well.

Two things remain open in the pulse-EPR routes:

- ~~**`spidyan` multi-spin phase drift.**~~ **Resolved, and it was never a
  library defect.** The residual reported in 0.5.0 was a test transcription
  error: `generate_spidyan_refs.m` sets `Sys.g = gfree` and the Python tests had
  rounded it to `2.00232`. That 3.5e-7 error in g shifts the precession
  frequency by 11.6 kHz, which accumulates as phase over the 0.5 µs delay.
  torchspin reproduces the analytic Larmor frequency exactly. Every `spidyan`
  case now agrees to 1e-14..1e-12.
- ~~**Real pulses do not reproduce EasySpin, and have a different output
  contract.**~~ **Resolved.** The route is now EasySpin-equivalent and
  validated on 18 MATLAB cases at 3e-13..3e-12 on both the raw and the
  mean-centered signal. The 0.5.0 diagnosis was right that no reduction of
  the old per-increment transients could recover EasySpin's trace, and the
  reason was that both the detection stage and the propagation were wrong:
  `Exp.mwFreq` never reached the sequencer (so the pulses were DC fields in
  the lab frame and nothing was excited), the Hamiltonian was built in the
  molecular frame while the state and operators stayed lab-frame, the
  orientation sum was divided by the total grid weight where EasySpin leaves
  the `4*pi` in, and the detection phase, `Exp.DetIntegrate`,
  down-conversion and `Exp.DetWindow` axis were all absent. See the CHANGELOG
  for the full list.
- ~~**Predefined sequence + real pulses is an unvalidated extension.**~~
  **Resolved, and it was the fast path that was wrong.** EasySpin rejects a
  predefined sequence with nonzero `Exp.tp` in *fast* mode only; in thyme mode
  `s_predefinedexperiments.m` builds the sequence from rectangular pulses, and
  2pESEEM, 3pESEEM and HYSCORE through the real-pulse engine are now MATLAB
  cases. torchspin previously *ignored* `Exp.tp` for a predefined sequence and
  returned an ideal-pulse result; it now raises EasySpin's error in fast mode
  and runs the real-pulse engine when `Opt.SimulationMode='thyme'`.
  4pESEEM and MimsENDOR have no thyme template in EasySpin either, and raise.
- ~~**Shaped pulses are still not implemented.**~~ **Resolved, and nothing
  had to be implemented.** `torchspin/pulse.py` already built shaped pulses
  for `spidyan`, and a pulse dict in a list-valued `Exp.Sequence` passes
  through to it unchanged; what had kept EasySpin's own `saffron_thyme_*`
  cases from running was the propagation and detection. All ten of them now
  agree at 3e-13..8e-12
  (`tests/data/generate_saffron_thyme_shaped_refs.m`), covering
  `quartersin/linear` chirps with `trise` and a swept `Frequency`, relaxation
  during a chirp, an explicit `Opt.SimFreq`, a pulse-parameter sweep along an
  indirect dimension, `Sys.gFrame`/`Sys.AFrame` tilts, and a predefined
  sequence detected over a transient window. Shaped pulses are expressed
  through the list-valued `Exp.Sequence` (EasySpin's own syntax); saffron's
  `Exp.Flip`/`Exp.tp` array shorthand builds rectangular pulses only, as it
  does in EasySpin.
- **The ideal-pulse and real-pulse routes do not converge as `Exp.tp -> 0`,
  and are not meant to be compared.** This was expected to work as a
  reference-free check on the normalization; it does not, and that is
  EasySpin's behavior, not torchspin's — at matched settings torchspin
  reproduces EasySpin's own ideal-vs-real mismatch to 1e-7 in scale. Three
  reasons: with single-point detection EasySpin does not down-convert, so the
  detected point carries the simulation-frame carrier phase (~50 cycles per
  20 ns increment); a single detected point contains coherence pathways the
  ideal route excludes analytically, which the two-step cycles in EasySpin's
  own templates do not remove for this observable; and as `tp -> 0` at fixed
  flip angle the required B1 eventually exceeds the carrier frequency, so the
  rotating-wave picture breaks down. Moving the simulation frame onto the
  carrier to remove the first effect is not neutral — it removes the carrier
  from the pulse too, doubling the effective field. Use the two routes for
  what each is for, and compare either against EasySpin rather than against
  each other.
- **Two EasySpin behaviors the real-pulse route cannot reach.** `saffron.m`
  line 1937 reads `Exp.DetWindow(1)` unconditionally once detection is not
  single-point, so a *transient* `Exp.DetSequence` errors in EasySpin itself
  (`Unrecognized field name "DetWindow"`); torchspin accepts it, and that
  combination is therefore not cross-validated. `Exp.DetOperator` is ignored
  by EasySpin's saffron (only `spidyan` transfers it); torchspin warns rather
  than ignoring it silently, and always detects `S+`.

The pulse-EPR suites no longer score agreement with a cosine. `saffron`,
`spidyan` and the `spidyan` tp sweep all assert the relative L2 error against
their MATLAB references, and agreement is 1e-16..1e-7 throughout.

Getting there meant fixing three kinds of harness fault that the cosine had
hidden, **none of them a library fault**: twelve tests ran `TimeDomain=True`
against references generated without `Opt.TimeDomain` (direct evolution vs
EasySpin's own binned approximation, a real 7–60% difference in the
modulation); six compared unnormalized output against references their
generator had normalized with `y = y/max(abs(y))`; and three used a `GridSize`
that did not match their generator (181 vs 91, and EasySpin's default 31 vs
20). `GridSize` is now explicit in every case so the two codes' defaults cannot
drift apart unnoticed.

One committed fixture is known to be stale rather than wrong-in-code:
regenerating `saffron_orientation_selective.mat` from the local EasySpin gives a
reference torchspin matches to 4.4e-15, while the stored one differs by 1.4e-6
in overall scale. That test checks its amplitude at 1e-5 and its
peak-normalized shape strictly, and says so, rather than loosening everything.

The former `saffron_2psimple` xfail (14N + quadrupole) is resolved: the test
had used a quadrupole coupling ten times smaller than EasySpin's own test, and
`SpinSystem.Q` now converts EasySpin's `[e2qQ/h eta]` shorthand (cosine 0.99995).

`spidyan()` is cross-validated on 7 MATLAB cases (cosine > 0.99) after the
simulation-frame port (ZeemanFreq→g translation, `SimFreq`, pulse carriers
via `pulse()`/`rfmixer`, phase cycling, `DetFreq` down-conversion, `Dim`
sweeps of pulse parameters including the pulse length `tp`; 8 MATLAB cases).

`saffron_thyme()` (real-pulse engine) is cross-validated on 28 MATLAB cases —
18 rectangular (`test_saffron_thyme_matlab_validation.py`) and EasySpin's own
10 `saffron_thyme_*` cases with shaped pulses
(`test_saffron_thyme_shaped_matlab_validation.py`) — in addition to its
internal-consistency tests. Every case states `Opt.SimulationMode`,
`Opt.GridSize`, `Opt.IntTimeStep`, `Exp.DetWindow`, `Exp.DetPhase` and (where
used) `Opt.SimFreq` and `Exp.DetIntegrate` explicitly in both codes, and
`test_thyme_references_present` / `test_shaped_references_present` fail rather
than skip if a reference file is missing, so a skipped comparison cannot look
like a passing one.

Three relations hold without any reference and guard the normalization: a
single-site P1 crystal equals `4*pi` times the independently validated
`spidyan` result (1e-12), an isotropic powder equals a single crystal, and a
nominal `Pulse.Flip` of `pi/2` delivers full transverse magnetization while
`pi` delivers none.

One caveat on reading those numbers: the references pin
`Opt.IntTimeStep = 1e-4 µs`, which is coarse enough that EasySpin warns about
it (its own default for this sequence is 4.3e-6 µs). What is validated is that
the two codes do the *same* calculation, not that 1e-4 µs is converged;
`eseem2p_fine` repeats one case at 2e-5 µs and agrees equally well.

**Relaxation with complex (S+) detection was wrong until this round, in
`spidyan` as well.** The Liouville-space commutator superoperator used
EasySpin's column-major `kron()` ordering against numpy's row-major
flattening, so the relaxation path propagated `conj(rho)`. An `Sz` trace is
real and so unaffected, which is why the existing `spidyan` relaxation
reference — `Exp.DetOperator = {'z1'}`, compared on its real part only —
could not see it. There is now a reference that detects `S+` under relaxation
and is checked on its imaginary part too.

**A sample tilt and the equivalent tensor tilt agree to ~1e-12, not to
machine precision.** `Sys.gFrame`/`Sys.AFrame` and an equal `Exp.SampleFrame`
describe the same physics, and EasySpin's own test asserts the two signals
agree to `1e-12` absolute. torchspin reaches them by different arithmetic:
EasySpin rotates the interaction tensors, while torchspin conjugates the
molecular-frame Hamiltonian into the lab-frame basis, which is the identity
for an untilted sample and two matrix exponentials for a tilted one. The two
Hamiltonians agree to 2e-16 relative, but integrating that over a detection
window turns it into a phase error growing linearly in time — 2e-11 absolute
on a signal of magnitude 12 after 0.3 µs. Both sides are separately validated
against MATLAB at ~1e-12 relative L2, so this is a round-off floor and not a
frame-handling error; it is worth knowing only if you compare the two setups
to each other over a long transient.

---

## Cardamom — validated against EasySpin (2026-09-02)

`tests/generate_cardamom_matlab_refs.m` (EasySpin 6 parameter names) was run with
`rng(42)`; `test_cardamom_matlab_validation.py` compares three cases (nitroxide
diffusion/fast, two-site jump/fast, diffusion/ISTOs) and reaches cosine 0.9998,
≥0.97 and 0.9992 with amplitude ratio 1.00 — the residual differences are the
random-number streams, which cannot be matched across MATLAB and NumPy.

Three EasySpin conventions were adopted to get there: field-swept cardamom
spectra are the imaginary FFT of the time-weighted FID and ignore
`Exp.Harmonic`; the spectrum scale is 1/nOrients of the weighted orientation sum
(EasySpin weights and then averages again); the jump model takes its dynamics
from `Sys.TransRates`/`Sys.Orientations` and needs no correlation time.

Remaining caveats audited in v0.1.0:
- The `SzFilter` secular projection used in MATLAB cardamom ISTOs is
  not yet applied in torchspin; for hyperfine systems this can cause subtle
  pseudo-secular differences.
- The `stochtraj_diffusion` Wigner-potential torque is computed via Euler-
  angle finite differences, which is exact only for axial L=2,M=0,K=0
  potentials. Generic potentials produce a known approximation.

---

## Pepper — round 3 (2026-09-02)

Added and MATLAB-validated (`test_pepper_round3_matlab_validation.py`, 32 cases):
`Experiment.mwMode` (tilted linear, unpolarised and circular excitation for
powders and crystals on the matrix, frequency-sweep and perturbation paths),
photoselection with perturbation theory, `Options.separate='transitions'`,
automatic field and frequency sweep ranges, ordering in frequency sweeps.

Two EasySpin-side approximations surfaced and are documented rather than
copied: (1) EasySpin's matrix path interpolates eigenvectors across the field
segment; for 63Cu+14N its intensities are 3–4 % off the exact value and its
spectrum disagrees with its own perturb2 (cosine 0.951), while torchspin's exact
intensities agree with EasySpin perturb2 at 0.988 (`septrans_CuN_matrix` xfail).
(2) For S=5/2 in parallel mode torchspin and EasySpin agree exactly on the coarse
grid; with the default [19 4] interpolation EasySpin sits farther from its own
converged spectrum (0.9947) than torchspin (0.9983); the test threshold is 0.995.

Internals aligned with EasySpin in the process: global transition pre-/post-
selection (`Opt.Threshold`) instead of per-orientation thresholds, a
Hermite-cubic tangent test that finds paired resonances of looping transitions,
linear (L1) / local-cubic (L3) interpolation of intensities and widths, and
`hamsymm` equality at 1e-12 relative so finite-difference steps break symmetry.

---

## Pepper — features validated in round 2 (2026-09-01)

All of the following are MATLAB-validated (`test_pepper_crystal_*`,
`test_pepper_parallel_*`, `test_pepper_features_*`, `test_pepper_features2_*`):
single crystals (`SampleFrame`/`CrystalSymmetry`/`MolFrame`/`SampleRotation`,
`Options.Sites`, `separate='orientations'|'sites'`), parallel-mode detection,
natural-abundance and custom isotopologues (`'Cu'`, `'(63,65)Cu'` + `Abund`,
`Options.IsoCutoff`), non-equilibrium populations (`SpinSystem.initState` in the
zero-field/eigen/xyz/coupled/uncoupled bases, density matrices, `'T0'`,
`'singlet'`), partial ordering (`Experiment.Ordering`, λ or callable),
photoselection (`lightBeam`, `lightScatter`, `tdm`), field modulation
(`ModAmp`), dispersion (`mwPhase`), `separate='components'`, and frequency
sweeps with strain/interpolation/projection through the same code path as
field sweeps. Not ported: `separate='transitions'`, unpolarised/circular
excitation, ordering in frequency sweeps, photoselection with perturbation
theory, automatic sweep ranging in pepper (garlic auto-ranges).

---

## eprload — EasySpin oracle and native units (behavior change, 2026-09-01)

`eprload` is validated against EasySpin's output for every readable file in
`tests/eprfiles` (75 files). To achieve parity the Bruker readers now return
the abscissa in the file's native units (Gauss for BES3T/ESP, MHz for ENDOR,
seconds for time sweeps) instead of converting to mT, multi-dimensional data
are returned as arrays with one abscissa per dimension (a list), several data
values per point (`IKKF 'CPLX,CPLX'`) give a list of datasets, and JEOL /
specman axes follow EasySpin's conventions. Callers that relied on mT must
divide by 10.  The `eprload` docstring now states the units per format and shows
that conversion in a worked example, and `eprload_info` prints the unit and
flags a gauss axis, so this no longer has to be found here first.

---

## Stochastic trajectories — reproducibility

`DiffusionPar` and `JumpPar` now expose a `seed: Optional[int]` field that
threads through all RNG calls. Setting `seed=42` produces bit-reproducible
trajectories across runs. Without a seed, output is non-deterministic.

This was added in v0.1.0 to support reproducible benchmarks; users porting
from earlier development snapshots should pass `seed=` explicitly.

---

## Autograd — differentiable pepper

Since 2026-09-05 `differentiable_spectrum()` builds a `SpinSystem` from the
parameter tensors and evaluates it with `pepper_autograd()`, which is `pepper`'s
own forward path (resonance-field search with transition tracking, EasySpin
grid interpolation, SOPHE triangle/zone projection, EasySpin convolution and
harmonic) written on the autograd graph. Forward output equals `pepper()` to
1e-11 relative for every grid symmetry, `GridSize` (int or `[N, Ninterp]`) and
harmonic; gradients through the resonance-field search come from the converged
Newton step (implicit-function theorem), no custom backward.

Gradient validation against central finite differences of `pepper()`
(`test_pepper_autograd.py`): A, D, Gaussian and Lorentzian lw and Temperature
agree to ~1e-8 relative norm error; g gradients agree to the finite-difference
limit set by the piecewise-linear projection (a vertex crossing a field bin is
a kink): relative norm error 6e-2 / 8e-3 / 1.5e-4 / 2e-8 for steps 1e-4 /
1e-5 / 1e-6 / 1e-7 in g, identically for `pepper` and `pepper_autograd`.
Optimizers see an almost-everywhere exact gradient of a piecewise-smooth model.

The earlier stand-alone models remain as `method='broadband'` and
`method='analytical'` (deprecated). They sum discrete orientations without
interpolation or projection and carry orientation-grid ripple at GridSizes
where pepper is converged: for a Cu(II) hyperfine system at GridSize 31 the
cosine to EasySpin is 0.985–0.999 (absorption) and 0.73–0.88 (first
derivative), against 0.9998–1.0000 / 0.999 for `pepper`; the ripple only falls
below 1 % of the signal around GridSize 91. Do not train models on their output.

Strains are differentiable (HStrain, gStrain, AStrain; DStrain magnitudes
are constants). Not yet differentiable (`pepper_autograd` raises
`NotImplementedError`, `pepper()` itself is unchanged): `Exp.mwPhase`,
partial ordering, photoselection, crystals, frequency sweeps, `Sys.initState`.
`salt` (fixed-field mode) and `garlic` (all modes) are
differentiable through the same approach (`test_salt_autograd.py`,
`test_garlic_autograd.py`); `garlic` switches its nearest-bin stick spectrum to
the two-bin linear split for grad tensors (sub-bin quantisation difference).
`saffron` is differentiable for ESEEM/HYSCORE with grad tensors (returns tensors
and uses `TimeDomain=True` accumulation; Mims-ENDOR raises). `chili` has no
differentiable path yet (needs a torch SLE solve); `curry` is pure torch and
differentiable.

## Platform and dependency support

`pyproject.toml` declares Python >= 3.10, `torch>=2.0,<3`, `numpy>=1.23,<3`,
`scipy>=1.10`.  Two combinations are exercised directly:

| | development host | clean-room host |
|---|---|---|
| OS | Ubuntu 22.04 (x86-64) | Ubuntu 22.04 (x86-64) |
| Python | 3.12 | 3.12 |
| NumPy / SciPy | 2.4 / 1.17 | 2.5 / 1.18 |
| PyTorch | 2.5 (CUDA 12.1) | 2.14 (CUDA 13.0) |
| GPU | 2× RTX 4090 | 2× RTX 3090 |

CI additionally runs the suite on Linux, macOS and Windows across Python
3.10-3.13 on every push, which is the full range the package claims support
for (`requires-python = ">=3.10"`).

The clean-room host installs the package into an empty environment from the
declared dependencies alone and runs the full suite; it is what the release is
gated on.  NumPy 1.x is still supported (the `np.trapezoid` / `np.trapz`
fallbacks) but is not routinely exercised.

Platform-specific breakage is real and worth testing for rather than assuming
away: the first CI run the project ever completed found that `eprload` could
not pair Bruker `.SPC`/`.PAR` files on a case-sensitive filesystem, which no
amount of testing on macOS would have surfaced.

CUDA is used opportunistically.  Batched eigendecompositions of matrices
larger than 32×32 are routed to the CPU thread pool (the cuSOLVER
`syevjBatched` limit), and `pepper`'s grid interpolation, triangle projection
and strain accumulation are host-side regardless of `device`; see the device
audit in
`benchmarks/results/workstation_20260904_verified/BENCHMARK_VERIFICATION.md`.
`chili` and `cardamom` have no CUDA path at all.

## Expensive tests are opt-in

Two cases in `test_pepper_cupc_matlab_validation.py` are exact diagonalizations
of a 72-dimensional Hilbert space over a converged grid, and one of them takes
about ten minutes (EasySpin needs 630 s for the same case).  CI's `full-suite`
job has no marker filter, so they are skipped unless `TORCHSPIN_RUN_EXPENSIVE=1`
is set rather than merely marked `slow`.  They are kept because
`cupc_matrix_aframe` is the largest exact comparison run to completion in both
codes — torchspin 590 s against EasySpin's 630 s — which is the evidence that
torchspin's matrix path is not the slower of the two.

---

## Test counts (v0.3.0, 2026-10-04, clean-room host)

| Outcome | Count |
|---|---|
| Passed | 2659 |
| Failed | 0 |
| Skipped | 4 |
| xfailed (documented known issues) | 3 |

The documented xfails are:
1. `test_pepper_round3_matlab_validation.py::[septrans_CuN_matrix]` — EasySpin's
   matrix-path eigenvector interpolation (see § Pepper — round 3)
2. `test_potential_biases_distribution` — stochastic Wigner-potential biasing
   requires longer equilibration than the test budget
3. `test_cardamom.py::test_fast_motion` — stochastic fast-motion narrowing
   needs more trajectories than the test budget

The 4 skips are environment-dependent, not unimplemented behavior:

* `test_fitgui.py` used to skip because `ipywidgets` shipped only in the `gui`
  extra; it is now in `test` and `dev` as well, so those tests run.  A missing
  GUI dependency also no longer surfaces as a bare `ModuleNotFoundError` — the
  panel raises an `ImportError` naming the module and the extra that provides
  it.
* 1 in `test_gpu_consistency.py` — it exercises the CPU fallback taken when
  CUDA is *missing*, so it skips on a machine that has a GPU.

Counts come from the clean-room host described above, with `pip install
-e ".[test]"` and both GPUs visible.  On a CPU-only machine the CUDA-variant
tests skip instead of running, so the skip count rises and the pass count
falls correspondingly; no test fails.

---

## How this document is maintained

This file is updated as caveats are addressed or new ones are discovered.
Each release notes which limitations changed in `CHANGELOG.md`.

This document is the distilled public record of an internal audit (~50 pages
of findings and synthesis notes) carried out during the port.  The raw audit
notes are not distributed; everything they established that bears on using
torchspin is stated here, and the evidence is the test suite plus the stored
EasySpin references in `tests/data/`.
