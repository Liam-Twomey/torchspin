# Pitfalls

Settings that are valid input but give a quietly wrong or degraded result. Where torchspin can detect
the problem it warns or raises, as noted below. The rest is up to the user.

## Simulation

### Perturbation theory at large A/ν

Low-order perturbation theory converges slowly when the hyperfine coupling is a sizable fraction of the
microwave frequency. Relative L2 error against the exact result, for isotropic ⁹¹Zr (I = 5/2),
A = 780 MHz, 9.65 GHz (A/ν = 0.08), line widths [1, 2] mT:

| Simulator, `Options.Method` | Relative L2 error |
|---|---|
| garlic `'perturb2'` | 0.82 |
| garlic `'perturb3'` | 0.11 |
| garlic `'perturb4'` | 0.018 |
| garlic `'perturb5'` | 0.0032 |
| pepper `'perturb'` (2nd order), vs `'matrix'` | 0.18 |

Use garlic `'exact'` (Breit–Rabi, the default) or pepper `'matrix'` (the default) for systems like this.
As a cross-check, pepper `'matrix'` and garlic `'exact'` agree to 6e-3 on the same isotropic system.

### garlic's `AccumMethod='binning'` in fits

In the isotropic limit garlic places each line in its nearest field bin (`'binning'`, EasySpin's default).
The spectrum is then piecewise-constant in g and A. A finite-difference derivative with respect to a
line position is zero at any step smaller than a bin. Levenberg–Marquardt and TRF then do not move g or
A, and the uncertainty estimate has nothing to work with.

- Dict-style `esfit` that calls garlic directly switches an unset `AccumMethod` to `'linear'`.
- A custom model that wraps garlic should pass `Options(AccumMethod='linear')` or `'explicit'`.
- esfit warns when a parameter has no effect on the residuals at the finite-difference step. It reports
  that parameter's `pstd` as `nan`, not 0.
- On the autograd graph, garlic switches to `'linear'` itself.

## Fitting

### Model sign with `autoscale='lsq'`

The least-squares scale is constrained to be non-negative, as in EasySpin. A model with the wrong sign,
such as an inversion-recovery curve written as `exp(-t/T)` and not `-exp(-t/T)`, cannot be fitted by a
negative scale. The best fit is then scale 0 plus the baseline, and esfit warns. Give the model the sign
of the signal. (`FitOptions(lsq_negative_scale='abs')` restores EasySpin's `abs()` behavior. That keeps a
baseline solved for the opposite sign, so it is not a least-squares fit.)

### Weak satellites next to a strong central line

With natural-abundance zirconium, about 89 % of the intensity sits in a central I = 0 line that does not
depend on A. The ⁹¹Zr satellites are weak and spread widely, so the RMSD as a function of A has false
minima wherever a shifted satellite comb overlaps part of the data. A local optimizer started tens of
MHz off can stop in one of them. `method='global'` has also been seen to lock onto a wrong assignment.
A staged fit is faster and more reliable:

1. fit g and the line widths with A fixed (the central line does not depend on A),
2. scan A on a 1-D grid with the rest fixed,
3. polish all parameters with a local optimizer.

[`examples/fitting/large_hyperfine_fit.py`](../examples/fitting/large_hyperfine_fit.py) runs this recipe
on synthetic data.

### Quadrupole fits started at zero

For allowed (ΔmI = 0) transitions the spectrum is even in Q, so dL/dQ = 0 exactly at Q = 0. A
gradient-based fit started at Q = 0 never moves Q, even with the autograd graph intact. Start from a
nonzero value.

## Pulse simulations (spidyan, saffron real pulses)

### Integration step

A user-set `SpidyanOptions.IntTimeStep` coarser than the Nyquist step of the highest simulation-frame
frequency undersamples the pulses. For a rectangular π pulse at Q-band, the error in ⟨Sz⟩ is 5e-4 at
Nyquist/5, 1.4 % at Nyquist/2.5, about 3 % at Nyquist/2 and 40 % at Nyquist. spidyan warns above
Nyquist/2. Leave `IntTimeStep` unset for the automatic step (Nyquist/50, as in EasySpin).

### Simulation frame close to the carrier

`SimFreq` close to the pulse carrier breaks the separation of the co- and counter-rotating components.
A π pulse with ν1 = 25 MHz leaves an error of 0.8 % at a carrier offset of 4·ν1, 3.6 % at 2·ν1, and
does nothing at offset 0. spidyan warns when the offset is below 5·ν1.

### Predefined sequences and real pulses

`saffron` with a predefined `Sequence` (`'2pESEEM'`, `'3pESEEM'`, `'HYSCORE'`, …) runs the ideal-pulse
algorithm. Together with nonzero `Exp.tp` it raises an error, as EasySpin does. For real pulses, write
the sequence explicitly (`Sequence=[P90, tau, P180, tau]` with pulse dicts, `Dim1`, `DetWindow=0`) and
set `SimulationMode='thyme'`.
