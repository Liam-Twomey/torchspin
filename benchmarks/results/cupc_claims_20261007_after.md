# CuPc pepper claims

- date: 2026-10-07
- torchspin: 0.3.0 (torchspin/__init__.py)
- torch: 2.14.1, threads: 10
- python: 3.14.8 on Darwin arm64

System (Pc_Fits.m): S=1/2, g = [2.04894 2.04894 2.181], A(Cu) = [15.3311 15.3311 646.629] MHz,
A(N) = 45 MHz isotropic, lw = 0.542209 mT Gaussian, 9.347144 GHz, 233.8-433.8 mT, 2667 points.

## Claim 1 - the same system the matrix method could not diagonalize

```
  hybrid, 4 explicit N  , GridSize=[91,4]:     1.96 s  ->  ok
  hybrid, N with n=[1,4], GridSize=[91,4]:     0.28 s  ->  ok
```

## Claim 2 - no hybrid method

```
Options(Method='hybrid')  ->  accepted
```

## Claim 3 - pepper rejects equivalent nuclei (Sys.n > 1)

```
Nucs=['Cu','N'], n=[1,4], Method=perturb
      0.02 s  ->  ok
```

## Claim 4 - perturbation cost is dominated by per-component setup

```
  system                     GridSize   components      time
  ------------------------------------------------------------
  63Cu + 4x14N (explicit)     [7, 4]            1      0.05 s
  63Cu + 4x14N (explicit)    [91, 4]            1      0.06 s
  Cu + 4xN  (natural)        [7, 4]           10      0.34 s
  Cu + 4xN  (natural)       [91, 4]           10      0.41 s

  The cost barely moves between a 7-knot and a 91-knot grid, so it is
  per-component setup rather than orientation work, and it is paid once
  per isotopologue -- which is why natural abundance costs several times
  more than naming the isotopes.
```

## Extra - does Options.IsoCutoff reach the isotopologue expansion?

```
  IsoCutoff   isotopologues()   expand_components()   agree?
  ----------------------------------------------------------
  0.0001                 10                    10   yes
  0.001                  10                    10   yes
  0.01                    2                     2   yes
  0.5                     1                     1   yes
```
