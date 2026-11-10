# CuPc pepper claims

- date: 2026-10-07
- torchspin: 0.3.0 (torchspin/__init__.py)
- torch: 2.14.1, threads: 10
- python: 3.14.8 on Darwin arm64

System (Pc_Fits.m): S=1/2, g = [2.04894 2.04894 2.181], A(Cu) = [15.3311 15.3311 646.629] MHz,
A(N) = 45 MHz isotropic, lw = 0.542209 mT Gaussian, 9.347144 GHz, 233.8-433.8 mT, 2667 points.

## Claim 1 - matrix diagonalization crashes on Cu + 4x14N

```
Cu(nat) + 4xN(nat), Method=matrix, GridSize=[19,4] (default)
     36.03 s  ->  _LinAlgError: linalg.eigh: (Batch element 5): The algorithm failed to converge becau
```

## Claim 2 - no hybrid method

```
Options(Method='hybrid')  ->  ValueError: Unknown Options.Method 'hybrid'. Valid: exact, matrix, perturb, perturb1, pertur
```

## Claim 3 - pepper rejects equivalent nuclei (Sys.n > 1)

```
Nucs=['Cu','N'], n=[1,4], Method=perturb
      0.00 s  ->  ValueError: pepper does not support sets of equivalent nuclei (SpinSystem.n > 1). 
```

## Claim 4 - perturbation cost is dominated by per-component setup

```
  system                     GridSize   components      time
  ------------------------------------------------------------
  63Cu + 4x14N (explicit)     [7, 4]            1      0.81 s
  63Cu + 4x14N (explicit)    [91, 4]            1      0.59 s
  Cu + 4xN  (natural)        [7, 4]           10      2.57 s
  Cu + 4xN  (natural)       [91, 4]           10      2.70 s

  Cost is nearly independent of GridSize -> it is a fixed per-component
  setup cost, paid once per isotopologue, not orientation work.
```

## Claim 5 - matrix cost grows steeply with the number of nuclei

```
  system                      components      time
  ------------------------------------------------------
  63Cu + 0x14N (explicit)             1      0.03 s
  Cu + 0xN  (natural)                2      0.08 s
  63Cu + 1x14N (explicit)             1      0.36 s
  Cu + 1xN  (natural)                4      1.56 s
  63Cu + 2x14N (explicit)             1      5.94 s
  Cu + 2xN  (natural)                6     20.51 s
```

## Extra - does Options.IsoCutoff reach the isotopologue expansion?

```
  IsoCutoff   isotopologues()   expand_components()   agree?
  ----------------------------------------------------------
  0.0001                 10                    10   yes
  0.001                  10                    10   yes
  0.01                    2                    10   NO - cutoff ignored
  0.5                     1                    10   NO - cutoff ignored
```
