"""Reproduce the reported Cu(II) phthalocyanine ``pepper`` defects.

A user fitting CuPc diluted in ZnPc (S=1/2 Cu + 4 equivalent 14N, A_par(Cu) about
647 MHz) reported five problems with torchspin 0.3.0.  This script measures all
five so that the fixes have a before/after ledger:

1. ``Method='matrix'`` raises ``torch._C._LinAlgError`` on Cu + 4x14N.
2. ``Method='hybrid'`` is rejected (EasySpin ``Opt.Method='hybrid'`` is missing).
3. ``pepper`` rejects sets of equivalent nuclei (``SpinSystem.n > 1``).
4. The perturbation path is dominated by a fixed per-component cost, so natural
   abundance (several isotopologues) is several times slower than explicit
   isotopes and the cost barely depends on ``GridSize``.
5. The matrix path cost grows steeply with the number of nuclei.

The reference system is the one in the MATLAB fit script ``Pc_Fits.m``.

Usage::

    python benchmarks/python/cupc_claims.py
    python benchmarks/python/cupc_claims.py --output benchmarks/results
"""
from __future__ import annotations

import argparse
import datetime as _dt
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Resolve the source checkout when the script is run without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _where(path) -> str:
    """Repo-relative location of an imported module, so reports carry no local paths."""
    p = Path(path).resolve()
    return str(p.relative_to(_REPO_ROOT)) if p.is_relative_to(_REPO_ROOT) else 'installed package'
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torchspin
from torchspin import Experiment, Options, SpinSystem, pepper
from torchspin.isotopologues import expand_components

# Pc_Fits.m: Sys0.g = [2.04894 2.181], A = [15.3311 646.629; 45 45; ...],
# lw = 0.542209 mT, Exp.mwFreq = 9.347144 GHz, Exp.Range = [233.8 433.8] mT.
G_CUPC = [2.04894, 2.04894, 2.181]
A_CU = [15.3311, 15.3311, 646.629]
A_N = [45.0, 45.0, 45.0]
LW_MT = 0.542209
EXP_KWARGS = dict(mwFreq=9.347144, Range=[233.8, 433.8], nPoints=2667, Harmonic=1)


def _system(nucs: list[str], n_nitrogen: int, n: list[int] | None = None) -> SpinSystem:
    return SpinSystem(S=[0.5], g=[G_CUPC], Nucs=nucs,
                      A=[A_CU] + [A_N] * n_nitrogen, n=n, lw=[LW_MT, 0.0])


def _time(sys_, opt) -> tuple[float, str, np.ndarray | None]:
    """Run one simulation, returning (seconds, status, spectrum)."""
    exp = Experiment(**EXP_KWARGS)
    t0 = time.perf_counter()
    try:
        _, spc = pepper(sys_, exp, opt)
    except Exception as exc:                      # noqa: BLE001 - we are cataloguing failures
        return time.perf_counter() - t0, f'{type(exc).__name__}: {str(exc).splitlines()[0][:70]}', None
    return time.perf_counter() - t0, 'ok', np.asarray(spc, dtype=float)


def claim_1_matrix_crash() -> list[str]:
    """Matrix diagonalization on Cu + 4x14N, the exact call that was reported.

    Only run with --slow: once the eigensolver no longer aborts, this call
    *completes*, and exact diagonalization of a 648-dimensional Hilbert space
    over a powder grid takes a very long time in either code.  That is the
    point of the hybrid method, not a thing to wait for on every run.
    """
    sys_ = _system(['Cu'] + ['N'] * 4, 4)
    dt, status, _ = _time(sys_, Options())
    return ['Cu(nat) + 4xN(nat), Method=matrix, GridSize=[19,4] (default)',
            f'  {dt:8.2f} s  ->  {status}']


def claim_1_hybrid_handles_it() -> list[str]:
    """The same system through the hybrid method, for scale."""
    rows = []
    for label, nucs, n_n, n in (('4 explicit N  ', ['Cu'] + ['N'] * 4, 4, None),
                                ('N with n=[1,4]', ['Cu', 'N'], 1, [1, 4])):
        sys_ = _system(nucs, n_n, n=n)
        dt, status, _ = _time(sys_, Options(Method='hybrid', HybridCoreNuclei=[1],
                                            GridSize=[91, 4], Verbosity=0))
        rows.append(f'  hybrid, {label}, GridSize=[91,4]:  {dt:7.2f} s  ->  {status}')
    return rows


def claim_2_hybrid_missing() -> list[str]:
    try:
        Options(Method='hybrid')
        head = 'accepted'
    except Exception as exc:                      # noqa: BLE001
        head = f'{type(exc).__name__}: {str(exc).splitlines()[0][:80]}'
    return [f"Options(Method='hybrid')  ->  {head}"]


def claim_3_equivalent_nuclei() -> list[str]:
    sys_ = _system(['Cu', 'N'], 1, n=[1, 4])
    dt, status, _ = _time(sys_, Options(Method='perturb', GridSize=[91, 4]))
    return [f"Nucs=['Cu','N'], n=[1,4], Method=perturb",
            f'  {dt:8.2f} s  ->  {status}']


def claim_4_perturb_cost() -> list[str]:
    rows = ['  system                     GridSize   components      time',
            '  ' + '-' * 60]
    for label, nucs, n_n in (('63Cu + 4x14N (explicit)', ['63Cu'] + ['14N'] * 4, 4),
                             ('Cu + 4xN  (natural)   ', ['Cu'] + ['N'] * 4, 4)):
        sys_ = _system(nucs, n_n)
        n_comp = len(expand_components(sys_))
        for grid in ([7, 4], [91, 4]):
            dt, status, _ = _time(sys_, Options(Method='perturb', GridSize=grid, Verbosity=0))
            note = '' if status == 'ok' else f'  ({status})'
            rows.append(f'  {label}   {str(grid):>8}   {n_comp:10d}   {dt:7.2f} s{note}')
    rows.append('')
    rows.append('  The cost barely moves between a 7-knot and a 91-knot grid, so it is')
    rows.append('  per-component setup rather than orientation work, and it is paid once')
    rows.append('  per isotopologue -- which is why natural abundance costs several times')
    rows.append('  more than naming the isotopes.')
    return rows


def claim_5_matrix_scaling() -> list[str]:
    rows = ['  system                      components      time',
            '  ' + '-' * 54]
    for n_n in (0, 1, 2):
        for label, nucs in ((f'63Cu + {n_n}x14N (explicit)', ['63Cu'] + ['14N'] * n_n),
                            (f'Cu + {n_n}xN  (natural)   ', ['Cu'] + ['N'] * n_n)):
            sys_ = _system(nucs, n_n)
            n_comp = len(expand_components(sys_))
            dt, status, _ = _time(sys_, Options(GridSize=[91, 4], Verbosity=0))
            note = '' if status == 'ok' else f'  ({status})'
            rows.append(f'  {label}    {n_comp:10d}   {dt:7.2f} s{note}')
    return rows


def extra_isocutoff() -> list[str]:
    """Opt.IsoCutoff should prune isotopologues; check that it reaches pepper."""
    from torchspin.isotopologues import isotopologues
    sys_ = _system(['Cu'] + ['N'] * 4, 4)
    rows = ['  IsoCutoff   isotopologues()   expand_components()   agree?',
            '  ' + '-' * 58]
    for cut in (1e-4, 1e-3, 1e-2, 0.5):
        direct = len(isotopologues(sys_, rel_threshold=cut))
        routed = len(expand_components(sys_, cut))
        rows.append(f'  {cut:<11g} {direct:>13d}   {routed:>19d}   '
                    f'{"yes" if direct == routed else "NO - cutoff ignored"}')
    return rows


SLOW_SECTIONS = (
    ('Claim 1 - matrix diagonalization on Cu + 4x14N (the reported call)', claim_1_matrix_crash),
    ('Claim 5b - matrix cost with two ligand nuclei', claim_5_matrix_scaling),
)

SECTIONS = (
    ('Claim 1 - the same system the matrix method could not diagonalize', claim_1_hybrid_handles_it),
    ('Claim 2 - no hybrid method', claim_2_hybrid_missing),
    ('Claim 3 - pepper rejects equivalent nuclei (Sys.n > 1)', claim_3_equivalent_nuclei),
    ('Claim 4 - perturbation cost is dominated by per-component setup', claim_4_perturb_cost),
    ('Extra - does Options.IsoCutoff reach the isotopologue expansion?', extra_isocutoff),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=None,
                        help='directory for a Markdown report (default: print only)')
    parser.add_argument('--label', default=None,
                        help="tag for the report filename, e.g. 'before' or 'after'")
    parser.add_argument('--slow', action='store_true',
                        help='also run the exact-diagonalization cases, which take '
                             'tens of minutes for Cu + 4 nitrogens')
    args = parser.parse_args()

    lines = [
        '# CuPc pepper claims',
        '',
        f'- date: {_dt.date.today().isoformat()}',
        f'- torchspin: {torchspin.__version__} ({_where(torchspin.__file__)})',
        f'- torch: {torch.__version__}, threads: {torch.get_num_threads()}',
        f'- python: {platform.python_version()} on {platform.system()} {platform.machine()}',
        '',
        'System (Pc_Fits.m): S=1/2, g = [2.04894 2.04894 2.181], '
        'A(Cu) = [15.3311 15.3311 646.629] MHz,',
        'A(N) = 45 MHz isotropic, lw = 0.542209 mT Gaussian, '
        '9.347144 GHz, 233.8-433.8 mT, 2667 points.',
        '',
    ]
    for title, fn in (SECTIONS + SLOW_SECTIONS if args.slow else SECTIONS):
        lines.append(f'## {title}')
        lines.append('')
        lines.append('```')
        lines.extend(fn())
        lines.append('```')
        lines.append('')

    report = '\n'.join(lines)
    print(report)

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        tag = f'_{args.label}' if args.label else ''
        path = args.output / f'cupc_claims_{_dt.date.today():%Y%m%d}{tag}.md'
        path.write_text(report, encoding='utf-8')
        print(f'wrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
