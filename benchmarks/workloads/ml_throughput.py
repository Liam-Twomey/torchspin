"""Training-data throughput of torchspin.ml.simulate_batch: spectra per second.

Simulates ``--n`` random parameter rows of one system and reports spectra/s for each
``--workers`` value (0 = sequential in this process). The process pool is started
inside every simulate_batch call, as it is in a data-generation job, so its start-up
time is included. On a GPU run (``--device cuda``) the rows are simulated one after
another on the device (workers must be 0).

    python benchmarks/workloads/ml_throughput.py --system cu --n 512 --workers 0 8 32 --threads 1
    python benchmarks/workloads/ml_throughput.py --system cu2n --device cuda --n 128 --workers 0
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path


def _systems():
    from torchspin import Experiment, Options, SpinSystem
    from torchspin.ml import ParamSpec
    exp = Experiment(mwFreq=9.5, Range=[250, 370], nPoints=1024, Harmonic=1)
    return {
        # Cu(II), one nucleus: the typical training-set system
        'cu': (SpinSystem(S=[0.5], g=[[2.05, 2.05, 2.25]], Nucs='63Cu', A=[[30, 30, 500]], lw=[0.5, 0.0]),
               ParamSpec({'g': [(2.0, 2.1), (2.0, 2.1), (2.15, 2.35)], 'A': [(20, 60), (20, 60), (350, 650)],
                          'lw': [(0.2, 1.5), (0, 0)]}, transforms={'lw': 'softplus'}),
               exp, Options(GridSize=10, Verbosity=0)),
        # Cu(II) with two 14N: 72x72 matrices, the expensive case
        'cu2n': (SpinSystem(S=[0.5], g=[[2.05, 2.05, 2.25]], Nucs='63Cu,14N,14N',
                            A=[[30, 30, 500], [40, 40, 40], [40, 40, 40]], lw=[0.5, 0.0]),
                 ParamSpec({'g': [(2.0, 2.1), (2.0, 2.1), (2.15, 2.35)], 'A': [(20, 60), (20, 60), (350, 650),
                                                                                 (30, 50), (30, 50), (30, 50),
                                                                                 (30, 50), (30, 50), (30, 50)],
                            'lw': [(0.2, 1.5), (0, 0)]}, transforms={'lw': 'softplus'}),
                 exp, Options(GridSize=10, Verbosity=0, Method='matrix')),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--system', choices=['cu', 'cu2n'], default='cu')
    ap.add_argument('--n', type=int, default=256, help='spectra per measurement')
    ap.add_argument('--workers', type=int, nargs='+', default=[0])
    ap.add_argument('--threads', type=int, default=0, help='torch threads in this process (0 = leave)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--repeat', type=int, default=2)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torchspin  # noqa: F401  (before torch: sets OMP_WAIT_POLICY)
    import dataclasses
    import torch
    from torchspin.ml import simulate_batch
    if a.threads:
        torch.set_num_threads(a.threads)
    template, spec, exp, opt = _systems()[a.system]
    if a.device != 'cpu':
        if any(a.workers):
            ap.error('GPU runs simulate rows sequentially on the device: use --workers 0')
        opt = dataclasses.replace(opt, device=a.device)
    params = spec.sample(a.n, torch.Generator().manual_seed(0))
    simulate_batch('pepper', template, exp, opt, spec, params[:2])          # warm up this process

    rows = []
    for w in a.workers:
        times = []
        for _ in range(a.repeat):
            if a.device.startswith('cuda'):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            simulate_batch('pepper', template, exp, opt, spec, params, workers=w)
            if a.device.startswith('cuda'):
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        best = min(times)
        rows.append(dict(workers=w, seconds=times, spectra_per_s=a.n / best))
        print(f'{a.system:5s} {a.device:6s} workers={w:4d}  {a.n / best:8.1f} spectra/s  '
              f'({best:.2f} s for {a.n})', flush=True)
    if a.out:
        meta = dict(system=a.system, n=a.n, device=a.device, threads=torch.get_num_threads(),
                    torch=torch.__version__, python=platform.python_version(),
                    n_affinity_cpus=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(),
                    gpu=torch.cuda.get_device_name(0) if a.device.startswith('cuda') else None,
                    omp_wait_policy=os.environ.get('OMP_WAIT_POLICY'))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(meta=meta, rows=rows), open(a.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
