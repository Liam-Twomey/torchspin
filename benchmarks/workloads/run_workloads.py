#!/usr/bin/env python3
"""Run the heavy workloads on CPU or CUDA, with replicated timing and a device audit.

    python benchmarks/workloads/run_workloads.py --device cpu --threads 8 --warmup 1 --repeat 5 --out results/cpu8.json
    python benchmarks/workloads/run_workloads.py --device cuda --threads 32 --warmup 1 --repeat 5 --out results/gpu.json
    python benchmarks/workloads/run_workloads.py --device cuda --only pepper_mn_S52_matrix --profile

Protocol (one process per configuration):
  * ``--threads N`` sets ``torch.set_num_threads(N)`` and, unless already set in the
    environment, OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=N *before* torch/numpy are
    imported; the effective native thread pools are recorded with threadpoolctl.
  * every workload gets ``--warmup`` untimed calls, then ``--repeat`` timed calls;
    every replicate is saved and ``seconds`` holds the median (min/quartiles in
    ``stats``).  CUDA calls are bracketed by ``torch.cuda.synchronize()``.
  * a workload that does not support the requested device (``WORKLOAD_DEVICES``)
    is recorded as ``effective_mode='unsupported'`` and not timed on it.
  * on CUDA every supported workload is verified: a one-call ``torch.profiler``
    trace records whether CUDA kernels ran, the kernel time, device-to-host
    copies and the ``torch.linalg.eigh``/``eigvalsh`` calls that were routed to
    the CPU pool (``torchspin/_linalg.py``); the CUDA output is compared with a
    CPU run (max abs deviation, NRMSE).  ``effective_mode`` is ``cuda``,
    ``hybrid`` (CPU fallbacks or < 50 % of the wall time in CUDA kernels) or
    ``cpu`` (no kernels observed).

Writes ``<out>`` (JSON) and, with ``--profile``, cProfile summaries to
``<out>.profile.txt``; CUDA profiler tables go to ``<out>.<workload>.cuda_profile.txt``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS')


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--only', default='', help='comma-separated workload names')
    ap.add_argument('--repeat', type=int, default=1, help='timed replicates per workload')
    ap.add_argument('--warmup', type=int, default=0, help='untimed calls before the replicates')
    ap.add_argument('--profile', action='store_true', help='cProfile one extra call per workload')
    ap.add_argument('--threads', type=int, default=0,
                    help='torch.set_num_threads and OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS (overrides the environment)')
    ap.add_argument('--out', default='')
    ap.add_argument('--save-outputs', default='',
                    help='write every workload\'s last result (numeric arrays) to this .npz, '
                         'for cross-host or CPU/CUDA comparison')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--no-verify', action='store_true',
                    help='skip the CUDA profiler trace and CPU/CUDA consistency check')
    ap.add_argument('--procs', type=int, default=0,
                    help='embarrassingly parallel run of the fit loop: N worker processes (CPU), each with --threads threads')
    return ap.parse_args()


def _apply_thread_env(threads: int) -> dict:
    """Set the native thread-pool variables before numpy/torch import; return them.

    The variables are overwritten, not defaulted: a batch script that exports
    OMP_NUM_THREADS for the whole allocation must not turn a "1-thread" run into
    a run with every core in the native pools."""
    if threads:
        for k in _THREAD_ENV:
            os.environ[k] = str(threads)
    return {k: os.environ.get(k) for k in _THREAD_ENV}


# --------------------------------------------------------------------------- worker pool
_W = {}


def _init_worker(threads, device):
    import torch as _t
    _t.set_num_threads(threads)
    _W['device'] = device


def _init_probe(_):
    # imports torchspin in the worker so the first timed replicate does not pay for it
    import torchspin  # noqa: F401
    return 0


def _one_fit_call(k):
    import numpy as np
    from torchspin import SpinSystem, Experiment, Options, pepper
    g = np.linspace(2.000, 2.010, 20)[k]
    sys_ = SpinSystem(S=[0.5], g=[float(g), 2.006, 2.003], Nucs='14N', A=[[20, 20, 85]], lwpp=[0.3, 0])
    exp = Experiment(mwFreq=9.5, Range=[330, 350], nPoints=1024)
    return pepper(sys_, exp, Options(Verbosity=0, GridSize=[31, 4], device=_W.get('device', 'cpu')))[1].numpy()


# --------------------------------------------------------------------------- helpers
def _flatten_outputs(obj):
    """All numeric arrays inside a simulator result (tensor, ndarray, tuple/list/dict)."""
    import numpy as np
    import torch
    out = []
    if isinstance(obj, torch.Tensor):
        out.append(obj.detach().cpu().numpy())
    elif isinstance(obj, np.ndarray):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            out.extend(_flatten_outputs(o))
    elif isinstance(obj, dict):
        for o in obj.values():
            out.extend(_flatten_outputs(o))
    elif isinstance(obj, (int, float, complex)):
        out.append(np.asarray([obj]))
    return out


def _compare(ref, test):
    """Scale-aware deviation between two simulator results (CPU reference vs CUDA)."""
    import numpy as np
    a_list, b_list = _flatten_outputs(ref), _flatten_outputs(test)
    if len(a_list) != len(b_list):
        return dict(comparable=False, reason=f'{len(a_list)} vs {len(b_list)} arrays')
    max_abs = 0.0
    sq_diff = 0.0
    sq_ref = 0.0
    n = 0
    max_scale = 0.0
    for a, b in zip(a_list, b_list):
        a = np.asarray(a); b = np.asarray(b)
        if a.shape != b.shape:
            return dict(comparable=False, reason=f'shape {a.shape} vs {b.shape}')
        if a.size == 0:
            continue
        d = (a.astype(np.complex128) - b.astype(np.complex128)).ravel()
        max_abs = max(max_abs, float(np.max(np.abs(d))))
        sq_diff += float(np.sum(np.abs(d) ** 2))
        sq_ref += float(np.sum(np.abs(a.astype(np.complex128)) ** 2))
        n += d.size
        max_scale = max(max_scale, float(np.max(np.abs(a))))
    nrmse = float(np.sqrt(sq_diff / max(n, 1)) / max(np.sqrt(sq_ref / max(n, 1)), 1e-300))
    return dict(comparable=True, n_values=n, max_abs_deviation=max_abs,
                max_abs_over_max_ref=max_abs / max(max_scale, 1e-300), nrmse=nrmse)


def _gpu_snapshot():
    """nvidia-smi utilization/memory of every GPU (contention record); None if unavailable."""
    import subprocess
    try:
        txt = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'], text=True, timeout=20)
    except Exception:
        return None
    rows = []
    for line in txt.strip().splitlines():
        i, u, m, t = [x.strip() for x in line.split(',')]
        rows.append(dict(index=int(i), utilization_pct=float(u), memory_used_mib=float(m), memory_total_mib=float(t)))
    return rows


class _EighCounter:
    """Count torch.linalg.eigh/eigvalsh calls by device and matrix size.

    ``torchspin/_linalg.py`` calls ``torch.linalg.eigh`` through the module
    attribute, so wrapping it here sees every batched diagonalization, including
    the ones it moved from CUDA to the CPU pool (matrices larger than 32x32).
    """

    def __init__(self):
        self.calls = {}

    def _wrap(self, fn):
        def wrapped(H, *a, **k):
            key = (('cuda' if H.is_cuda else 'cpu'), int(H.shape[-1]),
                   int(H.shape[:-2].numel()) if H.ndim > 2 else 1)
            self.calls[key] = self.calls.get(key, 0) + 1
            return fn(H, *a, **k)
        return wrapped

    def __enter__(self):
        import torch
        self._orig = (torch.linalg.eigh, torch.linalg.eigvalsh)
        torch.linalg.eigh = self._wrap(self._orig[0])
        torch.linalg.eigvalsh = self._wrap(self._orig[1])
        return self

    def __exit__(self, *exc):
        import torch
        torch.linalg.eigh, torch.linalg.eigvalsh = self._orig
        return False

    def summary(self):
        rows = [dict(device=d, dim=n, batch=b, calls=c) for (d, n, b), c in sorted(self.calls.items())]
        cpu_large = sum(r['calls'] for r in rows if r['device'] == 'cpu' and r['dim'] > 32)
        cpu_any = sum(r['calls'] for r in rows if r['device'] == 'cpu')
        cuda_any = sum(r['calls'] for r in rows if r['device'] == 'cuda')
        return dict(by_device_dim_batch=rows, cpu_calls=cpu_any, cpu_calls_dim_gt_32=cpu_large, cuda_calls=cuda_any)


def _verify_cuda(name, fn_cuda, fn_cpu, out_stem):
    """One profiled CUDA call + one CPU call: kernels, CPU routing, consistency."""
    import time
    import torch
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize()
    with _EighCounter() as counter:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            res_cuda = fn_cuda()
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
    eigh = counter.summary()
    kernel_events = 0
    kernel_us = 0.0
    memcpy_d2h = 0
    memcpy_h2d = 0
    for ev in prof.events():
        dev = getattr(ev, 'device_type', None)
        is_cuda = dev is not None and str(dev).endswith('CUDA')
        nm = ev.name.lower()
        if 'memcpy' in nm:
            if 'dtoh' in nm:
                memcpy_d2h += 1
            elif 'htod' in nm:
                memcpy_h2d += 1
            continue
        if is_cuda:
            kernel_events += 1
            kernel_us += float(ev.time_range.elapsed_us()) if hasattr(ev, 'time_range') else 0.0
    kernel_s = kernel_us * 1e-6
    ka = prof.key_averages()
    try:
        table = ka.table(sort_by='self_device_time_total', row_limit=25)
    except Exception:
        table = ka.table(sort_by='self_cuda_time_total', row_limit=25)
    table_cpu = ka.table(sort_by='self_cpu_time_total', row_limit=25)
    if out_stem:
        Path(out_stem + f'.{name}.cuda_profile.txt').write_text(
            f'=== {name}: profiled CUDA call, wall {wall:.3f} s, CUDA kernel time {kernel_s:.3f} s, '
            f'{kernel_events} kernel events, {memcpy_d2h} DtoH / {memcpy_h2d} HtoD copies\n'
            f'eigh/eigvalsh calls: {eigh}\n\n--- by device time ---\n{table}\n\n--- by CPU time ---\n{table_cpu}\n')
    # consistency with the CPU implementation
    res_cpu = fn_cpu()
    cmp = _compare(res_cpu, res_cuda)
    frac = kernel_s / wall if wall > 0 else 0.0
    if kernel_events == 0:
        mode = 'cpu'
    elif eigh['cpu_calls_dim_gt_32'] > 0 or frac < 0.5:
        mode = 'hybrid'
    else:
        mode = 'cuda'
    return dict(
        cuda_kernels_observed=kernel_events > 0, cuda_kernel_events=kernel_events,
        cuda_kernel_seconds=kernel_s, profiled_wall_seconds=wall, cuda_kernel_fraction_of_wall=frac,
        memcpy_dtoh_events=memcpy_d2h, memcpy_htod_events=memcpy_h2d,
        eigh=eigh, cpu_routed_eigh_dim_gt_32=eigh['cpu_calls_dim_gt_32'] > 0,
        consistency_vs_cpu=cmp, effective_mode=mode,
    )


# --------------------------------------------------------------------------- main
def main():
    a = _parse()
    thread_env = _apply_thread_env(a.threads)
    # imports after the thread environment is fixed
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # prefer the checkout's torchspin over any (possibly stale) site-packages install
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import cProfile, io, json, platform, pstats, statistics, subprocess, time
    import numpy as np
    import torch
    from workloads import WORKLOADS, WORKLOAD_DEVICES, STOCHASTIC, supports_device

    if a.list:
        for n in WORKLOADS:
            print(f'{n:32s} devices={",".join(WORKLOAD_DEVICES[n])}')
        return 0
    if a.threads:
        torch.set_num_threads(a.threads)
    use_cuda = a.device.startswith('cuda')
    if use_cuda and not torch.cuda.is_available():
        print('CUDA requested but torch.cuda.is_available() is False', file=sys.stderr)
        return 2
    names = [n for n in a.only.split(',') if n] or list(WORKLOADS)

    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = None
    repo = Path(__file__).resolve().parents[2]

    def _git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    meta = dict(
        host=platform.node(), python=platform.python_version(), torch=torch.__version__,
        torch_cuda=torch.version.cuda, numpy=np.__version__,
        requested_device=a.device, device=a.device,
        threads=torch.get_num_threads(), torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
        thread_env=thread_env, native_threadpools=pools,
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        n_affinity_cpus=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(),
        gpu=torch.cuda.get_device_name(0) if use_cuda else None,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        torchspin_commit=_git('rev-parse', 'HEAD'), torchspin_status=_git('status', '--short'),
        warmup=a.warmup, repeat=a.repeat, protocol='median of replicates; every replicate saved',
        started=time.strftime('%Y-%m-%d %H:%M:%S %Z'),
    )
    results, replicates, stats, audit = {}, {}, {}, {}
    outputs = {}
    prof_txt = []

    if a.procs:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        reps = []
        # the pool is started once, outside the timed replicates (as esfit caches its
        # pool); its start-up cost is reported separately
        t0 = time.perf_counter()
        pool = ctx.Pool(a.procs, initializer=_init_worker, initargs=(a.threads or 1, a.device))
        pool.map(_init_probe, range(a.procs))
        spawn_s = time.perf_counter() - t0
        try:
            for r in range(a.warmup + a.repeat):
                t0 = time.perf_counter()
                pool.map(_one_fit_call, list(range(20)))
                if r >= a.warmup:
                    reps.append(time.perf_counter() - t0)
        finally:
            pool.close(); pool.join()
        key = 'pepper_fit_loop_20_procs%d' % a.procs
        results[key] = statistics.median(reps); replicates[key] = reps
        q1, med, q3 = [float(x) for x in np.percentile(reps, [25, 50, 75])]
        stats[key] = dict(n=len(reps), median=med, q1=q1, q3=q3, iqr=q3 - q1, min=min(reps), max=max(reps),
                          pool_start_seconds=spawn_s)
        print(f"pepper_fit_loop_20 with {a.procs} processes x {a.threads or 1} threads: {results[key]:.2f} s "
              f"(median of {len(reps)}; pool start {spawn_s:.2f} s, not included)", flush=True)
        names = []

    for name in names:
        entry = dict(requested_device=a.device, supported_devices=list(WORKLOAD_DEVICES[name]),
                     torch_num_threads=torch.get_num_threads(), stochastic=name in STOCHASTIC)
        if not supports_device(name, a.device):
            entry.update(effective_mode='unsupported', cuda_kernels_observed=False,
                         note=f'{name.split("_")[0]} has no {a.device} implementation; not timed on {a.device}')
            audit[name] = entry
            print(f'{name:32s} unsupported on {a.device} (CPU-only implementation) - skipped', flush=True)
            continue
        fn = WORKLOADS[name](device=a.device)
        if use_cuda:
            entry['gpu_before'] = _gpu_snapshot()
            if not a.no_verify:
                fn_cpu = WORKLOADS[name](device='cpu')
                entry.update(_verify_cuda(name, fn, fn_cpu, a.out))
            else:
                entry['effective_mode'] = 'cuda (unverified)'
        else:
            entry.update(effective_mode='cpu', cuda_kernels_observed=False)
        for _ in range(a.warmup):
            fn()
            if use_cuda:
                torch.cuda.synchronize()
        reps = []
        for _ in range(a.repeat):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            last = fn()
            if use_cuda:
                torch.cuda.synchronize()
            reps.append(time.perf_counter() - t0)
        if a.profile:
            pr = cProfile.Profile(); pr.enable(); fn(); pr.disable()
            if use_cuda:
                torch.cuda.synchronize()
            s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(25)
            prof_txt.append(f'=== {name} ===\n' + s.getvalue())
        if use_cuda:
            entry['gpu_after'] = _gpu_snapshot()
        # Use the same linear percentile estimator as the MATLAB results and
        # the figure generator. With five replicates, the default exclusive
        # statistics.quantiles estimator can pull a quartile toward an extreme.
        q1, med, q3 = [float(x) for x in np.percentile(reps, [25, 50, 75])]
        results[name] = med; replicates[name] = reps
        stats[name] = dict(n=len(reps), median=med, q1=q1, q3=q3, iqr=q3 - q1, min=min(reps), max=max(reps),
                           mean=statistics.fmean(reps), stdev=statistics.stdev(reps) if len(reps) > 1 else 0.0)
        audit[name] = entry
        if a.save_outputs and a.repeat:
            for k, arr in enumerate(_flatten_outputs(last)):
                outputs[f'{name}__{k}'] = arr
        mode = entry['effective_mode']
        print(f'{name:32s} {med:9.3f} s  (median of {len(reps)}, IQR {q1:.3f}-{q3:.3f}, mode={mode})', flush=True)

    meta['finished'] = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    if a.save_outputs and outputs:
        Path(a.save_outputs).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.save_outputs, **outputs)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(meta=meta, seconds=results, replicates=replicates, stats=stats, audit=audit),
                  open(a.out, 'w'), indent=2, default=str)
        if prof_txt:
            open(a.out + '.profile.txt', 'w').write('\n'.join(prof_txt))
    return 0


if __name__ == '__main__':
    sys.exit(main())
