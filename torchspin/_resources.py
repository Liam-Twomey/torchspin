"""How many worker processes this process can afford: cores and memory.

``n_workers='auto'`` used to mean one worker per core on the machine. Each worker
is a separate Python process with its own torch (about 0.3-0.4 GB before the model
allocates anything), so on a many-core workstation or cluster node 'auto' could ask
for tens of GB, and inside a SLURM allocation it ignored the cores actually granted.
"""
from __future__ import annotations

import os
import sys

#: Memory planned per worker process (torch + torchspin + a small model).
WORKER_BYTES = 512 * 2**20
#: Fraction of the available memory the workers may take.
MEMORY_FRACTION = 0.5
#: Beyond this, more workers did not make fits faster in our measurements.
MAX_AUTO_WORKERS = 32


def cpu_budget() -> int:
    """Cores this process may use: CPU affinity (Linux), then SLURM, then the machine."""
    n = None
    if hasattr(os, 'sched_getaffinity'):
        try:
            n = len(os.sched_getaffinity(0))
        except OSError:
            n = None
    slurm = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm and slurm.isdigit():
        n = min(n, int(slurm)) if n else int(slurm)
    return max(1, n or os.cpu_count() or 1)


def _cgroup_limit() -> int | None:
    """Bytes left under the tightest cgroup v2 memory limit above this process.

    SLURM and container runtimes set the limit on a parent of the process's own
    cgroup (e.g. the job, not the step's task), so every ancestor is checked.
    """
    try:
        with open('/proc/self/cgroup') as f:
            path = next((line.split('::', 1)[1].strip() for line in f if line.startswith('0::')), None)
    except OSError:
        return None
    if path is None:
        return None
    best = None
    parts = [p for p in path.split('/') if p]
    for k in range(len(parts), -1, -1):
        base = '/sys/fs/cgroup/' + '/'.join(parts[:k])
        try:
            with open(base + '/memory.max') as f:
                limit = f.read().strip()
            if limit == 'max':
                continue
            with open(base + '/memory.current') as f:
                used = int(f.read().strip())
        except (OSError, ValueError):
            continue
        left = max(0, int(limit) - used)
        best = left if best is None else min(best, left)
    return best


def _slurm_limit() -> int | None:
    """The job's memory request from SLURM's environment, in bytes (an upper bound)."""
    try:
        if os.environ.get('SLURM_MEM_PER_NODE'):
            return int(os.environ['SLURM_MEM_PER_NODE']) * 2**20
        if os.environ.get('SLURM_MEM_PER_CPU'):
            cpus = int(os.environ.get('SLURM_CPUS_ON_NODE') or os.environ.get('SLURM_CPUS_PER_TASK') or 1)
            return int(os.environ['SLURM_MEM_PER_CPU']) * cpus * 2**20
    except ValueError:
        pass
    return None


def available_memory() -> int | None:
    """Bytes of memory new processes can use, or None if unknown."""
    avail = None
    if sys.platform.startswith('linux'):
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        avail = int(line.split()[1]) * 1024
                        break
        except (OSError, ValueError):
            pass
        cg = _cgroup_limit()
        if cg is not None:
            avail = cg if avail is None else min(avail, cg)
    slurm = _slurm_limit()
    if slurm is not None:
        avail = slurm if avail is None else min(avail, slurm)
    if avail is None:
        try:
            import psutil  # optional
            avail = int(psutil.virtual_memory().available)
        except Exception:
            try:  # macOS and other POSIX: physical memory as an upper bound
                avail = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
            except (ValueError, OSError, AttributeError):
                avail = None
    return avail


def auto_workers(n_tasks: int) -> int:
    """Worker count for ``n_workers='auto'``: cores, memory and task count, at most 32."""
    n = min(cpu_budget(), MAX_AUTO_WORKERS, max(1, int(n_tasks)))
    mem = available_memory()
    if mem is not None:
        n = min(n, max(1, int(mem * MEMORY_FRACTION // WORKER_BYTES)))
    return max(1, n)
