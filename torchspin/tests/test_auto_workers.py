"""n_workers='auto' sizes the worker pool by allocated cores and by memory.

Each worker is a separate process with its own torch, so 'auto' must not start one
per core of a large machine, and inside a SLURM allocation it must count the cores
that were granted, not the whole node.
"""
import importlib

import pytest

R = importlib.import_module('torchspin._resources')
E = importlib.import_module('torchspin.esfit')


@pytest.fixture
def machine(monkeypatch):
    """Fake a machine: cores (affinity), SLURM grant and available memory."""
    def make(cores, mem_gb, slurm=None):
        monkeypatch.setattr(R.os, 'sched_getaffinity', lambda pid: set(range(cores)), raising=False)
        monkeypatch.setattr(R.os, 'cpu_count', lambda: cores)
        if slurm is None:
            monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
        else:
            monkeypatch.setenv('SLURM_CPUS_PER_TASK', str(slurm))
        monkeypatch.setattr(R, 'available_memory', lambda: None if mem_gb is None else int(mem_gb * 2**30))
    return make


def test_laptop_uses_its_cores(machine):
    machine(cores=10, mem_gb=12)
    assert E._resolve_workers('auto', 100) == 10


def test_big_node_is_capped_at_32(machine):
    machine(cores=224, mem_gb=900)
    assert E._resolve_workers('auto', 1000) == 32


def test_slurm_grant_not_the_whole_node(machine):
    machine(cores=224, mem_gb=900, slurm=16)
    assert E._resolve_workers('auto', 1000) == 16


def test_memory_caps_workers(machine):
    # 4 GB free: half of it at 0.5 GB per worker = 4 workers, despite 64 cores
    machine(cores=64, mem_gb=4)
    assert E._resolve_workers('auto', 1000) == 4


def test_always_at_least_one_and_never_more_than_tasks(machine):
    machine(cores=64, mem_gb=0.1)
    assert E._resolve_workers('auto', 1000) == 1
    machine(cores=64, mem_gb=64)
    assert E._resolve_workers('auto', 3) == 3


def test_unknown_memory_falls_back_to_cores(machine):
    machine(cores=8, mem_gb=None)
    assert E._resolve_workers('auto', 100) == 8


def test_explicit_count_is_respected(machine):
    machine(cores=4, mem_gb=1)
    assert E._resolve_workers(12, 100) == 12


def test_available_memory_is_plausible():
    mem = R.available_memory()
    assert mem is None or 2**28 < mem < 2**44


def test_slurm_memory_request_caps_available(monkeypatch):
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '2048')
    assert R.available_memory() <= 2 * 2**30
    monkeypatch.delenv('SLURM_MEM_PER_NODE')
    monkeypatch.setenv('SLURM_MEM_PER_CPU', '1000')
    monkeypatch.setenv('SLURM_CPUS_ON_NODE', '3')
    assert R.available_memory() <= 3000 * 2**20
