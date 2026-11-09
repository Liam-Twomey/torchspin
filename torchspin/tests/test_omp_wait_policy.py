"""``import torchspin`` makes idle OpenMP threads sleep instead of spin-waiting.

Spinning OpenMP threads starved the Python thread and torchspin's worker threads, so
small-spectrum fits got slower as threads were added. The runtime reads
OMP_WAIT_POLICY once, when torch loads, so torchspin can set it only when it is
imported before torch. Each case runs in a fresh interpreter.
"""
import os
import subprocess
import sys


def _policy_after(code: str, env_extra=None) -> str:
    env = {k: v for k, v in os.environ.items() if k != 'OMP_WAIT_POLICY'}
    env.update(env_extra or {})
    out = subprocess.run([sys.executable, '-c', code + "\nimport os; print(os.environ.get('OMP_WAIT_POLICY'))"],
                         env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_import_sets_passive_when_torch_not_loaded():
    assert _policy_after('import torchspin') == 'PASSIVE'


def test_explicit_user_value_is_kept():
    assert _policy_after('import torchspin', {'OMP_WAIT_POLICY': 'ACTIVE'}) == 'ACTIVE'


def test_nothing_set_when_torch_came_first():
    # too late to affect the runtime, so torchspin must not pretend it did
    assert _policy_after('import torch, torchspin') == 'None'
