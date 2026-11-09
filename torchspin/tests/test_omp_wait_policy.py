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


def test_warns_only_where_it_matters():
    from torchspin._omp import should_warn
    assert should_warn(True, None, 'linux', 8)
    assert not should_warn(False, None, 'linux', 8)        # torchspin came first: it set the policy
    assert not should_warn(True, 'ACTIVE', 'linux', 8)     # the user chose
    assert not should_warn(True, None, 'darwin', 8)        # macOS torch does not spin this way
    assert not should_warn(True, None, 'linux', 1)         # one thread: nothing to spin


def test_torch_first_warns_on_linux_only():
    import subprocess
    env = {k: v for k, v in os.environ.items() if k not in ('OMP_WAIT_POLICY', 'OMP_NUM_THREADS')}
    out = subprocess.run([sys.executable, '-W', 'always', '-c',
                          'import torch; torch.set_num_threads(4); import torchspin'],
                         env=env, capture_output=True, text=True, check=True)
    assert ('OMP_WAIT_POLICY' in out.stderr) == sys.platform.startswith('linux'), out.stderr


def _child_policy(_):
    import os
    return os.environ.get('OMP_WAIT_POLICY')


def test_spawned_workers_get_passive_even_if_parent_has_none():
    """Workers load their own OpenMP runtime, so they get the policy whatever the parent's import order."""
    import multiprocessing as mp
    from torchspin._omp import passive_for_children
    saved = os.environ.pop('OMP_WAIT_POLICY', None)
    try:
        with passive_for_children():
            pool = mp.get_context('spawn').Pool(2)
        assert 'OMP_WAIT_POLICY' not in os.environ          # parent restored
        with pool:
            assert pool.map(_child_policy, range(2)) == ['PASSIVE', 'PASSIVE']
    finally:
        if saved is not None:
            os.environ['OMP_WAIT_POLICY'] = saved
