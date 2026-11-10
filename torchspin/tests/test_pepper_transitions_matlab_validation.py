"""Options.Transitions against EasySpin's Opt.Transitions (tests/data/ref_pepper_transitions.mat,
generate_pepper_transitions_refs.m): S=3/2 central / outer / all transitions (powder, strain,
crystal), the S=1 half-field line ('all' and [1 3]) and a frequency sweep with one pair."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from torchspin.tests.test_pepper_matlab_validation_ext2 import (
    sys_from_mat, exp_from_mat, opt_from_mat, cosine, _get)
from torchspin.pepper import pepper
from dataclasses import replace

REF = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'ref_pepper_transitions.mat'


def _load():
    m = loadmat(REF, squeeze_me=True, struct_as_record=False)
    return {str(c.name): c for c in np.atleast_1d(m['cases'])}


@pytest.fixture(scope='module')
def cases():
    if not REF.exists():
        pytest.skip(f'reference file missing: {REF}')
    return _load()


def _case_names():
    return list(_load().keys()) if REF.exists() else []


def _transitions(O):
    t = _get(O, 'Transitions')
    if t is None:
        return None
    if isinstance(t, str):
        return t
    a = np.atleast_2d(np.asarray(t, dtype=int))
    return [tuple(int(v) for v in row) for row in a]


@pytest.mark.parametrize('name', _case_names())
def test_pepper_transitions_vs_matlab(cases, name):
    c = cases[name]
    exp = exp_from_mat(c.Exp, c.Sys, x_axis=np.asarray(c.x, dtype=float))
    opt = replace(opt_from_mat(c.Opt), Transitions=_transitions(c.Opt))
    x, spc = pepper(sys_from_mat(c.Sys), exp, opt)
    x = x.detach().cpu().numpy(); spc = spc.detach().cpu().numpy()
    ref = np.asarray(c.spc, dtype=float); ref_x = np.asarray(c.x, dtype=float)
    assert np.allclose(x, ref_x, rtol=1e-9, atol=1e-9 * abs(ref_x).max()), 'sweep axis differs'
    cs = cosine(spc, ref)
    amp = abs(spc).max() / abs(ref).max()
    print(f'{name}: cosine {cs:.6f}, amplitude ratio {amp:.4f}')
    assert cs > 0.999, (name, cs)
    assert amp == pytest.approx(1.0, abs=0.02), (name, amp)
