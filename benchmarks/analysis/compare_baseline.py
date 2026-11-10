"""Compare a benchmark campaign with a stored baseline and flag regressions.

A campaign is the JSON written by ``benchmarks/workloads/run_workloads.py --out``.
A baseline is a reduced copy of a campaign: per-workload median and IQR, the
torchspin commit, thread count and device. It holds no host name, paths or
affinity lists, so it is safe to keep in version control.

Save a baseline from a campaign::

    python benchmarks/analysis/compare_baseline.py save campaign.json \\
        --label cuda-workstation-cpu32 --out benchmarks/baselines/cuda-workstation-cpu32.json

Compare a new campaign with it::

    python benchmarks/analysis/compare_baseline.py compare new.json \\
        benchmarks/baselines/cuda-workstation-cpu32.json [--tolerance 0.20]

A workload counts as a regression when its median is more than ``tolerance``
slower than the baseline median *and* the two interquartile ranges do not
overlap. Both conditions are needed so that replicate noise is not reported as a
change. Improvements are reported the same way. The exit status is 1 if any
workload regressed (usable as a CI or campaign gate), and 0 otherwise.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path


def _summary(campaign: dict) -> dict:
    """Per-workload {median, q1, q3, n} from a campaign JSON."""
    stats = campaign.get('stats') or {}
    out = {}
    for name, sec in (campaign.get('seconds') or {}).items():
        st = stats.get(name, {})
        med = float(st.get('median', sec))
        out[name] = dict(median=med, q1=float(st.get('q1', med)), q3=float(st.get('q3', med)),
                         n=int(st.get('n', 1)))
    return out


def save(campaign_path: str, label: str, out: str) -> int:
    campaign = json.loads(Path(campaign_path).read_text())
    meta = campaign.get('meta', {})
    baseline = dict(
        label=label,
        created=_dt.date.today().isoformat(),
        torchspin_commit=meta.get('torchspin_commit'),
        torch=meta.get('torch'),
        device=meta.get('device'),
        threads=meta.get('threads'),
        protocol=meta.get('protocol'),
        workloads=_summary(campaign),
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(baseline, indent=2) + '\n')
    print(f'baseline "{label}" with {len(baseline["workloads"])} workloads -> {out}')
    return 0


def compare(campaign_path: str, baseline_path: str, tolerance: float) -> int:
    new = _summary(json.loads(Path(campaign_path).read_text()))
    base_doc = json.loads(Path(baseline_path).read_text())
    base = base_doc['workloads']
    rows, regressed = [], []
    for name in sorted(set(new) | set(base)):
        if name not in new or name not in base:
            rows.append((name, base.get(name, {}).get('median'), new.get(name, {}).get('median'), None,
                         'missing in ' + ('campaign' if name not in new else 'baseline')))
            continue
        b, n = base[name], new[name]
        ratio = n['median'] / b['median'] if b['median'] > 0 else float('inf')
        disjoint = n['q1'] > b['q3'] or n['q3'] < b['q1']
        if ratio > 1 + tolerance and disjoint:
            verdict = 'REGRESSION'
            regressed.append(name)
        elif ratio < 1 / (1 + tolerance) and disjoint:
            verdict = 'faster'
        else:
            verdict = 'unchanged'
        rows.append((name, b['median'], n['median'], ratio, verdict))

    print(f'Baseline: {base_doc.get("label")} (commit {str(base_doc.get("torchspin_commit"))[:8]}, '
          f'{base_doc.get("device")}, {base_doc.get("threads")} threads); tolerance {tolerance:.0%}\n')
    print('| Workload | baseline median (s) | new median (s) | new / baseline | verdict |')
    print('|---|---:|---:|---:|---|')
    for name, bm, nm, r, v in rows:
        f = (lambda x: '–' if x is None else f'{x:.4g}')
        print(f'| `{name}` | {f(bm)} | {f(nm)} | {"–" if r is None else f"{r:.2f}"} | {v} |')
    if regressed:
        print(f'\n{len(regressed)} regression(s): {", ".join(regressed)}')
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('save', help='reduce a campaign JSON to a baseline')
    s.add_argument('campaign')
    s.add_argument('--label', required=True, help='generic host-class label, e.g. cuda-workstation-cpu32')
    s.add_argument('--out', required=True)
    c = sub.add_parser('compare', help='compare a campaign with a baseline')
    c.add_argument('campaign')
    c.add_argument('baseline')
    c.add_argument('--tolerance', type=float, default=0.20)
    a = ap.parse_args(argv)
    if a.cmd == 'save':
        return save(a.campaign, a.label, a.out)
    return compare(a.campaign, a.baseline, a.tolerance)


if __name__ == '__main__':
    sys.exit(main())
