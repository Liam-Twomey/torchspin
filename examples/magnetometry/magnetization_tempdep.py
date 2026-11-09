"""Temperature-Dependent Effective Magnetic Moment
Computes mu_eff = sqrt(8*chi*T) vs temperature for S=1/2, 1, 3/2 systems.
Used in SQUID magnetometry to diagnose spin states; plateau = g*sqrt(S(S+1)).

EasySpin equivalent: examples/magnetometry/magnetization_tempdep.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem
from torchspin.curry import curry


def run():
    B_field = np.array([10.0])       # 10 mT (linear regime)
    T_arr   = np.linspace(2.0, 300.0, 100)

    # (system, label, g*sqrt(S*(S+1)) analytical mu_eff)
    systems = [
        (SpinSystem(S=[0.5], g=[[2.0, 2.0, 2.0]]), 'S=1/2', 2.0 * np.sqrt(0.5 * 1.5)),
        (SpinSystem(S=[1.0], g=[[2.0, 2.0, 2.0]]), 'S=1',   2.0 * np.sqrt(1.0 * 2.0)),
        (SpinSystem(S=[1.5], g=[[2.0, 2.0, 2.0]]), 'S=3/2', 2.0 * np.sqrt(1.5 * 2.5)),
    ]

    results = []
    for Sys, label, mu_theory in systems:
        _, chi_mol = curry(Sys, B_field, T_arr, grid_size=10)
        chi_cm3 = chi_mol[0, :].numpy() * 1e6 / (4 * np.pi)   # m³/mol (SI) -> cm³/mol (CGS)
        chiT    = chi_cm3 * T_arr
        mu_eff  = np.sqrt(8.0 * chiT)
        results.append((mu_eff, label, mu_theory))

    return T_arr, results


def _plot(T_arr, results):
    colors = ['b', 'r', 'g']
    fig, ax = plt.subplots(figsize=(8, 5))

    for (mu_eff, label, mu_theory), col in zip(results, colors):
        ax.plot(T_arr, mu_eff, color=col, lw=2,
                label=f'{label}  (theory = {mu_theory:.3f} $\\mu_B$)')
        ax.axhline(mu_theory, color=col, lw=0.8, ls='--')

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('$\\mu_{eff}$ ($\\mu_B$)')
    ax.set_title('Effective magnetic moment $\\mu_{eff}$(T)')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
