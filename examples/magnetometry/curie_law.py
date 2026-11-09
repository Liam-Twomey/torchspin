"""Curie Law Magnetic Susceptibility
Verifies the Curie law chi_mol = C/T for S=1/2 by comparing curry() output
against the analytical Curie constant.

EasySpin equivalent: examples/magnetometry/curie_law.m
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchspin import SpinSystem
from torchspin.curry import curry
from torchspin.constants import BMAGN, BOLTZMANN


def run():
    g_val = 2.0
    S_val = 0.5

    Sys = SpinSystem(S=[S_val], g=[[g_val, g_val, g_val]])

    T_arr    = np.linspace(2.0, 300.0, 100)
    B_fields = np.array([0.1])   # small field (linear regime), mT

    mu_BM, chi_mol = curry(Sys, B_fields, T_arr, grid_size=15)

    # chi_mol shape: (nB, nT) in m³/mol; convert to cm³/mol
    chi_cm3 = chi_mol[0, :].numpy() * 1e6 / (4 * np.pi)   # m³/mol (SI) -> cm³/mol (CGS)

    # Analytical Curie constant (SI):
    #   C [m³·K/mol] = mu0 * N_A * mu_B^2 * g^2 * S(S+1) / (3 * k_B)
    mu0 = 4 * np.pi * 1e-7
    N_A = 6.02214076e23
    C_SI  = mu0 * N_A * BMAGN**2 * g_val**2 * S_val * (S_val + 1) / (3 * BOLTZMANN)
    C_cm3 = C_SI * 1e6 / (4 * np.pi)   # cm³·K/mol (CGS); 0.375 for S = 1/2, g = 2
    chi_curie = C_cm3 / T_arr

    chiT = chi_cm3 * T_arr

    return T_arr, chi_cm3, chi_curie, chiT, C_cm3


def _plot(T_arr, chi_cm3, chi_curie, chiT, C_cm3):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(T_arr, chi_cm3,   'b',   lw=2,   label='torchspin curry()')
    axes[0].plot(T_arr, chi_curie, 'r--', lw=1.5, label=f'Curie law (C={C_cm3:.4f})')
    axes[0].set_xlabel('Temperature (K)')
    axes[0].set_ylabel('$\\chi_{mol}$ (cm$^3$/mol)')
    axes[0].set_title('Magnetic susceptibility $\\chi$(T)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(T_arr, chiT, 'b', lw=2)
    axes[1].axhline(C_cm3, color='r', lw=1.5, ls='--', label=f'C = {C_cm3:.4f} cm$^3$·K/mol')
    axes[1].set_xlabel('Temperature (K)')
    axes[1].set_ylabel('$\\chi T$ (cm$^3$·K/mol)')
    axes[1].set_title('Curie product $\\chi T$ (should be constant)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Curie Law: S=1/2, g=2.0', fontsize=12)
    plt.tight_layout()
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    result = run()
    _plot(*result)
