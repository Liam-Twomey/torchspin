"""torchspin — PyTorch EPR/ESR spin Hamiltonian engine.

A Python/PyTorch port of EasySpin's core physics layer.
All energies are in **MHz**; magnetic fields in **mT**; Euler angles in **radians**.

Quick start::

    import torch
    from torchspin import SpinSystem, ham

    # S=1/2 nitroxide-like spin system
    sys = SpinSystem(
        S=[0.5],
        g=[[2.009, 2.006, 2.002]],
        Nucs=['14N'],
        A=[[10.0, 10.0, 95.0]],   # MHz
    )

    B0 = [0.0, 0.0, 340.0]         # mT, along z
    H = ham(sys, B0)
    energies = torch.linalg.eigvalsh(H)   # MHz
"""
from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PkgNotFound
try:
    __version__ = _pkg_version("torchspin")
except _PkgNotFound:
    # Editable install during development, or version metadata unavailable.
    __version__ = "0.6.1+dev"

from torchspin._compile import maybe_compile, set_compile_enabled, is_compile_available
from torchspin.autograd import differentiable_spectrum
from torchspin.batch import batch_pepper, batch_simulate
from torchspin.cardamom import cardamom, CardamomPar, CardamomOptions, MDInput
from torchspin.ctafft import ctafft
from torchspin.ewrls import ewrls
from torchspin.lpsvd import lpsvd, LPSVDResult
from torchspin.rapidscan2spc import rapidscan2spc
from torchspin.stochtraj_diffusion import stochtraj_diffusion, DiffusionPar
from torchspin.stochtraj_jump import stochtraj_jump, JumpPar
from torchspin.autoguess import (estimate_parameters, field_from_g, g_from_field,
                                  print_parameter_report)
from torchspin.lineshape import (gaussian, lorentzian, voigtian, apowin, addnoise, deriv, lshape)
from torchspin.dataproc import basecorr, fieldmod, rescaledata
from torchspin.utils import (datasmooth, equivcouple, equivsplit,
                              hilberttrans, larmorfrq, rcfilt, unitconvert,
                              mhz2mt, mt2mhz, hsdim, eprconvert)
from torchspin.nucdata import nucabund, nucspin, nucgval, nucqmom
from torchspin.blochsteady import blochsteady, BlochOptions
from torchspin.chili import chili, ChiliOptions
from torchspin.endorfrq import endorfrq
from torchspin.salt import salt
from torchspin.constants import (BMAGN, BOLTZMANN, CLIGHT, EVOLT, GFREE, NMAGN, PLANCK,
                                   HBAR, AMU, ECHARGE, EMASS, PMASS, NMASS, EPS0, MU0,
                                   FARADAY, AVOGADRO, BOHRRAD, HARTREE, RYDBERG, GAMMAE, GAMMAN,
                                   ANGSTROM, DEGREE, BARN, MOLGAS)
from torchspin.convspec import convspec
from torchspin.eprload import eprload, eprload_info
from torchspin.eprsave import eprsave, eprsave_info
from torchspin.esfit import esfit, FitOptions, FitResult
from torchspin.experiment import Experiment, Options
from torchspin.curry import curry
from torchspin.fastmotion import fastmotion
from torchspin.garlic import garlic
from torchspin.ham import ham
from torchspin.levels import levels
from torchspin.resfreqs_matrix import resfreqs_matrix
from torchspin.ham_cf import ham_cf
from torchspin.ham_ee import ham_ee
from torchspin.ham_ez import ham_ez
from torchspin.ham_ezho import ham_ezho
from torchspin.ham_hf import ham_hf
from torchspin.ham_nn import ham_nn
from torchspin.ham_nq import ham_nq
from torchspin.ham_nz import ham_nz
from torchspin.ham_oz import ham_oz
from torchspin.ham_so import ham_so
from torchspin.ham_zf import ham_zf, zfsframes
from torchspin.hamsymm import hamsymm
from torchspin.makespec import makespec
from torchspin.pepper import pepper
from torchspin.gridcheck import GridConvergence, grid_convergence
from torchspin.pepper_autograd import pepper_autograd
from torchspin.rotations import erot
from torchspin.rotutils import (eulang, euler2quat, quat2euler, quat2rotmat,
                                 rotmat2quat, rotaxi2mat, rotmat2axi,
                                 quatinv, quatmult, quatvecmult,
                                 vec2ang, ang2vec,
                                 tensor_cart2sph, tensor_sph2cart, rotateframe)
from torchspin.diptensor import diptensor
from torchspin.spinladder import spinladder, SpinManifold
from torchspin.orisel import orisel, OriselOptions
from torchspin.spinops import sop, commute
from torchspin.sphgrid import sphgrid, sphrand
from torchspin.angmom import (wigner3j, wigner6j, wignerd, spherharm, clebschgordan, cgmatrix,
                               isto, isto2stev)
from torchspin.endorfrq_perturb import endorfrq_perturb
from torchspin.resfreqs_perturb import resfreqs_perturb
from torchspin.resfields_eig import resfields_eig, EigOptions
from torchspin.spidyan import spidyan, SpidyanOptions, SpidyanInfo
from torchspin.mdload import mdload, MDData, MDInfo, MDOptions
from torchspin.mdhmm import mdhmm, HMMResult, HMMOptions
from torchspin.mdtraj2oripot import mdtraj2oripot
from torchspin.orca2torchspin import orca2torchspin, OrcaData
from torchspin.mlpsvd import mlpsvd, MLPSVDResult
from torchspin.oripotentialplot import oripotentialplot
from torchspin.saffron import saffron, PulseExperiment, SaffronOptions
from torchspin.signalprocessing import signalprocessing
from torchspin.saffron_pathways import find_refocusing_pathways
from torchspin.spinsystem import SpinSystem, nucspinrmv, nucspinkeep, nucspinadd, isotopologues, spinvec
from torchspin.sigeq import sigeq
from torchspin.fdaxis import fdaxis
from torchspin.plegendre import plegendre
from torchspin.propint import propint
from torchspin.resonatorprofile import resonatorprofile
from torchspin.rfmixer import rfmixer
from torchspin.transmitter import transmitter
from torchspin.pulse import pulse
from torchspin.exciteprofile import exciteprofile
from torchspin.resonator import resonator
from torchspin.photoselect import photoselect
from torchspin.levelsplot import levelsplot, LevelsPlotOptions
from torchspin.stackplot import stackplot
from torchspin.dipkernel import dipkernel
from torchspin.dipbackground import dipbackground
from torchspin.exponfit import exponfit
from torchspin.nucfrq2d import nucfrq2d
from torchspin.evolve import evolve

__all__ = [
    # Version
    '__version__',
    # Differentiable simulator
    'differentiable_spectrum',
    'pepper_autograd',
    # Core
    'SpinSystem',
    'nucspinrmv',
    'nucspinkeep',
    'nucspinadd',
    'isotopologues',
    'spinvec',
    'sop',
    'erot',
    # Hamiltonians
    'ham',
    'ham_cf',
    'ham_ez',
    'ham_ezho',
    'ham_hf',
    'ham_ee',
    'ham_nz',
    'ham_nq',
    'ham_nn',
    'ham_oz',
    'ham_so',
    'ham_zf',
    # Spectrum simulation
    'blochsteady',
    'BlochOptions',
    'pepper',
    'grid_convergence',
    'GridConvergence',
    'garlic',
    'fastmotion',
    'chili',
    'ChiliOptions',
    'endorfrq',
    'salt',
    'levels',
    'resfreqs_matrix',
    'hamsymm',
    'Experiment',
    'Options',
    'sphgrid',
    'sphrand',
    'makespec',
    'convspec',
    # Magnetometry
    'curry',
    # Data I/O (save)
    'eprsave',
    'eprsave_info',
    # Signal processing utilities
    'larmorfrq',
    'equivsplit',
    'equivcouple',
    'datasmooth',
    'rcfilt',
    'hilberttrans',
    'unitconvert',
    'mhz2mt',
    'mt2mhz',
    'hsdim',
    # Nuclear data
    'nucabund',
    'nucspin',
    'nucgval',
    'nucqmom',
    # Lineshape functions
    'gaussian',
    'lorentzian',
    'voigtian',
    'apowin',
    'addnoise',
    'deriv',
    'lshape',
    # Data processing
    'basecorr',
    'fieldmod',
    'rescaledata',
    # Angular momentum functions
    'wigner3j',
    'wigner6j',
    'wignerd',
    'spherharm',
    'clebschgordan',
    'cgmatrix',
    'isto',
    'isto2stev',
    # Perturbation-theory resonance/ENDOR
    'endorfrq_perturb',
    'resfreqs_perturb',
    'resfields_eig',
    'EigOptions',
    # Spin dynamics (spidyan)
    'spidyan',
    'SpidyanOptions',
    'SpidyanInfo',
    # MD trajectory analysis (cardamom Phase 5)
    'mdload',
    'MDData',
    'MDInfo',
    'MDOptions',
    'mdhmm',
    'HMMResult',
    'HMMOptions',
    'mdtraj2oripot',
    # Rotation utilities
    'eulang',
    'euler2quat',
    'quat2euler',
    'quat2rotmat',
    'rotmat2quat',
    'rotaxi2mat',
    'rotmat2axi',
    'quatinv',
    'quatmult',
    'quatvecmult',
    'vec2ang',
    'ang2vec',
    'tensor_cart2sph',
    'tensor_sph2cart',
    'rotateframe',
    # Spin operators
    'commute',
    # Dipolar coupling
    'diptensor',
    # ZFS analysis
    'zfsframes',
    # Spin ladder
    'spinladder',
    'SpinManifold',
    # Orientation selectivity
    'orisel',
    'OriselOptions',
    # Data I/O
    'eprload',
    'eprload_info',
    'orca2torchspin',
    'OrcaData',
    # Pulse EPR
    'saffron',
    'PulseExperiment',
    'SaffronOptions',
    'find_refocusing_pathways',
    # Parameter estimation
    'estimate_parameters',
    'field_from_g',
    'g_from_field',
    'print_parameter_report',
    # Fitting
    'esfit',
    'FitOptions',
    'FitResult',
    # Batch simulation
    'batch_pepper',
    'batch_simulate',
    # Trajectory-based simulation (cardamom)
    'cardamom',
    'CardamomPar',
    'CardamomOptions',
    'MDInput',
    'stochtraj_diffusion',
    'DiffusionPar',
    'stochtraj_jump',
    'JumpPar',
    # Compilation
    'maybe_compile',
    'set_compile_enabled',
    'is_compile_available',
    # Constants
    'BMAGN',
    'PLANCK',
    'GFREE',
    'NMAGN',
    'BOLTZMANN',
    'CLIGHT',
    'EVOLT',
    'HBAR',
    'AMU',
    'ECHARGE',
    'EMASS',
    'PMASS',
    'NMASS',
    'EPS0',
    'MU0',
    'FARADAY',
    'AVOGADRO',
    'BOHRRAD',
    'HARTREE',
    'RYDBERG',
    'GAMMAE',
    'GAMMAN',
    'ANGSTROM',
    'DEGREE',
    'BARN',
    'MOLGAS',
    # EPR conversion
    'eprconvert',
    # Tier 1 utilities
    'sigeq',
    'fdaxis',
    'plegendre',
    'photoselect',
    'levelsplot',
    'LevelsPlotOptions',
    'stackplot',
    # Tier 2 utilities
    'dipkernel',
    'dipbackground',
    'exponfit',
    'nucfrq2d',
    'evolve',
    # Tier 3b — Signal Processing
    'ctafft',
    'ewrls',
    'lpsvd',
    'LPSVDResult',
    'mlpsvd',
    'MLPSVDResult',
    'rapidscan2spc',
    'signalprocessing',
    # Visualization
    'oripotentialplot',
    # Tier 3c — Higher-order Zeeman
    'ham_ezho',
    # Tier 3d-i — Pulse EPR primitives
    'resonatorprofile',
    'transmitter',
    'rfmixer',
    'propint',
    # Tier 3d-ii — Pulse waveforms
    'pulse',
    'exciteprofile',
    'resonator',
]
