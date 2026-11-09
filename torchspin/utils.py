"""Signal-processing and EPR utility functions.

Ports of EasySpin's ``larmorfrq``, ``equivsplit``, ``equivcouple``,
``datasmooth``, ``rcfilt``, ``hilberttrans``, ``unitconvert``,
``mhz2mt``, ``mt2mhz``, and ``hsdim``.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from scipy.signal import lfilter

from torchspin.constants import BMAGN, BOLTZMANN, CLIGHT, EVOLT, GFREE, NMAGN, PLANCK
from torchspin.nucdata import nucdata


# ---------------------------------------------------------------------------
# mhz2mt / mt2mhz — field ↔ frequency conversion
# ---------------------------------------------------------------------------

def mhz2mt(
    mhz: Union[float, "array-like"],
    g: float = GFREE,
) -> np.ndarray:
    """Convert frequency in MHz to resonance field in mT.

    Uses the resonance condition  ``B [mT] = hν / (g μ_B)`` with
    the sign convention matching EasySpin: ``g`` is positive.

    Parameters
    ----------
    mhz:
        Frequency in MHz (scalar or array).
    g:
        g-factor.  Defaults to the free-electron g-factor.

    Returns
    -------
    B : ndarray
        Resonance field in mT.

    Examples
    --------
    >>> mhz2mt(9500.0)           # X-band, free electron ≈ 338.9 mT
    >>> mhz2mt(9500.0, g=2.0)
    """
    mhz = np.asarray(mhz, dtype=float)
    # B [mT] = hν [J] / (g μ_B [J/T]) * 1e-3
    return mhz * 1e6 * PLANCK / (g * BMAGN) * 1e3


def mt2mhz(
    mt: Union[float, "array-like"],
    g: float = GFREE,
) -> np.ndarray:
    """Convert resonance field in mT to frequency in MHz.

    Parameters
    ----------
    mt:
        Magnetic field in mT (scalar or array).
    g:
        g-factor.  Defaults to the free-electron g-factor.

    Returns
    -------
    freq : ndarray
        Resonance frequency in MHz.

    Examples
    --------
    >>> mt2mhz(340.0)            # free electron at 340 mT ≈ 9527 MHz
    >>> mt2mhz(340.0, g=2.0)
    """
    mt = np.asarray(mt, dtype=float)
    # ν [MHz] = g μ_B B [J] / h / 1e6
    return mt * 1e-3 * g * BMAGN / PLANCK / 1e6


# ---------------------------------------------------------------------------
# hsdim — Hilbert space dimension
# ---------------------------------------------------------------------------

def hsdim(sys_or_spins) -> int:
    """Return the Hilbert space dimension of a spin system.

    Parameters
    ----------
    sys_or_spins:
        Either a :class:`~torchspin.SpinSystem` instance, or a list/array
        of spin quantum numbers (e.g. ``[0.5, 1.0]``).

    Returns
    -------
    n : int
        ``prod(2*J + 1)`` over all spins.

    Examples
    --------
    >>> from torchspin import SpinSystem
    >>> hsdim(SpinSystem(S=[0.5], Nucs=['14N']))   # 2 × 3 = 6
    >>> hsdim([0.5, 1.0])                          # 2 × 3 = 6
    >>> hsdim(0.5)                                 # 2
    """
    # SpinSystem duck-typing: look for Spins attribute
    if hasattr(sys_or_spins, 'Spins'):
        spins = list(sys_or_spins.Spins)
    elif hasattr(sys_or_spins, '__iter__'):
        spins = list(sys_or_spins)
    else:
        spins = [sys_or_spins]
    n = 1
    for j in spins:
        n *= int(round(2 * j + 1))
    return n


# ---------------------------------------------------------------------------
# larmorfrq — Larmor frequency of nuclear spins
# ---------------------------------------------------------------------------

def larmorfrq(
    Nucs: Union[str, list],
    Fields: Union[float, "array-like"],
) -> np.ndarray:
    """Compute nuclear Larmor frequencies (MHz) at given field(s) (mT).

    Parameters
    ----------
    Nucs:
        Nuclear isotope symbol(s).  Either a string ``'14N'``, a
        comma-separated string ``'1H,14N'``, or a list ``['1H', '14N']``.
    Fields:
        Magnetic field(s) in mT.  Scalar or array.

    Returns
    -------
    freqs : ndarray, shape (nFields, nNuclei)
        Larmor frequencies in MHz.  Positive (sign convention: absolute value
        of g_n used, same as EasySpin).

    Examples
    --------
    >>> larmorfrq('1H', 340.0)    # 1H Larmor at 340 mT ≈ 14.47 MHz
    >>> larmorfrq('1H,14N', [300, 350])
    """
    # Parse nucleus list
    if isinstance(Nucs, list):
        nuc_list = Nucs
    else:
        nuc_list = [s.strip() for s in Nucs.split(',')]

    Fields = np.atleast_1d(np.asarray(Fields, dtype=float)).ravel()

    # prefac = |gn| * nmagn / (1e3 * planck * 1e6)  [MHz/mT] per unit gn
    # nmagn/planck = 7.62259 MHz/T, /1e3 → MHz/mT, *gn → actual Larmor
    prefac_per_gn = NMAGN / PLANCK / 1e3 / 1e6   # MHz/mT for gn=1

    freqs = np.zeros((len(Fields), len(nuc_list)), dtype=float)
    for j, nuc in enumerate(nuc_list):
        # nucdata returns (spins, gn, quad, abundance); gn is index 1
        gn = abs(float(nucdata(nuc)[1]))
        freqs[:, j] = gn * prefac_per_gn * Fields

    if freqs.shape[0] == 1:
        freqs = freqs[0]   # scalar-field case: shape (nNuclei,)
    return freqs


# ---------------------------------------------------------------------------
# equivsplit — EPR splitting pattern for n equivalent spins-I
# ---------------------------------------------------------------------------

def equivsplit(I: float, n: int) -> np.ndarray:
    """Return the EPR line intensity pattern for n equivalent spins-I.

    Uses the generalized Pascal's triangle (iterative convolution).

    Parameters
    ----------
    I:
        Nuclear spin quantum number (positive half-integer).
    n:
        Number of equivalent nuclei (positive integer).

    Returns
    -------
    intensities : ndarray, shape (2*n*I + 1,)
        Relative line intensities (first-order perturbation theory).

    Examples
    --------
    >>> equivsplit(0.5, 3)   # 3 protons → [1, 3, 3, 1]
    >>> equivsplit(1, 2)     # 2 × 14N → [1, 2, 3, 2, 1]
    """
    if not (n >= 1 and int(n) == n):
        raise ValueError("equivsplit: n must be a positive integer.")
    if not (I > 0 and (2 * I) == int(2 * I)):
        raise ValueError("equivsplit: I must be a positive multiple of 1/2.")

    dim = int(round(2 * I + 1))
    E = np.ones(dim, dtype=float)
    pattern = E.copy()
    for _ in range(1, n):
        pattern = np.convolve(pattern, E)
    return pattern


# ---------------------------------------------------------------------------
# equivcouple — Clebsch-Gordan decomposition of n equivalent spins-I
# ---------------------------------------------------------------------------

def equivcouple(I: float, n: int):
    """Decompose n equivalent spins-I into independent effective spins.

    Implements the Clebsch-Gordan reduction of the n-fold tensor product
    representation of spin-I.

    Parameters
    ----------
    I:
        Nuclear spin quantum number (positive half-integer).
    n:
        Number of equivalent nuclei (positive integer).

    Returns
    -------
    F : ndarray
        Effective spin quantum numbers in decreasing order.
    N : ndarray
        Multiplicities (number of times each effective spin appears).

    Examples
    --------
    >>> F, N = equivcouple(0.5, 5)
    >>> print(F)  # [2.5, 1.5, 0.5]
    >>> print(N)  # [1, 4, 5]
    """
    if n == 1:
        return np.array([I], dtype=float), np.array([1], dtype=int)

    dim = int(round(2 * I + 1))
    E = np.ones(dim, dtype=float)
    row_vec = np.array([1.0])
    for _ in range(n):
        row_vec = np.convolve(row_vec, E)

    row_vec_ext = np.concatenate([[0.0], row_vec])
    d_row = np.diff(row_vec_ext)

    F_max = I * n
    F = np.arange(F_max, -0.5, -1.0)   # I*n, I*n-1, ..., 0
    N = d_row[:len(F)].astype(int)

    # Remove zero entries
    mask = N > 0
    return F[mask], N[mask]


# ---------------------------------------------------------------------------
# datasmooth — Moving average smoothing (flat, binomial, Savitzky-Golay)
# ---------------------------------------------------------------------------

def datasmooth(
    y: "array-like",
    m: int,
    method: str = 'binom',
    poly_order: int = 2,
    deriv: int = 0,
) -> np.ndarray:
    """Smooth a 1D signal using a moving-window filter.

    Parameters
    ----------
    y:
        Input signal (1D array or 2D column array).
    m:
        Half-width of the filter window.  The full window is 2*m+1 points.
    method:
        ``'binom'`` (default) — binomial-weighted moving average.
        ``'flat'`` — unweighted (uniform) moving average.
        ``'savgol'`` — Savitzky-Golay polynomial filter.
    poly_order:
        Polynomial order for Savitzky-Golay (default 2).
    deriv:
        Derivative order for Savitzky-Golay (default 0 = smoothing only).

    Returns
    -------
    y_smooth : ndarray, same shape as input *y*.

    Examples
    --------
    >>> x = np.linspace(0, 2*np.pi, 200)
    >>> y = np.sin(x) + 0.2*np.random.randn(200)
    >>> y_s = datasmooth(y, 5)
    """
    y = np.asarray(y, dtype=float)
    row_vec = y.ndim == 1 or y.shape[0] == 1
    if y.ndim == 1:
        y = y[:, np.newaxis]
    elif y.shape[0] == 1:
        y = y.T

    m = int(m)
    if m == 0:
        return y.squeeze() if row_vec else y

    n_win = 2 * m + 1

    if method == 'flat':
        weights = np.ones(n_win) / n_win
    elif method == 'binom':
        # Binomial weights = diagonal of Pascal's matrix of size n_win
        # Equivalent to conv([1,1], [1,1], ...) n_win-1 times
        row = np.array([1.0])
        for _ in range(n_win - 1):
            row = np.convolve(row, [1.0, 1.0])
        weights = row / 2 ** (n_win - 1)
    elif method == 'savgol':
        if poly_order < deriv:
            raise ValueError("datasmooth: poly_order must be ≥ deriv.")
        t = np.arange(-m, m + 1, dtype=float)
        X = t[:, np.newaxis] ** np.arange(poly_order + 1)  # (n_win, p+1)
        F_mat = np.linalg.pinv(X)   # (p+1, n_win)
        weights = ((-1) ** deriv) * F_mat[deriv, :]
    else:
        raise ValueError(f"datasmooth: unknown method '{method}'. Use 'flat', 'binom', or 'savgol'.")

    ndata, ncols = y.shape

    # Pad boundaries with edge values
    y_pad = np.concatenate([
        np.tile(y[[0], :], (m, 1)),
        y,
        np.tile(y[[-1], :], (m + 1, 1)),
    ], axis=0)

    y_filt = lfilter(weights, [1.0], y_pad, axis=0)
    y_filt = y_filt[n_win - 1:n_win - 1 + ndata]

    if row_vec:
        return y_filt.squeeze()
    return y_filt


# ---------------------------------------------------------------------------
# rcfilt — RC low-pass filter
# ---------------------------------------------------------------------------

def rcfilt(
    y: "array-like",
    sample_time: float,
    time_constant: float,
    direction: str = 'up',
) -> np.ndarray:
    """Apply an RC low-pass filter to a signal.

    Models the analogue RC filter in CW EPR spectrometers.

    Parameters
    ----------
    y:
        Input signal (1D or 2D; filtered along axis 0 for 2D).
    sample_time:
        Dwell time per point (same units as *time_constant*).
    time_constant:
        RC filter time constant (same units as *sample_time*).
    direction:
        ``'up'`` (default) — forward filter (field swept upward).
        ``'down'`` or ``'dn'`` — reverse filter (field swept downward).

    Returns
    -------
    y_filt : ndarray, same shape as *y*.

    Examples
    --------
    >>> y_filt = rcfilt(spectrum, sample_time=1e-4, time_constant=5e-4)
    """
    if sample_time <= 0:
        raise ValueError("rcfilt: sample_time must be positive.")
    if time_constant < 0:
        raise ValueError("rcfilt: time_constant must be non-negative.")
    if direction not in ('up', 'down', 'dn'):
        raise ValueError("rcfilt: direction must be 'up', 'down', or 'dn'.")

    y = np.asarray(y, dtype=float)
    row_vec = y.ndim == 1
    if row_vec:
        y = y[:, np.newaxis]

    invert = direction in ('down', 'dn')
    if invert:
        y = y[::-1, :]

    if time_constant == 0:
        y_filt = y.copy()
    else:
        e = np.exp(-sample_time / time_constant)
        b_filt = np.array([1.0 - e])
        a_filt = np.array([1.0, -e])
        y_filt = lfilter(b_filt, a_filt, y, axis=0)

    if invert:
        y_filt = y_filt[::-1, :]

    if row_vec:
        return y_filt[:, 0]
    return y_filt


# ---------------------------------------------------------------------------
# hilberttrans — FFT-based Hilbert transform
# ---------------------------------------------------------------------------

def hilberttrans(y: "array-like") -> np.ndarray:
    """Compute the analytic signal (Hilbert transform) of a real 1D array.

    The output is a complex array: ``real(out) ≡ y``,
    ``imag(out)`` = Hilbert transform (90° phase-shifted version of *y*).

    Useful for computing the dispersion signal from an absorption spectrum.

    Parameters
    ----------
    y:
        Real-valued 1D input vector.

    Returns
    -------
    y_h : ndarray (complex), same length as *y*.

    Examples
    --------
    >>> x = np.linspace(-3, 3, 512)
    >>> yabs  = np.exp(-x**2)
    >>> y_h   = hilberttrans(yabs)
    >>> ydisp = np.imag(y_h)
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("hilberttrans: input must be a 1D array.")

    n = len(y)
    h = np.zeros(n, dtype=float)
    if n % 2 == 0:
        # Even length
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        # Odd length
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0

    y_h = np.fft.ifft(np.fft.fft(y) * h)
    return y_h


# ---------------------------------------------------------------------------
# unitconvert — unit conversion for EPR quantities
# ---------------------------------------------------------------------------

#: Supported unit conversion strings for :func:`unitconvert`.
UNIT_CONVERSIONS = [
    'cm^-1->eV', 'cm^-1->K', 'cm^-1->mT', 'cm^-1->MHz',
    'eV->cm^-1', 'eV->K',    'eV->mT',    'eV->MHz',
    'K->cm^-1',  'K->eV',    'K->mT',     'K->MHz',
    'mT->cm^-1', 'mT->eV',   'mT->K',     'mT->MHz',
    'MHz->cm^-1','MHz->eV',  'MHz->K',    'MHz->mT',
]


def unitconvert(
    value: "array-like",
    units: str,
    g: float = GFREE,
) -> np.ndarray:
    """Convert EPR-relevant energy/field quantities between units.

    Supported units: ``cm^-1``, ``eV``, ``K``, ``mT``, ``MHz``.
    Use the form ``'unit_a->unit_b'``.

    Parameters
    ----------
    value:
        Input value(s) in the source unit.  Scalar or array.
    units:
        Conversion string, e.g. ``'cm^-1->MHz'`` or ``'mT->K'``.
    g:
        g-factor used for conversions involving field (mT).  Defaults to
        the free-electron g-value.

    Returns
    -------
    out : ndarray
        Converted value(s) in the target unit.

    Examples
    --------
    >>> unitconvert(1.0, 'cm^-1->MHz')        # 29979.2458 MHz
    >>> unitconvert(340.0, 'mT->MHz', g=2.0)  # ≈ 9520 MHz
    """
    value = np.asarray(value, dtype=float)

    # Conversion table (output in target unit)
    conversions = {
        'cm^-1->eV':  lambda v: v * CLIGHT * PLANCK / EVOLT,
        'cm^-1->K':   lambda v: v * CLIGHT * PLANCK / BOLTZMANN,
        'cm^-1->mT':  lambda v: v / g * (PLANCK / BMAGN / 1e-3) * CLIGHT,
        'cm^-1->MHz': lambda v: v * CLIGHT / 1e6,

        'eV->cm^-1':  lambda v: v * EVOLT / (CLIGHT * PLANCK),
        'eV->K':      lambda v: v * EVOLT / BOLTZMANN,
        'eV->mT':     lambda v: v / g / BMAGN / 1e-3 * EVOLT,
        'eV->MHz':    lambda v: v * EVOLT / PLANCK / 1e6,

        'K->cm^-1':   lambda v: v * BOLTZMANN / (CLIGHT * PLANCK),
        'K->eV':      lambda v: v * BOLTZMANN / EVOLT,
        'K->mT':      lambda v: v / g / BMAGN / 1e-3 * BOLTZMANN,
        'K->MHz':     lambda v: v * BOLTZMANN / PLANCK / 1e6,

        'mT->cm^-1':  lambda v: v * g / (PLANCK / BMAGN / 1e-3) / CLIGHT,
        'mT->eV':     lambda v: v * g * BMAGN * 1e-3 / EVOLT,
        'mT->K':      lambda v: v * g * BMAGN * 1e-3 / BOLTZMANN,
        'mT->MHz':    lambda v: v * g * (1e-3 * BMAGN / PLANCK / 1e6),

        'MHz->cm^-1': lambda v: v * 1e6 / CLIGHT,
        'MHz->eV':    lambda v: v * 1e6 * PLANCK / EVOLT,
        'MHz->K':     lambda v: v * 1e6 * PLANCK / BOLTZMANN,
        'MHz->mT':    lambda v: v / g * (PLANCK / BMAGN / 1e-3) * 1e6,
    }

    if units not in conversions:
        # Case-insensitive suggestion
        units_lower = units.lower()
        suggestions = [k for k in conversions if k.lower() == units_lower]
        if suggestions:
            raise ValueError(
                f"unitconvert: unknown conversion '{units}'. Did you mean '{suggestions[0]}'?"
            )
        raise ValueError(
            f"unitconvert: unknown conversion '{units}'. "
            f"Supported: {', '.join(UNIT_CONVERSIONS)}"
        )

    return conversions[units](value)


# ---------------------------------------------------------------------------
# eprconvert — resonance condition calculator
# ---------------------------------------------------------------------------

def eprconvert(
    *,
    field: float | None = None,
    freq: float | None = None,
    g: float | None = None,
) -> dict:
    """Compute the missing EPR resonance quantity from the other two.

    Uses the resonance condition ``h * freq = g * mu_B * B``.

    Exactly two of the three keyword arguments must be given; the third
    is computed and all three are returned in a dict.

    Parameters
    ----------
    field:
        Magnetic field in mT.
    freq:
        Microwave frequency in GHz.
    g:
        g-factor (dimensionless).

    Returns
    -------
    dict
        ``{'field': ..., 'freq': ..., 'g': ...}`` with all three values.

    Examples
    --------
    >>> eprconvert(freq=9.8, g=2.0023)       # compute field
    >>> eprconvert(field=340.0, g=2.0023)     # compute frequency
    >>> eprconvert(field=340.0, freq=9.5)     # compute g-factor
    """
    given = sum(x is not None for x in (field, freq, g))
    if given != 2:
        raise ValueError(
            "eprconvert: provide exactly two of field=, freq=, g="
        )

    if field is None:
        # B = h * freq / (g * mu_B), freq in GHz, result in mT
        field = PLANCK * freq * 1e9 / (g * BMAGN) * 1e3
    elif freq is None:
        # freq = g * mu_B * B / h, B in mT, result in GHz
        freq = g * BMAGN * field * 1e-3 / PLANCK / 1e9
    else:
        # g = h * freq / (mu_B * B)
        g = PLANCK * freq * 1e9 / (BMAGN * field * 1e-3)

    return {'field': field, 'freq': freq, 'g': g}
