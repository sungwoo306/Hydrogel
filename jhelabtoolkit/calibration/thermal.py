import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy.special import kv


def Re(omega: float, w: float, nu: float) -> float:
    return omega * w**2 / (4 * nu)


def hydrodynamic_rectangular(Re: float) -> complex:
    return hydrodynamic_circular(Re) * correction_rectangular(Re)


def hydrodynamic_circular(Re: float) -> complex:
    z = -1j * np.sqrt(1j * Re)
    return 1 + 4 * kv(1, z) / (z * kv(0, z))


def correction_rectangular(Re: float) -> complex:
    tau = np.log10(Re)
    numer_real = polyval(
        tau, [0.91324, -0.48274, 0.46842, -0.12886, 0.044055, -0.0035117, 0.00069085]
    )
    denom_real = polyval(
        tau, [1, -0.56964, 0.48690, -0.13444, 0.045155, -0.0035862, 0.00069085]
    )
    numer_imag = polyval(
        tau, [-0.024134, -0.029256, 0.016294, -0.00010961, 0.000064577, -0.000044510]
    )
    denom_imag = polyval(
        tau, [1, -0.59702, 0.55182, -0.18357, 0.079156, -0.014369, 0.0028361]
    )
    return numer_real / denom_real + 1j * numer_imag / denom_imag


def resonance_curve(
    f: np.ndarray, f0: float, Q: float, P_dc: float, P_white: float
) -> np.ndarray:
    denominator = (f**2 - f0**2) ** 2 + f**2 * f0**2 / Q**2
    return P_white + P_dc * f0**4 / denominator
