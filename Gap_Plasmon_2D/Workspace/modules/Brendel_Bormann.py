
import numpy as np
from scipy.special import wofz



def BrendelBormann(wav, f0, omega_p, Gamma0, f, omega, gamma, sigma):
    """
    Brendel & Bormann model, using Voigt functions to model lorentzian
    resonances potentially widened with a gaussian distribution.
    f0, Gamma0 and omega_p are the chi_f parameters (eps_inf, plasma frequency)
    f, gamma, omega, sigma are the chi_b parameters (Lorentz resonances)
    f, gamma, omega, sigma must be lists (np arrays) of the same lengths
    They are given in eV (wav in nm)
    """
    # Brendel-Bormann model with n resonances
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / wav
    a = np.sqrt(w * (w + 1j * gamma))
    x = (a - omega) / (np.sqrt(2) * sigma)
    y = (a + omega) / (np.sqrt(2) * sigma)
    # Polarizability due to bound electrons
    chi_b = np.sum(
        1j
        * np.sqrt(np.pi)
        * f
        * omega_p**2
        / (2 * np.sqrt(2) * a * sigma)
        * (wofz(x) + wofz(y))
    )
    # Equivalent polarizability linked to free electrons (Drude model)
    chi_f = -(omega_p**2) * f0 / (w * (w + 1j * Gamma0))
    epsilon = 1 + chi_f + chi_b
    return epsilon