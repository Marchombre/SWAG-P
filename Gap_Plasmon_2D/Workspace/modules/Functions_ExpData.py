import json
import numpy as np

def get_n_k(material_name, lam, json_path):
    with open(json_path) as file:
        data = json.load(file)
    if material_name not in data:
        raise ValueError(f"Material '{material_name}' is not in the database.")
    material = data[material_name]
    if material["model"] == "ExpData":
        wl = np.array(material["wavelength_list"])
        epsilon_real = np.array(material["permittivities"])
        epsilon_imag = np.array(material.get("permittivities_imag", np.zeros_like(epsilon_real)))
        if lam < wl[0] or lam > wl[-1]:
            raise ValueError(f"Wavelength {lam} nm is out of the range [{wl[0]}, {wl[-1]}] nm.")
        eps_r = np.interp(lam, wl, epsilon_real)
        eps_i = np.interp(lam, wl, epsilon_imag)
        eps_complex = eps_r + 1.0j * eps_i
        n_complex = np.sqrt(eps_complex)
        return np.real(n_complex), np.imag(n_complex)
    else:
        raise ValueError(f"Model '{material['model']}' for '{material_name}' is not supported.")

from Brendel_Bormann_Faddeeva import BrendelBormann_Faddeeva

# Numerical approximation:
# The faddeeva function uses a spectral method based on a Fast Fourier Transform (FFT)
# to approximate the Faddeeva function.
#
# Parameter N determines the number of terms used in this approximation.
#
# Accuracy: A larger N means more terms are used in the approximation, improving the accuracy of the Faddeeva function calculation.
#
# Performance: A larger N also increases the computation time, so there is a trade-off between accuracy and performance.

def compute_permittivity(lam, f0, omega_p, Gamma0, f, omega, gamma, sigma, N=50):
    """
    Computes the complex permittivity ε for a material modeled by
    the Brendel-Bormann model using the Faddeeva approximation.
    
    Parameters:
      - lam : wavelength in nm.
      - f0, omega_p, Gamma0 : model parameters (in eV).
      - f, omega, gamma, sigma : lists or numpy arrays of resonance parameters (in eV).
      - N : numerical parameter for the FFT calculation in the faddeeva function (default = 50).
    
    Returns:
      - ε : complex permittivity computed via BrendelBormann_Faddeeva.
    """
    return BrendelBormann_Faddeeva(lam, f0, omega_p, Gamma0, f, omega, gamma, sigma, N)
