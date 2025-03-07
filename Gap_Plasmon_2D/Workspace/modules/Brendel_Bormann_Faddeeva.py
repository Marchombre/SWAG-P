import numpy as np
import numpy.fft as fft
from scipy.special import erf
from scipy.special import wofz

def BrendelBormann(wav, f0, omega_p, Gamma0, f, omega, gamma, sigma):
    """
    Brendel & Bormann model, using Voigt functions to model Lorentzian
    resonances potentially broadened by a Gaussian distribution.
    f0, Gamma0, and omega_p are the chi_f parameters (ε_inf, plasma frequency),
    while f, gamma, omega, sigma are the chi_b parameters (Lorentz resonances).
    f, gamma, omega, sigma must be lists (or numpy arrays) of the same length.
    They are provided in eV (with wav in nm).
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
    # Equivalent polarizability from free electrons (Drude model)
    chi_f = -(omega_p**2) * f0 / (w * (w + 1j * Gamma0))
    epsilon = 1 + chi_f + chi_b
    return epsilon

def BrendelBormann_Faddeeva(lambda_test, f0, omega_p, Gamma0, f, omega, gamma, sigma, N):
    """
    Brendel & Bormann model using the Voigt function to model Lorentzian
    resonances broadened by a Gaussian distribution.
    
    Parameters:
      - lambda_test : wavelength in nm
      - f0, omega_p, Gamma0 : parameters for chi_f (in eV)
      - f, gamma, omega, sigma : lists or numpy arrays of chi_b parameters (in eV)
      - N : numerical parameter for the FFT in faddeeva
    Returns:
      - epsilon : computed complex permittivity
    """
    # Convert wavelength to energy (eV): E ≈ 1240 / λ (with λ in nm)
    E = 1240.0 / lambda_test  # energy in eV
    w = E  # Use w as energy in eV
    
    chi_b = 0.0 + 0.0j  # Initialization
    
    f = np.array(f, dtype=float)
    omega = np.array(omega, dtype=float)
    gamma = np.array(gamma, dtype=float)
    sigma = np.array(sigma, dtype=float)
    
    for i in range(len(f)):
        a = (omega[i] - 1j * gamma[i]) / (np.sqrt(2) * sigma[i])
        x = (w - omega[i]) / (np.sqrt(2) * sigma[i])
        y = (w + omega[i]) / (np.sqrt(2) * sigma[i])
        prefactor = 1j * np.sqrt(np.pi) * f[i] * omega_p**2 / (2 * np.sqrt(2) * a * sigma[i])
        chi_b += prefactor * (faddeeva(x, N) + faddeeva(y, N))
    
    chi_f = - (omega_p**2) * f0 / (w * (w + 1j * Gamma0))
    epsilon = 1 + chi_f + chi_b
    return epsilon

def faddeeva(z, N):
    """Approximation of the Faddeeva function using an FFT-based method.
    
    Parameters:
      - z : complex argument (scalar or array)
      - N : number of modes used in the calculation (controls precision)
      
    Returns:
      - w(z) ≈ exp(-z^2) erfc(-i z)
    """
    # Ensure that z is an array
    z_arr = np.atleast_1d(z)
    w_val = np.zeros(z_arr.shape, dtype=complex)
    
    idx = (np.real(z_arr) == 0)
    w_val[idx] = np.exp(-np.abs(z_arr[idx])**2) * (1 - erf(np.imag(z_arr[idx])))
    
    idx_non = ~idx
    idx1 = np.where(np.imag(z_arr[idx_non]) < 0)[0]
    if idx1.size > 0:
        z_arr[idx_non][idx1] = np.conj(z_arr[idx_non][idx1])
    
    M = 2 * N
    M2 = 2 * M
    k = np.arange(-M + 1, M)
    L = np.sqrt(N / np.sqrt(2))
    theta = k * np.pi / M
    t = L * np.tan(theta / 2)
    f_val = np.exp(-t**2) * (L**2 + t**2)
    f_val = np.append(0, f_val)
    a_coeff = np.real(np.fft.fft(np.fft.fftshift(f_val))) / M2
    a_coeff = np.flipud(a_coeff[1:N+1])
    
    Z = (L + 1.0j * z_arr[idx_non]) / (L - 1.0j * z_arr[idx_non])
    p = np.polyval(a_coeff, Z)
    w_val[idx_non] = 2 * p / (L - 1.0j * z_arr[idx_non])**2 + (1 / np.sqrt(np.pi)) / (L - 1.0j * z_arr[idx_non])
    
    if idx1.size > 0:
        overall_idx = np.where(idx_non)[0][idx1]
        w_val[overall_idx] = np.conj(2 * np.exp(-z_arr[overall_idx]**2) - w_val[overall_idx])
    
    if np.ndim(z) == 0:
        return w_val[0]
    else:
        return w_val
