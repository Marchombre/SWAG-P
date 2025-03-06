import numpy as np
import numpy.fft as fft
from scipy.special import erf



import numpy as np
from scipy.special import wofz



def BrendelBormann_Faddeeva(lambda_test, f0, omega_p, Gamma0, f, omega, gamma, sigma, N):
    """
    Modèle Brendel & Bormann utilisant la fonction de Voigt pour modéliser des résonances Lorentz
    élargies par une distribution gaussienne.
    
    Paramètres :
      - lambda_test : longueur d'onde (nm)
      - f0, omega_p, Gamma0 : paramètres de chi_f (en eV)
      - f, gamma, omega, sigma : listes ou tableaux (numpy) des paramètres de chi_b (en eV)
      - N : paramètre numérique pour la FFT dans faddeeva
    Retourne :
      - epsilon : permittivité complexe
    """
    # Conversion de la longueur d'onde en énergie (eV) : E ≈ 1240/λ (λ en nm)
    E = 1240.0 / lambda_test  # énergie en eV
    w = E  # On utilise w comme énergie en eV
    
    chi_b = 0.0 + 0.0j  # Initialisation
    
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
    """Approximation de la fonction de Faddeeva en utilisant une méthode FFT.
    
    Paramètres :
      - z : argument complexe (scalaire ou array)
      - N : nombre de modes pour le calcul (contrôle la précision)
      
    Retourne :
      - w(z) ≈ exp(-z^2) erfc(-i z)
    """
    # On s'assure que z est un array
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