import json
import numpy as np
import numpy.fft as fft
from scipy.special import erf, wofz

#########################################
# Module Brendel_Bormann-Faddeeva
#########################################
def BrendelBormann(wav, f0, omega_p, Gamma0, f, omega, gamma, sigma):
    """
    Modèle Brendel & Bormann utilisant des fonctions de Voigt pour modéliser des résonances lorentziennes
    potentiellement élargies par une distribution gaussienne.
    
    Paramètres:
      - wav : longueur d'onde en nm
      - f0, Gamma0, omega_p : paramètres de chi_f (ε_inf, fréquence plasmonique)
      - f, gamma, omega, sigma : listes (ou numpy arrays) de paramètres pour chi_b (résonances lorentziennes, en eV)
      
    Retourne:
      - epsilon : permittivité complexe calculée
    """
    # Conversion de la longueur d'onde (nm) en énergie (eV)
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / wav
    a = np.sqrt(w * (w + 1j * gamma))
    x = (a - omega) / (np.sqrt(2) * sigma)
    y = (a + omega) / (np.sqrt(2) * sigma)
    
    # Polarizabilité due aux électrons liés
    chi_b = np.sum(
        1j * np.sqrt(np.pi) * f * omega_p**2 / (2 * np.sqrt(2) * a * sigma) * (wofz(x) + wofz(y))
    )
    # Polarizabilité équivalente issue des électrons libres (modèle de Drude)
    chi_f = - (omega_p**2) * f0 / (w * (w + 1j * Gamma0))
    epsilon = 1 + chi_f + chi_b
    return epsilon

def faddeeva(z, N):
    """
    Approximation de la fonction de Faddeeva en utilisant une méthode basée sur la FFT.
    
    Paramètres:
      - z : argument complexe (scalaire ou tableau)
      - N : nombre de termes utilisés dans le calcul (définit la précision)
      
    Retourne:
      - w(z) ≈ exp(-z^2) erfc(-i z)
    """
    # On s'assure que z est un tableau
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

def BrendelBormann_Faddeeva(lambda_test, f0, omega_p, Gamma0, f, omega, gamma, sigma, N):
    """
    Modèle Brendel & Bormann utilisant la fonction de Voigt pour modéliser des résonances lorentziennes
    élargies par une distribution gaussienne, avec l'approximation FFT pour la fonction de Faddeeva.
    
    Paramètres:
      - lambda_test : longueur d'onde en nm
      - f0, omega_p, Gamma0 : paramètres pour chi_f (en eV)
      - f, gamma, omega, sigma : listes ou numpy arrays pour chi_b (en eV)
      - N : paramètre numérique pour le calcul FFT dans la fonction faddeeva
      
    Retourne:
      - epsilon : permittivité complexe calculée
    """
    # Conversion de la longueur d'onde en énergie (eV) : E ≈ 1240 / λ (avec λ en nm)
    E = 1240.0 / lambda_test  # énergie en eV
    w = E  # on utilise w comme énergie en eV
    
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

#########################################
# Module Function_ExpData
#########################################
def get_n_k(material_name, lam, json_path):
    """
    Extrait l'indice (n) et l'extinction (k) pour un matériau à partir d'un fichier JSON.
    
    Paramètres:
      - material_name : nom du matériau dans la base de données
      - lam : longueur d'onde en nm
      - json_path : chemin vers le fichier JSON contenant les données
      
    Retourne:
      - n, k : parties réelle et imaginaire de l'indice complexe calculé à partir de ε
    """
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

def compute_permittivity(lam, f0, omega_p, Gamma0, f, omega, gamma, sigma, N=50):
    """
    Calcule la permittivité complexe ε pour un matériau modélisé par le modèle Brendel-Bormann
    en utilisant l'approximation de la fonction de Faddeeva.
    
    Paramètres:
      - lam : longueur d'onde en nm.
      - f0, omega_p, Gamma0 : paramètres du modèle (en eV).
      - f, omega, gamma, sigma : listes ou numpy arrays des paramètres de résonance (en eV).
      - N : paramètre numérique pour le calcul FFT dans la fonction faddeeva (défaut = 50).
    
    Retourne:
      - ε : permittivité complexe calculée
    """
    return BrendelBormann_Faddeeva(lam, f0, omega_p, Gamma0, f, omega, gamma, sigma, N)
