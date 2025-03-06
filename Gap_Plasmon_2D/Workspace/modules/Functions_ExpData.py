import json
import numpy as np

def get_n_k(material_name, lam, json_path):
    with open(json_path) as file:
        data = json.load(file)
    if material_name not in data:
        raise ValueError(f"Le matériau '{material_name}' est pas dans la base de données.")
    material = data[material_name]
    if material["model"] == "ExpData":
        wl = np.array(material["wavelength_list"])
        epsilon_real = np.array(material["permittivities"])
        epsilon_imag = np.array(material.get("permittivities_imag", np.zeros_like(epsilon_real)))
        if lam < wl[0] or lam > wl[-1]:
            raise ValueError(f"La longueur d'onde {lam} nm est hors de l'intervalle [{wl[0]}, {wl[-1]}] nm.")
        eps_r = np.interp(lam, wl, epsilon_real)
        eps_i = np.interp(lam, wl, epsilon_imag)
        eps_complex = eps_r + 1.0j * eps_i
        n_complex = np.sqrt(eps_complex)
        return np.real(n_complex), np.imag(n_complex)
    else:
        raise ValueError(f"Le modèle '{material['model']}' pour '{material_name}' est pas supporté.")
    



from Brendel_Bormann_Faddeeva import BrendelBormann_Faddeeva



# Approximation numérique: La fonction faddeeva utilise une méthode spectrale 
# basée sur une transformée de Fourier rapide (FFT) pour approximer la fonction Faddeeva. 

# Le paramètre N détermine le nombre de termes dans cette approximation. 

# Précision: Un N plus grand signifie que plus de termes sont utilisés dans l'approximation, 
# ce qui améliore la précision du calcul de la fonction Faddeeva. 

# Performance: Un N plus grand augmente également le temps de calcul, 
# donc il y a un compromis entre précision et performance.

def compute_permittivity(lam, f0, omega_p, Gamma0, f, omega, gamma, sigma, N=50):
    """
    Calcule la permittivité complexe ε pour un matériau modélisé par
    le modèle Brendel-Bormann utilisant l'approximation par Faddeeva.
    
    Paramètres :
      - lam : longueur d'onde en nm.
      - f0, omega_p, Gamma0 : paramètres du modèle (en eV).
      - f, omega, gamma, sigma : listes ou tableaux numpy des paramètres de résonance (en eV).
      - N : paramètre numérique pour le calcul FFT dans la fonction faddeeva (défaut = 50).
    
    Retourne :
      - ε : permittivité complexe calculée via BrendelBormann_Faddeeva.
    """
    return BrendelBormann_Faddeeva(lam, f0, omega_p, Gamma0, f, omega, gamma, sigma, N)

