# Characterization.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq

def find_hwhm_points(wavelength, reflectance, kind_interp='cubic'):
    # Pour identifier la résonance
    idx_min = np.argmin(reflectance)
    lambda_res = wavelength[idx_min]
    R_min = reflectance[idx_min]
    
    # Estimer les max
    R_max = max(reflectance[0], reflectance[-1])
    
    # Calculer la mi hauteur
    R_half = (R_max + R_min) / 2
    
    # Fonction pour interpoller qui donne une fonction continue pour f(λ) = reflectance(λ) - R_half
    f_interp = interp1d(wavelength, reflectance - R_half, kind=kind_interp)
    
    # Rechercher les points où la fonction d'interpolation est nulle
    # On utilise la méthode de Brent pour trouver les racines
    lambda_left = brentq(f_interp, wavelength[0], lambda_res)
    lambda_right = brentq(f_interp, lambda_res, wavelength[-1])
    
    # Calcul de la largeur FWHM
    fwhm = lambda_right - lambda_left

    return lambda_left, lambda_right, fwhm, lambda_res, R_half


def relative_spectral_ratio(delta_wavelength, lambda_ref):
    """
    Calcule le rapport spectral relatif en fonction de la variation de longueur d'onde et de la longueur d'onde de référence.
    
    Parameters:
    delta_wavelength (float): La variation de longueur d'onde.
    lambda_ref (float): La longueur d'onde de référence.
    
    Returns:
    float: Le rapport spectral relatif.
    """
    return delta_wavelength / lambda_ref

def Q_factor(lambda_res, fwhm):
    """
    Calcule le facteur Q à partir de la longueur d'onde de résonance et de la largeur FWHM.
    
    Parameters:
    lambda_res (float): La longueur d'onde de résonance.
    fwhm (float): La largeur FWHM.
    
    Returns:
    float: Le facteur Q.
    """
    return lambda_res / fwhm

def FOM(Sensitivity, delta_lmbd):
    """
    Calcule le figure of merit (FOM) à partir de la sensibilité et de la variation de longueur d'onde.
    
    Parameters:
    Sensitivity (float): La sensibilité.
    delta_lmbd (float): La variation de longueur d'onde.
    
    Returns:
    float: Le figure of merit (FOM).
    """
    return Sensitivity / delta_lmbd


def Exp_sensitivity(reflectance, n, delta_n=1e-4):
    """
    Calcule la sensibilité expérimentale à partir de la réflectance, 
    de la longueur d'onde de résonance et de l'indice de réfraction.
    
    Parameters:
    reflectance (function): Une fonction qui donne la réflectance en 
    fonction de l'indice de réfraction.
    lmbd_res (float): La longueur d'onde de résonance.
    n (float): L'indice de réfraction.
    delta_n (float): Une petite variation de l'indice de réfraction pour 
    calculer la dérivée (par défaut 1e-4).
    
    Returns:
    float: La sensibilité expérimentale (dérivée de R par rapport à n).
    """
    # Calculer la réflectance pour n et n + delta_n
    R_n = reflectance(n)
    R_n_plus_delta = reflectance(n + delta_n)
    
    # Calculer la dérivée de R par rapport à n
    sensitivity = (R_n_plus_delta - R_n) / delta_n
    
    return sensitivity

def Exp_sensitivity_with_resonance(reflectance, wavelength, n, delta_n=1e-4):
    """
    Calcule la sensibilité expérimentale en utilisant la formule 
    (dR/dλ) * (dλ_res/dn_i).
    
    Parameters:
    reflectance (function): Une fonction qui donne la réflectance en 
    fonction de la longueur d'onde et de l'indice de réfraction.
    wavelength (array): Tableau des longueurs d'onde.
    n (float): L'indice de réfraction.
    delta_n (float): Une petite variation de l'indice de réfraction pour 
    calculer la dérivée (par défaut 1e-4).
    
    Returns:
    float: La sensibilité expérimentale.
    """
    # Étape 1 : Calculer dR/dλ
    # Interpolation de la réflectance en fonction de la longueur d'onde
    reflectance_interp = interp1d(wavelength, reflectance(n, wavelength), kind='cubic')
    
    # Calcul de la dérivée numérique dR/dλ
    delta_lambda = np.gradient(wavelength)
    dR_dlambda = np.gradient(reflectance_interp(wavelength), delta_lambda)

    # Étape 2 : Calculer dλ_res/dn_i
    # Trouver la longueur d'onde de résonance pour n et n + delta_n
    lambda_res_n = wavelength[np.argmin(reflectance(n, wavelength))]
    lambda_res_n_plus_delta = wavelength[np.argmin(reflectance(n + delta_n, wavelength))]
    
    # Calcul de la dérivée numérique dλ_res/dn_i
    dlambda_res_dn = (lambda_res_n_plus_delta - lambda_res_n) / delta_n

    # Étape 3 : Calculer la sensibilité
    # Utiliser la valeur de dR/dλ à la longueur d'onde de résonance
    dR_dlambda_at_res = reflectance_interp(lambda_res_n)
    sensitivity = dR_dlambda_at_res * dlambda_res_dn

    return sensitivity





