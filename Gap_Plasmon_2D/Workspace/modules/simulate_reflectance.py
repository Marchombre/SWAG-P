# simulate_reflectance.py
import numpy as np
from Material_Configuration import build_material_configuration_dynamic
from Function_reflectance_SWAG import reflectance

def simulate_reflectance(lambda_range, geometry, wave, df_config, json_path, n_mod):
    """
    Effectue la simulation de la réflectance pour une plage de longueurs d'onde.
    
    Pour chaque valeur de lambda_range, on reconstruit dynamiquement le dictionnaire
    des permittivités à partir de df_config et on calcule la réflectance.
    
    Retourne deux listes : Rup_values et Rdown_values.
    """
    Rup_values = []
    Rdown_values = []
    
    for lam in lambda_range:
        # Mise à jour dynamique de la configuration des matériaux
        materials = build_material_configuration_dynamic(df_config, lam, json_path)
        # Mise à jour de la longueur d'onde dans le dictionnaire d'onde
        wave["wavelength"] = lam
        # Calcul de la réflectance
        Rup, Rdown = reflectance(geometry, wave, materials, n_mod)
        Rup_values.append(Rup)
        Rdown_values.append(Rdown)
    
    return Rup_values, Rdown_values
