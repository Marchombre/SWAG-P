# simulate_reflectance.py
import numpy as np
from Material_Configuration import build_material_configuration_dynamic
from Function_reflectance_SWAG import reflectance

def simulate_reflectance(lambda_range, geometry, wave, df_config, json_path, n_mod):
    """
    Make the reflectance simulation for a range of wavelengths.
    For each value of lambda_range, the permittivity dictionary is dynamically rebuilt
    from df_config and the reflectance is calculated.
    
    Two lists are returned: Rup_values and Rdown_values.
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
