import numpy as np
from Material_Configuration import build_material_configuration_dynamic
from Function_reflectance_SWAG import reflectance

def simulate_reflectance(lambda_range, geometry, wave, df_config, json_path, n_mod, ri_overrides=None):
    """
    Exécute la simulation de réflectance pour une plage de longueurs d'onde.
    Pour chaque valeur de lambda_range, la configuration des matériaux est reconstruite
    dynamiquement à partir de df_config (et éventuellement ri_overrides) et la réflectance est calculée.

    Parameters
    ----------
    lambda_range : array_like
        Plage de longueurs d'onde (en nm).
    geometry : dict
        Dictionnaire définissant la géométrie du système.
    wave : dict
        Dictionnaire des paramètres de l'onde (angle, polarisation, etc.).
    df_config : DataFrame
        Configuration des matériaux (issu du widget MATERIALS_CONFIG).
    json_path : str
        Chemin vers le fichier JSON combiné contenant les données ExpData.
    n_mod : int
        Nombre de modes RCWA.
    ri_overrides : dict, optional
        Dictionnaire d'overrides pour les matériaux RefractiveIndex (par défaut None).

    Returns
    -------
    Rup_values, Rdown_values : lists
        Listes des valeurs de réflectance calculées pour chaque longueur d'onde.
    """
    Rup_values = []
    Rdown_values = []
    
    for lam in lambda_range:
        # Mise à jour dynamique de la configuration des matériaux, avec overrides si définis.
        materials = build_material_configuration_dynamic(df_config, lam, json_path, ri_overrides=ri_overrides)
        # Mise à jour de la longueur d'onde dans le dictionnaire wave
        wave["wavelength"] = lam
        # Calcul de la réflectance
        Rup, Rdown = reflectance(geometry, wave, materials, n_mod)
        Rup_values.append(Rup)
        Rdown_values.append(Rdown)
    
    return Rup_values, Rdown_values
