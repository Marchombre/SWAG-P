#!/usr/bin/env python3
"""
Module: simulate_reflectance.py

Ce module se charge de la simulation de la réflectance pour une ou plusieurs combinaisons 
géométrie/matériaux. Il utilise une compréhension de liste pour maximiser les performances 
et retourne à la fois les résultats et les détails complets de la simulation.
"""

import os
import json
import numpy as np
import pandas as pd
from Material_Configuration import build_material_configuration_dynamic
from Function_reflectance_SWAG import reflectance
from Saving_Functions import save_simulation_summary

def simulate_reflectance_single(lambda_range, geometry, wave, df_config, json_combined_path, n_mod, ri_overrides=None):
    """
    Simule la réflectance (Rup, Rdown) sur une plage de longueurs d'onde.
    
    Args:
        lambda_range (list): Liste (ou itérable) de longueurs d'onde.
        geometry (dict): Configuration géométrique.
        wave (dict): Paramètres d'onde.
        df_config (pd.DataFrame): Configuration des matériaux.
        json_combined_path (str): Chemin vers le JSON combiné.
        n_mod (int): Nombre de modes RCWA.
        ri_overrides (dict, optionnel): Remplacements pour l'indice de réfraction.
        
    Returns:
        Tuple (Rup_values, Rdown_values) sous forme de listes.
    """
    if ri_overrides is None:
        ri_overrides = {}
    # Utilisation d'une compréhension de liste pour obtenir les tuples (Rup, Rdown)
    result = [
        reflectance(geometry, {**wave, "wavelength": lam},
                    build_material_configuration_dynamic(df_config, lam, json_combined_path, ri_overrides), n_mod)
        for lam in lambda_range
    ]
    Rup_values, Rdown_values = zip(*result)
    return list(Rup_values), list(Rdown_values)

