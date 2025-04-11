#!/usr/bin/env python3
"""
Module: simulate_reflectance.py

Ce module se charge de la simulation de la réflectance pour une ou plusieurs combinaisons 
géométrie/matériaux. Il utilise une compréhension de liste pour maximiser les performances 
et retourne à la fois les résultats et les détails complets de la simulation.
"""

import os
import json
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

def simulate_reflectance_all_combos(lambda_range, wave, n_mod, json_combined_path):
    """
    Charge le fichier de configurations geom_mat_combinations.json et simule la réflectance 
    pour chaque combinaison géométrie/matériaux.
    
    Retourne un tuple (results, simulation_details, all_configs) où :
      - results: dict { combo_name: (Rup_values, Rdown_values) }
      - simulation_details: dict contenant toutes les données de simulation par combo.
      - all_configs: liste des configurations lues depuis le JSON.
    
    Le résumé de simulation est sauvegardé via save_simulation_summary.
    """
    # Construction des chemins d'accès
    module_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(module_dir)
    combos_file = os.path.join(workspace_dir, "CONFIGURATIONS", "geom_mat_combinations.json")
    summary_dir = os.path.join(workspace_dir, "notebooks", "Summary_Simulation")

    if not os.path.isfile(combos_file):
        raise FileNotFoundError(f"Le fichier de combinaisons est introuvable: {combos_file}")
    
    with open(combos_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_combos = data.get("ALL_COMBINED_CONFIGS", [])
    if not all_combos:
        raise ValueError("Aucune combinaison trouvée dans geom_mat_combinations.json.")

    results = {}
    simulation_details = {}

    # Boucle sur chaque configuration pour simuler la réflectance
    for combo in all_combos:
        combo_name = combo["config_name"]
        geometry_dict = combo["geometry"]["geometry"]
        df_config = pd.DataFrame(combo["material"]["MATERIALS_CONFIG"])
        ri_overrides = combo["material"].get("RI_OVERRIDES", {})
        Rup, Rdown = simulate_reflectance_single(lambda_range, geometry_dict, wave, df_config, json_combined_path, n_mod, ri_overrides)
        results[combo_name] = (Rup, Rdown)
        simulation_details[combo_name] = {
            "geometry": geometry_dict,
            "material_config": df_config.to_dict(orient="records"),
            "ri_overrides": ri_overrides,
            "Rup": Rup,
            "Rdown": Rdown
        }

    summary_file = save_simulation_summary(simulation_details, lambda_range, wave, n_mod, summary_dir)
    print(f"Résumé de la simulation sauvegardé dans : {summary_file}")
    return results, simulation_details, all_combos

if __name__ == "__main__":
    # Exemple d'utilisation
    lambda_range = list(range(400, 701, 10))
    wave = {"parameter": "valeur_exemple"}
    n_mod = 3
    json_combined_path = "chemin/vers/json_combined.json"
    simulate_reflectance_all_combos(lambda_range, wave, n_mod, json_combined_path)
