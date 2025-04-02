#!/usr/bin/env python3
# simulate_reflectance.py

import os
import json
import datetime
import pandas as pd
import re

from Material_Configuration import build_material_configuration_dynamic
#from Function_reflectance_SWAG_V1 import reflectance
#from Function_reflectance_SWAG import reflectance
from Function_reflectance_SWAG_V2bis import reflectance

def simulate_reflectance_single(lambda_range, geometry, wave, df_config, json_combined_path, n_mod, ri_overrides=None):
    """
    Simule la réflectance (Rup, Rdown) pour une plage de longueurs d'onde,
    en utilisant une configuration géométrique 'geometry' (dict),
    une configuration matériaux 'df_config' (DataFrame),
    et les paramètres d'onde 'wave'.
    
    Retourne (Rup_values, Rdown_values).
    """
    if ri_overrides is None:
        ri_overrides = {}
    
    Rup_values = []
    Rdown_values = []
    for lam in lambda_range:
        materials_perm = build_material_configuration_dynamic(df_config, lam, json_combined_path, ri_overrides)
        wave["wavelength"] = lam
        Rup, Rdown = reflectance(geometry, wave, materials_perm, n_mod)
        Rup_values.append(Rup)
        Rdown_values.append(Rdown)

    return Rup_values, Rdown_values

def simulate_reflectance_all_combos(lambda_range, wave, n_mod, json_combined_path):
    """
    Charge le fichier geom_mat_combinations.json dans CONFIGURATIONS,
    puis simule la réflectance pour chaque combinaison géométrie-matériaux.
    
    Retourne un dict { combo_name: (Rup_values, Rdown_values) }
    et génère un fichier texte de résumé dans CONFIGURATIONS.
    """
    # Utiliser __file__ pour déterminer le chemin
    module_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(module_dir)
    CONFIGURATIONS_dir = os.path.join(workspace_dir, "CONFIGURATIONS")
    combos_file = os.path.join(CONFIGURATIONS_dir, "geom_mat_combinations.json")
    
    notebooks_dir = os.path.join(workspace_dir, "notebooks")
    summary_dir = os.path.join(notebooks_dir, "Summary_Simulation")

    if not os.path.isfile(combos_file):
        raise FileNotFoundError(f"Le fichier de combinaisons {combos_file} est introuvable. Créez-le via le widget Geometry/Material.")

    with open(combos_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_combos = data.get("ALL_COMBINED_CONFIGS", [])
    if not all_combos:
        raise ValueError("Aucune combinaison trouvée dans geom_mat_combinations.json.")

    results = {}
    simulation_details = {}

    for combo in all_combos:
        combo_name = combo["config_name"]
        geometry_dict = combo["geometry"]["geometry"]
        material_config_list = combo["material"]["MATERIALS_CONFIG"]
        df_config = pd.DataFrame(material_config_list)
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

    roles_order = [
        "perm_env", "perm_reso", "perm_gap", "perm_mol", "perm_func",
        "perm_diel", "perm_metalliclayer", "perm_accroche", "perm_sub"
    ]
    suffix_parts = []
    if simulation_details:
        first_combo = next(iter(simulation_details.values()))
        for role in roles_order:
            val = ""
            for entry in first_combo["material_config"]:
                if entry.get("key", "").strip() == role:
                    mat_info = entry.get("material", {})
                    mtype = mat_info.get("type", "").strip().lower()
                    if mtype == "none":
                        val = ""
                    elif mtype == "standard":
                        val = mat_info.get("material", "").strip()
                    elif mtype == "custom":
                        val = mat_info.get("expression", "").strip()
                    break
            if val.lower() != "none" and val != "":
                val_clean = re.sub(r'[^A-Za-z0-9\.\*\+]', '', val)
                suffix_parts.append(val_clean)
    filtered_parts = [part for part in suffix_parts if part]
    material_str_clean = "_".join(filtered_parts)

    summary_filename = os.path.join(summary_dir, f"simulation_summary_RCWA_V2bis_{material_str_clean}.txt")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = []
    lines.append("Simulation Summary - All Geometry/Material Combos")
    lines.append(f"Timestamp: {timestamp}")
    lines.append(f"Wave parameters: {wave}")
    lines.append(f"Number of RCWA modes: {n_mod}\n")
    lines.append("---- COMBINATIONS ----\n")

    for combo_name, details in simulation_details.items():
        lines.append(f"Combo name: {combo_name}")
        lines.append("Geometry:")
        lines.append(str(details["geometry"]))
        lines.append("Material config (df_config):")
        lines.append(str(details["material_config"]))
        lines.append(f"RI Overrides: {details['ri_overrides']}")
        lines.append("Reflectance points (Rup, Rdown):")
        for i in range(len(details["Rup"])):
            lines.append(f"  λ={lambda_range[i]} nm -> Rup={details['Rup'][i]}, Rdown={details['Rdown'][i]}")
        lines.append("-" * 40)

    with open(summary_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Résumé de la simulation multi-combos sauvegardé dans : {summary_filename}")
    return results
