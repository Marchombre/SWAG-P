#!/usr/bin/env python3
# simulate_reflectance.py

import os
import json
import datetime
import pandas as pd
import re

from Material_Configuration import build_material_configuration_dynamic
from Function_reflectance_SWAG import reflectance


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
        # 1. Pour chaque longueur d'onde lam, on reconstruit les permittivités des matériaux.
        #    Cette étape se fait via la fonction build_material_configuration_dynamic.
        #    Elle prend le DataFrame de configuration (df_config), la longueur d'onde actuelle,
        #    le chemin vers le JSON combiné (contenant vos données ExpData/BrendelBormann) et 
        #    d'éventuels overrides pour l'indice (ri_overrides).
        #    Le résultat, materials_perm, est un dictionnaire où chaque clé correspond à un rôle 
        #    (par exemple, "perm_sub", "perm_reso", etc.) et la valeur est la permittivité calculée.
        materials_perm = build_material_configuration_dynamic(df_config, lam, json_combined_path, ri_overrides)
        
        # 2. Mise à jour de la longueur d'onde dans le dictionnaire wave.
        #    Cela permet de passer la valeur lam (qui varie dans la boucle) à la fonction de calcul.
        wave["wavelength"] = lam
        
        # 3. Appel de la fonction reflectance qui, à partir de la géométrie, des paramètres d'onde
        #    et des permittivités calculées, retourne Rup (la réflectance vers le haut) et
        #    Rdown (la réflectance vers le bas).
        Rup, Rdown = reflectance(geometry, wave, materials_perm, n_mod)
        
        # 4. On accumule les résultats pour chaque lam dans des listes.
        Rup_values.append(Rup)
        Rdown_values.append(Rdown)

    # 5. Retourne les listes de résultats.
    return Rup_values, Rdown_values





def simulate_reflectance_all_combos(lambda_range, wave, n_mod, json_combined_path):
    """
    Charge le fichier geom_mat_combinations.json dans CONFIGURATIONS,
    puis simule la réflectance pour chaque combinaison géométrie-matériaux.
    
    Retourne un dict { combo_name: (Rup_values, Rdown_values) }
    et génère un fichier texte de résumé dans CONFIGURATIONS.
    """
    # 1. Détermination du chemin du fichier JSON de combinaisons.
    module_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(module_dir)
    CONFIGURATIONS_dir = os.path.join(workspace_dir, "CONFIGURATIONS")
    combos_file = os.path.join(CONFIGURATIONS_dir, "geom_mat_combinations.json")
    
    notebooks_dir = os.path.join(workspace_dir, "notebooks")
    summary_dir = os.path.join(notebooks_dir, "Summary_Simulation")

    if not os.path.isfile(combos_file):
        raise FileNotFoundError(f"Le fichier de combinaisons {combos_file} est introuvable. "
                                "Créez-le via le widget Geometry/Material.")

    # 2. Chargement du fichier JSON.
    with open(combos_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_combos = data.get("ALL_COMBINED_CONFIGS", [])
    if not all_combos:
        raise ValueError("Aucune combinaison trouvée dans geom_mat_combinations.json.")

    # 3. Pour chaque configuration enregistrée dans le JSON...
    results = {}          # Pour stocker les résultats de simulation pour chaque configuration.
    simulation_details = {}  # Pour sauvegarder les détails de chaque configuration (pour résumé).

    for combo in all_combos:
        # Extraction du nom de la configuration, par exemple "Geom_S1_based_on_schema - Mat_S1".
        combo_name = combo["config_name"]
        
        # Extraction de la géométrie.
        # Dans le JSON, la structure est : 
        # "geometry": { "config_name": "...", "geometry": { ... } }
        # On extrait le dictionnaire de paramètres géométriques.
        geometry_dict = combo["geometry"]["geometry"]
        
        # Extraction de la configuration matérielle.
        # Dans le JSON, "material" contient "MATERIALS_CONFIG" qui est une liste de dictionnaires.
        material_config_list = combo["material"]["MATERIALS_CONFIG"]
        
        # Conversion de la liste en DataFrame (ce qui est requis par build_material_configuration_dynamic).
        df_config = pd.DataFrame(material_config_list)
        
        # Extraction d'éventuels RI_OVERRIDES (pour des ajustements spécifiques de l'indice).
        ri_overrides = combo["material"].get("RI_OVERRIDES", {})

        # 4. Simulation pour cette configuration en appelant simulate_reflectance_single.
        Rup, Rdown = simulate_reflectance_single(
            lambda_range, geometry_dict, wave, df_config, json_combined_path, n_mod, ri_overrides
        )
        
        # Stockage des résultats dans le dictionnaire results, clé = nom de configuration.
        results[combo_name] = (Rup, Rdown)

        # Stockage des détails pour le résumé
        simulation_details[combo_name] = {
            "geometry": geometry_dict,
            "material_config": df_config.to_dict(orient="records"),
            "ri_overrides": ri_overrides,
            "Rup": Rup,
            "Rdown": Rdown
        }

    # 4) Construction du nom de fichier résumé
    # Ordre général souhaité : env, reso, diélectrique, molecule, functionalisation, accroche, metallic layer, sub.
    roles_order = [
        "perm_env",
        "perm_reso",
        "perm_dielec",
        "perm_mol",
        "perm_func",
        "perm_accroche",
        "perm_metalliclayer",
        "perm_sub"
    ]
    suffix_parts = []
    if simulation_details:
        # On utilise la configuration matérielle du premier combo (supposé représentatif)
        first_combo = next(iter(simulation_details.values()))
        for role in roles_order:
            val = ""
            # Recherche dans material_config de l'entrée correspondant au rôle
            for entry in first_combo["material_config"]:
                if entry.get("key", "").strip() == role:
                    mat = entry.get("material", {})
                    mtype = mat.get("type", "").strip().lower()
                    if mtype == "none" or mtype == "":
                        val = ""
                    elif mtype == "standard":
                        val = mat.get("material", "").strip()
                    elif mtype == "custom":
                        val = mat.get("expression", "").strip()
                    break

            # Si la valeur est "None" on ne l'inclut pas.
            # Ici, on suppose que si la valeur ne correspond pas à une formule chimique simple
            # (lettres, chiffres, éventuellement des points ou des astérisques), on l'ignore.
            if val.lower() == "none" or val == "":
                suffix_parts.append("")
            else:
                # On conserve uniquement les caractères alphanumériques, points, astérisques et signes de multiplication
                # (pour conserver par exemple "1.45**2")
                val_clean = re.sub(r'[^A-Za-z0-9\.\*\+]', '', val)
                suffix_parts.append(val_clean)
    # On joint les valeurs avec un underscore en respectant l'ordre
    # On ignore les segments vides
    filtered_parts = [part for part in suffix_parts if part]
    material_str_clean = "_".join(filtered_parts)

    summary_filename = os.path.join(summary_dir, f"simulation_summary_{material_str_clean}.txt")

    # 5) Création du résumé dans un fichier texte
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
