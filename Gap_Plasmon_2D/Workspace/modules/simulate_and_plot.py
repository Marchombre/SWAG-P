#!/usr/bin/env python3
# simulate_and_plot.py

import os
import re
import json
import matplotlib.pyplot as plt

from simulate_reflectance import simulate_reflectance_all_combos
from Geometry_Material_Config import load_json_config


def build_ordered_material_list(material_details):
    """
    Extrait la liste ordonnée des matériaux à partir d'une configuration de matériau.
    L'ordre imposé est : 
      perm_env, perm_reso, perm_dielec, perm_mol, perm_func, perm_accroche, perm_metalliclayer, perm_sub.
    
    Pour chaque rôle, on récupère :
      - la valeur de "material" si type == "Standard"
      - la valeur de "expression" si type == "Custom"
      - on ignore si type == "None" ou vide.
    
    Retourne, par exemple, ["Silver", "1.45**2", "Gold", "ITO"].
    """
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
    material_list = []
    if material_details and "MATERIALS_CONFIG" in material_details:
        for role in roles_order:
            val = ""
            for entry in material_details["MATERIALS_CONFIG"]:
                if entry.get("key", "").strip() == role:
                    mat_info = entry.get("material", {})
                    mtype = mat_info.get("type", "").strip().lower()
                    if mtype in ("none", ""):
                        val = ""
                    elif mtype == "standard":
                        val = mat_info.get("material", "").strip()
                    elif mtype == "custom":
                        val = mat_info.get("expression", "").strip()
                    break  # Passage au rôle suivant dès qu'une entrée correspondante est trouvée
            if val:
                material_list.append(val)
    return material_list

def run_simulation_all_combos(lambda_range, wave, n_mod, json_combined_path, geom_mat_combinations_path=None):
    """
    Exécute la simulation de réflectance pour toutes les combinaisons géométrie-matériaux et affiche le résultat.
    
    Le titre de la figure et le nom du fichier intègrent les informations de configuration
    extraites du fichier geom_mat_combinations.json, qui est chargé depuis CONFIGURATIONS_dir
    si geom_mat_combinations_path n'est pas fourni.
    
    Paramètres
    ----------
    lambda_range : array_like
        Plage de longueurs d'onde (en nm).
    wave : dict
        Paramètres de l'onde (angle, polarisation, etc.).
    n_mod : int
        Nombre de modes RCWA.
    json_combined_path : str
        Chemin vers le fichier JSON combiné (données ExpData/BrendelBormann).
    geom_mat_combinations_path : str, optionnel
        Chemin vers le fichier geom_mat_combinations.json. Si None, il sera chargé depuis CONFIGURATIONS_dir.
    
    Retourne
    -------
    results : dict
        Dictionnaire {combo_name: (Rup_values, Rdown_values)} renvoyé par simulate_reflectance_all_combos.
    """
    # 1) Exécution de la simulation multi-combos
    results = simulate_reflectance_all_combos(lambda_range, wave, n_mod, json_combined_path)
    
    # 2) Chargement du fichier geom_mat_combinations.json
    if geom_mat_combinations_path is None:
        geom_mat_data = load_json_config("geom_mat_combinations.json")
    else:
        with open(geom_mat_combinations_path, "r", encoding="utf-8") as f:
            geom_mat_data = json.load(f)
    
    all_combined_configs = geom_mat_data.get("ALL_COMBINED_CONFIGS", [])
    config_strings = []
    for config in all_combined_configs:
        # Extraction de la configuration matériau (la clé "material")
        material_details = config.get("material", {})
        config_name = material_details.get("config_name", "")
        mat_list = build_ordered_material_list(material_details)
        if mat_list:
            # Construit une chaîne comme "Mat_S1: Silver 1.45**2 Gold ITO"
            config_str = f"{config_name}: " + " ".join(mat_list) if config_name else " ".join(mat_list)
            config_strings.append(config_str)
    if config_strings:
        title_material_str = " | ".join(config_strings)
    else:
        title_material_str = "NoMaterial"
    
    # Construction du nom de fichier en remplaçant espaces et caractères spéciaux
    filename_material_str = re.sub(r'\s+', '_', title_material_str)
    filename_material_str = re.sub(r'[^A-Za-z0-9_]', '', filename_material_str)
    
    # 3) Création de la figure
    plt.figure(figsize=(10, 6))
    for combo_name, (Rup, Rdown) in results.items():
        plt.plot(lambda_range, Rup, label=f'Rup - {combo_name}')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title(f"Simulation Reflectance: {title_material_str}")
    plt.legend()
    plt.grid(True)
    
    # 4) Sauvegarde de la figure dans le dossier Figures
    module_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(module_dir)
    figures_dir = os.path.join(workspace_dir, "Figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    fig_path = os.path.join(figures_dir, f"reflectance_{filename_material_str}.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.show()
    print(f"Figure sauvegardée dans : {fig_path}")
    
    return results

if __name__ == "__main__":
    import numpy as np
    # Exemple d'appel
    lambda_range = np.linspace(450, 1000, 200)
    wave = {"angle": 0, "polarization": 1}
    n_mod = 10  # Valeur fixée d'après l'étude de convergence de n_mod
    json_combined_path = "path/to/json_combined.json"  # Chemin vers votre JSON combiné
    # geom_mat_combinations sera chargé depuis CONFIGURATIONS_dir via load_json_config
    results = run_simulation_all_combos(lambda_range, wave, n_mod, json_combined_path)
