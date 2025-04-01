#!/usr/bin/env python3
"""
Module: Saving_function.py

Ce module encapsule les fonctions dédiées à la génération des noms de fichiers pour les
résumés de simulation et les figures, basées sur la configuration matérielle.
"""

import os
import re
from datetime import datetime

def get_material_str_clean(simulation_details):
    """
    Extrait et retourne une chaîne (material_str_clean) construite à partir des configurations
    matérielles (la première configuration) contenues dans simulation_details.
    
    On parcourt les clés dans l'ordre spécifié et on nettoie les valeurs pour ne conserver
    que les caractères alphanumériques, points, astérisques et signes plus.
    """
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
                    if mtype == "standard":
                        val = mat_info.get("material", "").strip()
                    elif mtype == "custom":
                        val = mat_info.get("expression", "").strip()
                    break
            if val.lower() != "none" and val != "":
                val_clean = re.sub(r'[^A-Za-z0-9\.\*\+]', '', val)
                suffix_parts.append(val_clean)
    filtered_parts = [part for part in suffix_parts if part]
    return "_".join(filtered_parts)

def save_simulation_summary(simulation_details, lambda_range, wave, n_mod, summary_dir):
    """
    Enregistre le résumé de simulation dans un fichier texte.
    Le nom du fichier est construit en utilisant le material_str_clean extrait de simulation_details.
    
    Retourne le chemin du fichier enregistré.
    """
    material_str_clean = get_material_str_clean(simulation_details)
    summary_filename = os.path.join(summary_dir, f"simulation_summary_RCWA_V1_{material_str_clean}.txt")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    print(f"Résumé de la simulation sauvegardé dans : {summary_filename}")
    return summary_filename

def save_figure(fig, title, figures_dir, material_str_clean=None):
    """
    Enregistre la figure 'fig' dans le dossier 'figures_dir'.
    Si material_str_clean est fourni et non vide, il est utilisé pour nommer la figure ;
    sinon, le titre est nettoyé et utilisé comme nom.
    
    Retourne le chemin du fichier enregistré.
    """
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    if material_str_clean:
        filename_tag = material_str_clean
    else:
        filename_tag = re.sub(r'[^A-Za-z0-9_]', '', title)
    fig_path = os.path.join(figures_dir, f"Simulation_Reflectance_Spectra_{filename_tag}.png")
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Figure saved in: {fig_path}")
    return fig_path
