#!/usr/bin/env python3
"""
Module: Saving_Functions.py

Ce module encapsule les fonctions dédiées à la génération des noms de fichiers pour les
résumés de simulation et les figures, basées sur la configuration matérielle.
"""

import os
import re
from datetime import datetime
import numpy as np

from data_readers import get_material_str_clean



def save_simulation_summary(simulation_details, lambda_range, wave, n_mod, summary_dir, custom_name=None,
                            fwhm_summaries=None, lam_summaries=None, S_lam_summaries=None,
                            Q_factor=None, raw_score_summaries=None, comp_summaries=None):
    """
    Enregistre le résumé de simulation dans un fichier texte.

    - simulation_details : dict {config_name: details_dict}
    - lambda_range        : array des longueurs d'onde
    - wave                : dict des paramètres d'onde
    - n_mod               : int OU list[int] de même ordre que simulation_details
    - summary_dir         : dossier de sauvegarde
    - custom_name         : nom personnalisé (sinon auto-généré)
    - *_summaries         : listes de métriques (alignées sur simulation_details)
    """
    # --- construction du nom de fichier ---
    if custom_name and custom_name.strip():
        base_custom = custom_name.strip()
        if not base_custom.startswith("simulation_summary_RCWA_"):
            base_custom = "simulation_summary_RCWA_" + base_custom
        base_filename = base_custom
    else:
        material_str_clean = get_material_str_clean(simulation_details)
        base_filename = f"simulation_summary_RCWA_{material_str_clean}"
    summary_filename = os.path.join(summary_dir, f"{base_filename}.txt")

    # --- entête ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = [
        "Simulation Summary - All Geometry/Material Combos",
        f"Timestamp: {timestamp}",
        f"Wave parameters: {wave}"
    ]

    # gère n_mod int ou list
    multiple = isinstance(n_mod, (list, tuple, np.ndarray))
    if not multiple:
        lines.append(f"Number of RCWA modes: {n_mod}\n")

    lines.append("---- COMBINATIONS ----\n")

    # --- bouclage sur chaque configuration ---
    for idx, (combo_name, details) in enumerate(simulation_details.items()):
        lines.append(f"Combo name: {combo_name}")
        if multiple:
            nm = n_mod[idx]
            lines.append(f"  Number of RCWA modes: {nm}")
        lines.append("Geometry:")
        lines.append(str(details["geometry"]))
        lines.append("Material config (df_config):")
        lines.append(str(details["material_config"]))
        lines.append(f"RI Overrides: {details['ri_overrides']}")

        # métriques
        if fwhm_summaries is not None:
            lines.append("Metrics:")
            lines.append(f"  FWHM          : {fwhm_summaries[idx]}")
        if lam_summaries is not None:
            lines.append(f"  Lam_res      : {lam_summaries[idx]}")
        if S_lam_summaries is not None:
            lines.append(f"  S_lam        : {S_lam_summaries[idx]}")
        if Q_factor is not None:
            lines.append(f"  Q-factor     : {Q_factor[idx]}")
        if raw_score_summaries is not None:
            lines.append(f"  Score interne: {raw_score_summaries[idx]}")
        if comp_summaries is not None:
            lines.append(f"  Score total  : {comp_summaries[idx]}")

        # reflectance points
        lines.append("Reflectance points (Rup, Rdown):")
        if not (len(lambda_range) == len(details["Rup"]) == len(details["Rdown"])):
            raise ValueError(
                f"Nombre de points mismatch pour '{combo_name}': "
                f"{len(lambda_range)} vs {len(details['Rup'])} vs {len(details['Rdown'])}"
            )
        for i_pt, lam_pt in enumerate(lambda_range):
            lines.append(f"  λ={lam_pt} nm -> Rup={details['Rup'][i_pt]}, Rdown={details['Rdown'][i_pt]}")
        lines.append("-" * 40)

    # --- écriture du fichier ---
    os.makedirs(summary_dir, exist_ok=True)
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
