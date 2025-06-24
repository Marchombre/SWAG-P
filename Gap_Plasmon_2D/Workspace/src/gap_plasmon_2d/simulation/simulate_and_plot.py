#!/usr/bin/env python3

"""
Module: simulate_and_plot.py

Ce module simule la réflectance pour toutes les combinaisons géométrie/matériaux 
ou uniquement pour un sous-ensemble choisi et construit un graphique récapitulatif 
(composé du tracé des spectres et d'un tableau). Le résumé de simulation et la figure 
sont sauvegardés via les fonctions utilitaires de Saving_Functions.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from gap_plasmon_2d.simulation.simulate_reflectance import simulate_reflectance_single


# Liste ordonnée des paramètres géométriques avec leur libellé convivial
ordered_params = [
    ("thick_super",         "Superstrate"),          # épaisseur du superstrat
    ("thick_reso",          "Nanocube height"),      # hauteur du nanocube résonant
    ("width_reso",          "Nanocube width"),       # largeur du nanocube
    ("thick_gap",           "Gap (polymer)"),        # épaisseur de polymère entre strate
    ("thick_mol",           "Molecule"),             # épaisseur de couche moléculaire
    ("thick_func",          "Functionalisation"),    # épaisseur de couche fonctionnalisante
    ("thick_diel",          "Dielectric"),           # épaisseur du diélectrique
    ("thick_metalliclayer", "Metallic"),             # épaisseur de la couche métallique
    ("thick_accroche",      "Accroche"),             # épaisseur de la couche d’accroche
    ("thick_sub",           "Substrate"),            # épaisseur du substrat
    ("period",              "Period")                # période du réseau
]
# ──────────────────────────────────────────────────────────────────────────────
# Ce tableau permet de donner un ordre constant et des noms lisibles dans
# les résumés de géométrie ou dans le tableau final.
# ──────────────────────────────────────────────────────────────────────────────

def format_geometry_summary(geometry):
    """
    Formate un résumé de la géométrie à partir d'un dictionnaire.

    - geometry : dict avec clés correspondant aux clés de ordered_params
    - renvoie une chaîne multi-lignes “DispName: valeur”
    """
    # On parcourt ordered_params et on ne conserve que les clés présentes
    # pour produire une ligne “DispName: valeur” par paramètre.
    return "\n".join(
        f"{disp}: {geometry.get(key, 'NA')}"
        for key, disp in ordered_params
        if key in geometry
    )

def format_material_summary(material_config_list):
    """
    Formate un résumé des matériaux à partir d'une liste de configurations.

    - material_config_list : liste de dict, chacun décrivant un matériau
    - renvoie une chaîne multi-lignes “DispName: matériau/expressions”
    """
    lines = []  # liste des lignes de résumé à construire
    for entry in material_config_list:
        key = entry.get("key", "")
        # on cherche le disp_name correspondant dans ordered_params, sinon on prend la clé brute
        disp_name = next((dname for k, dname in ordered_params if k == key), key)
        mat = entry.get("material", {})  # dict décrivant le matériau
        mtype = mat.get("type", "").strip().lower()  # standard ou custom
        if mtype == "standard":
            val = mat.get("material", "").strip()     # nom du matériau standard
        elif mtype == "custom":
            val = mat.get("expression", "").strip()   # expression personnalisée
        elif mtype == "refractiveindex":
            book = mat.get("book", "")
            page = mat.get("page", "")
            val = f"Book: {book}, Page: {page}"            
        else:
            val = ""                                  # cas non défini
        if val:
            # on ajoute la ligne “DispName: valeur”
            lines.append(f"{disp_name}: {val}")
    # on joint toutes les lignes par un saut de ligne
    return "\n".join(lines)

def build_simulation_figure(simulation_details, lambda_range, title, all_configs):
    """
    Construit une figure matplotlib composée :

      1) d’un tracé des spectres de réflectance (axe du haut)
      2) d’un tableau récapitulatif des paramètres géométriques et matériaux (axe du bas)

    Paramètres :
      simulation_details (dict) : pour chaque config_name, un dict avec Rup, geometry, material_config…
      lambda_range     (array)  : vecteur des longueurs d’onde simulées
      title            (str)    : titre de la figure
      all_configs      (list)   : liste de configurations (dict contenant “config_name”)
    Retour :
      fig : instance matplotlib.figure.Figure
    """
    # initialisation des listes qui deviendront colonnes du tableau
    config_labels      = []  # libellés de colonnes (noms des configurations)
    geometry_summaries = []  # résumé géométrie par config
    material_summaries = []  # résumé matériaux par config

    # palette de couleurs cyclique de matplotlib
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # ──────────────────────────────────────────────────────────────────────────
    # On prépare d’abord tous les labels et résumés à afficher
    # ──────────────────────────────────────────────────────────────────────────
    for config in all_configs:
        combo_name = config["config_name"]
        # remplacement de “ - ” par saut de ligne pour alléger la légende
        config_labels.append(combo_name.replace(" - ", "\n"))
        details = simulation_details.get(combo_name, {})

        # formattage de la géométrie
        geometry_summaries.append(
            format_geometry_summary(details.get("geometry", {}))
        )
        # formattage des matériaux
        material_summaries.append(
            format_material_summary(details.get("material_config", []))
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Création de la figure et définition d’une grille 2x1
    # ──────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 10))
    # GridSpec : 2 lignes × 1 colonne, ratio hauteur 3:2.5
    gs  = GridSpec(2, 1, height_ratios=[3, 2.5])

    # ──────────────────────────────────────────────────────────────────────────
    # 1) Tracé des spectres de réflectance sur le premier sous-plot (ax1)
    # ──────────────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    for idx, config in enumerate(all_configs):
        combo_name = config["config_name"]
        # récupération des données Rup pour cette config
        Rup = simulation_details.get(combo_name, {}).get("Rup")
        if Rup is not None:
            # choix cyclique de la couleur
            color = colors[idx % len(colors)]
            # tracé de Rup vs lambda_range
            ax1.plot(lambda_range, Rup,
                     label=config_labels[idx],
                     color=color)
    # étiquettes et grille
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Reflectance")
    ax1.set_title(title)
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True)

    # ──────────────────────────────────────────────────────────────────────────
    # 2) Construction du tableau récapitulatif sur le second sous-plot (ax2)
    # ──────────────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')  # on masque les axes pour ne garder que le tableau

    if config_labels:
        # données du tableau : première ligne = geometry, deuxième ligne = material
        table_data = [geometry_summaries, material_summaries]
        table = ax2.table(
            cellText=table_data,
            colLabels=config_labels,
            rowLabels=["Geometry", "Material"],
            loc="center",
            cellLoc="left"
        )
        # forcer la taille de police et ajuster colonnes
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(col=list(range(len(config_labels))))

        # personnalisation du style des cellules
        for (row, col), cell in table.get_celld().items():
            if row == -1:
                # en-têtes de colonnes
                cell.set_facecolor("#40466e")
                cell.set_text_props(weight='bold', color='white',
                                    fontsize=10, ha='center')
            elif col == -1:
                # en-têtes de lignes
                cell.set_facecolor("#40466e")
                cell.set_text_props(weight='bold', color='white',
                                    fontsize=10)
            else:
                # cellules de contenu
                cell.set_facecolor("whitesmoke")
                cell.set_edgecolor("lightgray")
                cell.set_linewidth(0.5)

        # coloration du texte des cellules selon la colonne
        for (row, col), cell in table.get_celld().items():
            if row >= 0 and col >= 0:
                cell.get_text().set_color(
                    colors[col % len(colors)]
                )

        # calcul dynamique de la hauteur de chaque ligne en fonction
        # du nombre de lignes de texte (sauts de ligne)
        row_heights = {}
        for (row, col), cell in table.get_celld().items():
            if row >= 0:
                nb_lines = cell.get_text().get_text().count('\n') + 1
                row_heights[row] = max(row_heights.get(row, 0), nb_lines)
        # application de la hauteur calculée
        for (row, col), cell in table.get_celld().items():
            if row in row_heights:
                cell.set_height(0.04 * row_heights[row])
    else:
        # message affiché si aucune configuration n’est présente
        ax2.text(0.5, 0.5, "Aucune configuration simulée",
                 horizontalalignment='center')

    # ajustement final de la mise en page pour éviter les chevauchements
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # on retourne la figure prête à être affichée ou sauvegardée
    return fig

def run_simulation_one_combo(lam_range, wave, n_mod, combo, json_combined_path):
    """
    Simule la réflectance pour une configuration unique.

    Utilise les informations de géométrie et de matériau contenues dans combo.

    Returns:
        Rup          : réflectance incidente simulée (liste ou array)
        Absorption   : absorption calculée = 1 - (Rup + Rdown)
        simulation_details : dict complet avec geometry, material_config, Rup, Rdown, Absorption
    """
    # ──────────────────────────────────────────────────────────────────────────
    # Extraction des paramètres de géométrie et de matériau depuis le dict combo
    # ──────────────────────────────────────────────────────────────────────────
    geometry             = combo["geometry"]["geometry"]
    material_config_list = combo["material"]["MATERIALS_CONFIG"]
    # on convertit la liste de configs matériaux en DataFrame pour l’appel simulate_reflectance_single
    df_config    = pd.DataFrame(material_config_list)
    # eventuelles surcharges d’indices de réfraction
    ri_overrides = combo["material"].get("RI_OVERRIDES", {})

    # ──────────────────────────────────────────────────────────────────────────
    # Appel de la fonction de simulation pour une seule configuration
    # renvoie Rup (réflexion vers l’air) et Rdown (réflexion vers le substrat)
    # ──────────────────────────────────────────────────────────────────────────
    Rup, Rdown = simulate_reflectance_single(
        lam_range, geometry, wave,
        df_config, json_combined_path,
        n_mod, ri_overrides
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Calcul de l’absorption comme 1 – (Rup + Rdown), élément par élément
    # ──────────────────────────────────────────────────────────────────────────
    Rup_arr      = np.array(Rup)
    Rdown_arr    = np.array(Rdown)
    Absorption   = 1.0 - (Rup_arr + Rdown_arr)

    # ──────────────────────────────────────────────────────────────────────────
    # On assemble ensuite un dict 'simulation_details' qui contiendra
    # geometry, material_config, ri_overrides, Rup, Rdown, Absorption
    # ──────────────────────────────────────────────────────────────────────────
    simulation_details = {
        "geometry":        geometry,
        "material_config": df_config.to_dict(orient="records"),
        "ri_overrides":    ri_overrides,
        "Rup":             Rup,
        "Rdown":           Rdown,
        "Absorption":      Absorption
    }

    # on retourne Rup, Absorption et le dict complet pour usage ultérieur
    return Rup, Absorption, simulation_details

