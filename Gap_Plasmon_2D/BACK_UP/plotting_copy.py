#!/usr/bin/env python3
# -*- coding: utf‑8 -*-
"""
plotting.py  – Onglet « Plot » de l’application interactive

Fonctionnalités
---------------
• Sélection multiple de spectres enregistrés (simulation ou expérimental)
• Choix fin des métriques et overlays à afficher (identique à l’onglet Simulation)
• Affichage optionnel du spectre Rup_dn (calculé avec n₀+Δn) lorsqu’il existe
• Tableau synthétique à hauteur dynamique, mise en forme cohérente
• Export de la figure en .png
"""

# ------------------------------------------------------------------ #
#                             Imports                                #
# ------------------------------------------------------------------ #
import os, io, base64
from datetime import datetime
import re
import numpy     as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import Layout, HBox, VBox, HTML
from IPython.display import HTML as DHTML, display
from pyparsing import line
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from data_readers       import get_all_spectra_and_summaries
from simulate_and_plot  import ordered_params
from Characterization   import _find_dip_core, compute_half_point


# ------------------------------------------------------------------ #
#                        Chemins par défaut                           #
# ------------------------------------------------------------------ #
module_dir    = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(module_dir)
notebooks_dir = os.path.join(workspace_dir, "notebooks")
summary_dir   = os.path.join(notebooks_dir, "Summary_Simulation")
exp_data_dir  = os.path.join(notebooks_dir, "Experimental_Data")


# ------------------------------------------------------------------ #
#                     utilitaire download link                       #
# ------------------------------------------------------------------ #
def _download_link(fig, fname="plot.png"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return DHTML(f'<a download="{fname}" href="data:image/png;base64,{b64}" '
                 f'target="_blank">Télécharger l’image</a>')


# ------------------------------------------------------------------ #
#                     utilitaire pour netoyer les labels             #
# ------------------------------------------------------------------ #
def clean_config_label(label):
    """
    Enlève la partie ' ( ... )' à la fin d'un label, typique des configs géométrie/matériau
    Exemple : "Structure 1 - ITO / AU (10nm) (Structure 1 - ITO )" → "Structure 1 - ITO / AU (10nm)"
    """
    return re.sub(r'\s*\([^\)]*\)\s*$', '', label or '')


# ------------------------------------------------------------------ #
#                    Construction de l’onglet Plot                   #
# ------------------------------------------------------------------ #
def create_plot_tab():
    global custom_labels_dict, custom_labels_dn_dict, labels_editors_box, update_labels_btn

    custom_labels_dict = {}  # {original_label: custom_label}
    custom_labels_dn_dict = {}   # {original_label: custom_label for Rup_dn}
    custom_marker_labels = {}  # {label: {type: valeur}}
    custom_colors = {}           # {lab: "#rrggbb"}
    
    # --------------------- widgets principaux ---------------------- #
    spectra_select = widgets.SelectMultiple(
        options=[], description="Available spectra:",
        layout=Layout(width='80%', height='150px'),
        style={'description_width':'initial'})
    spectra_select.observe(lambda change: update_label_editors(), names="value")

    verbose_chk = widgets.Checkbox(
        value=True, description="Verbose",
        layout=Layout(width='100%'), indent=False,
        style={'description_width':'initial'})

    draw_b   = widgets.Button(description="Draw", button_style="info")
    
    # nouvelle checkbox pour afficher les labels sur le plot
    show_labels_chk = widgets.Checkbox(
        value=False,
        description="Show labels",
        indent=False,
        layout=Layout(width='auto'),
        style={'description_width':'initial'}
    )
    
    # === Variables globales pour l’édition interactive des labels ===
    
    custom_labels_dict = {}
    custom_labels_dn_dict = {}
    labels_editors_box = widgets.VBox()
    update_labels_btn = widgets.Button(
        description="Mettre à jour les labels",
        button_style="primary",
        layout=Layout(width="170px")
)
   # contiendra les inputs
   
    def on_update_labels(_=None):
        for row in labels_editors_box.children:
            for txt in row.children:
                if isinstance(txt, widgets.ColorPicker):
                    custom_colors[txt._orig_lab] = txt.value
                    continue
                
                if hasattr(txt, "_carre_type"):
                    lab = txt._orig_lab
                    typ = txt._carre_type
                    if lab not in custom_marker_labels:
                        custom_marker_labels[lab] = {}
                    custom_marker_labels[lab][typ] = txt.value
                elif getattr(txt, "_is_dn", False):
                    custom_labels_dn_dict[txt._orig_lab] = txt.value
                else:
                    custom_labels_dict[txt._orig_lab] = txt.value
        _draw(None)


    update_labels_btn.on_click(on_update_labels)
    
    toggle_labels_editors_btn = widgets.ToggleButton(
        value=False,
        description="Afficher/masquer les éditeurs de labels",
        icon="chevron-down",
        layout=Layout(width="270px")
    )
    
    labels_editors_panel = VBox([
        HTML("<b>Modifier les labels des spectres :</b>"),
        labels_editors_box,
        update_labels_btn 
    ], layout=Layout(display='none'))  # Masqué par défaut
        
    def on_toggle_labels_panel(change):
        if change["new"]:
            labels_editors_panel.layout.display = "block"
            toggle_labels_editors_btn.icon = "chevron-up"
        else:
            labels_editors_panel.layout.display = "none"
            toggle_labels_editors_btn.icon = "chevron-down"

    toggle_labels_editors_btn.observe(on_toggle_labels_panel, names="value")
    

    def update_label_editors():
        labels = list(spectra_select.value) if 'spectra_select' in locals() else []
        if not labels and 'Rup_dict' in locals():
            labels = list(Rup_dict.keys())

        # Détermine les types de carré à afficher
        marker_types = []
        if show_half_level_metrics.value:
            marker_types = ["half-base", "half-dn"]
        else:
            marker_types = ["dip-base", "dip-dn"]

        children = []
        for lab in labels:
            row = []
            
            # ── ColorPicker ──
            col_init = custom_colors.get(lab, None)
            color_picker = widgets.ColorPicker(
                concise=True,
                description='Couleur:',
                value=col_init or '#1f77b4',  # couleur par défaut si non définie
                layout=Layout(width='120px')
            )
            color_picker._orig_lab = lab
            row.append(color_picker)


            # Label principal du spectre
            default_lab = custom_labels_dict.get(lab, clean_config_label(lab))
            txt = widgets.Text(value=default_lab, description="Label:", layout=Layout(width="180px"))
            txt._orig_lab = lab
            txt._is_dn = False
            row.append(txt)

            # Label spectre Δn (optionnel)
            if 'Rup_dn_dict' in locals() and lab in Rup_dn_dict and Rup_dn_dict[lab] is not None:
                default_dn = custom_labels_dn_dict.get(lab, clean_config_label(lab) + " (R + Δn)")
                txt_dn = widgets.Text(value=default_dn, description="Label Δn:", layout=Layout(width="180px"))
                txt_dn._orig_lab = lab
                txt_dn._is_dn = True
                row.append(txt_dn)

            # Labels pour tous les types de carrés pertinents
            for typ in marker_types:
                val = custom_marker_labels.get(lab, {}).get(typ, f"{clean_config_label(lab)} S_R {typ.replace('-', ' ')}")
                txt_carre = widgets.Text(
                    value=val,
                    description=f"Carré {typ}:", layout=Layout(width="160px")
                )
                txt_carre._orig_lab = lab
                txt_carre._carre_type = typ
                row.append(txt_carre)

            children.append(HBox(row, layout=Layout(margin="2px 0")))
        labels_editors_box.children = children



    # Met à jour dynamiquement quand la sélection change
    spectra_select.observe(lambda change: update_label_editors(), names="value")    
    
    debug_out = widgets.Textarea(
    value='',
    placeholder='Logs verbose…',
    layout=Layout(
        width='100%',
        height='200px',      # fixe la hauteur
        overflow_y='scroll', # barre de scroll verticale
        border='1px solid darkred'
    ),
    disabled=False  
    )
    
    def _toggle_dbg(ch):
        debug_out.layout.display = 'block' if ch['new'] else 'none'
        if not ch['new']:
            debug_out.value = ''      # on vide simplement le texte
    verbose_chk.observe(_toggle_dbg, names='value')

    # ------------------ métriques / overlays (identiques) ---------- #
    def _cb(val, desc): return widgets.Checkbox(value=val, description=desc)

    # ---- métriques tableau ----
    show_fwhm_chk        = _cb(False, "FWHM")
    show_lambda0_chk     = _cb(True,  r"λ0")
    show_delta_lam_over_midLam_chk       = _cb(False, r"Δλ / λmin or λsym")
    show_S_lambda_chk    = _cb(True,  "Sλ (nm/RIU)")
    show_S_dn_chk        = _cb(True,  r"ΔR/Δn (1/RIU)")
    show_deltaR_half_chk = _cb(True,  r"ΔR_half")
    show_Q_chk           = _cb(False, "Q‑factor")

    # ---- overlays graphiques ----
    show_Rup_dn_overlay_chk   = _cb(True,  "Rup_dn dashed")
    show_hlines_chk       = _cb(False, "half‑level line")
    show_dips_chk         = _cb(False, "dips (×)")
    show_maxima_chk       = _cb(False, "maxima (×)")
    show_symmetry_pts_chk = _cb(False, "symmetric pts (×)")
    show_selected_dip_chk = _cb(True,  "selected dip (○)")
    show_sensitivity_marker_chk = _cb(True, "sensitivity marker (□)")
    
    # ---- metrics additionnelle : sensitivités au niveau demi-hauteur
    show_half_level_metrics = _cb(False, "S from fwhm")
    show_half_level_metrics.style.description_width = 'auto'
    show_half_level_metrics.layout.margin  = '0'
    show_half_level_metrics.layout.padding = '0'
    show_half_level_metrics.indent         = False
    show_half_level_metrics.observe(lambda ch: update_label_editors(), names="value")



    # 1) On fixe pour chaque Checkbox
    for cb in (
        show_fwhm_chk, show_lambda0_chk, show_delta_lam_over_midLam_chk, show_S_lambda_chk,
        show_S_dn_chk, show_deltaR_half_chk, show_Q_chk,
        show_Rup_dn_overlay_chk, show_hlines_chk, show_dips_chk,
        show_maxima_chk, show_symmetry_pts_chk, show_selected_dip_chk, show_sensitivity_marker_chk
    ):
        # description_width fixe la place allouée au texte
        cb.style.description_width = '60px'    # jouez sur la valeur (px) pour adapter
        # layout.margin supprime tout écart extérieur
        cb.layout.margin = '0'                 
        cb.layout.padding = '0'
        cb.indent = False                      # supprime l’indentation “intérieure”

    # 2) On met les deux HBox sur une seule ligne, sans gap
    metrics_hbox = HBox(
        [show_fwhm_chk, show_lambda0_chk, show_delta_lam_over_midLam_chk, show_S_lambda_chk,
        show_S_dn_chk, show_deltaR_half_chk, show_half_level_metrics, show_Q_chk],
        layout=Layout(
            display='flex',
            flex_flow='row nowrap',
            justify_content='space-around',
            margin='0',
            padding='0'
        )
    )
    overlays_hbox = HBox(
        [show_Rup_dn_overlay_chk, show_hlines_chk, show_dips_chk,
        show_maxima_chk, show_symmetry_pts_chk, show_selected_dip_chk, show_sensitivity_marker_chk],
        layout=Layout(
            display='flex',
            flex_flow='row nowrap',
            justify_content='space-around',
            margin='0',
            padding='0'
        )
    )


    # ─── Widgets pour zoom en λ ─────────────
    lambda_min_text = widgets.FloatText(
        value=0.0, description="λ min (nm):",
        layout=Layout(width='150px'),
        style={'description_width':'initial'}
    )
    lambda_max_text = widgets.FloatText(
        value=0.0, description="λ max (nm):",
        layout=Layout(width='150px'),
        style={'description_width':'initial'}
    )
    range_box = HBox([lambda_min_text, lambda_max_text],
                        layout=Layout(grid_gap='10px'))


    # 3) On assemble le tout
    controls_box = VBox([
        HTML("<h3>Plot</h3>"),
        spectra_select,
        verbose_chk,
        HTML("<b>Métriques à afficher :</b>"),
        metrics_hbox,
        HTML("<b>Overlays graphiques :</b>"),
        overlays_hbox, 
        range_box,
        
        HBox([
            show_labels_chk,
            draw_b,
            VBox([
                toggle_labels_editors_btn,  # Le bouton qui permet d'afficher/masquer
                labels_editors_panel        # Le panneau masquable qui contient la zone d’édition
            ], layout=Layout(align_items="flex-start", min_width="260px"))
        ], layout=Layout(grid_gap='12px'))

                
    ], layout=Layout(width='100%'))



    # -------------------- zone figure / tableau -------------------- #
    plot_out = widgets.Output(
        layout=Layout(
            border='2px solid #ccc',
            padding='10px',
            align_items='stretch',       # étire au maxi
            display='flex',             # active le mode flex
            flex_flow='column nowrap',  # empile verticalement
            #align_items='center',       # centre horizontalement
            width='100%',              # largeur pleine
            height='auto',              # laisse la hauteur s'ajuster
            justify_content='center'    # centre verticalement (si tu as un peu de marge)
        )
    )


    # ---------------------------------------------------------------- #
    #                    variables partagées              #
    # ---------------------------------------------------------------- #
    Rup_dict  = {}       # {label: (lam, Rup)}
    Rup_dn_dict   = {}   # {label: (lam, Rup_dn) or None}
    summaries = {}       # {label: (geom, mat)}
    metrics   = {}       # {label: {metric: value}}
    delta_ns = {}

    def _update_spectra():
        nonlocal Rup_dict, Rup_dn_dict, summaries, metrics, delta_ns
        (Rup_dict, Rup_dn_dict, summaries, metrics, delta_ns) = \
            get_all_spectra_and_summaries(summary_dir, exp_data_dir, ordered_params)
        spectra_select.options = list(Rup_dict.keys())
    _update_spectra()

    # ---------------------------------------------------------------- #
    #                     callback principal « Draw »                   #
    # ---------------------------------------------------------------- #
    def _draw(_btn):
        _update_spectra()                          # refresh (au cas où)
        verbose = verbose_chk.value

        labels = list(spectra_select.value) or list(Rup_dict.keys())
        if not labels:
            return

        # ──────────────────────────────────────────────────────────
        # 1) Figure dédiée au plot
        # ──────────────────────────────────────────────────────────
        fig_plot = plt.figure(figsize=(12, 6))
        ax_plot  = fig_plot.add_axes([0.10, 0.10, 0.80, 0.85])  # occupe presque tout [ left, bottom, width, height ]
        # … ici, on laisse TOUS les appels ax_plot.plot(), scatter(), grid(), légende, etc. …

        # ──────────────────────────────────────────────────────────
        # 2) Figure dédiée au tableau
        # ──────────────────────────────────────────────────────────
        fig_table = plt.figure(figsize=(9, 4))
        ax_table  = fig_table.add_axes([0.10, 0.05, 0.80, 0.90])  # occupe tout sauf un petit haut [ left, bottom, width, height ]
        ax_table.axis('off')

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # ----- accumulateurs tableau -------------------------------- #
        geom_sum = []; mat_sum = []; cfg_labels=[]
        fwhm_sum = []; lam0_sum=[]; delta_lam_over_midLam=[]
        S_lambda_sum   = []; dRdn_sum=[]; dRhalf_sum=[]
        Q_sum    = []
        S_lam_min_vals=[]; S_lam_sym_vals=[]
        debug_lines=[]

        use_half = show_half_level_metrics.value
        best_config = {'name': None, 'SR': float('-inf')}

        # ============================================================ #
        #                  BOUCLE SUR CHAQUE SPECTRE                   #
        # ============================================================ #
        for idx, lab in enumerate(labels):
            lam, Rup = Rup_dict[lab]
            Rup_dn_tuple = Rup_dn_dict.get(lab)

            # 1) calcul des bornes d’affichage
            min_w, max_w = lambda_min_text.value, lambda_max_text.value
            # si 0 => on prend toute la plage
            low  = lam.min() if min_w <= 0 else max(min_w, lam.min())
            high = lam.max() if max_w <= 0 else min(max_w, lam.max())

            # 2) on crée lam_plot/Rup_plot pour TOUTES les courbes
            mask       = (lam >= low) & (lam <= high)
            lam_plot   = lam[mask]
            Rup_plot   = Rup[mask]

            # 3) idem pour Rup_dn (si existant), pour l’affichage uniquement
            if Rup_dn_tuple is not None:
                lam_dn_full, Rup_dn_full = Rup_dn_tuple
                mask_dn     = (lam_dn_full >= low) & (lam_dn_full <= high)
                lam_dn_plot = lam_dn_full[mask_dn]
                Rup_dn_plot = Rup_dn_full[mask_dn]
            else:
                lam_dn_full = Rup_dn_full = lam_dn_plot = Rup_dn_plot = None


            dR_over_dn_list = []
            dLam_over_dn_list = []

            mets = metrics.get(lab, {})
            try_FWHM = mets.get("FWHM", "")
            try_lam0 = mets.get("Lam_res", mets.get("lam0", ""))

            # 1) Détection des dips sur Rup
            (dip_list_idx, lam_dip_list, R_dip_list,
            y_level_list,
            lam_left_list, lam_right_list, fwhm_list,
            lam_max_l_list, R_max_l_list,
            lam_max_r_list, R_max_r_list,
            lam_sym_list, R_sym_list,
            depth_list), _ = _find_dip_core(
                wavelength=lam,
                reflectance=Rup,
                smooth_win=0, polyorder=0,
                dip_prom=1e-2, dip_dist=1, peak_dist=1,
                verbose=verbose, cfg_name=lab
            )

            # Aucun dip trouvé
            if not dip_list_idx:
                if verbose:
                    debug_lines.append(f"[Plot] Aucun dip détecté pour « {lab} »")
                continue


            # 2) Vérification de la présence des données Rup_dn et delta_n
            if Rup_dn_tuple and lab in delta_ns:
                lam_dn, Rup_dn_vals = Rup_dn_tuple
                delta_n = delta_ns[lab]
                # dips on Rup_dn
                (dip_idx_list_dn, lam_dip_list_dn, R_dip_list_dn, y_level_list_dn,
                 lam_left_list_dn, lam_right_list_dn, fwhm_list_dn,
                 lam_max_l_list_dn, R_max_l_list_dn,
                 lam_max_r_list_dn, R_max_r_list_dn,
                 lam_sym_list_dn, R_sym_list_dn,
                 depth_list_dn), _ = _find_dip_core(lam_dn, Rup_dn_vals,0,0,1e-2,1,1,verbose,lab)
                interp0 = interp1d(lam, Rup, kind='cubic', bounds_error=False, fill_value='extrapolate')
                interp1 = interp1d(lam_dn, Rup_dn_vals, kind='cubic', bounds_error=False, fill_value='extrapolate')
                min_dips = min(len(lam_dip_list), len(lam_dip_list_dn))

                dR_over_dn_list=[]; dLam_over_dn_list=[]
                
                for i in range(min_dips):
                    if use_half:
                        half_pt = compute_half_point(lam, Rup, lam_left_list[i], lam_right_list[i])
                        R0 = float(interp0(half_pt)); R1 = float(interp1(half_pt))
                        lam0=half_pt
                        lam1 = compute_half_point(lam_dn, Rup_dn_vals, lam_left_list_dn[i], lam_right_list_dn[i])
                    else:
                        lam0 = lam_dip_list[i]; R0 = R_dip_list[i]
                        lam1 = lam_dip_list_dn[i]; R1 = float(interp1(lam0)) # R1 = R_dip_list_dn[i]
                    dR_over_dn_list.append(abs(R0-R1)/delta_n)
                    dLam_over_dn_list.append(abs(lam0-lam1)/delta_n)
                best_idx = int(np.nanargmax(dR_over_dn_list))
                best_SR, best_S_lambda = dR_over_dn_list[best_idx], dLam_over_dn_list[best_idx]
            else:
                # ──────────────────────────────────────────────────────────
                #   Pas de Δn : on sélectionne le meilleur dip avec raw_score
                # ──────────────────────────────────────────────────────────


                R_array = Rup  # tableau NumPy des réflectances
                lam_array = lam  # tableau NumPy des longueurs d’onde

                # 2) Créer un interpolateur spline sur R_array
                interp_R = interp1d(
                    lam_array, R_array,
                    kind='cubic',
                    bounds_error=False,
                    fill_value='extrapolate'
                )

                # 3) Définir un petit pas "delta" = pas de la grille en λ
                delta = lam_array[1] - lam_array[0]

                # 4) Paramètres de pondération
                alpha = 2.0
                beta  = 0.5

                best_raw_score = -np.inf
                best_idx_raw   = None

                # On vide d’abord dR_over_dn_list et dLam_over_dn_list pour éviter d’ajouter en double
                dR_over_dn_list.clear()
                dLam_over_dn_list.clear()

                # 5) Pour chaque dip candidat j, calculer raw_score
                for j in range(len(dip_list_idx)):
                    depth_j = depth_list[j]
                    fwhm_j  = fwhm_list[j]
                    lam_left_j  = lam_left_list[j]
                    lam_right_j = lam_right_list[j]

                    # ─── Calcul de la pente fine à demi-hauteur (flanc le plus abrupt) ───

                    # pente à gauche : (R(lam_left + delta) - R(lam_left - delta)) / (2*delta)
                    y_plus_L  = interp_R(lam_left_j + delta)
                    y_minus_L = interp_R(lam_left_j - delta)
                    slope_left  = abs((y_plus_L - y_minus_L) / (2 * delta))

                    # pente à droite : (R(lam_right + delta) - R(lam_right - delta)) / (2*delta)
                    y_plus_R  = interp_R(lam_right_j + delta)
                    y_minus_R = interp_R(lam_right_j - delta)
                    slope_right = abs((y_plus_R - y_minus_R) / (2 * delta))

                    # on retient le flanc le plus abrupt
                    slope = max(slope_left, slope_right)

                    # ─── Calcul du raw_score ───
                    # on ajoute un petit epsilon pour éviter division par zéro si fwhm_j = 0
                    raw_score = (depth_j ** alpha) * (slope ** (1.0 - depth_j)) / (fwhm_j ** beta + 1e-12)

                    if raw_score > best_raw_score:
                        best_raw_score = raw_score
                        best_idx_raw   = j

                    # comme on n’a pas de Δn ici, on stocke 0.0 pour la cohérence du tableau
                    dR_over_dn_list.append(0.0)
                    dLam_over_dn_list.append(0.0)

                # 6) On retient l’indice j qui maximise raw_score
                best_idx = best_idx_raw
                best_SR = None
                best_S_lambda = None

                if verbose:
                    debug_lines.append(
                        f"[Plot] Pas de Δn → sélection via raw_score, indice retenu = {best_idx}"
                    )

                
        

            # 4) Extraction des métriques du dip retenu
            lam_left = lam_left_list[best_idx]
            lam_right = lam_right_list[best_idx]
            fwhm = fwhm_list[best_idx]
            lam_dip = lam_dip_list[best_idx]
            R_dip = R_dip_list[best_idx]
            ylev = y_level_list[best_idx]
            lam_m_l = lam_max_l_list[best_idx]
            Rm_l = R_max_l_list[best_idx]
            lam_m_r = lam_max_r_list[best_idx]
            Rm_r = R_max_r_list[best_idx]
            lam_sym = lam_sym_list[best_idx]
            R_sym = R_sym_list[best_idx]
            depth = depth_list[best_idx]

            lam_max_l_list = np.array(lam_max_l_list)
            R_max_l_list = np.array(R_max_l_list)
            lam_max_r_list = np.array(lam_max_r_list)
            R_max_r_list = np.array(R_max_r_list)
            lam_sym_list = np.array(lam_sym_list)
            R_sym_list = np.array(R_sym_list)
            width_list = np.array(fwhm_list)
            depth_list = np.array(depth_list)

            lam_min = lam_m_l if Rm_l < Rm_r else lam_m_r
            lam_mid = lam_left if Rm_l < Rm_r else lam_right
            S_lam_min_abs = abs((lam_dip - lam_min) / lam_mid)
            S_lam_sym_abs = abs((lam_dip - lam_sym) / lam_mid)
            S_lam_min_vals.append(S_lam_min_abs)
            S_lam_sym_vals.append(S_lam_sym_abs)

            # Tracé graphique
            # priorité à la couleur custom si définie, sinon palette par défaut
            color = custom_colors.get(lab,
                     colors[idx % len(colors)])
            
            # Tracé principal restreint :
            label_affiche = custom_labels_dict.get(lab, clean_config_label(lab))
            
            ax_plot.plot(lam_plot, Rup_plot, color=color, label=label_affiche, zorder=1)



            # Dips :
            if show_dips_chk.value:
                # on ne montre que les dips DANS le range
                dip_lams  = lam[dip_list_idx]
                dip_rups  = Rup[dip_list_idx]
                dip_mask  = (dip_lams >= low) & (dip_lams <= high)
                ax_plot.scatter(dip_lams[dip_mask], dip_rups[dip_mask],
                                marker='x', s=40, color=color, zorder=3)


            if show_hlines_chk.value:
                # demi-hauteur sur Rup
                if lam_left_list[best_idx] >= low and lam_right_list[best_idx] <= high:
                    ax_plot.hlines(
                        y_level_list[best_idx],
                        lam_left_list[best_idx],
                        lam_right_list[best_idx],
                        linewidth=2, colors=color, zorder=2
                    )
                # idem sur Rup_dn dans le range
                if Rup_dn_tuple is not None and lam_left_list_dn[best_idx]>=low and lam_right_list_dn[best_idx]<=high:
                    ax_plot.hlines(
                        y_level_list_dn[best_idx],
                        lam_left_list_dn[best_idx],
                        lam_right_list_dn[best_idx],
                        linewidth=2, colors=color, zorder=2
                    )


                    
            if show_maxima_chk.value:
                ax_plot.scatter(lam_max_l_list, R_max_l_list, marker='x', s=30, color=color, zorder=3)
                ax_plot.scatter(lam_max_r_list, R_max_r_list, marker='x', s=30, color=color, zorder=3)

            if show_symmetry_pts_chk.value:
                ax_plot.scatter(lam_sym_list, R_sym_list, marker='x', s=30, color=color, zorder=3)

            if show_selected_dip_chk.value:
                # dip sur Rup
                ax_plot.scatter([lam_dip], [R_dip],
                                marker='o', s=70,
                                facecolor='none', edgecolor=color,
                                linewidths=2, zorder=4)
                # dip sur Rup_dn uniquement si disponible
                if Rup_dn_tuple is not None:
                    ax_plot.scatter([lam_dip_list_dn[best_idx]],
                                    [R_dip_list_dn[best_idx]],
                                    marker='o', s=70,
                                    facecolor='none', edgecolor=color,
                                    linewidths=2, zorder=4)


            if show_Rup_dn_overlay_chk.value and lam_dn_plot is not None:
                good = ~np.isnan(Rup_dn_plot)
                label_dn = custom_labels_dn_dict.get(lab, clean_config_label(lab) + " (R + Δn)")
                ax_plot.plot(lam_dn_plot[good], Rup_dn_plot[good], "--",
                            linewidth=2, color=color, alpha=0.7,
                            label=label_dn, zorder=0)


                
            if show_sensitivity_marker_chk.value and Rup_dn_tuple is not None:
                # on récupère d’abord les coordonnées de base (sur tout le spectre)
                if use_half:
                    # demi‐hauteur sur Rup
                    lam0 = compute_half_point(
                        lam, Rup,
                        lam_left_list[best_idx], lam_right_list[best_idx]
                    )
                    R0 = float(interp0(lam0))
                    # même λ sur Rup_dn
                    R1 = float(interp1(lam0))
                    # demi‐hauteur sur Rup_dn
                    lam1 = compute_half_point(
                        lam_dn_full, Rup_dn_full,
                        lam_left_list_dn[best_idx], lam_right_list_dn[best_idx]
                    )
                    y1 = y_level_list_dn[best_idx]
                    # on ne trace que si lam0 et lam1 sont dans [low,high]
                    if low <= lam0 <= high:
                        ax_plot.scatter([lam0], [R0],
                                        marker='s', s=70,
                                        facecolor='none', edgecolor=color, alpha=0.7,
                                        label=custom_marker_labels.get(lab, {}).get("half-base", 
                                                                                    f"{clean_config_label(lab)} S_R half R(λ; n)"))
                                      
                        ax_plot.scatter([lam0], [R1],
                                        marker='s', s=70,
                                        facecolor='none', edgecolor=color, alpha=0.7)
                    if low <= lam1 <= high:
                        ax_plot.scatter([lam1], [y1],
                                        marker='s', s=70,
                                        facecolor='none', edgecolor=color, alpha=0.7,
                                        label=custom_marker_labels.get(lab, {}).get("half-dn", 
                                                                                    f"{clean_config_label(lab)} S_R half R(λ; n+Δn)"))
                                        
                else:
                    # dip-mode
                    lam0 = lam_dip_list[best_idx]
                    R0   = R_dip_list[best_idx]
                    R1_at_lam0 = float(interp1(lam0))
                    lam1 = lam_dip_list_dn[best_idx]
                    R1   = R_dip_list_dn[best_idx]
                    # mêmes conditions de masque
                    if low <= lam0 <= high:
                        ax_plot.scatter([lam0], [R0],
                                        marker='s', s=70,
                                        facecolor='none', edgecolor=color, alpha=0.7,
                                        label=custom_marker_labels.get(lab, {}).get("dip-base", 
                                                                                    f"{clean_config_label(lab)} S_R dip R(λ; n)"))
                        
                        ax_plot.scatter([lam0], [R1_at_lam0],
                                        marker='s', s=70,
                                        facecolor='none', edgecolor=color, alpha=0.7)
                    if low <= lam1 <= high:
                        ax_plot.scatter([lam1], [R1],
                                        marker='s', s=70,
                                        facecolor='none', edgecolor=color, alpha=0.7,
                                        label=custom_marker_labels.get(lab, {}).get("dip-dn", 
                                                                                    f"{clean_config_label(lab)} S_R dip R(λ; n+Δn)"))

                
                

            # Debug texte
            if verbose:
                dips_nm = ", ".join(f"{lam[i]:.1f}" for i in dip_list_idx)
                depths_str = ", ".join(f"{x:.3f}" for x in depth_list)
                fwhm_str = ", ".join(f"{w:.3f}" for w in fwhm_list)
                dR_over_dn_str = ", ".join(f"{s:.3f}" for s in dR_over_dn_list)
                dLam_over_dn_str = ", ".join(f"{l:.3f}" for l in dLam_over_dn_list)

                dR_over_dn = f"{best_SR:.3f}" if best_SR is not None else "–"
                S_lambda = f"{best_S_lambda:.3f}" if best_S_lambda is not None else "–"
                
                debug_lines.append(
                    f"{clean_config_label(lab)}: dips[{dips_nm}]nm,  dip {lam_dip:.1f}nm  "
                    f"depths[{depths_str}], depth={depth:.3f}  "
                    #f"slopes[{slopes_str}] slope={slope:.3f}  "
                    f"FWHMs[{fwhm_str}], FWHM={fwhm:.1f}  "
                    f"ΔR/Δn[{dR_over_dn_str}], best ΔR/Δn={dR_over_dn}  "
                    f"Δλ/Δn[{dLam_over_dn_str}], best Δλ/Δn={S_lambda}"
                )
                debug_lines.append("")

                
            # ---------- alimenter tableau ---------------------------- #
            cfg_labels.append(clean_config_label(lab))
            geom_sum.append(summaries[lab][0])
            mat_sum .append(summaries[lab][1])

            fwhm_sum.append(try_FWHM or f"{fwhm:.1f} nm")
            lam0_sum.append(try_lam0 or f"{lam_dip:.1f} nm")
            delta_lam_over_midLam.append(f"{S_lam_min_abs:.3f} & {S_lam_sym_abs:.3f}")

            Q_sum.append(mets.get("Q-factor", f"{lam_dip/fwhm:.2f}"))
            
            # extras depuis Metrics:  (peuvent être vides)
            S_lambda_sum.append(f"{S_lambda}")
            dRdn_sum.append(f"{dR_over_dn}")  
            dRhalf_sum.append(mets.get("ΔR_half", ""))

        # ---------- BEST config  ----------- #
            # Enregistrement des meilleurs résultats
            try:
                sr_value = float(best_SR)
                if sr_value > best_config['SR']:
                    best_config = {'name': lab, 'SR': sr_value}
            except (TypeError, ValueError):
                continue

        # Affichage de la meilleure configuration après la boucle
        debug_lines.append(f"Meilleure configuration: {best_config['name']} avec S_R = {best_config['SR']:.3f}")

      
                    

        # --------- debug panel --------------------------------------- #
        if verbose:
            debug_out.value = "\n".join(debug_lines)


        # ---------------------- tableau ------------------------------ #
        flags = dict(
            show_fwhm = show_fwhm_chk.value,
            show_lambda0 = show_lambda0_chk.value,
            show_delta_lam_over_midLam = show_delta_lam_over_midLam_chk.value,
            show_S_lambda = show_S_lambda_chk.value,
            show_S_dn = show_S_dn_chk.value,
            show_deltaR_half = show_deltaR_half_chk.value,
            show_Q = show_Q_chk.value,
        )
        
        # avertissement si S_R demandé mais pas de ΔR/Δn dispo
        if flags['show_S_dn'] and not any(dRdn_sum):
            if verbose:
                debug_lines.append(
                    "[Plot] Aucune donnée ΔR/Δn trouvée, S_R ne sera pas affichée."
                )
                
        
        
        # === Filtrer “Geometry” ===
        base_geom = set(geom_sum[0].splitlines())
        new_geom = []
        for i, txt in enumerate(geom_sum):
            lines = txt.splitlines()
            if i == 0:
                new_geom.append(txt)
            else:
                diff = [l for l in lines if l not in base_geom]
                new_geom.append("\n".join(diff))
        geom_sum = new_geom

        # === Filtrer “Material” ===
        base_mat = set(mat_sum[0].splitlines())
        new_mat = []
        for i, txt in enumerate(mat_sum):
            lines = txt.splitlines()
            if i == 0:
                new_mat.append(txt)
            else:
                diff = [l for l in lines if l not in base_mat]
                new_mat.append("\n".join(diff))
        mat_sum = new_mat
        
        cellText=[]; rowLabels=[]
        cellText.append(geom_sum); rowLabels.append("Geometry")
        cellText.append(mat_sum ); rowLabels.append("Material")
        
        if flags['show_fwhm'] and any(fwhm_sum):
            cellText.append(fwhm_sum)
            rowLabels.append("FWHM (nm)")
        if flags['show_lambda0'] and any(lam0_sum):
            cellText.append(lam0_sum)
            rowLabels.append(r"$\lambda_0$")
        if flags['show_delta_lam_over_midLam'] and any(delta_lam_over_midLam):
            cellText.append(delta_lam_over_midLam)
            rowLabels.append(r"$\Delta\lambda/\lambda$")
        if flags['show_S_lambda'] and any(S_lambda_sum):
            cellText.append(S_lambda_sum)
            rowLabels.append(r"$S_{\lambda}$")
        if flags['show_S_dn'] and any(dRdn_sum):
            cellText.append(dRdn_sum)
            rowLabels.append(r"$S_R$")
        if flags['show_deltaR_half'] and any(dRhalf_sum):
            cellText.append(dRhalf_sum)
            rowLabels.append(r"$\Delta R_{half}$")
        if flags['show_Q'] and any(Q_sum):
            cellText.append(Q_sum)
            rowLabels.append("Q-factor")

        filtered = []
        for lbl, row in zip(rowLabels, cellText):
            if lbl in ("Geometry (nm)", "Material"):
                ref = row[0]
                new = [row[0]] + [("" if row[j]==ref else row[j]) for j in range(1,len(row))]
                filtered.append(new)
            else:
                filtered.append(row)
        cellText = filtered

                   
        # --- suppression des lignes totalement vides ---
        filtered_rows = []; filtered_labels = []
        for lbl, row in zip(rowLabels, cellText):
            if any(str(x).strip() for x in row):
                filtered_labels.append(lbl)
                filtered_rows.append(row)
        rowLabels, cellText = filtered_labels, filtered_rows
                        
        # === création du tableau ===          
                   
        n_cfg=len(cfg_labels)
        fs=8 if n_cfg<=5 else max(8-(n_cfg-5),3)
        
        table=ax_table.table(cellText=cellText,
                             colLabels=[clean_config_label(l.replace("Mat_","\nMat_")) for l in cfg_labels],
                             rowLabels=rowLabels, loc="center", cellLoc="left")
        table.auto_set_font_size(False); table.set_fontsize(fs)
        table.auto_set_column_width(col=list(range(n_cfg)))
        
        for (r,c),cell in table.get_celld().items():
            if r==-1 or c==-1:
                cell.set_facecolor("#40466e")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor("whitesmoke")
                cell.set_edgecolor("lightgray"); cell.set_linewidth(0.5)
                cell.get_text().set_color(colors[c%len(colors)])

        # hauteur dynamique
        h_row={}
        for (r,c),cell in table.get_celld().items():
            if r>=0:
                nb=cell.get_text().get_text().count("\n")+1
                h_row[r]=max(h_row.get(r,0),nb)
        for (r,c),cell in table.get_celld().items():
            if r in h_row:
                cell.set_height(0.04*h_row[r])

        # ---------------------- axes ----------------------- #
        ax_plot.set_xlabel("Wavelength (nm)", fontsize=16)
        ax_plot.set_ylabel("Reflectance", fontsize=16)
        ax_plot.grid(True)
        # si demandé, on affiche la légende avec les noms de config
        if show_labels_chk.value:
            ax_plot.legend(loc='best', fontsize=14)


        ax_plot.set_xlim(low, high)

        # --------------- rendu dans la zone de sortie ---------------- #
        plot_out.clear_output(wait=True)
        with plot_out:
            # affichage + lien pour la figure du spectre
            display(fig_plot)
            display(_download_link(fig_plot,
                    f"spectra_{datetime.now():%Y%m%d_%H%M%S}.png"))

            # affichage + lien pour la figure du tableau
            display(fig_table)
            update_label_editors()
            display(_download_link(fig_table,
                    f"tableau_{datetime.now():%Y%m%d_%H%M%S}.png"))
            
            
        update_label_editors()
    
            
        plt.close(fig_plot)
        plt.close(fig_table)



    # liaison bouton
    draw_b.on_click(_draw)

    # ---------------------------------------------------------------- #
    #                         assemblage final                          #
    # ---------------------------------------------------------------- #
    tab = VBox(
        [controls_box, debug_out, plot_out],
        layout=Layout(
            display='flex',
            flex_flow='column nowrap',
            width='100%'
        )
    )
    
    
    tab.update_spectra = _update_spectra
    return tab
