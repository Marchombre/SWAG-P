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
import os, io, base64, textwrap
from datetime import datetime

import numpy     as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import Layout, HBox, VBox, HTML
from IPython.display import HTML as DHTML, display
from pyparsing import line
from scipy.interpolate import interp1d

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
#                    Construction de l’onglet Plot                   #
# ------------------------------------------------------------------ #
def create_plot_tab():

    # --------------------- widgets principaux ---------------------- #
    spectra_select = widgets.SelectMultiple(
        options=[], description="Available spectra:",
        layout=Layout(width='80%', height='150px'),
        style={'description_width':'initial'})

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

    # 3) On assemble le tout
    controls_box = VBox([
        HTML("<h3>Plot</h3>"),
        spectra_select,
        verbose_chk,
        HTML("<b>Métriques à afficher :</b>"),
        metrics_hbox,
        HTML("<b>Overlays graphiques :</b>"),
        overlays_hbox,
        HBox([show_labels_chk, draw_b], layout=Layout(grid_gap='10px'))
    ], layout=Layout(width='100%'))


    # -------------------- zone figure / tableau -------------------- #
    plot_out = widgets.Output(
        layout=Layout(border='2px solid #ccc', padding='10px', min_height='400px'))

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
        fig_plot = plt.figure(figsize=(9, 6))
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
                best_idx=int(np.nanargmax(depth_list)); best_SR=best_S_lambda=None
                if verbose: debug_lines.append(f"[Plot] Δn ou Rup_dn absent pour « {lab} »")
                
                
        

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
            color = colors[idx % len(colors)]
            ax_plot.plot(lam, Rup, color=color, label=lab, zorder=1)

            if show_hlines_chk.value:
                # demi-hauteur sur Rup
                ax_plot.hlines(
                    y_level_list[best_idx],
                    lam_left_list[best_idx],
                    lam_right_list[best_idx],
                    linewidth=2, colors=color, zorder=2
                )
                # demi-hauteur sur Rup_dn uniquement si disponible
                if Rup_dn_tuple is not None:
                    ax_plot.hlines(
                        y_level_list_dn[best_idx],
                        lam_left_list_dn[best_idx],
                        lam_right_list_dn[best_idx],
                        linewidth=2, colors=color, zorder=2
                    )

                    
            if show_dips_chk.value:
                ax_plot.scatter(lam[dip_list_idx], Rup[dip_list_idx], marker='x', s=40, color=color, zorder=3)

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


            if show_Rup_dn_overlay_chk.value and Rup_dn_tuple is not None:
                good = ~np.isnan(Rup_dn_vals)
                ax_plot.plot(lam_dn[good], Rup_dn_vals[good], "--", linewidth=2, color=color, alpha=0.7, label=f"{lab} (R + Δn)", zorder=0)
                
                
            if show_sensitivity_marker_chk.value and Rup_dn_tuple is not None:
                # unpack pour être clair
                # lam0, R0 = point de référence (dip ou half) sur Rup
                # lam1, R1 = point équivalent pour Rup_dn
                if use_half:
                    # demi‐hauteur sur Rup
                    lam0 = compute_half_point(
                        lam, Rup,
                        lam_left_list[best_idx], lam_right_list[best_idx]
                    )
                    R0 = float(interp0(lam0))
                    # même λ sur Rup_dn
                    R1 = float(interp1(lam0))
                    # demi‐hauteur sur Rup_dn, même flank
                    lam1 = compute_half_point(
                        lam_dn, Rup_dn_vals,
                        lam_left_list_dn[best_idx], lam_right_list_dn[best_idx]
                    )
                    y1 = y_level_list_dn[best_idx]  # ylev sur Rp_dn

                    # 3 carrés half‐mode
                    ax_plot.scatter([lam0], [R0],
                                    marker='s', s=70,
                                    facecolor='none', edgecolor=color, alpha=0.7,
                                    label=f"{lab} sens. half-base")
                    ax_plot.scatter([lam0], [R1],
                                    marker='s', s=70,
                                    facecolor='none', edgecolor=color, alpha=0.7)
                    ax_plot.scatter([lam1], [y1],
                                    marker='s', s=70,
                                    facecolor='none', edgecolor=color, alpha=0.7,
                                    label=f"{lab} sens. half-Δn")
                else:
                    # creux sur Rup
                    lam0 = lam_dip_list[best_idx]
                    R0   = R_dip_list[best_idx]
                    
                    # même λ sur Rup_dn
                    R1_at_lam0 = float(interp1(lam0))
                    
                    # vrai creux sur Rup_dn
                    lam1 = lam_dip_list_dn[best_idx]
                    R1   = R_dip_list_dn[best_idx]

                    

                    # 3 carrés dip‐mode
                    ax_plot.scatter([lam0], [R0],
                                    marker='s', s=70,
                                    facecolor='none', edgecolor=color, alpha=0.7,
                                    label=f"{lab} sens. dip-base")
                    ax_plot.scatter([lam0], [R1_at_lam0],
                                    marker='s', s=70,
                                    facecolor='none', edgecolor=color, alpha=0.7)
                    ax_plot.scatter([lam1], [R1],
                                    marker='s', s=70,
                                    facecolor='none', edgecolor=color, alpha=0.7,
                                    label=f"{lab} sens. dip-Δn")
                            
                
                

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
                    f"{lab}: dips[{dips_nm}]nm,  dip {lam_dip:.1f}nm  "
                    f"depths[{depths_str}], depth={depth:.3f}  "
                    #f"slopes[{slopes_str}] slope={slope:.3f}  "
                    f"FWHMs[{fwhm_str}], FWHM={fwhm:.1f}  "
                    f"ΔR/Δn[{dR_over_dn_str}], best ΔR/Δn={dR_over_dn}  "
                    f"Δλ/Δn[{dLam_over_dn_str}], best Δλ/Δn={S_lambda}"
                )
                debug_lines.append("")

                
            # ---------- alimenter tableau ---------------------------- #
            cfg_labels.append(lab)
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
                             colLabels=[l.replace("Mat_","\nMat_") for l in cfg_labels],
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
        ax_plot.set_xlabel("Wavelength (nm)")
        ax_plot.set_ylabel("Reflectance")
        ax_plot.grid(True)
        # si demandé, on affiche la légende avec les noms de config
        if show_labels_chk.value:
            ax_plot.legend(loc='best', fontsize=8)

        # --------------- rendu dans la zone de sortie ---------------- #
        plot_out.clear_output(wait=True)
        with plot_out:
            # affichage + lien pour la figure du spectre
            display(fig_plot)
            display(_download_link(fig_plot,
                    f"spectra_{datetime.now():%Y%m%d_%H%M%S}.png"))

            # affichage + lien pour la figure du tableau
            display(fig_table)
            display(_download_link(fig_table,
                    f"tableau_{datetime.now():%Y%m%d_%H%M%S}.png"))
        plt.close(fig_plot)
        plt.close(fig_table)


    # liaison bouton
    draw_b.on_click(_draw)

    # ---------------------------------------------------------------- #
    #                         assemblage final                          #
    # ---------------------------------------------------------------- #
    tab = VBox([controls_box, debug_out, plot_out])
    tab.update_spectra = _update_spectra
    return tab
