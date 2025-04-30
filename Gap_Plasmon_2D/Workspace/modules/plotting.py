#!/usr/bin/env python3
"""
Module: plotting.py

Cette partie gère l'onglet Plot de l'application interactive, sans rien omettre du code d'origine.
"""

import os
import io, base64
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import HTML, display
from datetime import datetime
import textwrap

from data_readers import get_all_spectra_and_summaries
from simulate_and_plot import ordered_params
from Characterization import find_best_dip_fwhm

# Construction des chemins
module_dir    = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(module_dir)
notebooks_dir = os.path.join(workspace_dir, "notebooks")
summary_dir   = os.path.join(notebooks_dir, "Summary_Simulation")
exp_data_dir  = os.path.join(notebooks_dir, "Experimental_Data")


def create_download_link(fig, filename="figure.png"):
    buf = io.BytesIO()
    # ensure layout is tight and nothing is cut
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.05)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return HTML(f'<a download="{filename}" href="data:image/png;base64,{b64}" target="_blank">Télécharger l\'image</a>')


def create_plot_tab():

    # 1) Widgets
    spectra_select = widgets.SelectMultiple(
        options=[],
        description="Available spectra:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%', height='150px')
    )
    plot_verbose_toggle = widgets.Checkbox(
        value=True,
        description="Verbose",
        indent=False,
        layout=widgets.Layout(width='150px'),
        style={'description_width': 'initial'}
    )
    
    
    plot_debug = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            height='200px',
            overflow_y='auto',
            border='1px solid darkred',
            display='block' if plot_verbose_toggle.value else 'none'
        )
    )
    def toggle_plot_debug(change):
        plot_debug.layout.display = 'block' if change['new'] else 'none'
        if not change['new']:
            plot_debug.clear_output()
    plot_verbose_toggle.observe(toggle_plot_debug, names='value')    
    
    
    plot_button = widgets.Button(
        description="Draw", button_style="info",
        tooltip="Draw selected spectra"
    )
    plot_output = widgets.Output(
        layout=widgets.Layout(border="2px solid #ccc", padding="10px", min_height="400px")
    )
    plot_controls = widgets.VBox([
        widgets.HTML("<h3>Plot</h3>"),
        spectra_select,
        plot_verbose_toggle,
        plot_button
    ])

    # 2) Variables partagées
    plotted_lines = {}    # {label: (wavelength_array, reflectance_array)}
    summaries     = {}    # {label: (geom_summary, mat_summary)}
    metrics_all   = {}    # {label: metrics_dict}

    # 3) Fonction de mise à jour des spectres disponibles

    def update_spectra():
        nonlocal plotted_lines, summaries, metrics_all
        spectra, sums, mets = get_all_spectra_and_summaries(
            summary_dir, exp_data_dir, ordered_params
        )
        spectra_select.options = list(spectra.keys())
        plotted_lines = spectra
        summaries     = sums
        metrics_all   = mets

    # appel initial
    update_spectra()

    # 4) Callback de tracé

    def on_plot_button_clicked(b):
        # a) rafraîchir les données
        update_spectra()
        verbose = plot_verbose_toggle.value
        
        # b) marges et création de la figure + axes
        left_marges, width_marges = 0.10, 0.80
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        n_colors = len(colors)

        fig = plt.figure(figsize=(13, 9))
        
        ax_plot  = fig.add_axes([left_marges, 0.50, width_marges, 0.35])
        ax_table = fig.add_axes([left_marges, 0.05, width_marges, 0.35])
        ax_table.axis('off')
        
        # c) préparation des listes
        config_labels      = []
        geom_summaries     = []
        mat_summaries      = []
        fwhm_summaries     = []
        lam_summaries      = []
        S_lam_min_vals     = []
        S_lam_sym_vals     = []
        S_lam_summaries    = []
        Q_factor_list      = []
        raw_score_list     = []
        debug_lines        = []
        
        # d) détermination des labels à tracer
        labels = list(spectra_select.value) or list(plotted_lines.keys())
        
        # e) boucle de tracé et calcul des métriques
        for idx, label in enumerate(labels):
            # données
            wl, R  = plotted_lines[label]
            lam = np.array(wl)
            Rup = np.array(R)
            
            # calcul dip/FWHM
            (lam_left, lam_right, width_fwhm, lam_dip, Rdip, ylev,
            lam_m_l, Rm_l, lam_m_r, Rm_r, lam_sym, R_sym, slope,
            depth, raw_score, dips, scores_list, depths, slopes,
            widths, lam_max_ls, R_max_ls, lam_max_rs, R_max_rs,
            lam_syms, R_syms) = find_best_dip_fwhm(
                lam, Rup,
                smooth_win=0, polyorder=0,
                dip_prom=0.01, dip_dist=5,
                peak_dist=5, verbose=True
            )
            # On choisit le max de plus petite amplitude 
            if Rm_l < Rm_r:
                lam_min  = lam_m_l
                lam_middle = lam_left
            else:
                lam_min  = lam_m_r
                lam_middle = lam_right
            
            #  on ajoute S_lam
            S_lam_min_abs = abs((lam_dip - lam_min)   / lam_middle)
            S_lam_sym_abs = abs((lam_dip - lam_sym)   / lam_middle)
            # Ajout pour mémoriser les valeurs absolues
            S_lam_min_vals.append(S_lam_min_abs)
            S_lam_sym_vals.append(S_lam_sym_abs)
            
            
            color = colors[idx % n_colors]

            # tracé principal
            ax_plot.plot(lam, Rup, color=color)
            
            # tracés conditionnels (verbose)
            if verbose:
                ax_plot.hlines(ylev, xmin=lam_left, xmax=lam_right,
                            linewidth=2, colors=color)
                ax_plot.scatter(lam[dips], Rup[dips], marker='x', s=40, color=color)
                ax_plot.scatter(lam_max_ls, R_max_ls, marker='x', s=30, color=color)
                ax_plot.scatter(lam_max_rs, R_max_rs, marker='x', s=30, color=color)
                ax_plot.scatter(lam_syms, R_syms, marker='x', s=30, color=color)
                ax_plot.scatter([lam_dip], [Rdip], marker='o', s=70,
                                facecolor='none', edgecolor=color, linewidths=2)
                # ligne debug text
                dips_nm  = ", ".join(f"{l:.1f}" for l in lam[dips])
                scores_str = ", ".join(f"{s:.3e}" for s in scores_list)
                depths_str = ", ".join(f"{d:.3f}"  for d in depths)
                slopes_str = ", ".join(f"{s:.3e}" for s in slopes)
                widths_str = ", ".join(f"{w:.3f}" for w in widths)
            
                # Ligne unique résumé pour ce spectre
                debug_lines.append(
                    f"{label}:  "
                    f"dips=[{dips_nm}]  "
                    f"dip{lam_dip:.1f}nm  "
                    f"depths=[{depths_str}]  "
                    f"depth={depth:.3f}  "
                    f"slopes=[{slopes_str}]  "
                    f"slope={slope:.3e}  "
                    f"FWHMs=[{widths_str}]  "
                    f"FWHM={width_fwhm:.1f}  "
                    f"scores=[{scores_str}]  "
                    f"score={raw_score:.3e}  "
                )  
            
            # stockage pour le tableau
            config_labels.append(label)
            geom_summaries.append(summaries[label][0])
            mat_summaries.append(summaries[label][1])
            fwhm_summaries.append(f"{width_fwhm:.1f} nm")
            lam_summaries.append(f"{lam_dip:.1f} nm")
            S_lam_summaries.append(f"{S_lam_min_abs:.3f} & {S_lam_sym_abs:.3f}") 
            Q_factor_list.append(f"{lam_dip/width_fwhm:.1f}")
            raw_score_list.append(f"{raw_score:.2e}")
        
        
        
        if S_lam_min_vals:
            # norme euclidienne sur chaque couple
            norms = [np.hypot(a, b) for a, b in zip(S_lam_min_vals, S_lam_sym_vals)]
            best_idx   = int(np.argmin(norms))
            best_label = config_labels[best_idx]
            best_min   = S_lam_min_vals[best_idx]
            best_sym   = S_lam_sym_vals[best_idx]
            debug_lines.append(
                f"BEST_CONFIG → {best_label}  "
                f"(S_lam_min={best_min:.3f}, S_lam_sym={best_sym:.3f})"
            )
            
        
        # f) wrapping du debug text
        debug_txt = "\n".join(debug_lines)
        wrapper = textwrap.TextWrapper(width=100, break_long_words=True, replace_whitespace=False)
        wrapped = []
        for line in debug_txt.splitlines():
            wrapped.extend(wrapper.wrap(line) or [""])
        debug_txt = "\n".join(wrapped)
        
        # Affiche le debug dans le widget plot_debug
        plot_debug.clear_output()
        if verbose:
            with plot_debug:
                display(widgets.Textarea(
                    value=debug_txt,
                    layout=widgets.Layout(
                        width='100%',
                        height='200px',
                        overflow_y='auto'
                    )
                ))

        
        # h) finalisation du tracé
        ax_plot.set_xlabel("Wavelength (nm)")
        ax_plot.set_ylabel("Reflectance")
        ax_plot.set_title("Reflectance spectra")
        ax_plot.grid(True)
        
        # i) construction du tableau
        config_labels = [lbl.replace("Mat_","\nMat_") for lbl in config_labels]
        if config_labels:
            n = len(config_labels)
            fontsize = 8 if n <= 5 else max(8 - (n - 5), 3)
            table = ax_table.table(
                cellText=[
                    geom_summaries, mat_summaries,
                    fwhm_summaries, lam_summaries,
                    S_lam_summaries, Q_factor_list,
                    raw_score_list
                ],
                colLabels=config_labels,
                rowLabels=[
                    "Geometry", "Material", "FWHM", r"$\lambda_0$",
                    "S_lam abs: min & sym", "Q-factor", "Scoring dips per spectrum"
                ],
                loc="center", cellLoc="left"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(fontsize)
            table.auto_set_column_width(col=list(range(n)))
            for (r, c), cell in table.get_celld().items():
                if r == -1 or c == -1:
                    cell.set_facecolor("#40466e")
                    cell.set_text_props(weight="bold", color="white", fontsize=fontsize)
                else:
                    cell.set_facecolor("whitesmoke")
                    cell.set_edgecolor("lightgray")
                    cell.set_linewidth(0.5)
                    cell.get_text().set_color(colors[c % len(colors)])
            heights = {}
            for (r, c), cell in table.get_celld().items():
                if r >= 0:
                    lines = cell.get_text().get_text().count("\n") + 1
                    heights[r] = max(heights.get(r, 0), lines)
            for (r, c), cell in table.get_celld().items():
                if r in heights:
                    cell.set_height(0.04 * heights[r])

        # j) affichage final et lien
        with plot_output:
            plot_output.clear_output(wait=True)
            display(fig)
            link = create_download_link(
                fig,
                filename=f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            display(link)
            plt.close(fig)

    # 5) Bind du callback et assemblage du tab
    plot_button.on_click(on_plot_button_clicked)
    
    plot_tab = widgets.VBox([
        plot_controls,
        plot_debug,     # ← inséré juste ici
        plot_output
    ])
    plot_tab.update_spectra = update_spectra
    return plot_tab