from gap_plasmon_2d import paths
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plotting.py  – Onglet « Plot » de l’application interactive

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
import os, io, base64, re
from datetime import datetime

import numpy     as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import Layout, HBox, VBox, HTML, ToggleButton
from IPython.display import HTML as DHTML, display
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from gap_plasmon_2d.utils.data_readers       import get_all_spectra_and_summaries
from gap_plasmon_2d.simulation.simulate_and_plot  import ordered_params
from gap_plasmon_2d.analysis.characterization   import _find_dip_core, compute_half_point


# ------------------------------------------------------------------ #
#                        Chemins par défaut                          #
# ------------------------------------------------------------------ #
module_dir    = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(module_dir)
notebooks_dir = os.path.join(str(paths.RESULTS_DIR))
summary_dir   = os.path.join(notebooks_dir, "summary_simulation")
exp_data_dir  = os.path.join(notebooks_dir, "experimental")


# ------------------------------------------------------------------ #
#                     utilitaires globales                          #
# ------------------------------------------------------------------ #
def _download_link(fig, fname="plot.png"):
    """
    Retourne un widget HTML <a> pour télécharger la figure matplotlib en PNG.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return DHTML(
        f'<a download="{fname}" href="data:image/png;base64,{b64}" '
        f'target="_blank">Télécharger l’image</a>'
    )

def clean_config_label(label: str) -> str:
    """
    Enlève la partie ' ( ... )' à la fin d'un label.
    """
    return re.sub(r'\s*\([^\)]*\)\s*$', '', label or '')



# ───────────────────────────────────────────────────────────────────────
# Helper : génère le tableau « Verbose summary » façon onglet Simulation
# ───────────────────────────────────────────────────────────────────────
def verbose_summary_html(rows: list[dict], *, mode_label: str, best_cfg: str) -> str:
    """rows : [
        {'cfg','mode',
         'dips_all','dips_sel',
         'fwhm_all','fwhm_sel',
         'sr_all','sr_sel',
         'note'},
        …
    ]"""
    
    if not rows:
        return ""

    def _td(txt):  # garde les cellules non‑vides alignées
        return txt if txt not in (None, "", "–") else "&nbsp;"

    body = "\n".join(
        "<tr>"
        f"<td>{_td(r['cfg'])}</td>"
        f"<td>{_td(r['mode'])}</td>"
        f"<td>{_td(r['dips_all'])}</td>"
        f"<td>{_td(r['dips_sel'])}</td>"
        f"<td>{_td(r['fwhm_all'])}</td>"
        f"<td>{_td(r['fwhm_sel'])}</td>"
        f"<td>{_td(r['sr_all'])}</td>"
        f"<td>{_td(r['sr_sel'])}</td>"
        f"<td>{_td(r['note'])}</td>"
        "</tr>"
        for r in rows
    )

    return f"""
    <style>
      .vb-sum {{border-collapse:collapse;width:100%;font-family:Consolas,monospace;font-size:12px}}
      .vb-sum th,.vb-sum td{{border:1px solid #eee;padding:3px 6px;white-space:nowrap}}
      .vb-sum thead tr{{background:#37474f;color:#fff;font-weight:600}}
      .vb-sum tbody tr:nth-child(odd){{background:#fafafa}}
    </style>
    <details open class="debug-box">
      <summary><b>Verbose log — mode : {mode_label} | best config : {best_cfg}</b></summary>
      <div style='max-height:220px;overflow:auto'>
        <table class="vb-sum">
                <thead>
                <tr>
                    <th>Config</th><th>Mode</th>
                    <th>Dips – all (nm)</th><th>Dips – sel. (nm)</th>
                    <th>FWHM – all (nm)</th><th>FWHM – sel. (nm)</th>
                    <th>ΔR/Δn – all</th><th>ΔR/Δn – sel.</th>
                    <th>Note</th>
                </tr>
                </thead>
          <tbody>{body}</tbody>
        </table>
      </div>
    </details>
    """



# ------------------------------------------------------------------ #
#                           Classe PlotTab                           #
# ------------------------------------------------------------------ #
class PlotTab:
    """
    Encapsule l’onglet “Plot” : widgets, callbacks, méthodes.
    """

    def __init__(self):
        # ────────────────────────────────────────────────────────────
        # 1) États internes (dicts partagés)
        # ────────────────────────────────────────────────────────────
        self.custom_labels_dict    = {}
        self.custom_labels_dn_dict = {}
        self.custom_marker_labels  = {}
        self.custom_colors         = {}

        self.Rup_dict   = {}  # {label: (lam, Rup)}
        self.Rup_dn_dict= {}  # {label: (lam_dn, Rup_dn) or None}
        self.summaries  = {}  # {label: (geom, mat)}
        self.metrics    = {}  # {label: {metric: value}}
        self.delta_ns   = {}  # {label: delta_n}

        # ────────────────────────────────────────────────────────────
        # 2) Widgets principaux
        # ────────────────────────────────────────────────────────────
        # Sélecteur de spectres
        self.spectra_select = widgets.SelectMultiple(
            options=[],
            description="Available spectra:",
            layout=Layout(width='80%', height='150px'),
            style={'description_width':'initial'}
        )
        self.spectra_select.observe(self._on_spectra_change, names="value")

        # Verbose (HTML) – activé par défaut
        self.verbose_chk = widgets.Checkbox(
            value=True, description="Verbose log",
            layout=Layout(width='100%'), indent=False,
            style={'description_width':'initial'}
        )

        self.draw_b = widgets.Button(
            description="Draw",            # libellé du bouton
            button_style="info",           # couleur bleue Info
            icon="line-chart"              # facultatif : icône FontAwesome
        )
        self.draw_b.on_click(self._draw)   # callback


        # Zone HTML qui contiendra le verbose moderne
        self.verbose_html = widgets.HTML(
            value="",
            layout=Layout(
                width='100%', border='1px solid #ccc',
                padding='6px', margin='4px 0',
                display='none'          # masqué quand verbose=False
            )
        )
        self.verbose_chk.observe(self._toggle_dbg, names='value')

        # ────────────────────────────────────────────────────────────
        # 3) Éditeurs de labels custom
        # ────────────────────────────────────────────────────────────
        self.labels_editors_box = VBox()
        self.update_labels_btn  = widgets.Button(
            description="Updates labels",
            button_style="primary",
            layout=Layout(width="170px")
        )
        self.update_labels_btn.on_click(self._on_update_labels)

        self.toggle_labels_editors_btn = ToggleButton(
            value=False,
            description="Afficher/masquer les éditeurs de labels",
            icon="chevron-down",
            layout=Layout(width="270px")
        )
        self.toggle_labels_editors_btn.observe(
            self._on_toggle_labels_panel, names="value"
        )

        self.labels_editors_panel = VBox([
            HTML("<b> Modify labels spectra :</b>"),
            self.labels_editors_box,
            self.update_labels_btn
        ], layout=Layout(display='none'))

        # ────────────────────────────────────────────────────────────
        # 4) Métriques et overlays
        # ────────────────────────────────────────────────────────────
        def _cb(val, desc):
            return widgets.Checkbox(value=val, description=desc)

        # Métriques tableau
        self.show_fwhm_chk        = _cb(False, "FWHM")
        self.show_lambda0_chk     = _cb(True,  r"λ0")
        self.show_delta_lam_chk   = _cb(False, r"Δλ / λmin or λsym")
        self.show_S_lambda_chk    = _cb(True,  "Sλ (nm/RIU)")
        self.show_S_dn_chk        = _cb(True,  r"ΔR/Δn (1/RIU)")
        self.show_deltaR_half_chk = _cb(True,  r"ΔR_half")
        self.show_Q_chk           = _cb(False, "Q-factor")

        # Overlays graphiques
        self.show_Rup_dn_overlay_chk   = _cb(True,  "Rup_dn dashed")
        self.show_hlines_chk           = _cb(False, "half-level line")
        self.show_dips_chk             = _cb(False, "dips (×)")
        self.show_maxima_chk           = _cb(False, "maxima (×)")
        self.show_symmetry_pts_chk     = _cb(False, "symmetric pts (×)")
        self.show_selected_dip_chk     = _cb(True,  "selected dip (○)")
        self.show_sensitivity_marker_chk = _cb(True, "sensitivity marker (□)")

        # Sensitivités au demi-niveau
        self.show_half_level_metrics = _cb(False, "S from fwhm")
        self.show_half_level_metrics.style.description_width = 'auto'
        self.show_half_level_metrics.layout.margin  = '0'
        self.show_half_level_metrics.layout.padding = '0'
        self.show_half_level_metrics.indent         = False
        self.show_half_level_metrics.observe(
            lambda ch: self._update_label_editors(), names="value"
        )

        # Applique style « sans gap » sur tous
        for cb in (
            self.show_fwhm_chk, self.show_lambda0_chk, self.show_delta_lam_chk,
            self.show_S_lambda_chk, self.show_S_dn_chk, self.show_deltaR_half_chk,
            self.show_Q_chk, self.show_Rup_dn_overlay_chk, self.show_hlines_chk,
            self.show_dips_chk, self.show_maxima_chk, self.show_symmetry_pts_chk,
            self.show_selected_dip_chk, self.show_sensitivity_marker_chk
        ):
            cb.style.description_width = '60px'
            cb.layout.margin = '0'
            cb.layout.padding= '0'
            cb.indent = False

        # HBox métriques et overlays
        self.metrics_hbox = HBox([
            self.show_fwhm_chk, self.show_lambda0_chk, self.show_delta_lam_chk,
            self.show_S_lambda_chk, self.show_S_dn_chk, self.show_deltaR_half_chk,
            self.show_half_level_metrics, self.show_Q_chk
        ], layout=Layout(display='flex', flex_flow='row nowrap',
                        justify_content='space-around', margin='0', padding='0'))

        self.overlays_hbox = HBox([
            self.show_Rup_dn_overlay_chk, self.show_hlines_chk,
            self.show_dips_chk, self.show_maxima_chk,
            self.show_symmetry_pts_chk, self.show_selected_dip_chk,
            self.show_sensitivity_marker_chk
        ], layout=Layout(display='flex', flex_flow='row nowrap',
                        justify_content='space-around', margin='0', padding='0'))

        # ────────────────────────────────────────────────────────────
        # 5) Widgets de zoom en λ
        # ────────────────────────────────────────────────────────────
        self.lambda_min_text = widgets.FloatText(
            value=0.0, description="λ min (nm):",
            layout=Layout(width='150px'),
            style={'description_width':'initial'}
        )
        self.lambda_max_text = widgets.FloatText(
            value=0.0, description="λ max (nm):",
            layout=Layout(width='150px'),
            style={'description_width':'initial'}
        )
        self.range_box = HBox(
            [self.lambda_min_text, self.lambda_max_text],
            layout=Layout(grid_gap='10px')
        )

        # ────────────────────────────────────────────────────────────
        # 6) Zone figure / tableau
        # ────────────────────────────────────────────────────────────
        self.plot_out = widgets.Output(
            layout=Layout(
                border='2px solid #ccc', padding='10px',
                display='flex', flex_flow='column nowrap',
                width='100%', height='auto',
                justify_content='center'
            )
        )


        # Bouton pour (dé)masquer la légende matplotlib
        self.show_labels_chk = widgets.Checkbox(
            value=False,                # ou True si vous préférez
            description="Show labels",
            indent=False,
            layout=Layout(width="auto"),
            style={'description_width': 'initial'}
        )


        # ────────────────────────────────────────────────────────────
        # 7) Assemblage final
        # ────────────────────────────────────────────────────────────
        self.controls_box = VBox([
            HTML("<h3>Plot</h3>"),
            self.spectra_select,
            self.verbose_chk,
            HTML("<b>Shown metrics :</b>"), self.metrics_hbox,
            HTML("<b>Graphical overlays :</b>"), self.overlays_hbox,
            self.range_box,
            HBox([
                self.show_labels_chk,
                self.draw_b,
                VBox([
                    self.toggle_labels_editors_btn,
                    self.labels_editors_panel
                ], layout=Layout(align_items="flex-start", min_width="600px"))
            ], layout=Layout(grid_gap='12px'))
        ], layout=Layout(width='100%'))

        self.tab = VBox(
            [self.controls_box,     # panneau de contrôle
             self.verbose_html,     # verbose (au même endroit qu'avant)
             self.plot_out],        # figure + tableau
            layout=Layout(display='flex', flex_flow='column nowrap', width='100%')
        )

        # ────────────────────────────────────────────────────────────
        # 8) Bind callbacks finaux et initial update
        # ────────────────────────────────────────────────────────────
        self.draw_b.on_click(self._draw)
        self._update_spectra()



    # ====================================================================
    #                         Méthodes internes                           #
    # ====================================================================
    # la nouvelle fonction garde simplement l’affichage masqué/visible
    def _toggle_dbg(self, change):
        self.verbose_html.layout.display = 'block' if change['new'] else 'none'
        if not change['new']:
            self.verbose_html.value = ''

    def _on_toggle_labels_panel(self, change):
        show = change['new']
        self.labels_editors_panel.layout.display = 'block' if show else 'none'
        self.toggle_labels_editors_btn.icon = 'chevron-up' if show else 'chevron-down'

    def _on_update_labels(self, _=None):
        # Parcourt tous les champs d'édition pour mettre à jour les dicts
        for row in self.labels_editors_box.children:
            for txt in row.children:
                if isinstance(txt, widgets.ColorPicker):
                    self.custom_colors[txt._orig_lab] = txt.value
                elif hasattr(txt, "_carre_type"):
                    lab = txt._orig_lab; typ = txt._carre_type
                    self.custom_marker_labels.setdefault(lab, {})[typ] = txt.value
                elif getattr(txt, "_is_dn", False):
                    self.custom_labels_dn_dict[txt._orig_lab] = txt.value
                else:
                    self.custom_labels_dict[txt._orig_lab] = txt.value
        # Redessine immédiatement
        self._draw(None)

    def _on_spectra_change(self, change):
        # Met à jour les éditeurs de labels quand on change la sélection
        self._update_label_editors()

    def _update_label_editors(self):
        """
        Reconstruit self.labels_editors_box.children
        en fonction de la sélection courante et du mode (dip/half).
        """
        labels = list(self.spectra_select.value) or list(self.Rup_dict.keys())
        marker_types = (["half-base","half-dn"] if self.show_half_level_metrics.value
                        else ["dip-base","dip-dn"])
        rows = []

        for lab in labels:
            row = []

            # ColorPicker
            cp = widgets.ColorPicker(
                concise=True,
                description='Couleur:',
                value=self.custom_colors.get(lab, '#1f77b4'),
                layout=Layout(width='120px')
            )
            cp._orig_lab = lab
            row.append(cp)

            # Texte label principal
            txt_main = widgets.Text(
                value=self.custom_labels_dict.get(lab, clean_config_label(lab)),
                description="Label:", layout=Layout(width="180px")
            )
            txt_main._orig_lab = lab
            txt_main._is_dn = False
            row.append(txt_main)

            # Texte label Δn si existant
            if lab in self.Rup_dn_dict and self.Rup_dn_dict[lab] is not None:
                txt_dn = widgets.Text(
                    value=self.custom_labels_dn_dict.get(
                        lab,
                        clean_config_label(lab)+" (R + Δn)"
                    ),
                    description="Label Δn:", layout=Layout(width="250px")
                )
                txt_dn._orig_lab = lab
                txt_dn._is_dn = True
                row.append(txt_dn)

            # Textes pour chaque marker type
            for typ in marker_types:
                default = self.custom_marker_labels.get(lab, {}).get(
                    typ,
                    f"{clean_config_label(lab)} S_R {typ.replace('-', ' ')}"
                )
                txt_car = widgets.Text(
                    value=default,
                    description=f"Carré {typ}:", layout=Layout(width="250px")
                )
                txt_car._orig_lab = lab
                txt_car._carre_type = typ
                row.append(txt_car)

            rows.append(HBox(row, layout=Layout(margin="2px 0")))

        self.labels_editors_box.children = tuple(rows)


    def _update_spectra(self):
        """
        Charge ou recharge les spectres & métriques depuis les .npz et .json.
        Met à jour self.spectra_select.options.
        """
        (
            self.Rup_dict,
            self.Rup_dn_dict,
            self.summaries,
            self.metrics,
            self.delta_ns
        ) = get_all_spectra_and_summaries(
            summary_dir, exp_data_dir, ordered_params
        )
        self.spectra_select.options = list(self.Rup_dict.keys())
        # Après rafraîchissement, on reconstruit aussi les éditeurs
        self._update_label_editors()


    def _draw(self, _btn):
        """
        Callback principal pour dessiner courbes + tableau.
        """
        # 1) Mise à jour possibles des spectres
        self._update_spectra()
        verbose = self.verbose_chk.value
        # ── INITIALISATION DU VERBOSE
        summary_rows: list[dict] = []
        
        # On cache/affiche la boîte dès le départ (contenu vide pour l'instant)
        self.verbose_html.layout.display = 'block' if verbose else 'none'
        self.verbose_html.value = ""          # sera rempli en toute fin

        labels = list(self.spectra_select.value) or list(self.Rup_dict.keys())
        if not labels:
            return

        # 2) Préparation figures
        fig_plot  = plt.figure(figsize=(12, 6))
        ax_plot   = fig_plot.add_axes([0.10, 0.10, 0.80, 0.85])
        fig_table = plt.figure(figsize=(9, 4))
        ax_table  = fig_table.add_axes([0.10, 0.05, 0.80, 0.90])
        ax_table.axis('off')

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # 3) Collecteurs pour tableau & debug
        geom_sum = []; mat_sum = []; cfg_labels = []
        fwhm_sum = []; lam0_sum = []; delta_lam_over_midLam = []
        S_lambda_sum = []; dRdn_sum = []; dRhalf_sum = []; Q_sum = []
        S_lam_min_vals = []; S_lam_sym_vals = []
        
        best_config = {'name': None, 'SR': float('-inf')}
        use_half    = self.show_half_level_metrics.value

        # 4) Boucle sur chaque spectre sélectionné
        for idx, lab in enumerate(labels):
            lam, Rup = self.Rup_dict[lab]
            Rup_dn_tuple = self.Rup_dn_dict.get(lab)

            # 4.1) Fenêtre λ
            low  = (lam.min()
                    if self.lambda_min_text.value <= 0
                    else max(lam.min(), self.lambda_min_text.value))
            high = (lam.max()
                    if self.lambda_max_text.value <= 0
                    else min(lam.max(), self.lambda_max_text.value))
            mask     = (lam >= low) & (lam <= high)
            lam_plot = lam[mask]; Rup_plot = Rup[mask]

            # 4.2) Prépare Rup_dn si dispo
            if Rup_dn_tuple is not None:
                lam_dn_full, Rup_dn_full = Rup_dn_tuple
                mask_dn = (lam_dn_full >= low) & (lam_dn_full <= high)
                lam_dn_plot = lam_dn_full[mask_dn]
                Rup_dn_plot= Rup_dn_full[mask_dn]
            else:
                lam_dn_full = Rup_dn_full = lam_dn_plot = Rup_dn_plot = None

            # 4.3) Détection dips sur Rup
            (dip_idx, lam_dip_list, R_dip_list, y_level_list,
             lam_left_list, lam_right_list, fwhm_list,
             lam_max_l_list, R_max_l_list,
             lam_max_r_list, R_max_r_list,
             lam_sym_list, R_sym_list,
             depth_list), _ = _find_dip_core(
                wavelength=lam, reflectance=Rup,
                smooth_win=0, polyorder=0,
                dip_prom=1e-2, dip_dist=1, peak_dist=1,
                verbose=verbose, cfg_name=lab
            )
            if not dip_idx:
            
                continue

            # 4.4) Calcul métriques Δn ou raw_score
            dR_over_dn_list = []; dLam_over_dn_list = []
            if Rup_dn_tuple and lab in self.delta_ns:
                # On a des données Δn
                lam_dn, Rup_dn_vals = Rup_dn_tuple
                delta_n = self.delta_ns[lab]
                (dip_idx_dn, lam_dip_list_dn, R_dip_list_dn,
                 y_level_list_dn, lam_left_list_dn, lam_right_list_dn,
                 fwhm_list_dn, lam_max_l_list_dn, R_max_l_list_dn,
                 lam_max_r_list_dn, R_max_r_list_dn,
                 lam_sym_list_dn, R_sym_list_dn,
                 depth_list_dn), _ = _find_dip_core(
                    lam_dn, Rup_dn_vals,0,0,1e-2,1,1,verbose,lab
                )
                interp0 = interp1d(lam, Rup, kind='cubic',
                                  bounds_error=False, fill_value='extrapolate')
                interp1 = interp1d(lam_dn, Rup_dn_vals, kind='cubic',
                                  bounds_error=False, fill_value='extrapolate')
                min_dips = min(len(lam_dip_list), len(lam_dip_list_dn))

                for i in range(min_dips):
                    if use_half:
                        half0 = compute_half_point(
                            lam, Rup,
                            lam_left_list[i], lam_right_list[i]
                        )
                        R0 = float(interp0(half0))
                        lam1 = compute_half_point(
                            lam_dn, Rup_dn_vals,
                            lam_left_list_dn[i], lam_right_list_dn[i]
                        )
                        R1 = float(interp1(half0))
                    else:
                        lam1 = lam_dip_list_dn[i]
                        R0 = R_dip_list[i]
                        R1 = float(interp1(lam_dip_list[i]))
                    dR_over_dn_list.append(abs(R0-R1)/delta_n)
                    dLam_over_dn_list.append(abs(lam_dip_list[i]-lam1)/delta_n)

                best_idx = int(np.nanargmax(dR_over_dn_list))
                best_SR      = dR_over_dn_list[best_idx]
                best_S_lambda= dLam_over_dn_list[best_idx]

            else:
                # Pas de Δn → raw_score
                lam_array = lam; R_array = Rup
                interp_R = interp1d(lam_array, R_array,
                                    kind='cubic', bounds_error=False,
                                    fill_value='extrapolate')
                delta = lam_array[1] - lam_array[0]
                alpha, beta = 2.0, 0.5
                best_raw_score = -np.inf; best_idx_raw = None

                for j in range(len(dip_idx)):
                    depth_j = depth_list[j]
                    fwhm_j  = fwhm_list[j]
                    lam_left_j  = lam_left_list[j]
                    lam_right_j = lam_right_list[j]

                    y_pL  = interp_R(lam_left_j + delta)
                    y_mL  = interp_R(lam_left_j - delta)
                    slope_left  = abs((y_pL - y_mL)/(2*delta))
                    y_pR  = interp_R(lam_right_j + delta)
                    y_mR  = interp_R(lam_right_j - delta)
                    slope_right = abs((y_pR - y_mR)/(2*delta))
                    slope = max(slope_left, slope_right)

                    raw_score = ((depth_j**alpha) *
                                 (slope**(1.0-depth_j)) /
                                 (fwhm_j**beta + 1e-12))

                    dR_over_dn_list.append(0.0)
                    dLam_over_dn_list.append(0.0)

                    if raw_score > best_raw_score:
                        best_raw_score = raw_score
                        best_idx_raw  = j

                best_idx      = best_idx_raw
                best_SR       = None
                best_S_lambda = None


            # 4.5) Extraction métriques du dip retenu
            lam_left = lam_left_list[best_idx]
            lam_right= lam_right_list[best_idx]
            fwhm     = fwhm_list[best_idx]
            lam_dip  = lam_dip_list[best_idx]
            R_dip    = R_dip_list[best_idx]
            ylev     = y_level_list[best_idx]
            lam_m_l, Rm_l = lam_max_l_list[best_idx], R_max_l_list[best_idx]
            lam_m_r, Rm_r = lam_max_r_list[best_idx], R_max_r_list[best_idx]
            lam_sym, R_sym= lam_sym_list[best_idx], R_sym_list[best_idx]
            depth    = depth_list[best_idx]

            # Convertisseurs en numpy pour scatter
            lam_max_l_list = np.array(lam_max_l_list)
            R_max_l_list   = np.array(R_max_l_list)
            lam_max_r_list = np.array(lam_max_r_list)
            R_max_r_list   = np.array(R_max_r_list)
            lam_sym_list   = np.array(lam_sym_list)
            R_sym_list     = np.array(R_sym_list)
            width_list     = np.array(fwhm_list)
            depth_list_arr = np.array(depth_list)

            lam_min = lam_m_l if Rm_l < Rm_r else lam_m_r
            lam_mid = lam_left if Rm_l < Rm_r else lam_right
            S_lam_min_abs = abs((lam_dip - lam_min)/lam_mid)
            S_lam_sym_abs = abs((lam_dip - lam_sym)/lam_mid)
            S_lam_min_vals.append(S_lam_min_abs)
            S_lam_sym_vals.append(S_lam_sym_abs)

            # 4.6) Tracé principal
            color = self.custom_colors.get(lab, colors[idx%len(colors)])
            label_aff = self.custom_labels_dict.get(lab, clean_config_label(lab))
            ax_plot.plot(lam_plot, Rup_plot, color=color, label=label_aff, zorder=1)

            # 4.7) Overlays selon sélection
            if self.show_dips_chk.value:
                dip_lams = lam[dip_idx]
                dip_rups = Rup[dip_idx]
                mask_d   = (dip_lams>=low)&(dip_lams<=high)
                ax_plot.scatter(
                    dip_lams[mask_d], dip_rups[mask_d],
                    marker='x', s=40, color=color, zorder=3
                )

            if self.show_hlines_chk.value:
                if lam_left>=low and lam_right<=high:
                    ax_plot.hlines(
                        ylev, lam_left, lam_right,
                        linewidth=2, colors=color, zorder=2
                    )
                if Rup_dn_tuple is not None:
                    # idem pour Δn
                    lam_left_dn = lam_left_list_dn[best_idx]
                    lam_right_dn= lam_right_list_dn[best_idx]
                    ylev_dn     = y_level_list_dn[best_idx]
                    if lam_left_dn>=low and lam_right_dn<=high:
                        ax_plot.hlines(
                            ylev_dn, lam_left_dn, lam_right_dn,
                            linewidth=2, colors=color, zorder=2
                        )

            if self.show_maxima_chk.value:
                ax_plot.scatter(lam_max_l_list, R_max_l_list,
                                marker='x', s=30, color=color, zorder=3)
                ax_plot.scatter(lam_max_r_list, R_max_r_list,
                                marker='x', s=30, color=color, zorder=3)

            if self.show_symmetry_pts_chk.value:
                ax_plot.scatter(lam_sym_list, R_sym_list,
                                marker='x', s=30, color=color, zorder=3)

            if self.show_selected_dip_chk.value:
                ax_plot.scatter([lam_dip], [R_dip],
                                marker='o', s=70,
                                facecolor='none', edgecolor=color,
                                linewidths=2, zorder=4)
                if Rup_dn_tuple is not None:
                    ax_plot.scatter(
                        [lam_dip_list_dn[best_idx]],
                        [R_dip_list_dn[best_idx]],
                        marker='o', s=70,
                        facecolor='none', edgecolor=color,
                        linewidths=2, zorder=4
                    )

            if self.show_Rup_dn_overlay_chk.value and lam_dn_plot is not None:
                good = ~np.isnan(Rup_dn_plot)
                label_dn = self.custom_labels_dn_dict.get(
                    lab, clean_config_label(lab)+" (R + Δn)"
                )
                ax_plot.plot(
                    lam_dn_plot[good], Rup_dn_plot[good],
                    "--", linewidth=2, color=color, alpha=0.7,
                    label=label_dn, zorder=0
                )

            if self.show_sensitivity_marker_chk.value and Rup_dn_tuple is not None:
                # marquage en carré selon mode
                interp1 = interp1d(
                    lam_dn_full, Rup_dn_full,
                    kind='cubic', bounds_error=False, fill_value='extrapolate'
                )
                if use_half:
                    lam0 = compute_half_point(
                        lam, Rup, lam_left, lam_right
                    )
                    R0 = float(interp1d(
                        lam, Rup, kind='cubic',
                        bounds_error=False, fill_value='extrapolate'
                    )(lam0))
                    lam1 = compute_half_point(
                        lam_dn_full, Rup_dn_full,
                        lam_left_list_dn[best_idx],
                        lam_right_list_dn[best_idx]
                    )
                    y1 = y_level_list_dn[best_idx]
                    if low <= lam0 <= high:
                        ax_plot.scatter(
                            [lam0], [R0], marker='s', s=70,
                            facecolor='none', edgecolor=color,
                            alpha=0.7,
                            label=self.custom_marker_labels.get(
                                lab, {}
                            ).get("half-base",
                                  f"{clean_config_label(lab)} S_R half R(λ; n)")
                        )
                        ax_plot.scatter(
                            [lam0], [float(interp1(lam0))],
                            marker='s', s=70,
                            facecolor='none', edgecolor=color,
                            alpha=0.7
                        )
                    if low <= lam1 <= high:
                        ax_plot.scatter(
                            [lam1], [y1], marker='s', s=70,
                            facecolor='none', edgecolor=color,
                            alpha=0.7,
                            label=self.custom_marker_labels.get(
                                lab, {}
                            ).get("half-dn",
                                  f"{clean_config_label(lab)} S_R half R(λ; n+Δn)")
                        )
                else:
                    lam0 = lam_dip
                    R0   = R_dip
                    lam1 = lam_dip_list_dn[best_idx]
                    R1   = R_dip_list_dn[best_idx]
                    R1_at_lam0 = float(interp1(lam0))
                    if low <= lam0 <= high:
                        ax_plot.scatter(
                            [lam0], [R0], marker='s', s=70,
                            facecolor='none', edgecolor=color,
                            alpha=0.7,
                            label=self.custom_marker_labels.get(
                                lab, {}
                            ).get("dip-base",
                                  f"{clean_config_label(lab)} S_R dip R(λ; n)")
                        )
                        ax_plot.scatter(
                            [lam0], [R1_at_lam0],
                            marker='s', s=70,
                            facecolor='none', edgecolor=color,
                            alpha=0.7
                        )
                    if low <= lam1 <= high:
                        ax_plot.scatter(
                            [lam1], [R1], marker='s', s=70,
                            facecolor='none', edgecolor=color,
                            alpha=0.7,
                            label=self.custom_marker_labels.get(
                                lab, {}
                            ).get("dip-dn",
                                  f"{clean_config_label(lab)} S_R dip R(λ; n+Δn)")
                        )


            SR_txt    = f"{best_SR:.3f}" if best_SR is not None else "–"
            SL_txt    = f"{best_S_lambda:.3f}" if best_S_lambda is not None else "–"
            
            # 4.5) après avoir calculé lam_dip_list, fwhm_list, dR_over_dn_list, …
            dips_all   = ", ".join(f"{l:.1f}"   for l in lam_dip_list)
            fwhms_all  = ", ".join(f"{w:.1f}"   for w in fwhm_list)
            sr_all     = ", ".join(f"{s:.3f}"   for s in dR_over_dn_list)
            # valeur « retenue » (= best_idx) :
            dips_sel   = f"{lam_dip:.1f} nm"
            fwhm_sel   = f"{fwhm:.1f}"
            sr_sel     = f"{best_SR:.3f}" if best_SR is not None else "–"


            # 4.9) Prépare ligne de tableau pour ce spectre
            summary_rows.append({
                "cfg":      clean_config_label(lab),
                "mode":     "FWHM ½" if use_half else "Dip",
                "dips_all": ", ".join(f"{l:.1f}" for l in lam_dip_list),
                "dips_sel": f"{lam_dip:.1f} nm",
                "fwhm_all": ", ".join(f"{w:.1f}" for w in fwhm_list),
                "fwhm_sel": f"{fwhm:.1f}",
                "sr_all":   ", ".join(f"{s:.3f}" for s in dR_over_dn_list),
                "sr_sel":   (f"{best_SR:.3f}" if best_SR is not None else "–"),
                "note":     ("raw‑score" if best_SR is None else "")
            })

            
            cfg_labels.append(clean_config_label(lab))
            geom_sum.append(self.summaries[lab][0])
            mat_sum .append(self.summaries[lab][1])
            fwhm_sum.append(self.metrics.get(lab,{}).get("FWHM", f"{fwhm:.1f} nm"))
            lam0_sum.append(self.metrics.get(lab,{}).get(
                "Lam_res", self.metrics.get(lab,{}).get("lam0", f"{lam_dip:.1f} nm")
            ))
            delta_lam_over_midLam.append(f"{S_lam_min_abs:.3f} & {S_lam_sym_abs:.3f}")
            Q_sum.append(self.metrics.get(lab,{}).get("Q-factor", f"{lam_dip/fwhm:.2f}"))
            S_lambda_sum.append(SL_txt)
            dRdn_sum.append(SR_txt)
            dRhalf_sum.append(self.metrics.get(lab,{}).get("ΔR_half", ""))

            # 4.10) Mémorise meilleur config
            try:
                val = float(best_SR)
                if val > best_config['SR']:
                    best_config = {'name': lab, 'SR': val}
            except Exception:
                pass


        # 7) Construction du tableau final
        flags = {
            'show_fwhm' : self.show_fwhm_chk.value,
            'show_lambda0' : self.show_lambda0_chk.value,
            'show_delta_lam_over_midLam' : self.show_delta_lam_chk.value,
            'show_S_lambda' : self.show_S_lambda_chk.value,
            'show_S_dn' : self.show_S_dn_chk.value,
            'show_deltaR_half' : self.show_deltaR_half_chk.value,
            'show_Q' : self.show_Q_chk.value
        }

        # Filtrage geometry / material
        base_geom = set(geom_sum[0].splitlines())
        new_geom = [
            geom_sum[i] if i==0 else "\n".join(l for l in txt.splitlines() if l not in base_geom)
            for i, txt in enumerate(geom_sum)
        ]
        geom_sum = new_geom

        base_mat = set(mat_sum[0].splitlines())
        new_mat = [
            mat_sum[i] if i==0 else "\n".join(l for l in txt.splitlines() if l not in base_mat)
            for i, txt in enumerate(mat_sum)
        ]
        mat_sum = new_mat

        # Construction cellText / rowLabels
        cellText = []; rowLabels = []
        cellText.append(geom_sum);   rowLabels.append("Geometry")
        cellText.append(mat_sum);    rowLabels.append("Material")
        if flags['show_fwhm'] and any(fwhm_sum):
            cellText.append(fwhm_sum);       rowLabels.append("FWHM (nm)")
        if flags['show_lambda0'] and any(lam0_sum):
            cellText.append(lam0_sum);       rowLabels.append(r"$\lambda_0$")
        if flags['show_delta_lam_over_midLam'] and any(delta_lam_over_midLam):
            cellText.append(delta_lam_over_midLam); rowLabels.append(r"$\Delta\lambda/\lambda$")
        if flags['show_S_lambda'] and any(S_lambda_sum):
            cellText.append(S_lambda_sum);   rowLabels.append(r"$S_{\lambda}$")
        if flags['show_S_dn'] and any(dRdn_sum):
            cellText.append(dRdn_sum);       rowLabels.append(r"$S_R$")
        if flags['show_deltaR_half'] and any(dRhalf_sum):
            cellText.append(dRhalf_sum);     rowLabels.append(r"$\Delta R_{half}$")
        if flags['show_Q'] and any(Q_sum):
            cellText.append(Q_sum);          rowLabels.append("Q-factor")

        # Supprime lignes entièrement vides
        filtered_cells = []; filtered_labels = []
        for lbl, row in zip(rowLabels, cellText):
            if any(str(x).strip() for x in row):
                filtered_labels.append(lbl)
                filtered_cells.append(row)
        rowLabels, cellText = filtered_labels, filtered_cells

        # Création du tableau matplotlib
        n_cfg = len(cfg_labels)
        fs = 8 if n_cfg<=5 else max(8-(n_cfg-5),3)
        table = ax_table.table(
            cellText=cellText,
            colLabels=[clean_config_label(l.replace("Mat_","\nMat_")) for l in cfg_labels],
            rowLabels=rowLabels,
            loc="center",
            cellLoc="left"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(fs)
        table.auto_set_column_width(col=list(range(n_cfg)))

        for (r,c), cell in table.get_celld().items():
            if r==-1 or c==-1:
                cell.set_facecolor("#40466e")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor("whitesmoke")
                cell.set_edgecolor("lightgray")
                cell.set_linewidth(0.5)
                cell.get_text().set_color(colors[c%len(colors)])

        # Hauteur dynamique
        row_heights = {}
        for (r,c),cell in table.get_celld().items():
            if r>=0:
                nb = cell.get_text().get_text().count("\n")+1
                row_heights[r] = max(row_heights.get(r,0), nb)
        for (r,c),cell in table.get_celld().items():
            if r in row_heights:
                cell.set_height(0.04*row_heights[r])

        # 8) Mise en forme axes & légende
        ax_plot.set_xlabel("Wavelength (nm)", fontsize=16)
        ax_plot.set_ylabel("Reflectance", fontsize=16)
        ax_plot.grid(True)
        ax_plot.set_xlim(low, high)
        if self.show_labels_chk.value:
            ax_plot.legend(loc='best', fontsize=14)

        # 9) Affichage dans la zone de sortie
        self.plot_out.clear_output(wait=True)
        with self.plot_out:
            display(fig_plot)
            display(_download_link(
                fig_plot,
                f"spectra_{datetime.now():%Y%m%d_%H%M%S}.png"
            ))
            display(fig_table)
            # On reconstruit éditeurs pour rester synchrones
            self._update_label_editors()
            display(_download_link(
                fig_table,
                f"tableau_{datetime.now():%Y%m%d_%H%M%S}.png"
            ))
            
        # ── AFFICHAGE DU VERBOSE
        if verbose:
            mode_lbl = "FWHM ½" if use_half else "Dip"
            self.verbose_html.value = verbose_summary_html(
                summary_rows,
                mode_label=mode_lbl,
                best_cfg=(best_config['name'] or "—")
            )
                


        # ── FERMETURE DES FIGURES
        plt.close(fig_plot)
        plt.close(fig_table)


# ────────────────────────────────────────────────────────────────────────
# Export de la classe : création d’une instance et récupération du .tab
# ────────────────────────────────────────────────────────────────────────
plot_tab = PlotTab()
tab = plot_tab.tab   # à passer à votre application interactive
