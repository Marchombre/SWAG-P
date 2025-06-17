#!/usr/bin/env python3
# -*- coding: utf‑8 -*-
"""
Module : simulation.py
Onglet « Simulation » – version stable, après corrections successives :
  • aucun constructeur de widget appelé en positionnel
  • tableau à hauteur dynamique
  • sauvegarde complète des métriques calculées
"""
# --------------------------------------------------------------------- #
#                                imports                                #
# --------------------------------------------------------------------- #
import os, io, base64, json, textwrap
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import Layout, HBox, VBox, ToggleButton, HTML
from IPython.display import HTML as DHTML, display, Javascript
from scipy.interpolate import interp1d

# dépendances internes
from simulate_and_plot     import ordered_params, run_simulation_one_combo
from data_readers          import list_sim_summary_files
from convergence_analysis  import create_multi_convergence_widget
from Saving_Functions      import save_simulation_summary
from Characterization      import _find_dip_core, find_best_dip, simulate_delta_spectrum, compute_half_point

# --------------------------------------------------------------------- #
#                               chemins                                 #
# --------------------------------------------------------------------- #
module_dir         = os.path.dirname(os.path.abspath(__file__))
workspace_dir      = os.path.dirname(module_dir)
notebooks_dir      = os.path.join(workspace_dir, "notebooks")
summary_dir        = os.path.join(notebooks_dir, "Summary_Simulation")
exp_data_dir       = os.path.join(notebooks_dir, "Experimental_Data")
configurations_dir = os.path.join(workspace_dir, "CONFIGURATIONS")
data_dir           = os.path.join(workspace_dir, "data")
json_combined_path = os.path.join(data_dir, "combined_materials.json")




# --------------------------------------------------------------------- #
def _download_link(fig, fname="figure.png"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.05)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return DHTML(f'<a download="{fname}" href="data:image/png;base64,{b64}" '
                 f'target="_blank">Télécharger l’image</a>')

# --------------------------------------------------------------------- #
def create_simulation_tab(json_combined_path: str,
                          summary_dir: str,
                          exp_data_dir: str) -> widgets.VBox:
    """Construit l’onglet *Simulation* complet."""

    # ------------------ chargement des configurations -----------------
    cfg_file = os.path.join(configurations_dir, "geom_mat_combinations.json")
    if os.path.exists(cfg_file):
        with open(cfg_file, encoding="utf-8") as f:
            all_configs = json.load(f)["ALL_COMBINED_CONFIGS"]
    else:
        all_configs = []

    # =================== widgets paramètres généraux ==================
    sim_lambda_min = widgets.FloatText(
        value=300.0, description="λ min (nm):",
        layout=Layout(width='150px'), style={'description_width': 'initial'})
    sim_lambda_max = widgets.FloatText(
        value=1100.0, description="λ max (nm):",
        layout=Layout(width='150px'), style={'description_width': 'initial'})
    sim_n_points = widgets.IntText(
        value=400, description="Points:",
        layout=Layout(width='200px'), style={'description_width': 'initial'})
    sim_n_mod = widgets.IntText(
        value=5, description="Modes:",
        layout=Layout(width='200px'), style={'description_width': 'initial'})
    # mode de calcul : dip ou half-level
    mode_calc_radio = widgets.RadioButtons(
        options=[('Dip (min)', 'dip'), ('FWHM (half)', 'half')],
        value='dip',
        description='Mode calc:',
        style={'description_width':'initial'},
        layout=Layout(width='200px')
    )
    sim_run_button = widgets.Button(
        description="Run simulation", button_style="success",
        tooltip="Lancer la simulation")

    mode_selection = widgets.RadioButtons(
        options=[('Fixe', 'fixed'),
                 ('Personnalisé', 'custom'),
                 ('Automatique', 'auto')],
        value='fixed', description='Modes:',
        style={'description_width': 'initial'})
    custom_modes_box = VBox()

    # --------- empêcher valeurs négatives ----------------------------
    def _positive(ch):
        if ch['new'] < 0:
            ch['owner'].value = 0
    for w in (sim_lambda_min, sim_lambda_max, sim_n_points, sim_n_mod):
        w.observe(_positive, names='value')

    # =================== gestion des fichiers .npz ====================
    sim_files_dropdown = widgets.Dropdown(
        options=list_sim_summary_files(summary_dir),
        description="Simulation files:",
        layout=Layout(width='500px'),
        style={'description_width': 'initial'})
    sim_refresh_button = widgets.Button(
        description="Refresh files", button_style="info",
        tooltip="Rafraîchir la liste")
    sim_download_button = widgets.Button(
        description="Download", button_style="danger",
        tooltip="Télécharger le fichier")

    sim_refresh_button.on_click(
        lambda *_: sim_files_dropdown.set_trait(
            "options", list_sim_summary_files(summary_dir)))

    def _download(_):
        path = sim_files_dropdown.value
        if not path:
            print("Aucun fichier sélectionné."); return
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        fname = os.path.basename(path)
        js = (f"var a=document.createElement('a');"
              f"a.href='data:application/octet-stream;base64,{b64}';"
              f"a.download='{fname}';a.style.display='none';"
              "document.body.appendChild(a);a.click();a.remove();")
        display(Javascript(js))
    sim_download_button.on_click(_download)

    sim_files_box = HBox([sim_files_dropdown],
                         layout=Layout(width='100%', margin='10px 0'))
    download_refresh_box = HBox([sim_refresh_button, sim_download_button],
                                layout=Layout(width='50%', margin='10px 0'))

    sim_name_widget = widgets.Text(
        value="", placeholder="Nom de simulation (auto si vide)",
        description="Sim Name:", layout=Layout(width='500px'),
        style={'description_width': 'initial'})

    # ======================= sélecteur Config / Δn ====================
    config_rows, config_checkboxes, dn_checkboxes = [], {}, {}
    for cfg in all_configs:
        name = cfg["config_name"]
        chk = widgets.Checkbox(value=False, description=name, indent=False)
        dn  = widgets.Checkbox(value=False, description='Δn', indent=False,
                               layout=Layout(width='46px'))
        config_checkboxes[name] = chk
        dn_checkboxes[name] = dn
        config_rows.append(HBox([chk, dn], layout=Layout(grid_gap='5px')))

    visible = min(len(config_rows), 10)
    
        # 1) Crée les deux boutons
    select_all_cfg_btn = widgets.Button(
        description="Tout sélectionner Configs",
        button_style="info",
        layout=Layout(width='auto', margin='0 5px 5px 0')
    )
    select_all_dn_btn = widgets.Button(
        description="Tout sélectionner Δn",
        button_style="info",
        layout=Layout(width='auto', margin='0 0 5px 0')
    )

    # 2) Handler qui bascule l’état de toutes les checkboxes Configs
    def _toggle_all_cfg(_):
        all_sel = all(cb.value for cb in config_checkboxes.values())
        for cb in config_checkboxes.values():
            cb.value = not all_sel
    select_all_cfg_btn.on_click(_toggle_all_cfg)

    # 3) Handler qui bascule l’état de toutes les checkboxes Δn
    def _toggle_all_dn(_):
        all_sel = all(cb.value for cb in dn_checkboxes.values())
        for cb in dn_checkboxes.values():
            cb.value = not all_sel
    select_all_dn_btn.on_click(_toggle_all_dn)

    # 3) Reconstruit config_list en y glissant les boutons avant les lignes
    config_list = VBox(
        children=[
            HTML("<b>Configurations et Δn</b>"),
            HBox([select_all_cfg_btn, select_all_dn_btn],
                layout=Layout(grid_gap='10px')),
            *config_rows
        ],
        layout=Layout(
            width='500px',
            height=f'{30+visible*30}px',
            overflow_y='auto',
            border='1px solid lightgray',
            padding='5px',
            display='none'
        )
    )
    toggle_btn = ToggleButton(
        value=False, description="Sélection Configs / Δn",
        icon='caret-down', layout=Layout(width='520px'))

    def _toggle(ch):
        config_list.layout.display = 'block' if ch['new'] else 'none'
        toggle_btn.icon = 'caret-up' if ch['new'] else 'caret-down'
    toggle_btn.observe(_toggle, names='value')
    config_selector = VBox([toggle_btn, config_list],
                           layout=Layout(padding='5px'))

    # ======================== couche(s) Δn ============================
    layer_keys = [m['key'] for m in all_configs[0]['material']['MATERIALS_CONFIG']]
    layer_selector = widgets.SelectMultiple(
        options=layer_keys, description="Couche(s) Δn:",
        layout=Layout(width='300px', height='100px'),
        style={'description_width': 'initial'})
    
    delta_n_widget = widgets.FloatText(
        value=1e-2, description="Δn:",
        layout=Layout(width='150px'),
        style={'description_width': 'initial'})
    
    delta_n_widget.observe(_positive, names='value')

    # =================== custom modes rafraîchissement ================
    custom_n_mod_inputs = {}
    def _refresh_custom_modes(*_):
        sel = [c for c in all_configs
               if config_checkboxes[c["config_name"]].value]
        if mode_selection.value != 'custom' or not sel:
            custom_modes_box.children = []; return
        inputs = []
        for cfg in sel:
            name = cfg["config_name"]
            it = widgets.IntText(
                value=sim_n_mod.value, description=name,
                layout=Layout(width='300px'),
                style={'description_width': 'initial'})
            custom_n_mod_inputs[name] = it
            inputs.append(it)
        custom_modes_box.children = inputs
    mode_selection.observe(_refresh_custom_modes, names='value')
    for cb in config_checkboxes.values():
        cb.observe(_refresh_custom_modes, names='value')

    # =========================== verbose ==============================
    verbose_toggle = widgets.Checkbox(
        value=True, description="Verbose", indent=False,
        layout=Layout(width='100%'),
        style={'description_width': 'initial'})
    
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
            debug_out.value = ''    # on vide simplement le texte
    verbose_toggle.observe(_toggle_dbg, names='value')

    # ===================== métriques / overlays =======================
    def _cb(val, desc): return widgets.Checkbox(value=val, description=desc)
    show_fwhm_chk         = _cb(False, "FWHM")
    show_lambda0_chk      = _cb(True,  r"λ0")
    show_delta_lam_over_midLam_chk        = _cb(False, r"Δλ / λmin or λsym")
    show_S_lambda_chk     = _cb(True,  "Sλ = Δλ / Δn (nm/RIU)")
    show_S_dn_chk         = _cb(True,  r"ΔR/Δn (1/RIU)")
    show_deltaR_half_chk  = _cb(True,  r"ΔR_half")
    show_Q_chk            = _cb(False, "Q‑factor")

    show_Rup_dn_chk   = _cb(True,  "Rup_dn dashed")
    show_hlines_chk       = _cb(False, "half‑level line")
    show_dips_chk         = _cb(False, "dips (×)")
    show_maxima_chk       = _cb(False, "maxima (×)")
    show_symmetry_pts_chk = _cb(False, "symmetric pts (×)")
    show_selected_dip_chk = _cb(True,  "selected dip (○)")
    show_sensitivity_marker_chk = _cb(True,  "sensitivity marker (□)")
    
    # 1) enlève toute marge / padding / fixe la largeur à auto sur chaque checkbox
    for cb in (show_fwhm_chk, show_lambda0_chk, show_delta_lam_over_midLam_chk, show_S_lambda_chk,
            show_S_dn_chk, show_deltaR_half_chk, show_Q_chk):
        cb.layout.margin  = '0'
        cb.layout.padding = '0'
        cb.layout.width   = 'auto'
        cb.indent         = False  # si besoin d'enlever l'indentation interne

    # 2) utilise gap=0px sur la HBox
    metrics_hbox = HBox(
        [show_fwhm_chk, show_lambda0_chk, show_delta_lam_over_midLam_chk, show_S_lambda_chk,
        show_S_dn_chk, show_deltaR_half_chk, show_Q_chk],
        layout=Layout(
            display='flex',
            flex_flow='row nowrap',
            justify_content='space-around',
            gap='0px',      # colle les items les uns aux autres
            margin='0 10px 0 0',
            padding='0'
        )
    )

    # même chose pour tes overlays :
    for cb in (show_Rup_dn_chk, show_hlines_chk, show_dips_chk,
            show_maxima_chk, show_symmetry_pts_chk, show_selected_dip_chk, show_sensitivity_marker_chk):
        cb.layout.margin  = '0'
        cb.layout.padding = '0'
        cb.layout.width   = 'auto'
        cb.indent         = False

    overlays_hbox = HBox(
        [show_Rup_dn_chk, show_hlines_chk, show_dips_chk,
        show_maxima_chk, show_symmetry_pts_chk, show_selected_dip_chk, show_sensitivity_marker_chk],
        layout=Layout(
            display='flex',
            flex_flow='row nowrap',
            justify_content='space-around',
            gap='0px',
            margin='0 10px 0 0',
            padding='0'
        )
    )


    metrics_selector = VBox(
        children=[HTML("<b>Métriques à afficher :</b>"), metrics_hbox,
                  HTML("<b>Overlays graphiques :</b>"), overlays_hbox],
        layout=Layout(width='100%', border='1px solid lightgray',
                      padding='5px', margin='10px 0'))

    # ======================== refresh configs =========================
    def _load_configs():
        if os.path.exists(cfg_file):
            with open(cfg_file, encoding='utf-8') as f:
                return json.load(f)["ALL_COMBINED_CONFIGS"]
        return []

    def _refresh_cfgs(_):
        nonlocal all_configs
        for cb in config_checkboxes.values():
            try: cb.unobserve(_refresh_custom_modes, names='value')
            except Exception:
                pass
        all_configs = _load_configs()
        config_rows.clear(); config_checkboxes.clear(); dn_checkboxes.clear()
        for cfg in all_configs:
            name = cfg["config_name"]
            chk = widgets.Checkbox(value=False, description=name, indent=False)
            dn  = widgets.Checkbox(value=False, description='Δn', indent=False,
                                   layout=Layout(width='46px'))
            chk.observe(_refresh_custom_modes, names='value')
            config_checkboxes[name] = chk; dn_checkboxes[name] = dn
            config_rows.append(HBox([chk, dn], layout=Layout(grid_gap='5px')))
        config_list.children = [HTML("<b>Configurations et Δn</b>")] + config_rows
        _refresh_custom_modes()

    cfg_refresh_button = widgets.Button(
        description="Refresh Configs", button_style="info",
        tooltip="Rafraîchir la liste des configs")
    cfg_refresh_button.on_click(_refresh_cfgs)

    # ===================== conteneur gauche (controls) ================
    sim_controls = VBox(
        children=[
            HTML("<h3>Simulation – Paramètres</h3>"),
            sim_name_widget,
            sim_files_box,
            HBox([download_refresh_box]),
            HBox([sim_lambda_min, sim_lambda_max]),
            HBox([sim_n_points, sim_n_mod]),
            VBox([cfg_refresh_button]),
            config_selector,
            HBox([mode_selection, layer_selector]),
            custom_modes_box,
            HBox([delta_n_widget, mode_calc_radio, sim_run_button]),
            verbose_toggle],
        layout=Layout(padding='10px', border='1px solid lightgray'))

    # widget convergence (droite)
    conv_widget = create_multi_convergence_widget(json_combined_path, all_configs)

    # sortie figure + tableau
    sim_output = widgets.Output(
        layout=Layout(border='2px solid #ccc', padding='10px',
                      min_height='400px', margin='40px 0 0 0'))
    
    
    
    
    
    
    

    # ================================================================= #
    #                          callback RUN                             #
    # ================================================================= #
    
    def _run(_):
        # ---------- spinner -------------------------------------------
        with sim_output:
            sim_output.clear_output(wait=True)
            display(HTML("<div style='text-align:center'>"
                         "<img src='https://i.gifer.com/ZZ5H.gif' width='40'>"
                         "<br><em>Simulation en cours…</em></div>"))

        # ---------- paramètres de base --------------------------------
        lam_range = np.linspace(sim_lambda_min.value,
                                sim_lambda_max.value,
                                sim_n_points.value)
        wave = {"angle": 0, "polarization": 1}
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        sel_layers = list(layer_selector.value)
        delta_n = max(delta_n_widget.value, 1e-9)

        # ---------- modes automatiques --------------------------------
        auto_modes = {}
        conv_json = Path(workspace_dir) / "Convergence/convergence_results.json"
        if conv_json.exists():
            with open(conv_json, encoding='utf-8') as f:
                master = json.load(f)
            auto_modes = {n: r[-1]["optimal_n_mode"]
                          for n, r in master.get("configs", {}).items() if r}

        selected_cfgs = [c for c in all_configs
                         if config_checkboxes[c["config_name"]].value]
        dn_cfgs = {n for n, cb in dn_checkboxes.items() if cb.value}

        mode_by_cfg = {}
        if mode_selection.value == 'fixed':
            for cfg in selected_cfgs:
                mode_by_cfg[cfg["config_name"]] = sim_n_mod.value
        elif mode_selection.value == 'custom':
            for n, inp in custom_n_mod_inputs.items():
                mode_by_cfg[n] = inp.value
        else:
            for cfg in selected_cfgs:
                nom = cfg["config_name"]
                if nom in auto_modes:
                    mode_by_cfg[nom] = int(auto_modes[nom])
                else:
                    # Fallback : met un mode par défaut, ou affiche un warning
                    mode_by_cfg[nom] = sim_n_mod.value  # Ou la valeur par défaut que tu veux
                    print(f"[AVERTISSEMENT] Pas de mode auto trouvé pour '{nom}'. Mode par défaut utilisé.")


        # ---------- figure --------------------------------------------
        fig = plt.figure(figsize=(13, 9))
        ax_plot  = fig.add_axes([0.10, 0.50, 0.80, 0.35])
        ax_table = fig.add_axes([0.10, 0.05, 0.80, 0.35]); ax_table.axis('off')

        # ---------- accumulateurs  ----------------------------------
        cfg_labels, geom_sum, mat_sum = [], [], []
        fwhm_sum, lam_sum, delta_lam_over_midLam = [], [], []
        S_lambda_sum, S_R_sum, dR_half_sum = [], [], []
        S_R_vals = []
        S_lam_min, S_lam_sym = [], []
        Q_fac = []
        debug_lines = []
        simulation_details = {}

        verbose = verbose_toggle.value
        flags = dict(
            show_fwhm=show_fwhm_chk.value, show_lambda0=show_lambda0_chk.value,
            show_delta_lam_over_midLam=show_delta_lam_over_midLam_chk.value, show_S_lambda=show_S_lambda_chk.value,
            show_S_dn=show_S_dn_chk.value, show_deltaR_half=show_deltaR_half_chk.value,
            show_Q=show_Q_chk.value,
            show_Rup_dn=show_Rup_dn_chk.value, show_hlines=show_hlines_chk.value,
            show_dips=show_dips_chk.value, show_maxima=show_maxima_chk.value,
            show_symmetry_pts=show_symmetry_pts_chk.value,
            show_selected_dip=show_selected_dip_chk.value,
            show_sensitivity_marker=show_sensitivity_marker_chk.value)
        
        use_half = (mode_calc_radio.value == 'half')


        # ================== boucle configs ============================
        for idx, cfg in enumerate(selected_cfgs):
            color = colors[idx % len(colors)]
            name = cfg["config_name"]; n_modes = mode_by_cfg[name]
            Rup, _, details = run_simulation_one_combo(
                lam_range, wave, n_modes, cfg, json_combined_path)
            lam = lam_range; Rup = np.asarray(Rup, dtype=float)
            simulation_details[name] = details



            Best_values_out, who, best_dip_index = find_best_dip(
                cfg=cfg,
                wavelength=lam,
                reflectance=Rup,
                wave=wave,
                n_modes=n_modes,
                sel_layers=sel_layers,
                delta_n=delta_n,
                json_combined_path=json_combined_path,
                smooth_win=0,
                polyorder=0,
                dip_prom=1e-2,
                dip_dist=1,
                peak_dist=1,
                verbose=True,
                cfg_name=name,
                mode="half" if use_half else "dip"
            )

            (dip_list, lam_dip_list, R_dip_list,
            y_level_list,
            lam_left_list, lam_right_list, fwhm_list,
            lam_max_l_list, R_max_l_list,
            lam_max_r_list, R_max_r_list,
            lam_sym_list,   R_sym_list,
            depth_list), _ = _find_dip_core(
                wavelength=lam, reflectance=Rup,
                smooth_win=0, polyorder=0,
                dip_prom=1e-2, dip_dist=1, peak_dist=1,
                verbose=False, cfg_name=name
            )

            # si tu as vraiment besoin de tableaux NumPy pour tracer :
            lam_max_l_list = np.array(lam_max_l_list)
            R_max_l_list   = np.array(R_max_l_list)
            lam_max_r_list = np.array(lam_max_r_list)
            R_max_r_list   = np.array(R_max_r_list)
            lam_sym_list   = np.array(lam_sym_list)
            R_sym_list     = np.array(R_sym_list)
            width_list     = np.array(fwhm_list)    
            depth_list     = np.array(depth_list)

            if Best_values_out is None:             # ← aucun dip pour cette config
                debug_lines.append(f" Aucun dip sélectionné pour la configuration “{who}” – ignorée.")
                continue                # on saute au spectre suivant

            # sinon on déroule
            (lam_left, lam_right, fwhm, depth,
            lam_dip, R_dip, ylev,
            lam_m_l, Rm_l, lam_m_r, Rm_r,
            lam_sym, R_sym,
            best_S_R, S_lambda, dR_half,
            dips_idx_list, dR_over_dn_list, dLam_over_dn_list
            ) = Best_values_out


            lam_min = lam_m_l if Rm_l < Rm_r else lam_m_r
            lam_mid = lam_left if Rm_l < Rm_r else lam_right
            
            S_lam_min_abs = abs((lam_dip - lam_min) / lam_mid)
            S_lam_sym_abs = abs((lam_dip - lam_sym) / lam_mid)
            
            S_lam_min.append(S_lam_min_abs)
            S_lam_sym.append(S_lam_sym_abs)
            
            compute_delta = flags['show_Rup_dn'] and name in dn_cfgs and sel_layers

            if compute_delta:
                # on ne fait la simu Δn QUE si la case “Spectres Δn” est cochée
                Rup_dn, lam_calc, R0, lam_calc_dn, R1, S_lambda, S_R, dR_half = simulate_delta_spectrum(
                    cfg=cfg,
                    lam=lam_range,
                    wave=wave,
                    n_modes=n_modes,
                    sel_layers=sel_layers,
                    delta_n=delta_n,
                    lam_dip=lam_dip,
                    R_dip=R_dip,
                    lam_left=lam_left,
                    lam_right=lam_right,
                    base_spectrum=Rup,
                    json_combined_path=json_combined_path,
                    dip_index=best_dip_index,
                    mode="half" if use_half else "dip"
                )
                
                # → on redétecte les dips sur Rup_dn  si delta n est séléctionné
                if Rup_dn is not None:
                    (dip_idx_dn, lam_dip_dn_list, R_dip_dn_list,
                    ylev_dn_list,
                    lam_left_dn, lam_right_dn, fwhm_dn,
                    lam_max_l_dn, R_max_l_dn,
                    lam_max_r_dn, R_max_r_dn,
                    lam_sym_dn, R_sym_dn,
                    depth_dn), _ = _find_dip_core(
                        wavelength=lam,
                        reflectance=Rup_dn,
                        smooth_win=0, polyorder=0,
                        dip_prom=1e-2, dip_dist=1, peak_dist=1,
                        verbose=False, cfg_name=name + " (Δn)"
                    )  
                                  
                    if flags['show_Rup_dn'] and Rup_dn is not None:
                        ax_plot.plot(lam, Rup_dn, '--', color=color, linewidth=2, alpha=0.7, zorder=100, label=f"{name} (R + Δn)")
                        

                    # 1) half-level lines
                    if flags['show_hlines']:
                        ax_plot.hlines(ylev_dn_list[best_dip_index], lam_left_dn[best_dip_index], lam_right_dn[best_dip_index],
                                        linestyles='--', linewidth=2,
                                        color=color, alpha=0.7, zorder=99)

                    # 2) dips
                    if flags['show_dips']:
                        ax_plot.scatter(lam_dip_dn_list, R_dip_dn_list, #lam[dip_idx_dn]
                                        marker='x', s=40,
                                        color=color, alpha=0.7, zorder=101)

                    # 3) maxima
                    if flags['show_maxima']:
                        ax_plot.scatter(lam_max_l_dn, R_max_l_dn,
                                        marker='x', s=30,
                                        color=color, alpha=0.7, zorder=101)
                        ax_plot.scatter(lam_max_r_dn, R_max_r_dn,
                                        marker='x', s=30,
                                        color=color, alpha=0.7, zorder=101)

                    # 4) symmetry points
                    if flags['show_symmetry_pts']:
                        ax_plot.scatter(lam_sym_dn, R_sym_dn,
                                        marker='x', s=30,
                                        color=color, alpha=0.7, zorder=101)

                    # 5) selected dip
                    if flags['show_selected_dip']:
                        ax_plot.scatter(lam_dip_dn_list[best_dip_index], 
                                        R_dip_dn_list[best_dip_index],  
                                        marker='o', s=70,
                                        facecolor='none',
                                        edgecolor=color, alpha=0.7, zorder=102)
                        
                        
                
                    if flags['show_sensitivity_marker']:
                        if use_half:
                            # 3 marqueurs en half-mode :
                            # (1) base : demi-hauteur de Rup
                            lam_half = compute_half_point(lam, Rup, lam_left_list[best_dip_index], lam_right_list[best_dip_index])
                            ax_plot.scatter(
                                [lam_half], [R0],
                                marker='s', s=80,
                                facecolor='none', edgecolor=color,
                                alpha=0.7, zorder=102,
                                label=f"{name} sens. half-base"
                            )
                            # (2) sur Rup_dn au même λ
                            ax_plot.scatter(
                                [lam_half], [R1],
                                marker='s', s=80,
                                facecolor='none', edgecolor=color,
                                alpha=0.7, zorder=102
                            )
                            # (3) nouveau demi-point sur Rup_dn (même flank)
                            lam_half_dn = compute_half_point(lam, Rup_dn, lam_left_dn[best_dip_index], lam_right_dn[best_dip_index])
                            ax_plot.scatter(
                                [lam_half_dn], [ylev_dn_list[best_dip_index]],
                                marker='s', s=80,
                                facecolor='none', edgecolor=color,
                                alpha=0.7, zorder=102,
                                label=f"{name} sens. half-Δn"
                            )
                        else:
                            # 3 marqueurs en dip-mode :
                            # (1) base : creux de Rup
                            ax_plot.scatter(
                                [lam_dip], [R_dip],
                                marker='s', s=80,
                                facecolor='none', edgecolor=color,
                                alpha=0.7, zorder=102,
                                label=f"{name} sens. dip-base"
                            )
                            # (2) sur Rup_dn au même λ_dip
                            ax_plot.scatter(
                                [lam_dip], [R1],
                                marker='s', s=80,
                                facecolor='none', edgecolor=color,
                                alpha=0.7, zorder=102
                            )
                            # (3) vrai creux sur Rup_dn
                            ax_plot.scatter(
                                [lam_dip_dn_list[best_dip_index]], [R_dip_dn_list[best_dip_index]],
                                marker='s', s=80,
                                facecolor='none', edgecolor=color,
                                alpha=0.7, zorder=102,
                                label=f"{name} sens. dip-Δn"
                            )
                                                        
                        
                        
                                        
            else:
                # Si on ne fait pas Δn, on n’appelle jamais _find_dip_core sur None
                Rup_dn = None
                lam_calc_dn = S_lambda = S_R = dR_half = None
                # Il est préférable aussi de définir à vide ou None toutes les listes/variables
                # qui seraient utilisées plus bas pour éviter des NameError.
                dip_idx_dn = lam_dip_dn_list = R_dip_dn_list = []
                ylev_dn_list = lam_left_dn = lam_right_dn = []
                fwhm_dn = lam_max_l_dn = R_max_l_dn = []
                lam_max_r_dn = R_max_r_dn = lam_sym_dn = R_sym_dn = []
                depth_dn = []
                        
                
                
            # ----- accumulateurs tableau ------------------------------
            cfg_labels.append(name)
            geom = cfg["geometry"]["geometry"]
            geom_sum.append("\n".join(
                f"{d}: {geom[k]}" for k, d in ordered_params if k in geom))
            mat_lines = []
            for e in cfg["material"]["MATERIALS_CONFIG"]:
                key = e['key']
                disp = next((d for k, d in ordered_params if k == key), key)
                mat = e['material']; typ = mat['type'].lower()
                if typ == "standard": val = mat['material']
                elif typ == "custom": val = mat['expression']
                else: val = f"Book: {mat.get('book','')}, Page: {mat.get('page','')}"
                mat_lines.append(f"{disp}: {val}")
            mat_sum.append("\n".join(mat_lines))

            fwhm_sum.append(f"{fwhm:.1f} nm")
            lam_sum.append(f"{lam_dip:.1f} nm")
            delta_lam_over_midLam.append(f"{S_lam_min_abs:.3f} & {S_lam_sym_abs:.3f}")
            Q_fac.append(f"{lam_dip/fwhm:.1f}")

            def _a(lst, cond, val=""): lst.append(val if cond else "")
            _a(S_lambda_sum,
               flags['show_S_lambda'] and name in dn_cfgs and S_lambda is not None, 
               f"{S_lambda:.3f}" if S_lambda is not None else "")
            
            _a(S_R_sum,
               flags['show_S_dn'] and name in dn_cfgs and S_R is not None,
               f"{S_R:.3f}" if S_R is not None else "")
            S_R_vals.append(S_R if S_R is not None else np.nan)
            
            _a(dR_half_sum,
               flags['show_deltaR_half'] and name in dn_cfgs and dR_half is not None,
               f"{dR_half:.3f}" if dR_half is not None else "")



            # ------------------------------------------------------------
            #  AJOUT des métriques supplémentaires dans *details*
            # ------------------------------------------------------------
            details["extra_metrics"] = {
                "Sλ (nm/RIU)" : f"{S_lambda:.3f}" if S_lambda is not None else "",
                "ΔR/Δn (1/RIU)": f"{S_R:.3f}"    if S_R is not None else "",
                "ΔR_half"       : f"{dR_half:.3f}" if dR_half is not None else "",
                "S_lam_min"     : f"{S_lam_min_abs:.3f}",
                "S_lam_sym"     : f"{S_lam_sym_abs:.3f}",
                "Δn"            : f"{delta_n:.3e}"
            }
   

            # ----- tracé ------------------------------------------------
            ax_plot.plot(lam, Rup, color=color, zorder=1,)

            # ----- overlays indépendants du verbose -----------------------

            if flags['show_hlines']:
                ax_plot.hlines(ylev, lam_left, lam_right, color=color)

            if flags['show_dips']:
                ax_plot.scatter(lam[dips_idx_list], Rup[dips_idx_list], marker='x', color=color)

            if flags['show_maxima']:
                ax_plot.scatter(lam_max_l_list, R_max_l_list, marker='x', color=color)
                ax_plot.scatter(lam_max_r_list, R_max_r_list, marker='x', color=color)

            if flags['show_symmetry_pts']:
                ax_plot.scatter(lam_sym_list, R_sym_list, marker='x', color=color)

            if flags['show_selected_dip']:
                ax_plot.scatter([lam_dip], [R_dip], marker='o',
                                facecolor='none', edgecolor=color, s=70)
                
                

                  
                

            # ----- debug : uniquement si verbose --------------------------
            if verbose:
                dips_nm     = ", ".join(f"{lam[d]:.1f}"   for d in dips_idx_list)
                dR_over_dn_str      = ", ".join(f"{s:.3f}"    for s in dR_over_dn_list)
                dLam_over_dn_str    = ", ".join(f"{s1:.3f}"    for s1 in dLam_over_dn_list)
                depths_str  = ", ".join(f"{di:.3f}"        for di in depth_list)
                fwhm_str  = ", ".join(f"{w:.3f}"        for w in width_list)


                if lam_calc_dn is not None:
                    lam_dip_dn_str   = f"{lam_calc_dn:.2f}"
                    delta_lam_str  = f"{(lam_calc_dn - lam_dip):.3f}"
                    S_lambda_str   = f"{S_lambda:.2f}"
                else:
                    lam_dip_dn_str   = "–"
                    delta_lam_str  = "–"
                    S_lambda_str   = "–"
                                                                             # A VERIFIER
                if S_R is not None:
                    S_R_str        = f"{S_R:.3f}"
                else:
                    S_R_str        = "–"
        
    
                # ligne unique de résumé pour ce spectre
                debug_lines.append(
                    f"{name} : dips[{dips_nm}], λ0={lam_dip:.2f} nm, "
                    f"λΔn={lam_dip_dn_str} nm, "
                    f"Δλ={delta_lam_str},  "
                    f"Δλ/Δn[{dLam_over_dn_str}],  best Δλ/Δn={S_lambda_str},  "
                    f"depths[{depths_str}], depth={depth:.3f}  "
                    #f"slopes[{slopes_str}] slope={slope:.3f}  "
                    f"FWHMs[{fwhm_str}], FWHM={fwhm:.1f}  "
                    f"ΔR/Δn[{dR_over_dn_str}], best ΔR/Δn=={S_R_str}"
                )
                debug_lines.append("")
                
            # ----- sauvegarde de R + delta n ------------------------------    
            if Rup_dn is not None: 
                details["Rup_dn"] = Rup_dn.tolist()
                details["delta_n"] = delta_n
                
            # ----- sauvegarde par config ------------------------------
            save_simulation_summary(
                {name: details}, lam_range, wave, n_modes, summary_dir,
                custom_name=name,
                fwhm_summaries=[fwhm_sum[-1]],
                lam_summaries=[lam_sum[-1]],
                delta_lam_over_midLam_summaries=[delta_lam_over_midLam[-1]],
                Q_factor=[Q_fac[-1]],
                best_S_R=[S_R_sum[-1]]
                )

        # ----- meilleure dip with respect to S_R max ---------------------------------------
        
        # On convertit S_R_vals en array pour gérer plus facilement les NaN
        S_R_array = np.array(S_R_vals, dtype=float)

        # Vérifier qu'il y a au moins un élément non-NaN
        if S_R_array.size > 0 and not np.all(np.isnan(S_R_array)):
            best_idx = int(np.nanargmax(S_R_array))
            best_S_R_val = S_R_array[best_idx]
            debug_lines.append(
                f"→ BEST_CONFIG (max ΔR/Δn): {cfg_labels[best_idx]} (S_R = {best_S_R_val:.3f})"
            )        


        # ----- debug ---------------------------------------------------
        if verbose:
            debug_out.value = "\n".join(debug_lines)
            
        # ----- tableau final (uniquement si on a au moins une config) ----
        if not cfg_labels:
            sim_output.clear_output(wait=True)
            with sim_output:
                print("Aucun dip valide trouvé : pas de tableau à afficher.")
                plt.close(fig)
                return

        # ----- tableau final ------------------------------------------
        ax_plot.set_xlabel("Wavelength (nm)")
        ax_plot.set_ylabel("Reflectance"); ax_plot.set_title("Simulation")
        ax_plot.grid(True); ax_table.axis('off')
        
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
        

        col_labels = [lbl.replace("Mat_", "\nMat_") for lbl in cfg_labels]
        if col_labels:
            cellText, rowLabels = [], []
            
            cellText.append(geom_sum); rowLabels.append("Geometry (nm)")
            cellText.append(mat_sum);  rowLabels.append("Material")
            
            if flags['show_fwhm']:     cellText.append(fwhm_sum); rowLabels.append("FWHM (nm)")
            if flags['show_lambda0']:  cellText.append(lam_sum);  rowLabels.append(r"$\lambda_0$")
            if flags['show_delta_lam_over_midLam']: cellText.append(delta_lam_over_midLam); rowLabels.append(r"$\Delta_{\lambda}$ / $\lambda_{min}$ or $\lambda_{sym}$")
            if flags['show_S_lambda']: cellText.append(S_lambda_sum); rowLabels.append(r"$S_{\lambda}$ = Δλ / Δn (nm/RIU)")
            if flags['show_S_dn']:     cellText.append(S_R_sum); rowLabels.append(r"ΔR/Δn (1/RIU)")
            if flags['show_deltaR_half']: cellText.append(dR_half_sum); rowLabels.append(r"$\Delta R_{half}$")
            if flags['show_Q']:        cellText.append(Q_fac); rowLabels.append("Q‑factor")

            filtered = []
            for lbl, row in zip(rowLabels, cellText):
                if lbl in ("Geometry (nm)", "Material"):
                    ref = row[0]
                    new = [row[0]] + [("" if row[j]==ref else row[j]) for j in range(1,len(row))]
                    filtered.append(new)
                else:
                    filtered.append(row)
            cellText = filtered

                       
            n_cfg = len(col_labels)
            fs = 8 if n_cfg <= 5 else max(8 - (n_cfg - 5), 3)
            table = ax_table.table(
                cellText=cellText, colLabels=col_labels, rowLabels=rowLabels,
                loc="center", cellLoc="left")
            table.auto_set_font_size(False); table.set_fontsize(fs)
            table.auto_set_column_width(col=list(range(n_cfg)))

            for (r, c), cell in table.get_celld().items():
                if r == -1 or c == -1:
                    cell.set_facecolor("#40466e")
                    cell.get_text().set_color("white")
                    cell.get_text().set_weight("bold")
                else:
                    cell.set_facecolor("whitesmoke")
                    cell.set_edgecolor("lightgray")
                    cell.set_linewidth(0.5)
                    cell.get_text().set_color(colors[c % len(colors)])

            # --------- hauteur dynamique --------------------------------
            row_heights = {}
            for (r, c), cell in table.get_celld().items():
                if r >= 0:
                    nb = cell.get_text().get_text().count("\n") + 1
                    row_heights[r] = max(row_heights.get(r, 0), nb)
            for (r, c), cell in table.get_celld().items():
                if r in row_heights:
                    cell.set_height(0.04 * row_heights[r])

        # ----- affichage final ----------------------------------------
        sim_output.clear_output(wait=True)
        with sim_output:
            display(fig)
            display(_download_link(fig,
                   f"simulation_{datetime.now():%Y%m%d_%H%M%S}.png"))
        plt.close(fig)

    sim_run_button.on_click(_run)

    # ======================= assemblage final =========================
    columns = HBox([sim_controls, conv_widget],
                layout=Layout(align_items='flex-start'))
    return VBox([columns, metrics_selector, debug_out, sim_output])
