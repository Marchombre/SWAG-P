from gap_plasmon_2d import paths
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module : simulation.py
Onglet « Simulation » – version orientée objet, reprenant
exactement la logique de create_simulation_tab() originale.
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
import h5py

from ipywidgets import Layout, HBox, VBox, ToggleButton, HTML
from IPython.display import HTML as DHTML, display, Javascript
from scipy.interpolate import interp1d

# pour surveiller les changements de fichiers
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
from IPython import get_ipython

from gap_plasmon_2d.utils.file_watchers import start_watcher

# dépendances internes
from gap_plasmon_2d.simulation.simulate_and_plot     import ordered_params, run_simulation_one_combo
from gap_plasmon_2d.utils.data_readers          import list_sim_summary_files
from gap_plasmon_2d.analysis.convergence_analysis  import create_multi_convergence_widget
from gap_plasmon_2d.utils.saving__functions      import save_simulation_summary
from gap_plasmon_2d.analysis.characterization      import (
    _find_dip_core, find_best_dip,
    simulate_delta_spectrum, compute_half_point
)

# --------------------------------------------------------------------- #
#                               chemins                                 #
# --------------------------------------------------------------------- #
module_dir         = os.path.dirname(os.path.abspath(__file__))
workspace_dir      = os.path.dirname(module_dir)
notebooks_dir      = os.path.join(str(paths.RESULTS_DIR))
summary_dir        = os.path.join(notebooks_dir, "summary_simulation")
exp_data_dir       = os.path.join(notebooks_dir, "Experimental_Data")
configurations_dir = os.path.join(str(paths.CONFIGS_DIR))
data_dir           = os.path.join(str(paths.DATA_DIR))
json_combined_path = os.path.join(data_dir, "combined_materials.json")

h5_path = paths.H5_RESULTS_DIR / str(paths.H5_RESULTS_DIR / "simulation_results.h5")
h5_path.parent.mkdir(parents=True, exist_ok=True)

def _download_link(fig, fname="figure.png"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.05)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return DHTML(f'<a download="{fname}" href="data:image/png;base64,{b64}" '
                 f'target="_blank">Télécharger l’image</a>')


class SimulationTab:
    """
    Encapsule tout l’onglet Simulation : widgets, callback _run,
    et méthode cost(). Reproduit fidèlement la fonction
    create_simulation_tab() initiale.
    """

    def __init__(self):
        # ------------- charger les configs JSON ------------------
        cfg_file = os.path.join(configurations_dir, "geom_mat_combinations.json")
        if os.path.exists(cfg_file):
            with open(cfg_file, encoding="utf-8") as f:
                self.all_configs = json.load(f)["ALL_COMBINED_CONFIGS"]
        else:
            self.all_configs = []

        # ------------- garde-fou valeurs négatives ---------------
        def _positive(change):
            if change['new'] < 0:
                change['owner'].value = 0

        # --------- widgets paramètres généraux -------------------
        self.sim_lambda_min = widgets.FloatText(
            value=300.0, description="λ min (nm):",
            layout=Layout(width='150px'), style={'description_width': 'initial'})
        self.sim_lambda_max = widgets.FloatText(
            value=1100.0, description="λ max (nm):",
            layout=Layout(width='150px'), style={'description_width': 'initial'})
        self.sim_n_points = widgets.IntText(
            value=400, description="Points:",
            layout=Layout(width='200px'), style={'description_width': 'initial'})
        self.sim_n_mod = widgets.IntText(
            value=5, description="Modes:",
            layout=Layout(width='200px'), style={'description_width': 'initial'})

        for w in (self.sim_lambda_min,
                  self.sim_lambda_max,
                  self.sim_n_points,
                  self.sim_n_mod):
            w.observe(_positive, names='value')

        self.mode_calc_radio = widgets.RadioButtons(
            options=[('Dip (min)', 'dip'),
                     ('FWHM (half)', 'half')],
            value='dip',
            description='Mode calc:',
            style={'description_width':'initial'},
            layout=Layout(width='200px')
        )
        self.sim_run_button = widgets.Button(
            description="Run simulation", button_style="success",
            tooltip="Lancer la simulation")

        self.mode_selection = widgets.RadioButtons(
            options=[('Fixe', 'fixed'),
                     ('Personnalisé', 'custom'),
                     ('Automatique', 'auto')],
            value='fixed', description='Modes:',
            style={'description_width': 'initial'})
        self.custom_modes_box = VBox()

        # --------- gestion des fichiers .npz --------------------
        self.sim_files_dropdown = widgets.Dropdown(
            options=list_sim_summary_files(summary_dir),
            description="Simulation files:",
            layout=Layout(width='500px'),
            style={'description_width': 'initial'})

        self.sim_download_button = widgets.Button(
            description="Download", button_style="danger",
            tooltip="Télécharger le fichier")


        self.sim_download_button.on_click(self._download_file)

        # --------- nom de simulation -----------------------------
        self.sim_name_widget = widgets.Text(
            value="", placeholder="Nom de simulation (auto si vide)",
            description="Sim Name:", layout=Layout(width='500px'),
            style={'description_width': 'initial'})

        # --------- sélecteur Config / Δn -------------------------
        self.config_checkboxes = {}
        self.dn_checkboxes = {}
        config_rows = []
        for cfg in self.all_configs:
            name = cfg["config_name"]
            chk = widgets.Checkbox(value=False,
                                   description=name, indent=False)
            dn  = widgets.Checkbox(value=False,
                                   description='Δn', indent=False,
                                   layout=Layout(width='46px'))
            self.config_checkboxes[name] = chk
            self.dn_checkboxes[name] = dn
            config_rows.append(HBox([chk, dn],
                                    layout=Layout(grid_gap='5px')))

        visible = min(len(config_rows), 10)

        self.select_all_cfg_btn = widgets.Button(
            description="Tout sélectionner Configs",
            button_style="info",
            layout=Layout(width='auto', margin='0 5px 5px 0')
        )
        self.select_all_dn_btn = widgets.Button(
            description="Tout sélectionner Δn",
            button_style="info",
            layout=Layout(width='auto', margin='0 0 5px 0')
        )
        self.select_all_cfg_btn.on_click(self._toggle_all_cfg)
        self.select_all_dn_btn.on_click(self._toggle_all_dn)

        self.config_list = VBox(
            children=[
                HTML("<b>Configurations et Δn</b>"),
                HBox([self.select_all_cfg_btn,
                      self.select_all_dn_btn],
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
        self.toggle_btn = ToggleButton(
            value=False, description="Select your configuration and Δn",
            icon='caret-down', layout=Layout(width='520px'), button_style='warning')
        self.toggle_btn.observe(self._toggle_config_list, names='value')
        
        self.config_refresh_btn = widgets.Button(
            description="Refresh Configs",
            button_style="info",
            tooltip="Refresh configurations previously saved",
            layout=Layout(width='auto', margin='0 5px 5px 0')
        )
        self.config_refresh_btn.on_click(self._refresh_configs)
        
        
        self.config_selector = VBox(
            [ HBox([ self.toggle_btn, self.config_refresh_btn ]),  # ajoutez ici
            self.config_list ],
            layout=Layout(padding='5px')
        )

        # --------- couche(s) Δn ----------------------------------
        layer_keys = [
            m['key']
            for m in self.all_configs[0]['material']['MATERIALS_CONFIG']
        ]
        self.layer_selector = widgets.SelectMultiple(
            options=layer_keys, description="Couche(s) Δn:",
            layout=Layout(width='300px', height='100px'),
            style={'description_width': 'initial'}
        )
        self.delta_n_widget = widgets.FloatText(
            value=1e-2, description="Δn:",
            layout=Layout(width='150px'),
            style={'description_width': 'initial'}
        )
        self.delta_n_widget.observe(_positive, names='value')

        # --------- custom modes rafraîchissement ------------------
        self.custom_n_mod_inputs = {}
        self.mode_selection.observe(self._refresh_custom_modes,
                                    names='value')
        #for cb in self.config_checkboxes.values():
        #    cb.observe(self._refresh_custom_modes, names='value')

        # --------- verbose & debug -------------------------------
        self.verbose_toggle = widgets.Checkbox(
            value=False, description="Verbose", indent=False,
            layout=Layout(width='100%'),
            style={'description_width': 'initial'}
        )
        self.debug_out = widgets.Textarea(
            value='',
            placeholder='Logs verbose…',
            layout=Layout(
                width='100%',
                height='200px',
                overflow_y='scroll',
                border='1px solid darkred'
            ),
            disabled=False
        )
        self.verbose_toggle.observe(self._toggle_debug,
                                    names='value')

        # --------- métriques / overlays --------------------------
        def _cb(v, d): return widgets.Checkbox(value=v,
                                               description=d)
        self.show_fwhm_chk              = _cb(False, "FWHM")
        self.show_lambda0_chk           = _cb(True,  r"λ0")
        self.show_delta_lam_over_midLam = _cb(False,
                                              r"Δλ / λmin or λsym")
        self.show_S_lambda_chk          = _cb(True,
                                              "Sλ = Δλ / Δn (nm/RIU)")
        self.show_S_dn_chk              = _cb(True,
                                              r"ΔR/Δn (1/RIU)")
        self.show_deltaR_half_chk       = _cb(True, r"ΔR_half")
        self.show_Q_chk                 = _cb(False, "Q-factor")
        self.show_Rup_dn_chk            = _cb(True,
                                              "Rup_dn dashed")
        self.show_hlines_chk            = _cb(False,
                                              "half-level line")
        self.show_dips_chk              = _cb(False, "dips (×)")
        self.show_maxima_chk            = _cb(False, "maxima (×)")
        self.show_symmetry_pts_chk      = _cb(False,
                                              "symmetric pts (×)")
        self.show_selected_dip_chk      = _cb(True,
                                              "selected dip (○)")
        self.show_sensitivity_marker = _cb(True,
                                           "sensitivity marker (□)")

        # HBox metrics
        self.metrics_selector = VBox(
            children=[
                HTML("<b>Métriques à afficher :</b>"),
                HBox(
                    [ self.show_fwhm_chk,
                      self.show_lambda0_chk,
                      self.show_delta_lam_over_midLam,
                      self.show_S_lambda_chk,
                      self.show_S_dn_chk,
                      self.show_deltaR_half_chk,
                      self.show_Q_chk ],
                    layout=Layout(
                        display='flex',
                        flex_flow='row nowrap',
                        justify_content='space-around',
                        gap='0px',
                        margin='0 10px 0 0',
                        padding='0'
                    )
                ),
                HTML("<b>Overlays graphiques :</b>"),
                HBox(
                    [ self.show_Rup_dn_chk,
                      self.show_hlines_chk,
                      self.show_dips_chk,
                      self.show_maxima_chk,
                      self.show_symmetry_pts_chk,
                      self.show_selected_dip_chk,
                      self.show_sensitivity_marker ],
                    layout=Layout(
                        display='flex',
                        flex_flow='row nowrap',
                        justify_content='space-around',
                        gap='0px',
                        margin='0 10px 0 0',
                        padding='0'
                    )
                )
            ],
            layout=Layout(
                width='100%',
                border='1px solid lightgray',
                padding='5px',
                margin='10px 0'
            )
        )

        # --------- convergences (droite) -------------------------
        self.conv_widget = create_multi_convergence_widget(
            json_combined_path, self.all_configs
        )

        # --------- sortie figure & tableau ----------------------
        self.sim_output = widgets.Output(
            layout=Layout(
                border='2px solid #ccc',
                padding='10px',
                min_height='400px',
                margin='40px 0 0 0'
            )
        )

        # --------- conteneur gauche (controls) ------------------
        self.sim_controls = VBox(
            children=[
                HTML("<h3>Simulation – Paramètres</h3>"),
                self.sim_name_widget,
                HBox([ self.sim_files_dropdown ]),
                HBox([ self.sim_download_button ]),
                HBox([ self.sim_lambda_min,
                       self.sim_lambda_max ]),
                HBox([ self.sim_n_points,
                       self.sim_n_mod ]),
                self.config_selector,
                HBox([ self.mode_selection,
                       self.layer_selector ]),
                self.custom_modes_box,
                HBox([ self.delta_n_widget,
                       self.mode_calc_radio,
                       self.sim_run_button ]),
                self.verbose_toggle
            ],
            layout=Layout(
                padding='10px',
                border='1px solid lightgray'
            )
        )

        # --------- assemble final UI ---------------------------
        self.tab = VBox(
            [ HBox([ self.sim_controls,
                     self.conv_widget ],
                   layout=Layout(align_items='flex-start')),
              self.metrics_selector,
              self.debug_out,
              self.sim_output ]
        )

        # branchement du bouton Run
        self.sim_run_button.on_click(self._run)

    # ----------------------------------------------------------------- #
    #                    méthodes utilitaires                           #
    # ----------------------------------------------------------------- #
    
    
    def _get_n_modes_for(self, cfg_name):
        """
        Renvoie le nombre de modes à utiliser pour la config cfg_name,
        en suivant fixed/custom/auto comme dans _run().
        """
        # fixed
        if self.mode_selection.value == 'fixed':
            return int(self.sim_n_mod.value)

        # custom
        if self.mode_selection.value == 'custom':
            # si l'utilisateur a rafraîchi les custom_modes
            return int(self.custom_n_mod_inputs.get(cfg_name, 
                              widgets.IntText(value=self.sim_n_mod.value)).value)

        # auto
        # on reconstitue l'auto_modes exactement comme dans _run()
        conv_json = Path(workspace_dir) / "results/summary_convergence/convergence_results.json"
        if conv_json.exists():
            with open(conv_json, encoding='utf-8') as f:
                master = json.load(f)
            auto_modes = {
                name: r[-1]["optimal_n_mode"]
                for name, r in master.get("configs", {}).items() if r
            }
            if cfg_name in auto_modes:
                return int(auto_modes[cfg_name])
        # fallback
        return int(self.sim_n_mod.value)
    
    
    def _toggle_all_cfg(self, _):
        sel = all(cb.value for cb in self.config_checkboxes.values())
        for cb in self.config_checkboxes.values():
            cb.value = not sel

    def _toggle_all_dn(self, _):
        sel = all(dn.value for dn in self.dn_checkboxes.values())
        for dn in self.dn_checkboxes.values():
            dn.value = not sel

    def _toggle_config_list(self, change):
        show = 'block' if change['new'] else 'none'
        self.config_list.layout.display = show
        self.toggle_btn.icon = 'caret-up' if change['new'] else 'caret-down'


    def _refresh_configs(self, _):
            # 1) Recharge le JSON
            cfg_file = os.path.join(configurations_dir, "geom_mat_combinations.json")
            if os.path.exists(cfg_file):
                with open(cfg_file, encoding="utf-8") as f:
                    self.all_configs = json.load(f)["ALL_COMBINED_CONFIGS"]
            else:
                self.all_configs = []

            # 2) Reconstruit les cases à cocher
            self.config_checkboxes.clear()
            self.dn_checkboxes.clear()
            config_rows = []
            for cfg in self.all_configs:
                name = cfg["config_name"]
                chk = widgets.Checkbox(value=False, description=name, indent=False)
                dn  = widgets.Checkbox(value=False, description='Δn', indent=False,
                                    layout=Layout(width='46px'))
                self.config_checkboxes[name] = chk
                self.dn_checkboxes[name] = dn
                config_rows.append(HBox([chk, dn], layout=Layout(grid_gap='5px')))

            # 3) Met à jour le conteneur avec le header et les boutons
            header = HTML("<b>Configurations et Δn</b>")
            buttons = HBox([self.select_all_cfg_btn, self.select_all_dn_btn],
                        layout=Layout(grid_gap='10px'))
            self.config_list.children = [header, buttons, *config_rows]

            # 4) Conserve l’état ouvert/fermé et rafraîchit les modes custom
            self._toggle_config_list({'new': self.toggle_btn.value})
            self._refresh_custom_modes()



    def _refresh_custom_modes(self, *_):
        sel = [c for c in self.all_configs
               if self.config_checkboxes[c['config_name']].value]
        if self.mode_selection.value != 'custom' or not sel:
            self.custom_modes_box.children = []
            return
        inputs = []
        for cfg in sel:
            name = cfg['config_name']
            it = widgets.IntText(
                value=self.sim_n_mod.value,
                description=name,
                layout=Layout(width='300px'),
                style={'description_width':'initial'}
            )
            self.custom_n_mod_inputs[name] = it
            inputs.append(it)
        self.custom_modes_box.children = inputs

    def _toggle_debug(self, change):
        self.debug_out.layout.display = 'block' if change['new'] else 'none'
        if not change['new']:
            self.debug_out.value = ''

    def _download_file(self, _):
        path = self.sim_files_dropdown.value
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

    # ----------------------------------------------------------------- #
    #                              callback RUN                         #
    # ----------------------------------------------------------------- #
    def _run(self, _):
        with self.sim_output:
            self.sim_output.clear_output(wait=True)
            display(HTML(
                "<div style='text-align:center'>"
                "<img src='https://i.gifer.com/ZZ5H.gif' width='40'>"
                "<br><em>Simulation en cours…</em></div>"
            ))

        # base params
        lam_range   = np.linspace(
            self.sim_lambda_min.value,
            self.sim_lambda_max.value,
            self.sim_n_points.value
        )
        wave        = {"angle": 0, "polarization": 1}
        colors      = plt.rcParams['axes.prop_cycle'].by_key()['color']
        sel_layers  = list(self.layer_selector.value)
        delta_n     = max(self.delta_n_widget.value, 1e-9)

        # auto-modes
        auto_modes = {}
        conv_json = Path(workspace_dir) / "results/summary_convergence/convergence_results.json"
        if conv_json.exists():
            with open(conv_json, encoding='utf-8') as f:
                master = json.load(f)
            auto_modes = {
                n: r[-1]["optimal_n_mode"]
                for n, r in master.get("configs", {}).items() if r
            }

        # selected configs & Δn-configs
        selected_cfgs = [
            cfg for cfg in self.all_configs
            if self.config_checkboxes[cfg['config_name']].value
        ]
        dn_cfgs = {
            name for name, dn in self.dn_checkboxes.items() if dn.value
        }

        # mode_by_cfg
        mode_by_cfg = {}
        if self.mode_selection.value == 'fixed':
            for cfg in selected_cfgs:
                mode_by_cfg[cfg['config_name']] = self.sim_n_mod.value
        elif self.mode_selection.value == 'custom':
            for n, inp in self.custom_n_mod_inputs.items():
                mode_by_cfg[n] = inp.value
        else:
            for cfg in selected_cfgs:
                nom = cfg['config_name']
                if nom in auto_modes:
                    mode_by_cfg[nom] = int(auto_modes[nom])
                else:
                    mode_by_cfg[nom] = self.sim_n_mod.value
                    print(f"[AVERTISSEMENT] Pas de mode auto trouvé pour '{nom}'. Mode par défaut utilisé.")

        # préparer figure
        fig      = plt.figure(figsize=(13, 9))
        ax_plot  = fig.add_axes([0.10, 0.50, 0.80, 0.35])
        ax_table = fig.add_axes([0.10, 0.05, 0.80, 0.35])
        ax_table.axis('off')

        # accumulateurs
        cfg_labels            = []
        geom_sum              = []
        mat_sum               = []
        fwhm_sum              = []
        lam_sum               = []
        delta_lam_over_midLam = []
        S_lambda_sum          = []
        S_R_sum               = []
        dR_half_sum           = []
        S_R_vals              = []
        S_lam_min             = []
        S_lam_sym             = []
        Q_fac                 = []
        debug_lines           = []

        verbose = self.verbose_toggle.value

        # flags dict for convenience
        flags = dict(
            show_fwhm=self.show_fwhm_chk.value,
            show_lambda0=self.show_lambda0_chk.value,
            show_delta_lam_over_midLam=self.show_delta_lam_over_midLam.value,
            show_S_lambda=self.show_S_lambda_chk.value,
            show_S_dn=self.show_S_dn_chk.value,
            show_deltaR_half=self.show_deltaR_half_chk.value,
            show_Q=self.show_Q_chk.value,
            show_Rup_dn=self.show_Rup_dn_chk.value,
            show_hlines=self.show_hlines_chk.value,
            show_dips=self.show_dips_chk.value,
            show_maxima=self.show_maxima_chk.value,
            show_symmetry_pts=self.show_symmetry_pts_chk.value,
            show_selected_dip=self.show_selected_dip_chk.value,
            show_sensitivity_marker=self.show_sensitivity_marker.value
        )
        use_half = (self.mode_calc_radio.value == 'half')

        # boucle configs
        for idx, cfg in enumerate(selected_cfgs):
            color = colors[idx % len(colors)]
            name  = cfg['config_name']
            n_modes = mode_by_cfg[name]

            Rup, _, details = run_simulation_one_combo(
                lam_range, wave, n_modes,
                cfg, json_combined_path
            )
            Rup = np.asarray(Rup, dtype=float)
            simulation_details = details

            # find_best_dip
            Best_values_out, who, best_dip_index = find_best_dip(
                cfg=cfg,
                wavelength=lam_range,
                reflectance=Rup,
                wave=wave,
                n_modes=n_modes,
                sel_layers=sel_layers,
                delta_n=delta_n,
                json_combined_path=json_combined_path,
                smooth_win=0, polyorder=0,
                dip_prom=1e-2, dip_dist=1,
                peak_dist=1, verbose=True,
                cfg_name=name,
                mode=('half' if use_half else 'dip')
            )

            (dip_list, lam_dip_list, R_dip_list,
             y_level_list,
             lam_left_list, lam_right_list, fwhm_list,
             lam_max_l_list, R_max_l_list,
             lam_max_r_list, R_max_r_list,
             lam_sym_list, R_sym_list,
             depth_list), _ = _find_dip_core(
                wavelength=lam_range,
                reflectance=Rup,
                smooth_win=0, polyorder=0,
                dip_prom=1e-2, dip_dist=1,
                peak_dist=1, verbose=False,
                cfg_name=name
            )

            # numpy arrays
            lam_max_l = np.array(lam_max_l_list)
            R_max_l   = np.array(R_max_l_list)
            lam_max_r = np.array(lam_max_r_list)
            R_max_r   = np.array(R_max_r_list)
            lam_sym   = np.array(lam_sym_list)
            R_sym     = np.array(R_sym_list)
            width_arr = np.array(fwhm_list)
            depth_arr = np.array(depth_list)

            if Best_values_out is None:
                debug_lines.append(
                    f"Aucun dip sélectionné pour “{who}” – ignorée."
                )
                continue

            (lam_left, lam_right, fwhm, depth,
             lam_dip, R_dip, ylev,
             lam_m_l, Rm_l, lam_m_r, Rm_r,
             lam_sympt, R_sympt,
             best_S_R, S_lambda, dR_half,
             dips_idx_list, dR_over_dn_list, dLam_over_dn_list
            ) = Best_values_out

            lam_min = lam_m_l if Rm_l < Rm_r else lam_m_r
            lam_mid = lam_left if Rm_l < Rm_r else lam_right

            S_lam_min_abs = abs((lam_dip - lam_min) / lam_mid)
            S_lam_sym_abs = abs((lam_dip - lam_sympt) / lam_mid)
            S_lam_min.append(S_lam_min_abs)
            S_lam_sym.append(S_lam_sym_abs)

            compute_delta = (
                flags['show_Rup_dn'] and
                name in dn_cfgs and
                sel_layers
            )

            if compute_delta:
                Rup_dn, lam_calc, R0, lam_calc_dn, R1, \
                S_lambda_val, S_R_val, dR_half_val = simulate_delta_spectrum(
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
                    mode=('half' if use_half else 'dip')
                )
                if Rup_dn is not None:
                    (dip_idx_dn, lam_dip_dn_list,
                     R_dip_dn_list, ylev_dn_list,
                     lam_left_dn, lam_right_dn,
                     fwhm_dn,
                     lam_max_l_dn, R_max_l_dn,
                     lam_max_r_dn, R_max_r_dn,
                     lam_sym_dn, R_sym_dn,
                     depth_dn), _ = _find_dip_core(
                        wavelength=lam_range,
                        reflectance=Rup_dn,
                        smooth_win=0, polyorder=0,
                        dip_prom=1e-2, dip_dist=1,
                        peak_dist=1, verbose=False,
                        cfg_name=name + " (Δn)"
                    )
                    if flags['show_Rup_dn']:
                        ax_plot.plot(
                            lam_range, Rup_dn, '--',
                            color=color, linewidth=2,
                            alpha=0.7, zorder=100,
                            label=f"{name} (R + Δn)"
                        )
                    if flags['show_hlines']:
                        ax_plot.hlines(
                            ylev_dn_list[best_dip_index],
                            lam_left_dn[best_dip_index],
                            lam_right_dn[best_dip_index],
                            linestyles='--', linewidth=2,
                            color=color, alpha=0.7, zorder=99
                        )
                    if flags['show_dips']:
                        ax_plot.scatter(
                            lam_dip_dn_list, R_dip_dn_list,
                            marker='x', s=40,
                            color=color, alpha=0.7, zorder=101
                        )
                    if flags['show_maxima']:
                        ax_plot.scatter(
                            lam_max_l_dn, R_max_l_dn,
                            marker='x', s=30,
                            color=color, alpha=0.7, zorder=101
                        )
                        ax_plot.scatter(
                            lam_max_r_dn, R_max_r_dn,
                            marker='x', s=30,
                            color=color, alpha=0.7, zorder=101
                        )
                    if flags['show_symmetry_pts']:
                        ax_plot.scatter(
                            lam_sym_dn, R_sym_dn,
                            marker='x', s=30,
                            color=color, alpha=0.7, zorder=101
                        )
                    if flags['show_selected_dip']:
                        ax_plot.scatter(
                            lam_dip_dn_list[best_dip_index],
                            R_dip_dn_list[best_dip_index],
                            marker='o', s=70,
                            facecolor='none', edgecolor=color,
                            alpha=0.7, zorder=102
                        )
                    if flags['show_sensitivity_marker']:
                        if use_half:
                            lam_half = compute_half_point(
                                lam_range, Rup,
                                lam_left_list[best_dip_index],
                                lam_right_list[best_dip_index]
                            )
                            ax_plot.scatter(
                                [lam_half], [R0],
                                marker='s', s=80,
                                facecolor='none',
                                edgecolor=color,
                                alpha=0.7, zorder=102,
                                label=f"{name} sens. half-base"
                            )
                            ax_plot.scatter(
                                [lam_half], [R1],
                                marker='s', s=80,
                                facecolor='none',
                                edgecolor=color,
                                alpha=0.7, zorder=102
                            )
                            lam_half_dn = compute_half_point(
                                lam_range, Rup_dn,
                                lam_left_dn[best_dip_index],
                                lam_right_dn[best_dip_index]
                            )
                            ax_plot.scatter(
                                [lam_half_dn],
                                [ylev_dn_list[best_dip_index]],
                                marker='s', s=80,
                                facecolor='none',
                                edgecolor=color,
                                alpha=0.7, zorder=102,
                                label=f"{name} sens. half-Δn"
                            )
                        else:
                            ax_plot.scatter(
                                [lam_dip], [R_dip],
                                marker='s', s=80,
                                facecolor='none',
                                edgecolor=color,
                                alpha=0.7, zorder=102,
                                label=f"{name} sens. dip-base"
                            )
                            ax_plot.scatter(
                                [lam_dip], [R1],
                                marker='s', s=80,
                                facecolor='none',
                                edgecolor=color,
                                alpha=0.7, zorder=102
                            )
                            ax_plot.scatter(
                                [lam_dip_dn_list[best_dip_index]],
                                [R_dip_dn_list[best_dip_index]],
                                marker='s', s=80,
                                facecolor='none',
                                edgecolor=color,
                                alpha=0.7, zorder=102,
                                label=f"{name} sens. dip-Δn"
                            )
                # fin simulate_delta

            # accumulateurs tableau
            cfg_labels.append(name)
            geom = cfg["geometry"]["geometry"]
            geom_sum.append("\n".join(
                f"{d}: {geom[k]}" for k, d in ordered_params
                if k in geom
            ))
            mat_lines = []
            for e in cfg["material"]["MATERIALS_CONFIG"]:
                key = e['key']
                disp = next((d for k, d in ordered_params
                             if k == key), key)
                mat = e['material']; typ = mat['type'].lower()
                if typ == "standard":
                    val = mat['material']
                elif typ == "custom":
                    val = mat['expression']
                else:
                    val = (f"Book: {mat.get('book','')}, "
                           f"Page: {mat.get('page','')}")
                mat_lines.append(f"{disp}: {val}")
            mat_sum.append("\n".join(mat_lines))

            fwhm_sum.append(f"{fwhm:.1f} nm")
            lam_sum.append(f"{lam_dip:.1f} nm")
            delta_lam_over_midLam.append(
                f"{S_lam_min_abs:.3f} & {S_lam_sym_abs:.3f}"
            )
            Q_fac.append(f"{lam_dip/fwhm:.1f}")

            def _append(lst, cond, val=""):
                lst.append(val if cond else "")
            _append(
                S_lambda_sum,
                flags['show_S_lambda'] and name in dn_cfgs
                and S_lambda is not None,
                f"{S_lambda:.3f}" if S_lambda is not None else ""
            )
            _append(
                S_R_sum,
                flags['show_S_dn'] and name in dn_cfgs
                and best_S_R is not None,
                f"{best_S_R:.3f}" if best_S_R is not None else ""
            )
            S_R_vals.append(best_S_R if best_S_R is not None else np.nan)
            _append(
                dR_half_sum,
                flags['show_deltaR_half'] and name in dn_cfgs
                and dR_half is not None,
                f"{dR_half:.3f}" if dR_half is not None else ""
            )

            # enrichir details
            details["extra_metrics"] = {
                "Sλ (nm/RIU)"   : f"{S_lambda:.3f}"
                                    if S_lambda is not None else "",
                "ΔR/Δn (1/RIU)": f"{best_S_R:.3f}"
                                    if best_S_R is not None else "",
                "ΔR_half"      : f"{dR_half:.3f}"
                                    if dR_half is not None else "",
                "S_lam_min"    : f"{S_lam_min_abs:.3f}",
                "S_lam_sym"    : f"{S_lam_sym_abs:.3f}",
                "Δn"           : f"{delta_n:.3e}"
            }

            # tracé de base
            ax_plot.plot(lam_range, Rup, color=color, zorder=1)

            # overlays non-verbose
            if flags['show_hlines']:
                ax_plot.hlines(ylev, lam_left, lam_right,
                               color=color)
            if flags['show_dips']:
                ax_plot.scatter(
                    lam_range[dips_idx_list],
                    Rup[dips_idx_list],
                    marker='x', color=color
                )
            if flags['show_maxima']:
                ax_plot.scatter(
                    lam_max_l_list, R_max_l_list,
                    marker='x', color=color
                )
                ax_plot.scatter(
                    lam_max_r_list, R_max_r_list,
                    marker='x', color=color
                )
            if flags['show_symmetry_pts']:
                ax_plot.scatter(
                    lam_sym_list, R_sym_list,
                    marker='x', color=color
                )
            if flags['show_selected_dip']:
                ax_plot.scatter(
                    [lam_dip], [R_dip],
                    marker='o', facecolor='none',
                    edgecolor=color, s=70
                )

            # debug verbose
            if verbose:
                dips_nm = ", ".join(
                    f"{lam_range[d]:.1f}" for d in dips_idx_list
                )
                dR_over_str   = ", ".join(
                    f"{s:.3f}" for s in dR_over_dn_list
                )
                dLam_over_str = ", ".join(
                    f"{s1:.3f}" for s1 in dLam_over_dn_list
                )
                depths_str = ", ".join(
                    f"{di:.3f}" for di in depth_list
                )
                fwhm_str   = ", ".join(
                    f"{w:.3f}" for w in width_arr
                )

                if 'lam_calc_dn' in locals() and lam_calc_dn is not None:
                    lam_dip_dn_str = f"{lam_calc_dn:.2f}"
                    delta_lam_str  = f"{(lam_calc_dn - lam_dip):.3f}"
                    S_lambda_str   = f"{S_lambda:.2f}"
                else:
                    lam_dip_dn_str = delta_lam_str = S_lambda_str = "–"
                S_R_str = f"{best_S_R:.3f}" if best_S_R is not None else "–"

                debug_lines.append(
                    f"{name} : dips[{dips_nm}], "
                    f"λ0={lam_dip:.2f} nm, "
                    f"λΔn={lam_dip_dn_str} nm, "
                    f"Δλ={delta_lam_str},  "
                    f"Δλ/Δn[{dLam_over_str}],  "
                    f"best Δλ/Δn={S_lambda_str},  "
                    f"depths[{depths_str}], "
                    f"depth={depth:.3f}  "
                    f"FWHMs[{fwhm_str}], "
                    f"FWHM={fwhm:.1f}  "
                    f"ΔR/Δn[{dR_over_str}], "
                    f"best ΔR/Δn=={S_R_str}"
                )
                debug_lines.append("")

            # sauvegarde par config
            save_simulation_summary(
                {name: details},
                lam_range, wave, n_modes, summary_dir,
                custom_name=name,
                fwhm_summaries=[fwhm_sum[-1]],
                lam_summaries=[lam_sum[-1]],
                delta_lam_over_midLam_summaries=[
                    delta_lam_over_midLam[-1]
                ],
                Q_factor=[Q_fac[-1]],
                best_S_R=[S_R_sum[-1]]
            )

        # meilleur ΔR/Δn
        arr = np.array(S_R_vals, dtype=float)
        if arr.size and not np.all(np.isnan(arr)):
            best_idx = int(np.nanargmax(arr))
            debug_lines.append(
                f"→ BEST_CONFIG (max ΔR/Δn): "
                f"{cfg_labels[best_idx]} "
                f"(S_R = {arr[best_idx]:.3f})"
            )

        # afficher debug
        if verbose:
            self.debug_out.value = "\n".join(debug_lines)

        # si aucune config valide
        if not cfg_labels:
            with self.sim_output:
                print("Aucun dip valide trouvé : pas de tableau à afficher.")
            plt.close(fig)
            return

        # filtrer Geometry & Material
        base_geom = set(geom_sum[0].splitlines())
        new_geom  = []
        for i, txt in enumerate(geom_sum):
            lines = txt.splitlines()
            if i == 0:
                new_geom.append(txt)
            else:
                diff = [l for l in lines if l not in base_geom]
                new_geom.append("\n".join(diff))
        geom_sum = new_geom

        base_mat = set(mat_sum[0].splitlines())
        new_mat  = []
        for i, txt in enumerate(mat_sum):
            lines = txt.splitlines()
            if i == 0:
                new_mat.append(txt)
            else:
                diff = [l for l in lines if l not in base_mat]
                new_mat.append("\n".join(diff))
        mat_sum = new_mat

        # construire la table
        col_labels = [lbl.replace("Mat_", "\nMat_")
                      for lbl in cfg_labels]
        cellText, rowLabels = [], []
        cellText.append(geom_sum);       rowLabels.append("Geometry (nm)")
        cellText.append(mat_sum);        rowLabels.append("Material")
        if flags['show_fwhm']:
            cellText.append(fwhm_sum)
            rowLabels.append("FWHM (nm)")
        if flags['show_lambda0']:
            cellText.append(lam_sum)
            rowLabels.append(r"$\lambda_0$")
        if flags['show_delta_lam_over_midLam']:
            cellText.append(delta_lam_over_midLam)
            rowLabels.append(
                r"$\Delta_{\lambda}$ / $\lambda_{min}$ or $\lambda_{sym}$"
            )
        if flags['show_S_lambda']:
            cellText.append(S_lambda_sum)
            rowLabels.append(r"$S_{\lambda}$ = Δλ / Δn (nm/RIU)")
        if flags['show_S_dn']:
            cellText.append(S_R_sum)
            rowLabels.append("ΔR/Δn (1/RIU)")
        if flags['show_deltaR_half']:
            cellText.append(dR_half_sum)
            rowLabels.append(r"$\Delta R_{half}$")
        if flags['show_Q']:
            cellText.append(Q_fac)
            rowLabels.append("Q-factor")

        filtered = []
        for lbl, row in zip(rowLabels, cellText):
            if lbl in ("Geometry (nm)", "Material"):
                ref = row[0]
                filtered.append(
                    [row[0]] +
                    [("" if row[j]==ref else row[j])
                     for j in range(1, len(row))]
                )
            else:
                filtered.append(row)
        cellText = filtered

        n_cfg = len(col_labels)
        fs = 8 if n_cfg <= 5 else max(8 - (n_cfg-5), 3)
        table = ax_table.table(
            cellText=cellText,
            colLabels=col_labels,
            rowLabels=rowLabels,
            loc="center", cellLoc="left"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(fs)
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

        # ajuster hauteur des lignes
        row_heights = {}
        for (r, c), cell in table.get_celld().items():
            if r >= 0:
                nb = cell.get_text().get_text().count("\n") + 1
                row_heights[r] = max(row_heights.get(r, 0), nb)
        for (r, c), cell in table.get_celld().items():
            if r in row_heights:
                cell.set_height(0.04 * row_heights[r])

        # affichage final
        with self.sim_output:
            self.sim_output.clear_output(wait=True)
            display(fig)
            display(_download_link(
                fig,
                f"simulation_{datetime.now():%Y%m%d_%H%M%S}.png"
            ))
        plt.close(fig)
        
        # 1) Ouvrir/créer le fichier HDF5
        with h5py.File(h5_path, "a") as f:
            grp = f.require_group(
                f"Simulations/{cfg['config_name']}_{datetime.now():%Y%m%d_%H%M%S}"
            )

            # 2) Méta
            grp.attrs["date"]        = datetime.now().isoformat()
            grp.attrs["config_name"] = cfg["config_name"]

            # 3) Grille & spectres
            grp.create_dataset("wavelength",   data=lam_range, compression="gzip")
            grp.create_dataset("Rup_base",     data=Rup,       compression="gzip")
            # Rup_dn n'existe que si vous l'avez calculé
            if 'Rup_dn' in locals():
                grp.create_dataset("Rup_delta_n", data=Rup_dn,   compression="gzip")

            # 4) Métriques supplémentaires
            meta_grp = grp.require_group("extra_metrics")
            for k, v in details.get("extra_metrics", {}).items():
                if np.isscalar(v):
                    meta_grp.attrs[k] = v
                else:
                    meta_grp.create_dataset(k, data=v, compression="gzip")

    # ----------------------------------------------------------------- #
    #                            fonction de coût                       #
    # ----------------------------------------------------------------- #

    def cost(self, x, keys, mode="dip"):
        """
        Injection de x sur les clés `keys`, simulation, puis on choisit
        le dip optimal via find_best_dip (max ΔR/Δn ou Δλ/Δn selon mode).
        Retourne 1 – best_S_R.
        """
        # 1) Récupère la config cochée
        sel = [c for c in self.all_configs
            if self.config_checkboxes[c["config_name"]].value]
        if not sel:
            raise RuntimeError("Cochez une configuration.")
        cfg = deepcopy(sel[0])

        # 2) Injecte uniquement les paramètres optimisés
        for xi, k in zip(x, keys):
            cfg["geometry"]["geometry"][k] = float(xi)

        # 3) Prépare la grille & les autres paramètres
        lam = np.linspace(self.sim_lambda_min.value,
                        self.sim_lambda_max.value,
                        self.sim_n_points.value)
        wave    = {"angle": 0, "polarization": 1}
        n_modes = self._get_n_modes_for(cfg["config_name"])  # fixed/custom/auto
        sel_layers = list(self.layer_selector.value)
        delta_n    = max(self.delta_n_widget.value, 1e-9)

        # 4) Simule le spectre de base
        Rup0, _, _ = run_simulation_one_combo(
            lam, wave, n_modes, cfg, json_combined_path
        )
        Rup0 = np.asarray(Rup0, float)

        # 5) Trouve le dip le plus sensible
        best_out, _, _ = find_best_dip(
            cfg=cfg,
            wavelength=lam,
            reflectance=Rup0,
            wave=wave,
            n_modes=n_modes,
            sel_layers=sel_layers,
            delta_n=delta_n,
            json_combined_path=json_combined_path,
            smooth_win=0, polyorder=0,
            dip_prom=1e-2, dip_dist=1,
            peak_dist=1,
            verbose=False,
            cfg_name=cfg["config_name"],
            mode=('half' if mode=="half" else 'dip')
        )

        # 6) Si aucun dip trouvé, on pénalise au maximum
        if best_out is None:
            return 1.0

        # 7) best_out est un tuple :
        #    (lam_left, lam_right, fwhm, depth,
        #     lam_dip, R_dip, ylev,
        #     lam_max_l, R_max_l, lam_max_r, R_max_r,
        #     lam_sym, R_sym,
        #     best_dR (=ΔR/Δn), best_Slam (=Δλ/Δn), best_dR_half,
        #     dip_idx_list, dR_over_dn_list, dLam_over_dn_list)
        #    On récupère best_dR (position 13)
        best_dR = best_out[12]

        # 8) Retourne le coût 1 – ΔR/Δn (plus ΔR/Δn est grand, plus le coût est petit)
        return 1.0 - float(best_dR)


# instanciation globale
sim_tab = SimulationTab()
simulation_tab = sim_tab.tab
