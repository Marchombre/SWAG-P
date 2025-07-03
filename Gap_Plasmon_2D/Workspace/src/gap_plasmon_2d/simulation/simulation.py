import matplotlib as mpl
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
import matplotlib as mpl
mpl.use("module://matplotlib_widget") 
import matplotlib.pyplot as plt

import ipywidgets as widgets
import h5py


import multiprocessing as mp
import queue as q
import threading
from functools import partial   # si besoin d’init pool


from ipywidgets import Layout, HBox, VBox, ToggleButton, HTML
from IPython.display import HTML as DHTML, display, Javascript
from scipy.interpolate import interp1d

# pour surveiller les changements de fichiers
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
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








# -----------------------------------------------------------------------------#
#  Worker function — executed in each process                                  #
# -----------------------------------------------------------------------------#
def _simulate_worker(args):
    """
    Effectue la simulation d’un seul spectre dans un *process* fils puis
    renvoie le résultat sous forme sérialisable.

    Parameters
    ----------
    args : tuple
        (
            cfg,                # dict : configuration complète (geometry + materials)
            lam_range,          # np.ndarray : grille λ
            wave,               # dict       : {'angle': 0, 'polarization': 1}
            n_modes,            # int        : nombre de modes RCWA
            sel_layers,         # list[str]  : couches où appliquer Δn (peut être vide)
            delta_n,            # float      : valeur Δn (≥ 0)
            mode_calc,          # str        : 'dip' | 'half' | 'fixed_lambda' | 'range_lambda'
            lambda0,            # float      : λ₀ si mode_calc == 'fixed_lambda'
            flags,              # dict       : booléens show_* des check-boxes (pas utilisé ici)
            json_combined_path  # str        : chemin vers « combined_materials.json »
        )

    Returns
    -------
    dict
        {
            'cfg_name' : str,          # nom de la configuration
            'cfg'      : dict,         # la configuration complète (géométrie + matériaux)
            'Rup'      : np.ndarray,   # spectre R↑ simulé
            'details'  : dict          # méta-données brutes de run_simulation_one_combo
        }
    """
    # ─────────────── dépaquetage des arguments ───────────────────────
    (cfg, lam_range, wave, n_modes, sel_layers, delta_n,
     mode_calc, lambda0, flags, json_combined_path) = args

    # Nom court pour plus de clarté
    name = cfg["config_name"]

    # ───────────────────── simulation du spectre ─────────────────────
    # run_simulation_one_combo renvoie (Rup, Rdown, details)
    Rup, _, details = run_simulation_one_combo(
        lam_range,          # grille λ
        wave,               # dictionnaire angle/polarisation
        n_modes,            # nombre de modes RCWA
        cfg,                # configuration géométrie + matériaux
        json_combined_path  # propriétés optiques combinées
    )
    Rup = np.asarray(Rup, dtype=float)   # cast explicite en float64 numpy

    # -----------------------------------------------------------------
    # NOTES
    # -----
    # • Toute l’analyse lourde (détection de dip, calcul de ΔR/Δn, FWHM,
    #   simulation du spectre Δn, embelissements graphiques, etc.) n’est
    #   pas exécutée dans ce worker.  Elle sera reproduite *à l’identique*
    #   dans `_build_outputs`, dans le processus principal, exactement
    #   comme dans ta version séquentielle d’origine.  De cette façon :
    #     – on garde un code worker très compact et 100 % picklable ;
    #     – on évite de dupliquer des dépendances lourdes inutilement
    #       dans chaque process ;
    #     – on maintient la logique métier et l’UI sans la moindre perte
    #       de fonctionnalité.
    #
    # • Si tu tiens à déplacer également l’analyse (find_best_dip,
    #   simulate_delta_spectrum, etc.) côté worker pour soulager le
    #   thread principal, il suffira d’injecter cette logique ici puis
    #   de renvoyer les métriques supplémentaires dans le dictionnaire
    #   retourné.  Le design reste compatible.
    # -----------------------------------------------------------------

    # ─────────────────────── résultats picklables ─────────────────────
    return dict(
        cfg_name = name,
        cfg      = cfg,
        Rup      = Rup,
        details  = details,
    )











# --------------------------------------------------------------------- #
#                             main class                                #
# --------------------------------------------------------------------- #
class SimulationTab:
    """
    Encapsule tout l’onglet Simulation : widgets, callbacks, logique de
    calcul et, désormais, exécution parallèle annulable.
    """

    # ----------------------------------------------------------------- #
    #                            __init__                               #
    # ----------------------------------------------------------------- #
    def __init__(self):
        # ---- runtime flags & handles (ajoutés pour le parallélisme) ---
        self._is_running   = False
        self._cancelled    = False
        self._pool         = None          # mp.Pool      (processes)
        self._worker_thread = None         # threading.Thread
        self._result_queue  = None         # queue.Queue

        # ---------------- charger les configurations JSON --------------
        cfg_file = os.path.join(configurations_dir, "geom_mat_combinations.json")
        if os.path.exists(cfg_file):
            with open(cfg_file, encoding="utf-8") as f:
                self.all_configs = json.load(f)["ALL_COMBINED_CONFIGS"]
        else:
            self.all_configs = []

        self.json_combined_path = json_combined_path    

        # ------------ widgets paramétriques généraux (inchangés) -------
        def _positive(change):
            owner = change['owner']; val = change['new']
            if val < 0:
                owner.value = 0
            if owner is self.sim_lambda_min and val > self.sim_lambda_max.value:
                owner.value = self.sim_lambda_max.value
            if owner is self.sim_lambda_max and val < self.sim_lambda_min.value:
                owner.value = self.sim_lambda_min.value

        self.sim_lambda_min = widgets.FloatText(
            value=300.0, description="λ min (nm):",
            layout=Layout(width='150px'), style={'description_width': 'initial'})
        self.sim_lambda_max = widgets.FloatText(
            value=900.0, description="λ max (nm):",
            layout=Layout(width='150px'), style={'description_width': 'initial'})
        self.sim_n_points = widgets.IntText(
            value=400, description="Points:",
            layout=Layout(width='200px'), style={'description_width': 'initial'})
        self.sim_n_mod = widgets.IntText(
            value=5, layout=Layout(width='200px'),
            style={'description_width': 'initial'})

        for w in (self.sim_lambda_min,
                  self.sim_lambda_max,
                  self.sim_n_points,
                  self.sim_n_mod):
            w.observe(_positive, names='value')

        # ---------------- widgets pour la métrique λ0 / range ----------
        self.band_min_in  = widgets.FloatText(
            value=650.0, description="λmin:",
            layout=Layout(width='120px'),
            style={'description_width': 'initial'})
        self.band_max_in  = widgets.FloatText(
            value=750.0, description="λmax:",
            layout=Layout(width='120px'),
            style={'description_width': 'initial'})
        self.band_box_in  = HBox([self.band_min_in, self.band_max_in],
                                 layout=Layout(gap='5px'))

        self.mode_calc_radio = widgets.RadioButtons(
            options=[('Dip (ΔR/Δn)',    'dip'),
                     ('FWHM (half)',    'half'),
                     ('Custum fixed λ₀', 'fixed_lambda')],
            value='dip',
            description='Compute R(λ) at fixe λ:',
            style={'description_width': 'initial'},
            layout=Layout(width='220px'))

        self.lambda0_in = widgets.FloatText(
            value=700.0, description="λ₀ (nm):",
            layout=Layout(width='130px'),
            style={'description_width': 'initial'})

        def _toggle_lambda0(change):
            self.lambda0_in.layout.display = '' if change['new'] == 'fixed_lambda' else 'none'

        _toggle_lambda0({'new': self.mode_calc_radio.value})
        self.mode_calc_radio.observe(_toggle_lambda0, names='value')

        # ---------------- bouton RUN + NOUVEAU bouton CANCEL -----------
        self.sim_run_button = widgets.Button(
            description="Run simulation", button_style="success",
            tooltip="Start simulation")

        self.sim_cancel_button = widgets.Button(
            description="Cancel", button_style="warning",
            tooltip="Cancel running simulation", disabled=True)

        self.sim_cancel_button.on_click(self._on_cancel)

        # ------------- barre de progression + status HTML --------------
        self._status_html  = widgets.HTML("")
        self._progress_bar = widgets.FloatProgress(
            value=0, min=0, max=1, description="0 %",
            bar_style="info",
            layout=widgets.Layout(width='100%', display='none'))

        self.runtime_box = VBox([self._status_html, self._progress_bar],
                                layout=Layout(gap='4px'))

        # ------------------------- RCWA modes --------------------------
        self.mode_selection   = widgets.RadioButtons(
            options=[('Fixe', 'fixed'),
                     ('Personnalisé', 'custom'),
                     ('Automatique', 'auto')],
            value='fixed', style={'description_width': 'initial'})
        self.custom_modes_box = VBox()

        self.custom_n_mod_inputs = {}

        # ----------------- gestion des fichiers de résumé -------------
        self.sim_files_dropdown = widgets.Dropdown(
            options=list_sim_summary_files(summary_dir),
            description="Simulation files:",
            layout=Layout(width='500px'),
            style={'description_width': 'initial'})

        self.sim_download_button = widgets.Button(
            description="Download", button_style="danger")

        self.sim_download_button.on_click(self._download_file)

        # ---------------------- nom de simulation ----------------------
        self.sim_name_widget = widgets.Text(
            value="", placeholder="Nom de simulation (auto si vide)",
            description="Sim Name:", layout=Layout(width='500px'),
            style={'description_width': 'initial'})

        # -------------- sélecteur Config / Δn (inchangé) --------------
        self.config_checkboxes = {}
        self.dn_checkboxes     = {}
        config_rows = []


        for cfg in self.all_configs:
            name = cfg["config_name"]
            chk  = widgets.Checkbox(value=False, description=name, indent=False)
            dn   = widgets.Checkbox(value=False, description='Δn', indent=False,
                                    layout=Layout(width='46px'))
            self.config_checkboxes[name] = chk
            self.dn_checkboxes[name]     = dn
            config_rows.append(HBox([chk, dn], layout=Layout(grid_gap='5px')))

        for chk in self.dn_checkboxes.values():
            chk.observe(self._toggle_delta_widgets, names='value')




        visible = min(len(config_rows), 10)
        self.select_all_cfg_btn = widgets.Button(
            description="Tout sélectionner Configs", button_style="info",
            layout=Layout(width='auto', margin='0 5px 5px 0'))
        self.select_all_dn_btn  = widgets.Button(
            description="Tout sélectionner Δn", button_style="info",
            layout=Layout(width='auto', margin='0 0 5px 0'))
        self.select_all_cfg_btn.on_click(self._toggle_all_cfg)
        self.select_all_dn_btn.on_click(self._toggle_all_dn)

        self.config_list = VBox(
            children=[HTML("<b>Configurations et Δn</b>"),
                      HBox([self.select_all_cfg_btn,
                            self.select_all_dn_btn],
                           layout=Layout(grid_gap='10px')),
                      *config_rows],
            layout=Layout(width='500px',
                          height=f'{30+visible*30}px',
                          overflow_y='auto',
                          border='1px solid lightgray',
                          padding='5px',
                          display='none'))

        self.toggle_btn = ToggleButton(
            value=False, description="Select your configuration and Δn",
            icon='caret-down', layout=Layout(width='520px'),
            button_style='warning')
        self.toggle_btn.observe(self._toggle_config_list, names='value')

        self.config_refresh_btn = widgets.Button(
            description="Refresh Configs", button_style="info",
            layout=Layout(width='auto', margin='0 5px 5px 0'))
        self.config_refresh_btn.on_click(self._refresh_configs)

        self.config_selector = VBox(
            [HBox([self.toggle_btn, self.config_refresh_btn]),
             self.config_list],
            layout=Layout(padding='5px'))

        # ---------------- couche(s) Δn + delta_n widget ----------------
        layer_keys = [m['key']
                      for m in self.all_configs[0]['material']['MATERIALS_CONFIG']]
        self.layer_selector = widgets.SelectMultiple(
            options=layer_keys, description="Couche(s) Δn:",
            layout=Layout(width='300px', height='100px'),
            style={'description_width': 'initial'})

        self.delta_n_widget = widgets.FloatText(
            value=1e-2, description="Δn:",
            layout=Layout(width='150px'),
            style={'description_width': 'initial'})
        self.delta_n_widget.observe(_positive, names='value')

        # ---------------------- debug & verbose ------------------------
        self.verbose_toggle = widgets.Checkbox(
            value=False, description="Verbose", indent=False,
            layout=Layout(width='100%'),
            style={'description_width': 'initial'})
        self.debug_out = widgets.Textarea(
            value='', placeholder='Logs verbose…',
            layout=Layout(width='100%', height='200px',
                          overflow_y='scroll', border='1px solid darkred'))
        self.verbose_toggle.observe(self._toggle_debug, names='value')

        # ---------------- métriques / overlays (inchangé) --------------
        def _cb(v, d): return widgets.Checkbox(value=v, description=d)

        self.show_fwhm_chk              = _cb(False, "FWHM")
        self.show_lambda0_chk           = _cb(True,  r"λ0")
        self.show_delta_lam_over_midLam = _cb(False, r"Δλ / λmin or λsym")
        self.show_S_lambda_chk          = _cb(True,  "Sλ = Δλ / Δn (nm/RIU)")
        self.show_S_dn_chk              = _cb(True,  r"ΔR/Δn (1/RIU)")
        self.show_deltaR_half_chk       = _cb(True,  r"ΔR_half")
        self.show_Q_chk                 = _cb(False, "Q-factor")
        self.show_Rup_dn_chk            = _cb(True,  "Rup_dn dashed")
        self.show_hlines_chk            = _cb(False, "half-level line")
        self.show_dips_chk              = _cb(False, "dips (×)")
        self.show_maxima_chk            = _cb(False, "maxima (×)")
        self.show_symmetry_pts_chk      = _cb(False, "symmetric pts (×)")
        self.show_selected_dip_chk      = _cb(True,  "selected dip (○)")
        self.show_sensitivity_marker    = _cb(True,  "sensitivity marker (□)")

        self.metrics_selector = VBox(
            children=[
                HTML("<b>Métriques à afficher :</b>"),
                HBox([self.show_fwhm_chk, self.show_lambda0_chk,
                      self.show_delta_lam_over_midLam, self.show_S_lambda_chk,
                      self.show_S_dn_chk, self.show_deltaR_half_chk,
                      self.show_Q_chk],
                     layout=Layout(display='flex', flex_flow='row nowrap',
                                   justify_content='space-around', gap='0px')),
                HTML("<b>Overlays graphiques :</b>"),
                HBox([self.show_Rup_dn_chk, self.show_hlines_chk,
                      self.show_dips_chk, self.show_maxima_chk,
                      self.show_symmetry_pts_chk, self.show_selected_dip_chk,
                      self.show_sensitivity_marker],
                     layout=Layout(display='flex', flex_flow='row nowrap',
                                   justify_content='space-around', gap='0px'))
            ],
            layout=Layout(width='100%', border='1px solid lightgray',
                          padding='5px', margin='10px 0'))

        # ---------------- widget convergence (inchangé) ----------------
        self.conv_widget = create_multi_convergence_widget(
            json_combined_path, self.all_configs)

        # ---------------- zone d’affichage figure/table ----------------
        self.sim_output = widgets.Output(
            layout=Layout(border='2px solid #ccc', padding='10px',
                          min_height='400px', margin='40px 0 0 0'))

        # ------------------------- conteneur gauche --------------------
        self.sim_controls = VBox(
            children=[
                HTML("<h3>Simulation – Paramètres</h3>"),
                self.sim_name_widget,
                HBox([self.sim_files_dropdown]),
                HBox([self.sim_download_button]),
                HBox([self.sim_lambda_min, self.sim_lambda_max]),
                HBox([self.sim_n_points, self.sim_n_mod]),
                self.config_selector,
                widgets.HTML(value="<b>RCWA Fourier modes</b>"),
                HBox([self.mode_selection, self.layer_selector]),
                self.custom_modes_box,
                HBox([self.delta_n_widget, self.mode_calc_radio,
                      self.lambda0_in, self.sim_run_button,
                      self.sim_cancel_button]),
                self.runtime_box,  # ← barre de progression + status
                self.verbose_toggle
            ],
            layout=Layout(padding='10px', border='1px solid lightgray'))

        # --------------------- assemblage final UI ---------------------
        self.tab = VBox(
            [HBox([self.sim_controls, self.conv_widget],
                  layout=Layout(align_items='flex-start')),
             self.metrics_selector,
             self.debug_out,
             self.sim_output])

        # ---------------------- callbacks boutons ----------------------
        self.sim_run_button.on_click(self._on_run)   # NEW _on_run parallèle


        # appel initial
        self._toggle_delta_widgets()

    # ----------------------------------------------------------------- #
    #                    méthodes utilitaires                           #
    # ----------------------------------------------------------------- #
    
    # ----------------------------------------------------------------- #
    #  Callback : lancement des simulations parallèles                  #
    # ----------------------------------------------------------------- #
    def _on_run(self, _):
        """
        Start button : lance la/les simulation(s) dans un thread de travail,
        qui délègue les spectres à un Pool de *processes*.  
        – Ne bloque jamais l’event-loop Jupyter ;  
        – Met à jour la barre de progression via _check_process ;  
        – Autorise l’annulation à tout moment via _on_cancel.
        """

        # ─── garde anti double-click ────────────────────────────────────
        if self._is_running:
            return

        # ----------------------------------------------------------------
        # 1) Réinitialisation des widgets runtime
        # ----------------------------------------------------------------
        self._is_running            = True
        self._cancelled             = False
        self.sim_cancel_button.disabled = False

        self._status_html.value     = (
            "🚀 Simulation in progress… (you can Cancel)<br>"
            "The progress-bar will appear when the first result arrives."
        )
        self._progress_bar.value    = 0.0
        self._progress_bar.description = "0 %"
        self._progress_bar.bar_style   = "info"
        self._progress_bar.layout.display = "none"   # masquée tant que 1ᵉʳ PROG

        # ----------------------------------------------------------------
        # 2) Collecte des paramètres UI (identique à l’ancienne version)
        # ----------------------------------------------------------------
        lam_range = np.linspace(
            self.sim_lambda_min.value,
            self.sim_lambda_max.value,
            self.sim_n_points.value
        )
        wave        = {"angle": 0, "polarization": 1}
        sel_layers  = list(self.layer_selector.value)
        delta_n     = max(self.delta_n_widget.value, 1e-9)

        mode_calc   = self.mode_calc_radio.value       # 'dip' | 'half' | 'fixed_lambda'
        lambda0     = self.lambda0_in.value
        flags = {
            "show_fwhm"                : self.show_fwhm_chk.value,
            "show_lambda0"             : self.show_lambda0_chk.value,
            "show_delta_lam_over_midLam": self.show_delta_lam_over_midLam.value,
            "show_S_lambda"            : self.show_S_lambda_chk.value,
            "show_S_dn"                : self.show_S_dn_chk.value,
            "show_deltaR_half"         : self.show_deltaR_half_chk.value,
            "show_Q"                   : self.show_Q_chk.value,
            "show_Rup_dn"              : self.show_Rup_dn_chk.value,
            "show_hlines"              : self.show_hlines_chk.value,
            "show_dips"                : self.show_dips_chk.value,
            "show_maxima"              : self.show_maxima_chk.value,
            "show_symmetry_pts"        : self.show_symmetry_pts_chk.value,
            "show_selected_dip"        : self.show_selected_dip_chk.value,
            "show_sensitivity_marker"  : self.show_sensitivity_marker.value,
        }
        verbose     = self.verbose_toggle.value

        # Sélection des configurations
        selected_cfgs = [
            cfg for cfg in self.all_configs
            if self.config_checkboxes[cfg['config_name']].value
        ]
        if not selected_cfgs:
            self._status_html.value = "⚠️ Please select at least one configuration."
            self._is_running = False
            self.sim_cancel_button.disabled = True
            return
        
        # 2.bis  Configurations pour lesquelles on veut Δn
        self._cfgs_with_delta = {
            cfg['config_name']
            for cfg in self.all_configs
            if self.dn_checkboxes[cfg['config_name']].value          # ← case Δn cochée
}


        # ----------------------------------------------------------------
        # 3) Construction de la liste d’arguments pour chaque worker
        # ----------------------------------------------------------------
        args_list = []
        for cfg in selected_cfgs:
            n_modes = self._get_n_modes_for(cfg['config_name'])   # fixe/custom/auto
            args_list.append(
                (cfg, lam_range, wave, n_modes,
                sel_layers, delta_n,
                mode_calc, lambda0, flags,
                self.json_combined_path)
            )

        # ----------------------------------------------------------------
        # 4) Mise en cache des variables requises par _build_outputs
        # ----------------------------------------------------------------
        self._lam_range   = lam_range
        self._sel_layers  = sel_layers
        self._delta_n     = delta_n
        self._flags       = flags
        self._verbose     = verbose

        # ----------------------------------------------------------------
        # 5) Démarrage du thread worker + queue de communication
        # ----------------------------------------------------------------
        self._result_queue = q.Queue()

        self._worker_thread = threading.Thread(
            target=self._simulate_many,          # méthode définie plus loin
            kwargs=dict(args_list=args_list,
                        progress_queue=self._result_queue),
            daemon=True
        )
        self._worker_thread.start()

        # ----------------------------------------------------------------
        # 6) Boucle de polling non bloquante (via IOLoop Tornado)
        # ----------------------------------------------------------------
        loop = get_ipython().kernel.io_loop
        loop.add_timeout(loop.time() + 0.1, self._check_process)




    # ----------------------------------------------------------------- #
    #  Thread worker : lance les spectres dans un Pool de processes     #
    # ----------------------------------------------------------------- #
    def _simulate_many(self, *, args_list, progress_queue):
        """
        Exécuté dans un thread secondaire pour ne pas bloquer l’UI.
        Délègue chaque configuration au Pool de *processes* via
        _simulate_worker.  Publie les messages :
            ("PROG", frac)                ← progression 0–1
            ("DONE", results_list)        ← succès
            ("ERROR", traceback_str)      ← exception

        Parameters
        ----------
        args_list : list[tuple]
            Liste d’arguments positionnels pour _simulate_worker
            (un tuple par configuration sélectionnée).
        progress_queue : queue.Queue
            Canal de communication vers le thread principal / _check_process.
        """
        import traceback

        # 1) Création du Pool (fork sous Unix, spawn sous Windows)
        ctx  = mp.get_context("fork" if os.name != "nt" else "spawn")
        pool = ctx.Pool()
        self._pool = pool        # pour _on_cancel

        try:
            N = len(args_list)
            results = []

            # 2) Boucle d’évaluation parallèle
            for i, out in enumerate(pool.imap_unordered(_simulate_worker,
                                                        args_list,
                                                        chunksize=1)):
                # Annulation demandée ?
                if self._cancelled:
                    pool.terminate()
                    raise RuntimeError("cancelled by user")

                results.append(out)

                if progress_queue is not None:
                    progress_queue.put(("PROG", (i + 1) / N))

            # 3) Fin normale : on ferme le Pool proprement
            pool.close()
            pool.join()
            self._pool = None

            if progress_queue is not None:
                progress_queue.put(("DONE", results))

        except Exception:
            # 4) Exception : on transmet le traceback au thread principal
            trace = traceback.format_exc()
            if progress_queue is not None:
                progress_queue.put(("ERROR", trace))

            # Terminer/joindre le pool si nécessaire
            try:
                pool.terminate()
                pool.join()
            finally:
                self._pool = None



    # ----------------------------------------------------------------- #
    #  Polling : traite les messages de la queue                        #
    # ----------------------------------------------------------------- #
    def _check_process(self):
        """
        Appelée périodiquement (100 ms) par l’event-loop Tornado.
        Dépouille la queue _result_queue pour mettre à jour l’UI ou
        terminer le cycle de simulation.
        """
        try:
            tag, *payload = self._result_queue.get_nowait()
        except q.Empty:
            tag = None

        # ─── première mise à jour PROG → afficher la barre ─────────────
        if tag == "PROG" and self._progress_bar.layout.display == "none":
            self._status_html.value = ""               # cache le texte
            self._progress_bar.layout.display = ""     # montre la barre

        # ----------------------------------------------------------------
        # 1) Progression
        # ----------------------------------------------------------------
        if tag == "PROG":
            frac = payload[0]                          # valeur 0–1
            self._progress_bar.value       = frac
            self._progress_bar.description = f"{int(frac * 100)} %"

        # ----------------------------------------------------------------
        # 2) Fin normale
        # ----------------------------------------------------------------
        elif tag == "DONE":
            results = payload[0]                       # liste de dicts
            self._is_running = False
            self.sim_cancel_button.disabled = True

            # barre verte
            self._progress_bar.value = 1.0
            self._progress_bar.description = "100 %"
            self._progress_bar.bar_style = "success"

            # Construction des figures / tableaux (longue méthode)
            self._build_outputs(results)

            return   # stop polling

        # ----------------------------------------------------------------
        # 3) Erreur dans le thread / worker
        # ----------------------------------------------------------------
        elif tag == "ERROR":
            trace = payload[0]
            self._status_html.value = (
                "❌ Simulation aborted:<br>"
                f"<pre style='max-height:300px; overflow:auto'>{trace}</pre>"
            )
            self._progress_bar.bar_style = "danger"
            self.sim_cancel_button.disabled = True
            self._is_running = False
            return   # stop polling

        # ----------------------------------------------------------------
        # 4) Rescheduler : continuer tant que l’on n’a pas traité DONE/ERROR
        # ----------------------------------------------------------------
        if tag not in ("DONE", "ERROR"):
            loop = get_ipython().kernel.io_loop
            loop.add_timeout(loop.time() + 0.1, self._check_process)


    # ----------------------------------------------------------------- #
    #  Bouton Cancel : arrêt immédiat du calcul                         #
    # ----------------------------------------------------------------- #
    def _on_cancel(self, _=None):
        """
        Appelé lorsque l’utilisateur clique sur le bouton “Cancel”.
        - Signale l’annulation au thread worker (_cancelled=True)
        - Termine le Pool de processus si encore actif
        - Met à jour l’interface (barre rouge, message) et réactive l’UI
        """
        if not self._is_running:
            return

        # 1) Drapeau d’annulation pour le thread worker
        self._cancelled = True
        self._is_running = False
        self.sim_cancel_button.disabled = True

        # 2) Si un Pool est encore en vie, on le tue proprement
        if self._pool is not None:
            try:
                self._pool.terminate()
                self._pool.join()
            except Exception:
                pass
            finally:
                self._pool = None

        # 3) Feedback visuel
        self._status_html.value = "❌ Simulation cancelled by user."
        self._progress_bar.bar_style = "danger"

        # 4) Optionnel : vider la queue pour éviter des messages résiduels
        try:
            while True:
                self._result_queue.get_nowait()
        except q.Empty:
            pass


    # ----------------------------------------------------------------- #
    #  Post-traitement : figure, tableau, logs, sauvegardes             #
    # ----------------------------------------------------------------- #
    def _build_outputs(self, results):
        """
        Reconstruit les figures et tableaux exactement comme dans l’ancienne
        version séquentielle, mais en utilisant :

            results      ← liste de dicts {'cfg','cfg_name','Rup','details'}
            self._lam_range, self._flags, self._sel_layers, self._delta_n,
            self._verbose  ← caches posés au lancement

        Affiche le tout dans self.sim_output et écrit le HDF5.
        """
        # ─── variables mises en cache au lancement ───────────────────────
        lam_range  = self._lam_range
        flags      = self._flags
        sel_layers = self._sel_layers
        delta_n    = self._delta_n
        verbose    = self._verbose

        # ─── raccourcis utiles ───────────────────────────────────────────
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        wave   = {"angle": 0, "polarization": 1}
        use_half = (self.mode_calc_radio.value == 'half')

        # ─── accumulateurs pour le tableau / métriques -------------------
        cfg_labels, geom_sum, mat_sum = [], [], []
        fwhm_sum, lam_sum, delta_lam_over_midLam = [], [], []
        S_lambda_sum, S_R_sum, dR_half_sum = [], [], []
        S_R_vals, S_lam_min, S_lam_sym, Q_fac = [], [], [], []
        debug_lines = []


        # ------------------------------------------------------------------
        # Préparation de la zone d’affichage et désactivation interactive
        # ------------------------------------------------------------------
        with self.sim_output:
            self.sim_output.clear_output(wait=True)

            was_interactive = mpl.is_interactive()          
            mpl.interactive(False)                          


            # ─── création de la figure AVANT la boucle ───────────────────────────
            fig = plt.figure(figsize=(13, 9))
            ax_plot  = fig.add_axes([0.10, 0.50, 0.80, 0.35])
            ax_plot.set_xlabel("λ (nm)")
            ax_plot.set_ylabel("Reflectance R")
            ax_table = fig.add_axes([0.10, 0.05, 0.80, 0.35])
            ax_table.axis('off')


            # ─── boucle sur les configurations simulées ----------------------
            for idx, res in enumerate(results):
                cfg      = res['cfg']
                name     = res['cfg_name']
                Rup_base = res['Rup']
                details  = res['details']
                color    = colors[idx % len(colors)]
                n_modes  = self._get_n_modes_for(name)

                ax_plot.plot(lam_range, Rup_base, color=color,
                            linewidth=1.5, zorder=1, label=name)

                # ------------------------------------------------------------------
                # Recherche du dip optimal et, si demandé, spectre Δn
                # ------------------------------------------------------------------
                Best_values_out, who, best_dip_index = find_best_dip(
                    cfg=cfg,
                    wavelength=lam_range,
                    reflectance=Rup_base,
                    wave=wave,
                    n_modes=n_modes,
                    sel_layers=sel_layers,
                    delta_n=delta_n,
                    json_combined_path=self.json_combined_path,
                    smooth_win=0, polyorder=0,
                    dip_prom=1e-2, dip_dist=1,
                    peak_dist=1, verbose=False,
                    cfg_name=name,
                    mode=('half' if use_half else 'dip')
                )

                # extraction brute du spectre (pour overlays génériques)
                (dip_list, lam_dip_list, R_dip_list,
                y_level_list, lam_left_list, lam_right_list,
                fwhm_list, lam_max_l_list, R_max_l_list,
                lam_max_r_list, R_max_r_list,
                lam_sym_list, R_sym_list,
                depth_list), _ = _find_dip_core(
                    wavelength=lam_range,
                    reflectance=Rup_base,
                    smooth_win=0, polyorder=0,
                    dip_prom=1e-2, dip_dist=1,
                    peak_dist=1, verbose=False,
                    cfg_name=name
                )

                lam_max_l = np.array(lam_max_l_list)
                R_max_l   = np.array(R_max_l_list)
                lam_max_r = np.array(lam_max_r_list)
                R_max_r   = np.array(R_max_r_list)
                lam_sym   = np.array(lam_sym_list)
                R_sym     = np.array(R_sym_list)
                width_arr = np.array(fwhm_list)
                depth_arr = np.array(depth_list)

                # si aucun dip valide
                if Best_values_out is None:
                    debug_lines.append(
                        f"Aucun dip sélectionné pour “{who}” – ignorée.")
                    continue

                # dépaquetage
                (lam_left, lam_right, fwhm, depth,
                lam_dip, R_dip, ylev,
                lam_m_l, Rm_l, lam_m_r, Rm_r,
                lam_sympt, R_sympt,
                best_S_R, S_lambda, dR_half,
                dips_idx_list, dR_over_dn_list,
                dLam_over_dn_list) = Best_values_out

                lam_min    = lam_m_l if Rm_l < Rm_r else lam_m_r
                lam_mid    = lam_left if Rm_l < Rm_r else lam_right
                S_lam_min_abs = abs((lam_dip - lam_min) / lam_mid)
                S_lam_sym_abs = abs((lam_dip - lam_sympt) / lam_mid)
                S_lam_min.append(S_lam_min_abs)
                S_lam_sym.append(S_lam_sym_abs)

                compute_delta = (name in self._cfgs_with_delta
                                and sel_layers                        # au moins une couche choisie
                                and delta_n > 0)


                # ─── neutraliser si compute_delta == False ──────────────────────────
                best_S_R = best_S_R if compute_delta else None
                S_lambda = S_lambda if compute_delta else None
                dR_half  = dR_half  if compute_delta else None
                S_R_vals.append(best_S_R if best_S_R is not None else np.nan)


                # λ fixe : on écrase lam_dip/R_dip
                if self.mode_calc_radio.value == 'fixed_lambda':
                    lam_dip = self.lambda0_in.value
                    R_dip   = float(np.interp(lam_dip, lam_range, Rup_base))

                # ------------------------------------------------------------------
                # Simulation Δn (si demandée) et overlays
                # ------------------------------------------------------------------
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
                        base_spectrum=Rup_base,
                        json_combined_path=self.json_combined_path,
                        dip_index=best_dip_index,
                        mode=('half' if use_half else 'dip')
                    )
                    if Rup_dn is not None and flags['show_Rup_dn']:
                        ax_plot.plot(lam_range, Rup_dn, '--', color=color,
                                    linewidth=2, alpha=0.7, zorder=100,
                                    label=f"{name} (R + Δn)")

                # ------------------------------------------------------------------
                # Overlays graphiques divers
                # ------------------------------------------------------------------
                if flags['show_hlines']:
                    ax_plot.hlines(ylev, lam_left, lam_right,
                                color=color)
                if flags['show_dips']:
                    ax_plot.scatter(lam_range[dips_idx_list],
                                    Rup_base[dips_idx_list],
                                    marker='x', color=color)
                if flags['show_maxima']:
                    ax_plot.scatter(lam_max_l_list, R_max_l_list, marker='x',
                                    color=color)
                    ax_plot.scatter(lam_max_r_list, R_max_r_list, marker='x',
                                    color=color)
                if flags['show_symmetry_pts']:
                    ax_plot.scatter(lam_sym_list, R_sym_list, marker='x',
                                    color=color)
                if flags['show_selected_dip']:
                    ax_plot.scatter([lam_dip], [R_dip], marker='o',
                                    facecolor='none', edgecolor=color, s=70)

                # ------------------------------------------------------------------
                # Verbose : log détaillé
                # ------------------------------------------------------------------
                if verbose:
                    dips_nm = ", ".join(f"{lam_range[d]:.1f}" for d in dips_idx_list)
                    debug_lines.append(
                        f"{name} dips[{dips_nm}] λ0={lam_dip:.2f} nm "
                        f"FWHM={fwhm:.1f} ΔR/Δn={best_S_R if best_S_R is not None else '–'}")

                # ------------------------------------------------------------------
                # Accumulateurs pour le tableau
                # ------------------------------------------------------------------
                cfg_labels.append(name)

                # Geometry
                geom = cfg["geometry"]["geometry"]
                geom_sum.append("\n".join(
                    f"{d}: {geom[k]}" for k, d in ordered_params if k in geom))

                # Materials
                mat_lines = []
                for e in cfg["material"]["MATERIALS_CONFIG"]:
                    key = e['key']
                    disp = next((d for k, d in ordered_params if k == key), key)
                    mat  = e['material']; typ = mat['type'].lower()
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
                    f"{S_lam_min_abs:.3f} & {S_lam_sym_abs:.3f}")
                Q_fac.append(f"{lam_dip / fwhm:.1f}")

                # champs optionnels
                def _append(lst, cond, val=""):
                    lst.append(val if cond else "")

                _append(S_lambda_sum,
                        flags['show_S_lambda'] and S_lambda is not None,
                        f"{S_lambda:.3f}" if S_lambda is not None else "")

                _append(S_R_sum,
                        flags['show_S_dn'] and best_S_R is not None,
                        f"{best_S_R:.3f}" if best_S_R is not None else "")

                _append(dR_half_sum,
                        flags['show_deltaR_half'] and dR_half is not None,
                        f"{dR_half:.3f}" if dR_half is not None else "")
                
                # enrichir details pour la sauvegarde
                details["extra_metrics"] = {
                    "Sλ (nm/RIU)"   : f"{S_lambda:.3f}" if S_lambda is not None else "",
                    "ΔR/Δn (1/RIU)" : f"{best_S_R:.3f}" if best_S_R is not None else "",
                    "ΔR_half"       : f"{dR_half:.3f}"  if dR_half is not None else "",
                    "S_lam_min"     : f"{S_lam_min_abs:.3f}",
                    "S_lam_sym"     : f"{S_lam_sym_abs:.3f}",
                    "Δn"            : f"{delta_n:.3e}"
                }

                # sauvegarde individuelle (au fil de l’eau)
                save_simulation_summary(
                    {name: details}, lam_range, wave, n_modes, summary_dir,
                    custom_name=name,
                    fwhm_summaries=[fwhm_sum[-1]],
                    lam_summaries=[lam_sum[-1]],
                    delta_lam_over_midLam_summaries=[delta_lam_over_midLam[-1]],
                    Q_factor=[Q_fac[-1]],
                    best_S_R=[S_R_sum[-1]]
                )

            # ----------------------------------------------------------------------
            # Meilleur ΔR/Δn global
            # ----------------------------------------------------------------------
            arr = np.array(S_R_vals, dtype=float)
            if arr.size and not np.all(np.isnan(arr)):
                best_idx = int(np.nanargmax(arr))
                debug_lines.append(
                    f"→ BEST_CONFIG (max ΔR/Δn): "
                    f"{cfg_labels[best_idx]} (S_R = {arr[best_idx]:.3f})"
                )

            # ----------------------------------------------------------------------
            # Affichage verbose
            # ----------------------------------------------------------------------
            if verbose:
                self.debug_out.value = "\n".join(debug_lines)

            # ----------------------------------------------------------------------
            # Construction du tableau final 
            # ----------------------------------------------------------------------
            if not cfg_labels:                        # aucun dip retenu
                ax_plot.set_title("Spectres simulés – aucun dip valide")
                handles, labels = ax_plot.get_legend_handles_labels()
                if labels:
                    ax_plot.legend(handles, labels, loc='best',
                                fontsize=9, frameon=False) 


                return



            # filtrer Geometry / Material pour ne garder que les différences
            base_geom = set(geom_sum[0].splitlines())
            geom_sum  = ["\n".join([l for l in g.splitlines() if l not in base_geom] or ["–"])
                        if i else g
                        for i, g in enumerate(geom_sum)]

            base_mat = set(mat_sum[0].splitlines())
            mat_sum  = ["\n".join([l for l in m.splitlines() if l not in base_mat] or ["–"])
                        if i else m
                        for i, m in enumerate(mat_sum)]

            # tableau matplotlib
            col_labels = [lbl.replace("Mat_", "\nMat_") for lbl in cfg_labels]
            cellText, rowLabels = [], []

            cellText.append(geom_sum);  rowLabels.append("Geometry (nm)")
            cellText.append(mat_sum);   rowLabels.append("Material")
            if flags['show_fwhm']:
                cellText.append(fwhm_sum);  rowLabels.append("FWHM (nm)")

            if flags['show_lambda0']:
                cellText.append(lam_sum);   rowLabels.append(r"$\lambda_0$")

            if flags['show_delta_lam_over_midLam']:
                cellText.append(delta_lam_over_midLam)
                rowLabels.append(r"$\Delta_{\lambda}$ / $\lambda_{min}$ or $\lambda_{sym}$")

            if flags['show_S_lambda']:
                cellText.append(S_lambda_sum);  rowLabels.append(r"$S_{\lambda}$ (nm/RIU)")

            if flags['show_S_dn']:
                cellText.append(S_R_sum);      rowLabels.append("ΔR/Δn (1/RIU)")

            if flags['show_deltaR_half']:
                cellText.append(dR_half_sum);  rowLabels.append(r"$\Delta R_{half}$")

            if flags['show_Q']:
                cellText.append(Q_fac);        rowLabels.append("Q-factor")

            fs = 8 if len(cfg_labels) <= 5 else max(8 - (len(cfg_labels) - 5), 3)
            table = ax_table.table(cellText=cellText, colLabels=col_labels,
                                rowLabels=rowLabels, loc="center", cellLoc="left")
            table.auto_set_font_size(False)
            table.set_fontsize(fs)
            table.auto_set_column_width(col=np.arange(len(cfg_labels)))

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

            # légende (spectres)
            handles, labels = ax_plot.get_legend_handles_labels()
            handles = [h for h, lab in zip(handles, labels) if lab and not lab.startswith('_')]
            labels  = [lab for lab in labels if lab and not lab.startswith('_')]
            if labels:
                ax_plot.legend(handles, labels, loc='best', fontsize=9, frameon=False)


            # ----------------------------------------------------------------------
            # Affichage dans le widget Output
            # ----------------------------------------------------------------------

            fig.tight_layout()             
            display(fig)
            display(_download_link(fig,
                        f"simulation_{datetime.now():%Y%m%d_%H%M%S}.png"))

        # ------------------------------------------------------------------
        # 4) Rétablir l’interactivité et fermer proprement
        # ------------------------------------------------------------------
        mpl.interactive(was_interactive)           
        plt.close(fig)  
                         


        # ----------------------------------------------------------------------
        # Écriture HDF5 récap (un seul groupe pour la session)
        # ----------------------------------------------------------------------
        with h5py.File(h5_path, "a") as f:
            grp = f.require_group(f"Simulations/{datetime.now():%Y%m%d_%H%M%S}")
            grp.attrs["date"] = datetime.now().isoformat()

            for res in results:
                cfg = res['cfg']
                name = res['cfg_name']
                sub = grp.require_group(name)
                sub.create_dataset("wavelength", data=lam_range, compression="gzip")
                sub.create_dataset("Rup_base",   data=res['Rup'], compression="gzip")
                if "extra_metrics" in res['details']:
                    meta = sub.require_group("extra_metrics")
                    for k, v in res['details']["extra_metrics"].items():
                        meta.attrs[k] = v


    
    
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
        """
        (Dé)sélectionne d’un coup toutes les cases ‘Δn’.

        - Si elles sont toutes cochées → on les décoche.
        - Sinon → on les coche toutes.
        """
        # True si TOUTES les cases sont déjà cochées
        all_selected = all(chk.value for chk in self.dn_checkboxes.values())

        # On applique la valeur opposée à chacune
        for chk in self.dn_checkboxes.values():
            chk.value = not all_selected

        # ↳ force tout de suite la mise à jour des widgets Δn
        self._toggle_delta_widgets()




    def _toggle_delta_widgets(self, *_):
        sel_any_dn = any(chk.value for chk in self.dn_checkboxes.values())
        self.layer_selector.disabled = not sel_any_dn
        self.delta_n_widget.disabled = not sel_any_dn


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
    #                            fonction de coût                       #
    # ----------------------------------------------------------------- #

    def cost(self, x, keys, mode="dip", fixed_lambda=None, range_lambda=None):
        """
        Injection de x sur les clés `keys`, simulation, puis on choisit
        le dip optimal via find_best_dip (max ΔR/Δn ou Δλ/Δn selon mode).
        Retourne 1 – best_S_R.
        """
        # 1) Récupère la config cochée
        sel = [c for c in self.all_configs
            if self.config_checkboxes[c["config_name"]].value]
        if not sel:
            raise RuntimeError("Select configuration.")
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
        delta_n    = max(self.delta_n_widget.value, 1e-6)

        # 4) Simule le spectre de base
        Rup0, _, _ = run_simulation_one_combo(
            lam, wave, n_modes, cfg, self.json_combined_path
        )
        Rup0 = np.asarray(Rup0, float)

        # valeurs λ issues des widgets si non fournies
        fixed_lambda = fixed_lambda or self.lambda0_in.value

        if mode == 'fixed_lambda':
            R = float(np.interp(fixed_lambda, lam, Rup0))
            return 1.0 - R

        elif mode == 'range_lambda':
            lam_min, lam_max = range_lambda
            mask = (lam >= lam_min) & (lam <= lam_max)
            R_mean = float(np.mean(Rup0[mask]))
            return 1.0 - R_mean

        
        # 8) Retourne le coût 1 – ΔR/Δn (plus ΔR/Δn est grand, plus le coût est petit)
        # ↓ ne lance la recherche de dip que si nécessaire
        if mode in ('dip', 'half'):
            best_out, _, _ = find_best_dip(
            cfg=cfg,
            wavelength=lam,
            reflectance=Rup0,
            wave=wave,
            n_modes=n_modes,
            sel_layers=sel_layers,
            delta_n=delta_n,
            json_combined_path=self.json_combined_path,
            smooth_win=0, polyorder=0,
            dip_prom=1e-2, dip_dist=1,
            peak_dist=1,
            verbose=False,
            cfg_name=cfg["config_name"],
            mode=('half' if mode=="half" else 'dip')
        )
            
            
            if best_out is None:
                return 1.0
            
            idx_dR = 13 if mode == 'dip' else 15   # 15 = best_dR_half
            best_dR = float(best_out[idx_dR])
            return 1.0 - best_dR

        
    


# instanciation globale
sim_tab = SimulationTab()
simulation_tab = sim_tab.tab

