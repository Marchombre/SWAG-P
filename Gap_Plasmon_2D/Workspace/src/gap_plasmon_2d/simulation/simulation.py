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
import logging
logger = logging.getLogger(__name__)


from gap_plasmon_2d import paths
from pathlib import Path

import matplotlib as mpl
mpl.use('module://ipympl.backend_nbagg')

import os, io, base64, json, textwrap, sys
from copy import deepcopy
from datetime import datetime

import warnings

import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
import h5py

from matplotlib.figure import Figure
from matplotlib.backends.backend_nbagg import FigureCanvasNbAgg


from datetime import time

import multiprocessing as mp
import queue as q
import threading
from functools import partial   # si besoin d’init pool


from ipywidgets import Layout, HBox, VBox, ToggleButton, HTML
from IPython.display import HTML as DHTML, display, Javascript, clear_output, Image
from scipy.interpolate import interp1d

# pour surveiller les changements de fichiers
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from IPython import get_ipython

from gap_plasmon_2d.utils.file_watchers import start_watcher

# dépendances internes
from gap_plasmon_2d.optimisation.cost_function import compute_cost
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


# 1) Référentiels de base
# module_dir       = Path(__file__).resolve().parent             # …/Workspace/src/…/simulation
# workspace_dir    = module_dir.parent                            # …/Workspace/src/… 
# project_root     = workspace_dir.parent                         # …/Workspace

# 2) Dossiers « externes » fournis par gap_plasmon_2d
data_dir         = Path(paths.DATA_DIR)                         # …/data
configurations_dir = Path(paths.CONFIGS_DIR)                    # …/configs
results_dir      = Path(paths.RESULTS_DIR)                      # …/results

# 3) Sous-dossiers à l’intérieur de results_dir
summary_sim_dir       = results_dir / "summary_simulation"      # pour les .json & .png de simulation
summary_convergence   = results_dir / "summary_convergence"     # pour convergence_results.json
experimental_data_dir = results_dir / "Experimental_Data"       # datas expérimentales

# 4) Fichiers clés
json_combined_path = os.path.join(data_dir, "combined_materials.json")

CONFIG_LIST_JSON = Path(configurations_dir) / "geom_mat_combinations.json"
h5_path               = Path(paths.H5_RESULTS_DIR) / "simulation_results.h5"

# 5) Création automatique des dossiers si nécessaire
for d in (
    summary_sim_dir,
    summary_convergence,
    experimental_data_dir,
    h5_path.parent
):
    d.mkdir(parents=True, exist_ok=True)



    
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





def _load_available_configs() -> list[str]:
    """
    Lit CONFIG_LIST_JSON et retourne la liste triée des config_name disponibles.
    Gère à la fois l’ancien format “configs” et le format “ALL_COMBINED_CONFIGS”.
    """
    try:
        data = json.loads(CONFIG_LIST_JSON.read_text(encoding="utf-8"))
        if "configs" in data:
            return sorted(data["configs"].keys())
        if "ALL_COMBINED_CONFIGS" in data:
            return sorted(
                cfg["config_name"]
                for cfg in data["ALL_COMBINED_CONFIGS"]
                if "config_name" in cfg
            )
    except Exception as e:
        warnings.warn(f"Impossible de lire '{CONFIG_LIST_JSON}': {e}")
    return []






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

        # 1) runtime flags & handles pour le parallélisme
        self._init_runtime_flags()
        # 2) chargement des configs JSON
        self._load_configs()
        # 3) création des widgets (sans layout)
        self._init_common_widgets()
        
        self._init_metrics_overlays()


        # 4) construction des panneaux (panels)
        # ── Initialise 1 seule figure + 2 axes ───────────────────────────
        # 1) Active le backend 'widget' pour ipympl (une seule fois)
        get_ipython().run_line_magic('matplotlib', 'widget')
        # 2) Désactive l’auto-affichage des figures, sinon plt.subplots() 
        #    injecte une figure dans la cellule
        plt.ioff()

        # 3) Création de la figure & des axes (plus tard vous displayez
        #    uniquement via self.canvas_output)
        self.fig, (self.ax_plot, self.ax_table) = plt.subplots(
            nrows=2, figsize=(8, 7),
            gridspec_kw={'height_ratios': [1, 1]}
        )

        # ───> canvas réactif (s’adapte à la largeur disponible)
        self.fig.canvas.layout = widgets.Layout(width='100%', min_width='0')

        # ─────────── AJOUT (remplace la version précédente) ───────────
        # (a) on masque complètement le canvas…
        self.fig.canvas.layout.display = 'none'            # ← change « visibility » → « display »

        # (b) …et on le remet dans le flux dès le premier draw().
        def _show_canvas(_event):
            self.fig.canvas.layout.display = ''            # ← ré-affiche le canvas
            self.fig.canvas.mpl_disconnect(self._draw_cid)

        self._draw_cid = self.fig.canvas.mpl_connect('draw_event', _show_canvas)
        # ───────────────────────────────────────────────────────────────


        # ajustements de layout
        self.fig.subplots_adjust(
            left=0.15, right=0.98,
            top=0.90,  bottom=0.10,
            hspace=0.2
        )
        self.ax_table.axis('off')

        # 4) Réactive le mode interactif *sans* auto-affichage
        plt.ion()

        # 5) Prépare l’Output et n’affiche QUE ce canvas
        self.canvas_output = widgets.Output(
            layout=widgets.Layout(
                border='1px solid lightgray',
                min_height='250px',
                overflow_x='hidden',   # ← on interdit le scroll horizontal
                min_width='0'
            )
        )


        with self.canvas_output:
            clear_output()
            display(self.fig.canvas)


        self._build_panels()
        # 5) assemblage responsive en grille
        self._assemble_layout()
        # 6) liaison des signaux (on_click, observe)
        self._connect_signals()
        # 7) watcher automatique sur le dossier summary_sim_dir
        #    à chaque nouveau .json on reconstruit l’affichage
        start_watcher(
            path=summary_sim_dir,
            callback=lambda event: self._on_new_summary(event),
            extensions=[".json"]
        )

        # watcher sur le dossier de configs, pas sur le fichier lui-même
        self._cfg_watcher, self._cfg_handler = start_watcher(
            path=str(CONFIG_LIST_JSON),         # ← on regarde LE fichier
            callback=self._on_cfg_fs_event,     # ← on appelle cette méthode
            extensions=[".json"],
            recursive=False,
            debounce_interval=0.2,
        )




    def _on_new_summary(self, event):
        """
        Callback watchdog : lorsqu’un nouveau fichier summary_sim
        apparaît, on l’affiche automatiquement.
        """
        # recharge la liste des fichiers et sélectionne le dernier ajouté
        files = list_sim_summary_files(summary_sim_dir)
        if files:
            self.sim_files_dropdown.options = files
            self.sim_files_dropdown.value   = files[-1]
            # génère un nouveau download link
            self._download_file(None)
        
        # 7) état initial des widgets Δn
        self._toggle_delta_widgets()




    # ----------------------------------------------------------------- #
    #      1) initialisation runtime et chargement des configs         #
    # ----------------------------------------------------------------- #
    def _init_runtime_flags(self):
        self._is_running    = False
        self._cancelled     = False
        self._pool          = None
        self._worker_thread = None
        self._result_queue  = None

    def _load_configs(self):
        cfg_file = os.path.join(configurations_dir, "geom_mat_combinations.json")
        if os.path.exists(cfg_file):
            with open(cfg_file, encoding="utf-8") as f:
                self.all_configs = json.load(f)["ALL_COMBINED_CONFIGS"]
        else:
            self.all_configs = []
        self.json_combined_path = json_combined_path



    # ----------------------------------------------------------------- #
    #       2) toggle λ₀ (méthode de classe, pas locale)               #
    # ----------------------------------------------------------------- #
    def _toggle_lambda0(self, change):
        self.lambda0_in.layout.display = (
            '' if change['new'] == 'fixed_lambda' else 'none'
        )




    def _init_common_widgets(self):
        """
        Initialise tous les widgets sans construire leur layout final.
        """
        # ─── contraintes positives / bornées ─────────────────────────────────
        def _positive(change):
            owner, val = change['owner'], change['new']
            if val < 0:
                owner.value = 0
            if owner is self.sim_lambda_min and val > self.sim_lambda_max.value:
                owner.value = self.sim_lambda_max.value
            if owner is self.sim_lambda_max and val < self.sim_lambda_min.value:
                owner.value = self.sim_lambda_min.value

        # ─── Spectre & points────────────────────────────────────────────────
        self.sim_lambda_min = widgets.FloatText(
            value=450.0, description="λ min (nm):",
            layout=widgets.Layout(width='150px'),
            style={'description_width':'initial'}
        )
        self.sim_lambda_max = widgets.FloatText(
            value=900.0, description="λ max (nm):",
            layout=widgets.Layout(width='150px'),
            style={'description_width':'initial'}
        )
        self.sim_n_points = widgets.IntText(
            value=300, description="Points:",
            layout=widgets.Layout(width='150px'),
            style={'description_width':'initial'}
        )
        self.sim_n_mod = widgets.IntText(
            value=5,
            layout=widgets.Layout(width='150px'),
            style={'description_width':'initial'}
        )
        for w in (self.sim_lambda_min, self.sim_lambda_max, self.sim_n_points):
            w.observe(_positive, names='value')

        # ─── Métrique λ₀ / range ─────────────────────────────────────────────
        self.band_min_in = widgets.FloatText(
            value=650.0, description="λmin:",
            layout=widgets.Layout(width='150px'),
            style={'description_width':'initial'}
        )
        self.band_max_in = widgets.FloatText(
            value=750.0, description="λmax:",
            layout=widgets.Layout(width='150px'),
            style={'description_width':'initial'}
        )
        self.band_box_in = widgets.HBox(
            [self.band_min_in, self.band_max_in],
            layout=widgets.Layout(gap='5px')
        )
        self.mode_calc_radio = widgets.RadioButtons(
            options=[
                ('(ΔR/Δn) λ dip of reflectance',    'dip'),
                ('(ΔR/Δn) λ half slope',    'half'),
                ('Custom fixed λ₀','fixed_lambda')
            ],
            value='dip',
            description='Compute R(λ):',
            style={'description_width':'initial'},
            layout=widgets.Layout(width='220px')
        )
        self.lambda0_in = widgets.FloatText(
            value=700.0, description="λ₀ (nm):",
            layout=widgets.Layout(width='130px'),
            style={'description_width':'initial'}
        )
        # toggle λ₀ via méthode de classe
        self.mode_calc_radio.observe(self._toggle_lambda0, names='value')
        self._toggle_lambda0({'new': self.mode_calc_radio.value})

        # ─── Cancel + status ───────────────────────────────────────────
        self.sim_cancel_button = widgets.Button(
            description="Cancel", button_style="warning",
            tooltip="Cancel running simulation", disabled=True
        )
        self._status_html      = widgets.HTML("")
        self._progress_bar     = widgets.FloatProgress(
            value=0, min=0, max=1, description="0 %",
            bar_style="info",
            layout=widgets.Layout(width='100%', display='none')
        )
        self.runtime_box       = widgets.VBox(
            [self._status_html, self._progress_bar],
            layout=widgets.Layout(gap='4px')
        )

        # ─── RCWA modes fixe/custom/auto ─────────────────────────────────────
        self.mode_selection      = widgets.RadioButtons(
            options=[('Fixe','fixed'),('Custum','custom'),('Auto','auto')],
            value='fixed',
            description='RCWA modes',
            style={'description_width':'initial'},
            layout=widgets.Layout(width='220px')
        )
        self.custom_modes_box = widgets.VBox(
            value=5, min=1,
            description='n_mod',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
            )
        
        self.custom_n_mod_inputs = {}

        # ─── Fichiers résumé & download ─────────────────────────────────────
        self.sim_files_dropdown  = widgets.Dropdown(
            options=list_sim_summary_files(summary_sim_dir),
            description="Sim files:",
            layout=widgets.Layout(width='500px'),
            style={'description_width':'initial'}
        )

        self.sim_download_button = widgets.Button(
            description="Download", button_style="danger"
        )

        # ─── Bouton Run ──────────────────────────────────────────────────
        self.sim_run_button = widgets.Button(
            description="Run",
            button_style="success",
            tooltip="Lancer la simulation"
        )


        # ─── Nom de simulation ───────────────────────────────────────────────
        self.sim_name_widget     = widgets.Text(
            placeholder="Nom sim (auto si vide)",
            description="Sim Name:",
            layout=widgets.Layout(width='500px'),
            style={'description_width':'initial'}
        )

        # ─── Configs & Δn ────────────────────────────────────────────────────
        self.config_checkboxes, self.dn_checkboxes = {}, {}
        rows = []
        for cfg in self.all_configs:
            name = cfg["config_name"]
            chk = widgets.Checkbox(value=False, description=name, indent=False)
            dn  = widgets.Checkbox(value=False, description='Δn', indent=False,
                    layout=widgets.Layout(width='46px'))
            
            chk.observe(self._update_sim_run_button, names='value')
            
            self.config_checkboxes[name] = chk
            self.dn_checkboxes[name]     = dn
            rows.append(widgets.HBox([chk, dn],
                    layout=widgets.Layout(gap='5px')))
        
        
        self.select_all_cfg_btn = widgets.Button(
            description="Tout sélectionner Configs",
            button_style="info",
            layout=widgets.Layout(margin='0 5px 5px 0')
        )
        self.select_all_dn_btn = widgets.Button(
            description="Tout sélectionner Δn",
            button_style="info",
            layout=widgets.Layout(margin='0 0 5px 0')
        )
        visible = min(len(rows), 10)
        self.config_list = widgets.VBox(
            [widgets.HBox([self.select_all_cfg_btn, self.select_all_dn_btn],
                        layout=widgets.Layout(gap='10px')),
            *rows],
            layout=widgets.Layout(
                width='500px',
                height=f'{30 + visible*30}px',
                overflow_y='auto',
                border='1px solid lightgray',
                padding='5px',
                display='none'
            )
        )

        self._update_sim_run_button()
        
        self.toggle_btn = widgets.ToggleButton(
            description="Select Configs & Δn",
            value=True,                     # ← ouvert par défaut
            icon='caret-up',                # ← icône cohérente
            layout=widgets.Layout(width='520px'),
            button_style='warning'
        )

        self._rebuild_sim_config_selector()

        self._toggle_config_list({'new': self.toggle_btn.value})


        self.config_selector    = widgets.VBox(
            [self.toggle_btn, self.config_list],
            layout=widgets.Layout(padding='5px')
        )

        # ─── Couches Δn & delta_n ───────────────────────────────────────────
        layer_keys = [m['key'] for m in self.all_configs[0]['material']['MATERIALS_CONFIG']]
        self.layer_selector     = widgets.SelectMultiple(
            options=layer_keys,
            description="Add Δn to layers:",
            layout=widgets.Layout(width='300px', height='100px'),
            style={'description_width':'initial'},
            disabled=True
        )
        self.delta_n_widget     = widgets.FloatText(
            value=1e-2, description="Δn:",
            layout=widgets.Layout(width='150px'),
            style={'description_width':'initial'}
        )
        self.delta_n_widget.observe(_positive, names='value')

        # ─── Debug & verbose ─────────────────────────────────────────────────
        # ► 1.a  case à cocher  (active par défaut)
        self.verbose_toggle = widgets.Checkbox(
            value=True,                # ← ON par défaut
            description="Verbose log",
            indent=False,
            style={'description_width': 'initial'}
        )

        # ► 1.b  zone HTML plein‑écran
        self.debug_out = widgets.HTML(
            value="",
            layout=widgets.Layout(
                width='100%',          # occupe toute la largeur future
                border='1px solid #cfd8dc',
                padding='6px',
                overflow_y='auto',
                max_height='200px',
                display=''             # visible dès le début (car verbose = True)
            )
        )



    def _build_panels(self):
        """
        Ne construit plus que le panneau de contrôle (à gauche).
        Les metrics et debug seront placés par _assemble_layout.
        """
        self.panel_controls = widgets.VBox(
            [
                widgets.HTML("<h3>Simulation – Paramètres</h3>"),
                self.sim_name_widget,
                self.sim_files_dropdown,
                self.sim_download_button,
                widgets.HBox(
                    [self.sim_lambda_min, self.sim_lambda_max, self.sim_n_points],
                    layout=widgets.Layout(gap='10px')
                ),
                self.config_selector,
                widgets.HTML("<b>RCWA modes</b>"),
                widgets.HBox(
                    [self.mode_selection, self.sim_n_mod],
                    layout=widgets.Layout(gap='10px')
                ),
                self.custom_modes_box,
                widgets.HBox(
                    [self.layer_selector, self.delta_n_widget],
                    layout=widgets.Layout(gap='10px')
                ),
                widgets.HBox(
                    [self.mode_calc_radio, self.lambda0_in],
                    layout=widgets.Layout(gap='10px')
                ),
                widgets.HBox(
                    [self.sim_run_button, self.sim_cancel_button],
                    layout=widgets.Layout(gap='10px')
                ),
                self.runtime_box,
                self.verbose_toggle
            ],
            layout=widgets.Layout(
                padding='10px',
                border='1px solid lightgray',
                min_width='320px'
            )
        )





    def _init_metrics_overlays(self):
        """Crée un panneau moderne pour choisir métriques et overlays."""

        # Libellés & valeurs par défaut
        metric_labels = [
            "FWHM", "λ₀", "Δλ/λₘᵢₙ", "Sλ (nm/RIU)",
            "ΔR/Δn", "ΔR_half", "Q-factor"
        ]
        overlay_labels = [
            "Rup+Δn", "Half-level", "Dips",
            "Maxima", "Symmetry", "Selected",
            "Sensitivity"
        ]
        defaults = {lbl: False for lbl in metric_labels}
        defaults["λ₀"] = True

        # Création des Checkbox
        self.metric_checks = {
            lbl: widgets.Checkbox(value=True, description=lbl, indent=False)
            for lbl in metric_labels
        }
        self.overlay_checks = {
            lbl: widgets.Checkbox(value=True, description=lbl, indent=False)
            for lbl in overlay_labels
        }

        #  Autorise le retour à la ligne dans *tous* les labels
        for cb in (*self.metric_checks.values(), *self.overlay_checks.values()):
            cb.add_class("wrap-label")        # on pose une classe CSS
            cb.layout.width = "auto"          # pas de largeur minimale
            cb.style.description_width = "initial"

        # helper grid -------------------------------------------------------
        def _make_grid(children):
            return widgets.GridBox(
                list(children),
                layout=widgets.Layout(
                    grid_template_columns="repeat(auto-fit, minmax(110px, 1fr))",
                    gap="6px",
                    overflow_x="hidden",      # ⟵ bloque le scroll horizontal
                    width="100%",
                    min_width="0"             # ⟵ évite que le grid force sa largeur
                )
            )

        metrics_grid  = _make_grid(self.metric_checks.values())
        overlays_grid = _make_grid(self.overlay_checks.values())


        # ------------------------------------------------------------------
        #  accordéons indépendants, chacun
        #  contenant un seul panneau, tous deux ouverts par défaut
        #  (selected_index = 0).  De cette façon, les contenus « Métriques »
        #  *et* « Overlays » sont visibles dès le lancement.
        # ------------------------------------------------------------------
        acc_metrics = widgets.Accordion(
            children=[metrics_grid],
            selected_index=0,                       # ← ouvert d’office
            layout=widgets.Layout(width="100%")
        )
        acc_metrics.set_title(0, "Show metrics")

        acc_overlays = widgets.Accordion(
            children=[overlays_grid],
            selected_index=0,                       # ← ouvert d’office
            layout=widgets.Layout(width="100%")
        )
        acc_overlays.set_title(0, "Graphical overlays")

        # Regroupement : on remplace l’ancien unique Accordion
        # par un VBox qui contient les deux accordéons ci-dessus.
        self.metrics_panel = widgets.VBox(
            [acc_metrics, acc_overlays],
            layout=widgets.Layout(
                width="100%",
                padding="8px",
                border="1px solid lightgray",
                border_radius="6px",
                gap="4px",
                overflow_x="hidden"
            )
        )

        # règle CSS injectée une fois pour la classe « wrap-label »
        display(HTML("""
        <style>
        .wrap-label > label.widget-label {
            white-space: normal !important;     /* autorise le retour à la ligne */
            line-height : 1.1em;
        }
        </style>
        """))




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
            # Métriques
            "show_fwhm"                 : self.metric_checks["FWHM"].value,
            "show_lambda0"              : self.metric_checks["λ₀"].value,
            "show_delta_lam_over_midLam": self.metric_checks["Δλ/λₘᵢₙ"].value,
            "show_S_lambda"             : self.metric_checks["Sλ (nm/RIU)"].value,
            "show_S_dn"                 : self.metric_checks["ΔR/Δn"].value,
            "show_deltaR_half"          : self.metric_checks["ΔR_half"].value,
            "show_Q"                    : self.metric_checks["Q-factor"].value,
            # Overlays
            "show_Rup_dn"               : self.overlay_checks["Rup+Δn"].value,
            "show_hlines"               : self.overlay_checks["Half-level"].value,
            "show_dips"                 : self.overlay_checks["Dips"].value,
            "show_maxima"               : self.overlay_checks["Maxima"].value,
            "show_symmetry_pts"         : self.overlay_checks["Symmetry"].value,
            "show_selected_dip"         : self.overlay_checks["Selected"].value,
            "show_sensitivity_marker"   : self.overlay_checks["Sensitivity"].value,
        }

        verbose     = self.verbose_toggle.value


        # 1) on lit d'abord les noms cochés, toujours à jour avec le widget
        selected_names = [
            name for name, chk in self.config_checkboxes.items()
            if chk.value
        ]
        if not selected_names:
            self._status_html.value = "⚠️ Please select at least one configuration."
            self._is_running = False
            self.sim_cancel_button.disabled = True
            return

        # Sélection des configurations
        selected_cfgs = [
            cfg for cfg in self.all_configs
            if cfg['config_name'] in selected_names
        ]

        
        # 2.bis) mêmes pour Δn
        self._cfgs_with_delta = {
            name for name, chk in self.dn_checkboxes.items()
            if chk.value
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
            # -- état interne --------------------------------------------------
            self._is_running = False
            self.sim_cancel_button.disabled = True

            # -- cache la barre & remet à zéro ---------------------------------
            self._progress_bar.layout.display = "none"
            self._progress_bar.value          = 0
            self._progress_bar.bar_style      = "info"        # prêt pour le prochain run

            # -- message “Done” chic -------------------------------------------
            self._status_html.layout.display = ""             # (au cas où il était masqué)
            self._status_html.value = (
                "<span style='"
                "display:inline-flex; align-items:center; gap:6px; "
                "font-weight:600; color:#2E7D32; font-size:14px;'>"
                "&#x2705; Done"
                "</span>"
            )

            # -- continue avec l’affichage des résultats -----------------------
            self._build_outputs(payload[0])
            return




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

        # 1) Efface les anciens tracés
        self.ax_plot.clear()
        self.ax_table.clear()
        self.ax_table.axis('off')
        self.ax_plot.set_xlabel("λ (nm)")
        self.ax_plot.set_ylabel("Reflectance R")

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

        try:
                       
            # Ferme l’ancienne figure (celle en mémoire !)
            if hasattr(self, "_last_fig") and self._last_fig is not None:
                plt.close(self._last_fig)
                self._last_fig = None



            # ─── boucle sur les configurations simulées ----------------------
            for idx, res in enumerate(results):
                cfg      = res['cfg']
                name     = res['cfg_name']
                Rup_base = res['Rup']
                details  = res['details']
                color    = colors[idx % len(colors)]
                n_modes  = self._get_n_modes_for(name)

                self.ax_plot.plot(lam_range, Rup_base, color=color,
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
                    
                    # ─── injecte Rup_dn dans le dictionnaire details ───────────
                    details["Rup_dn"] = np.asarray(Rup_dn)   # save_simulation_summary lira ce champ
                    
                    
                    if Rup_dn is not None and flags['show_Rup_dn']:
                        self.ax_plot.plot(lam_range, Rup_dn, '--', color=color,
                                    linewidth=2, alpha=0.7, zorder=100,
                                    label=f"{name} (R + Δn)")

                # ------------------------------------------------------------------
                # Overlays graphiques divers
                # ------------------------------------------------------------------
                if flags['show_hlines']:
                    self.ax_plot.hlines(ylev, lam_left, lam_right,
                                color=color)
                if flags['show_dips']:
                    self.ax_plot.scatter(lam_range[dips_idx_list],
                                    Rup_base[dips_idx_list],
                                    marker='x', color=color)
                if flags['show_maxima']:
                    self.ax_plot.scatter(lam_max_l_list, R_max_l_list, marker='x',
                                    color=color)
                    self.ax_plot.scatter(lam_max_r_list, R_max_r_list, marker='x',
                                    color=color)
                if flags['show_symmetry_pts']:
                    self.ax_plot.scatter(lam_sym_list, R_sym_list, marker='x',
                                    color=color)
                if flags['show_selected_dip']:
                    self.ax_plot.scatter([lam_dip], [R_dip], marker='o',
                                    facecolor='none', edgecolor=color, s=70)

                # -------------------------------------------------
                #  Collecte pour la future table verbose
                # -------------------------------------------------
                debug_rows = getattr(self, "_debug_rows", [])
                debug_rows.append(dict(
                    name     = name,
                    mode     = "FWHM ½" if use_half else "Dip",
                    lambda0  = f"{lam_dip:.1f} nm",
                    fwhm     = f"{fwhm:.1f}",
                    SR       = f"{best_S_R:.3f}" if best_S_R is not None else "–",
                    comment  = "No valid dip !" if Best_values_out is None else ""
                ))
                self._debug_rows = debug_rows      # on persiste pour la fin

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
                    {name: details}, lam_range, wave, n_modes, summary_sim_dir,
                    custom_name=name,
                    fwhm_summaries=[fwhm_sum[-1]],
                    lam_summaries=[lam_sum[-1]],
                    delta_lam_over_midLam_summaries=[delta_lam_over_midLam[-1]],
                    Q_factor=[Q_fac[-1]],
                    best_S_R=[S_R_sum[-1]]
                )


            # 4) Met à jour le lien de téléchargement
            self._last_download_link = _download_link(self.fig, f"simulation_{datetime.now():%Y%m%d_%H%M%S}.png")
            
            with self.canvas_output:
                clear_output(wait=True)
                self.fig.canvas.draw()
                display(self.fig.canvas, self._last_download_link)
            
            
            
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
            #  VERBOSE  – tableau responsive plein‑écran
            # ----------------------------------------------------------------------
            if verbose:
                rows = getattr(self, "_debug_rows", [])
                mode_sel = "FWHM ½" if use_half else "Dip"

                best_cfg = next((cfg_labels[i] for i,v in enumerate(S_R_vals)
                                if v == np.nanmax(S_R_vals)), "—")

                table_body = "\n".join(
                    f"<tr>"
                    f"<td>{r['name']}</td>"
                    f"<td>{r['mode']}</td>"
                    f"<td>{r['lambda0']}</td>"
                    f"<td>{r['fwhm']}</td>"
                    f"<td>{r['SR']}</td>"
                    f"<td>{r['comment']}</td>"
                    f"</tr>"
                    for r in rows
                ) or "<tr><td colspan='6' style='text-align:center;'>No entry</td></tr>"

                verbose_html = f"""
                <style>
                .vlog * {{ font-family:Consolas, monospace; font-size:12px; }}
                .vlog table{{border-collapse:collapse;width:100%;}}
                .vlog th,.vlog td{{border:1px solid #ddd;padding:3px 6px;white-space:nowrap;}}
                .vlog thead th{{background:#455a64;color:#fff;}}
                .vlog tbody tr:nth-child(odd){{background:#f7f9fa;}}
                .vlog caption{{caption-side:top;text-align:left;font-weight:bold;
                                margin:2px 0 6px;color:#1565c0;}}
                </style>

                <div class="vlog">
                <caption>Verbose log — mode : <b>{mode_sel}</b> | best config : <b>{best_cfg}</b></caption>
                <table>
                    <thead>
                    <tr><th>Config</th><th>Mode</th><th>λ₀</th><th>FWHM (nm)</th><th>ΔR/Δn</th><th>Note</th></tr>
                    </thead>
                    <tbody>
                    {table_body}
                    </tbody>
                </table>
                </div>
                """

                self.debug_out.value = verbose_html
                # réinitialise pour la prochaine simulation
                self._debug_rows = []


            # ----------------------------------------------------------------------
            # Construction du tableau final 
            # ----------------------------------------------------------------------
            if not cfg_labels:                        # aucun dip retenu
                self.ax_plot.set_title("Spectres simulés – aucun dip valide")
                handles, labels = self.ax_plot.get_legend_handles_labels()
                if labels:
                    self.ax_plot.legend(handles, labels, loc='best',
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
            table = self.ax_table.table(cellText=cellText, colLabels=col_labels,
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
            handles, labels = self.ax_plot.get_legend_handles_labels()
            handles = [h for h, lab in zip(handles, labels) if lab and not lab.startswith('_')]
            labels  = [lab for lab in labels if lab and not lab.startswith('_')]
            if labels:
                self.ax_plot.legend(handles, labels, loc='best', fontsize=9, frameon=False)


        except Exception as e:
                import traceback
                display(HTML(f"<pre>{traceback.format_exc()}</pre>"))

                         


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
        conv_json = summary_convergence / "convergence_results.json"
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


    def _update_sim_run_button(self, *_):
        """Active le bouton Run si au moins une config est cochée."""
        any_selected = any(cb.value for cb in self.config_checkboxes.values())
        self.sim_run_button.disabled = not any_selected



    def _toggle_config_list(self, change):
        show = 'block' if change['new'] else 'none'
        self.config_list.layout.display = show
        self.toggle_btn.icon = 'caret-up' if change['new'] else 'caret-down'


    def _rebuild_sim_config_selector(self, *_):
        # 1) on mémorise l’état ouvert/fermé
        was_open = self.toggle_btn.value
        # 2) on garde d’anciennes sélections
        prev_cfg = {n: cb.value for n, cb in self.config_checkboxes.items()}
        prev_dn  = {n: cb.value for n, cb in self.dn_checkboxes.items()}

        # 3) on reconstruit les widgets ligne par ligne
        rows = []
        self.config_checkboxes.clear()
        self.dn_checkboxes.clear()
        for name in _load_available_configs():
            chk_cfg = widgets.Checkbox(
                value=prev_cfg.get(name, False),
                description=name, indent=False
            )
            chk_dn = widgets.Checkbox(
                value=prev_dn.get(name, False),
                description="Δn", indent=False,
                layout=widgets.Layout(width="46px")
            )
            # on rattache les observateurs existants
            chk_dn.observe(self._toggle_delta_widgets, names='value')
            chk_cfg.observe(self._update_sim_run_button, names='value')

            self.config_checkboxes[name] = chk_cfg
            self.dn_checkboxes[name]     = chk_dn
            rows.append(widgets.HBox([chk_cfg, chk_dn],
                          layout=widgets.Layout(gap='5px')))
        # 4) on réinjecte dans le VBox
        visible = min(len(rows), 10)
        self.config_list.layout.height = f"{30 + visible*30}px"
        # on veut juste les HBox, pas le header ni les boutons
        header = widgets.HBox([self.select_all_cfg_btn, self.select_all_dn_btn],
                               layout=widgets.Layout(gap='10px'))
        self.config_list.children = [header, *rows]

        # 5) on réapplique l’état ouvert/fermé et on refresh custom modes
        self.toggle_btn.value = was_open
        self._refresh_custom_modes()
        self._update_sim_run_button()


    def _on_cfg_fs_event(self, *args):
        """Callback du watcher : on reconstruit la liste dès que JSON change."""
        self._rebuild_sim_config_selector()




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
                layout=Layout(width='150px'),
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



    # ------------------------------------------------------------------
    #  Cost wrapper (appelle la fonction indépendante)
    # ------------------------------------------------------------------
    def cost(self, x, keys, mode="dip", fixed_lambda=None, range_lambda=None):
        """Délègue au module optimisation.cost_function.compute_cost"""
        return compute_cost(
            self, x, keys,
            mode=mode,
            fixed_lambda=fixed_lambda,
            range_lambda=range_lambda
        )

        
    
    def _assemble_layout(self):
        
        # ───────── colonne gauche : panneau de contrôle ──────────
        self.panel_controls.layout = widgets.Layout(
            flex="1 1 300px",     # min-width 300 px, grandit/rétrécit si besoin
            padding="0 10px 0 0",
            min_width="0"
        )

        # ───────── colonne droite : métriques + figure ───────────
        right_column = widgets.VBox(
            [self.metrics_panel, self.canvas_output],
            layout=widgets.Layout(
                flex="2 1 400px",  # s’étire 2 × plus vite que la colonne de gauche
                padding="0 0 0 5px",
                gap="5px",
                align_items="stretch",
                min_width  ="0",
                overflow_x ="hidden"
            )
        )

        # ───────── première ligne : controles + figure ───────────
        top_row = widgets.HBox(
            [self.panel_controls, right_column],
            layout=widgets.Layout(
                width="100%",
                flex_flow="row wrap",   # autorise le retour à la ligne pour le responsive
                align_items="stretch",
                overflow_x ="hidden"
            )
        )

        # ───────── assemblage final : log sous le reste ──────────
        self.tab = widgets.VBox(
            [top_row, self.debug_out],
            layout=widgets.Layout(
                width="100%",
                gap="6px",
                overflow_x="hidden"     # empêche toute barre de défilement horizontale
            )
        )




    def _connect_signals(self):
        self.sim_run_button.on_click(self._on_run)
        self.sim_cancel_button.on_click(self._on_cancel)
        self.sim_download_button.on_click(self._download_file)
        self.select_all_cfg_btn.on_click(self._toggle_all_cfg)
        self.select_all_dn_btn.on_click(self._toggle_all_dn)
        self.toggle_btn.observe(self._toggle_config_list, names='value')
        for chk in self.dn_checkboxes.values():
            chk.observe(self._toggle_delta_widgets, names='value')
        self.mode_selection.observe(self._refresh_custom_modes, names='value')
        self.mode_calc_radio.observe(self._toggle_lambda0, names='value')
        self.verbose_toggle.observe(self._toggle_debug, names='value')


# instanciation globale
sim_tab = SimulationTab()
simulation_tab = sim_tab.tab

