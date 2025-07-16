# -*- coding: utf-8 -*-
"""
Optimisation.py
===============

• Implémente l’onglet d’optimisation basé sur Differential Evolution (DE).
• Corrigé, commenté et fiabilisé : noms cohérents, gestion du mode de calcul,
  traçage, sauvegarde HDF5, etc.
• Ne modifie **aucune** capacité fonctionnelle ; seules clarté, robustesse et
  cohérence interne sont améliorées.
"""
from __future__ import annotations

import multiprocessing as mp
import multiprocessing.pool

from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

import warnings
import threading
import h5py
from copy import deepcopy
import json

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from IPython import get_ipython
import ipywidgets as widgets
#from tqdm.notebook import trange
import traceback, queue as q

from gap_plasmon_2d import paths

from gap_plasmon_2d.optimisation.cost_function import compute_cost
from gap_plasmon_2d.ui.geometry_settings import geometry_limits
from gap_plasmon_2d.ui.optimized_geometry import plot_geometry_static_from_run
from gap_plasmon_2d.simulation.simulation import SimulationTab, sim_tab
from gap_plasmon_2d.simulation.simulate_and_plot import run_simulation_one_combo
from gap_plasmon_2d.utils.saving__functions import save_optimization_hdf5
from gap_plasmon_2d.utils.data_readers import (
    read_optimization_hdf5,
    list_optimization_files,
    list_runs_in_h5
)
from gap_plasmon_2d.utils.file_watchers import start_watcher




class OptimizationCancelled(Exception):
    """Exception levée quand l'utilisateur annule l'optimisation."""
    pass



# -----------------------------------------------------------------------------#
#  PATHS & GLOBALS                                                             #
# -----------------------------------------------------------------------------#
# Ces variables globales définissent les chemins vers :
#  - le dossier où l’on stocke les résultats des optimisations
#  - le dossier des données (notamment le fichier JSON des propriétés optiques)

# BASE_NOTEBOOKS : chemin absolu vers le dossier racine des résultats Jupyter
# Numérique : 
#   Path(__file__).resolve() → path absolu de ce fichier
#   .parent.parent          → remonte de deux niveaux (src/.../optimisation.py → racine du package)
#   / str(paths.RESULTS_DIR)→ concatène le nom de dossier RESULTS_DIR (ex : "results")
BASE_NOTEBOOKS = (
    Path(__file__).resolve().parent.parent  
    / str(paths.RESULTS_DIR)
)

# summary_opt_dir : sous-dossier pour les fichiers de synthèse d’optimisation
# chaque run d’optimisation écrira ici son HDF5 résumé
summary_opt_dir = BASE_NOTEBOOKS / "summary_optimisation"

# Création du dossier s’il n’existe pas (parents=True gère la création récursive)
summary_opt_dir.mkdir(parents=True, exist_ok=True)

summary_convergence = Path(paths.RESULTS_DIR) / "summary_convergence"

# data_dir : dossier des données brutes (matériaux, géométries, etc.)
# contient notamment les fichiers de propriétés optiques
data_dir = Path(paths.DATA_DIR)

# json_combined_path : chemin vers le fichier JSON listant les matériaux standards
# et leurs propriétés optiques (indice de réfraction vs λ).
json_combined_path = data_dir / "combined_materials.json"

# dossier des fichiers « configs »
configurations_dir = Path(paths.CONFIGS_DIR)  
CONFIG_LIST_JSON = configurations_dir / "geom_mat_combinations.json"


# ------------------------------------------------------------------ #
# Fichier de combinaisons géométrie/matériaux : deux formats possibles
# ------------------------------------------------------------------ #
def _load_available_configs() -> list[str]:
    """
    Retourne la liste des noms de configuration présents dans
    « geom_mat_combinations.json ».

    • Ancien format  (clé “configs”) :
        {
            "configs": {
                "cfg_A": {...},
                "cfg_B": {...},
                ...
            }
        }

    • Format utilisé par l’onglet Simulation (clé “ALL_COMBINED_CONFIGS”) :
        {
            "ALL_COMBINED_CONFIGS": [
                { "config_name": "cfg_A", ... },
                { "config_name": "cfg_B", ... },
                ...
            ]
        }
    """
    try:
        with open(CONFIG_LIST_JSON, encoding="utf-8") as f:
            data = json.load(f)

        # ① ancien format
        if "configs" in data:
            return sorted(data["configs"].keys())

        # ② nouveau format (celui de SimulationTab)
        if "ALL_COMBINED_CONFIGS" in data:
            return sorted(
                cfg["config_name"]              # ← même champ que SimulationTab
                for cfg in data["ALL_COMBINED_CONFIGS"]
                if "config_name" in cfg
            )

        return []                               # format inattendu
    except Exception:
        return []                               # fichier illisible ? → vide

    

# -----------------------------------------------------------------------------#
#  Worker-side globals & helpers                                               #
# -----------------------------------------------------------------------------#

# On définit une variable globale `_WORKER_SIM` qui pointe vers l’instance
# `sim_tab` de SimulationTab importée depuis le module simulation.
# Grâce au fork, chaque sous-processus héritera de cette même instance en Copy-On-Write,
# évitant de devoir recharger/configurer le simulateur à chaque appel.
# --------------------------------------------------------------------------- #
# Worker-side globals (communs à tous les subprocess)
_WORKER_SIM: SimulationTab = sim_tab             # ← déjà présent



# --------------------------------------------------------------------------- #
#  cost_worker                                 #
# --------------------------------------------------------------------------- #
def cost_worker(
    args: Tuple[
        int,                 # idx dans la pop
        np.ndarray,          # x  (vecteur d’épaisseurs)
        List[str],           # keys optimisées
        str,                 # cfg_name
        str,                 # mode
        Dict[str, Any],      # mode_kw
        Dict[str, float],     # fixed_vals
        int,
        float,                          # delta_n à utiliser
        list[int]
    ]
) -> Tuple[int, float]:
    idx, x, keys, cfg_name, mode, mode_kw, fixed, n_modes, delta_n, sel_layers = args

    # 1) récupère la config voulue dans _WORKER_SIM
    cfg = next(c for c in _WORKER_SIM.all_configs
               if c["config_name"] == cfg_name)



    # 3) injection des valeurs fixes
    geom   = cfg["geometry"]["geometry"]
    backup = {k: geom[k] for k in fixed}
    for k, v in fixed.items():
        geom[k] = float(v)

    # 4) calcul du coût
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cost_val = compute_cost(
                _WORKER_SIM, x, keys,
                mode=mode,
                n_modes=n_modes,
                selected_cfg=cfg,
                delta_n=delta_n,
                sel_layers=sel_layers,
                **mode_kw
            )
        if not np.isfinite(cost_val):
            cost_val = 1e6
    except Exception:
        cost_val = 1e6
    finally:
        # restauration des valeurs fixes
        for k, v in backup.items():
            geom[k] = v

    return idx, float(cost_val)




class OptimizationFileArboWidget:
    """
    Parcours de l’arborescence :
      summary_opt_dir/
      └─ <family>/
         └─ <cost_mode>/
            └─ <config_name>/
               └─ budgetXX_popYY/        
                  └─ wavelength_range_<min>:<max>/
                     └─ BXX_PYY.h5
    """
    def __init__(self, summary_opt_dir: str):

        self.base_dir      = Path(summary_opt_dir)

        # 1) Dropdowns (inchangés)
        self.family_dd     = widgets.Dropdown(description="Family:")
        self.cost_mode_dd  = widgets.Dropdown(description="Cost mode:")
        self.name_dd       = widgets.Dropdown(description="Config:")
        self.budget_pop_dd = widgets.Dropdown(description="Budget-Pop:")
        self.wave_dd       = widgets.Dropdown(description="Wavelength range:")
        self.file_dd       = widgets.Dropdown(description="File:")
        self.run_dd        = widgets.Dropdown(description="Run:")

        self._user_selecting = False          # ← flag global au widget

        self.run_bounds_out = widgets.Output(layout=widgets.Layout(
            border="1px solid #ccc",
            padding="4px",
            max_height="250px",  
            overflow_y="auto"
        ))        
        # branchement du callback
        self.run_dd.observe(self._on_run_changed, names="value")
        
        # 2) Standardisation de la taille et des descriptions
        for dd in (
            self.family_dd, self.cost_mode_dd, self.name_dd,
            self.budget_pop_dd, self.wave_dd,
            self.file_dd, self.run_dd
        ):
            dd.layout = widgets.Layout(width="300px")
            # raccourcit un peu la zone de description
            dd.style = {"description_width": "initial"}

        # 1) Deux colonnes : 80% pour les sélecteurs, 20% pour le tableau déroulant
        row1 = widgets.HBox(
            [self.family_dd, self.cost_mode_dd, self.name_dd],
            layout=widgets.Layout(gap="20px")
        )
        row2 = widgets.HBox(
            [self.budget_pop_dd, self.wave_dd, self.file_dd],
            layout=widgets.Layout(gap="20px")
        )
        row3 = widgets.HBox(
            [self.run_dd],
            layout=widgets.Layout(gap="20px")
        )

        left_col = widgets.VBox(
            [row1, row2, row3],
            layout=widgets.Layout(
                display="flex",
                flex="4 1 auto",
                flex_flow="column",
                align_items="flex-start",
                gap="5px"
            )
        )
        right_col = widgets.VBox(
            [self.run_bounds_out],
            layout=widgets.Layout(
                display="flex",
                height="250px",
                flex="1 1 auto",
                align_items="flex-start"
            )
        )

        select_row = widgets.HBox(
            [left_col, right_col],
            layout=widgets.Layout(
                display="flex",
                width="100%",
                gap="20px",
                align_items="flex-start"
            )
        )

        # 2) Ligne du bouton Plot (déjà défini dans OptimizationTab, n'intervient pas ici)
        # plot_row = widgets.HBox([...])

        # 3) Empilement final de cet arborescence + (éventuel) plot_row
        self.widget = widgets.VBox(
            [select_row],  
            layout=widgets.Layout(
                display="flex",
                flex_flow="column",
                width="100%",
                border="1px solid #ccc",
                padding="8px",
                margin="4px 0px",
                gap="10px"
            )
        )


        # 5) Observers en cascade (inchangés)
        self.family_dd.observe(self._refresh_cost_modes,     names="value")
        self.cost_mode_dd.observe(self._refresh_configs,     names="value")
        self.name_dd.observe(self._refresh_budget_pops,      names="value")
        self.budget_pop_dd.observe(self._refresh_wavelengths, names="value")
        self.wave_dd.observe(self._refresh_files,            names="value")
        self.file_dd.observe(self._refresh_runs,             names="value")

        # 6) Lancement de la cascade
        self._refresh_families()

    def _list_subdirs(self, path: Path) -> list[str]:
        return sorted(p.name for p in path.iterdir() if p.is_dir())

    def _list_h5(self, path: Path) -> list[tuple[str,str]]:
        return sorted(
            [(f.name, str(f)) for f in path.glob("*.h5")],
            key=lambda x: x[0]
        )

    def _refresh_families(self, change=None):
        old = self.family_dd.value
        opts = self._list_subdirs(self.base_dir)
        self.family_dd.options = opts
        self.family_dd.value   = old if old in opts else (opts[0] if opts else None)
        self._refresh_cost_modes()

    def _refresh_cost_modes(self, change=None):
        old = self.cost_mode_dd.value
        base = self.base_dir / (self.family_dd.value or "")
        opts = self._list_subdirs(base) if base.is_dir() else []
        self.cost_mode_dd.options = opts
        self.cost_mode_dd.value   = old if old in opts else (opts[0] if opts else None)
        self._refresh_configs()

    def _refresh_configs(self, change=None):
        old = self.name_dd.value
        base = self.base_dir / self.family_dd.value / self.cost_mode_dd.value
        opts = self._list_subdirs(base) if base.is_dir() else []
        self.name_dd.options = opts
        self.name_dd.value   = old if old in opts else (opts[0] if opts else None)
        self._refresh_budget_pops()

    def _refresh_budget_pops(self, change=None):
        old = self.budget_pop_dd.value
        base = self.base_dir / self.family_dd.value / self.cost_mode_dd.value / self.name_dd.value
        opts = sorted(p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("budget")) if base.is_dir() else []
        self.budget_pop_dd.options = opts
        self.budget_pop_dd.value   = old if old in opts else (opts[0] if opts else None)
        self._refresh_wavelengths()

    def _refresh_wavelengths(self, change=None):
        old = self.wave_dd.value
        base = (
            self.base_dir /
            self.family_dd.value /
            self.cost_mode_dd.value /
            self.name_dd.value /
            self.budget_pop_dd.value
        )
        opts = sorted(p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("wavelength_range_")) if base.is_dir() else []
        self.wave_dd.options = opts
        self.wave_dd.value   = old if old in opts else (opts[0] if opts else None)
        self._refresh_files()

    def _refresh_files(self, change=None):
        old = self.file_dd.value
        base = (
            self.base_dir /
            self.family_dd.value /
            self.cost_mode_dd.value /
            self.name_dd.value /
            self.budget_pop_dd.value /
            self.wave_dd.value
        )
        opts = self._list_h5(base) if base.is_dir() else []
        paths = [p for (_, p) in opts]
        self.file_dd.options = opts
        self.file_dd.value   = old if old in paths else (paths[0] if paths else None)
        self._refresh_runs()

    def _refresh_runs(self, change=None):
        old = self.run_dd.value
        runs = list_runs_in_h5(self.file_dd.value) if self.file_dd.value else []
        self.run_dd.options = runs
        # -- ne touche pas à .value si l’utilisateur vient d’agir --
        if not self._user_selecting:
            self.run_dd.value = (
                old if old in runs else (runs[-1] if runs else None)
            )


    def get_selected_file(self) -> str | None:
        return self.file_dd.value
    

    def _on_run_changed(self, change):
        """Affiche Param / Min / Max / Best **+** Fixed dès qu’on sélectionne un run."""
        self._user_selecting = True              # l’utilisateur vient de cliquer
        try:
            self.run_bounds_out.clear_output()
            h5path  = self.file_dd.value
            run_key = change["new"]
            if not h5path or run_key is None:
                return

            data      = read_optimization_hdf5(h5path, run_key=run_key)
            opt_keys  = data["keys"]
            lowers    = data["lowers"]
            uppers    = data["uppers"]
            best_vals = data["best_final"]
            fixed     = data.get("fixed", {})

            # entête à 4 colonnes
            header = "<tr><th>Paramètre</th><th>Min</th><th>Max</th><th>Valeur</th></tr>"
            # ligne métrique
            best_cost    = data["best_cost"]
            metric_value = 1.0 - best_cost
            label_map = {
                "dip"          : "Sensitivity S (ΔR/Δn)",
                "half"         : "Sensitivity S½ (ΔR/Δn)",
                "fixed_lambda" : "Reflectance R(λ₀)",
                "range_lambda" : "Mean reflectance ⟨R⟩",
            }
            metric_label = label_map.get(data["mode"], "Metric")
            metric_row = (
            f"<tr><td><b>{metric_label}</b></td>"
            f"<td></td><td></td>"
            f"<td><b>{metric_value:.3g}</b></td></tr>"
            )

            # lignes pour les paramètres optimisés
            opt_rows = "\n".join(
            f"<tr><td>{k}</td><td>{l:.3g}</td><td>{u:.3g}</td><td>{v:.3g}</td></tr>"
            for k, l, u, v in zip(opt_keys, lowers, uppers, best_vals)
            )

            # lignes pour les paramètres fixés
            fixed_rows = "\n".join(
            f"<tr>"
            f"<td>{k} (fixé)</td>"
            f"<td>{v:.3g}</td>"
            f"<td>{v:.3g}</td>"
            f"<td>{v:.3g}</td>"
            f"</tr>"
            for k, v in fixed.items()
            )

            table_html = f"""
            <div style="max-height:200px; overflow-y:auto; border:1px solid #ccc; padding:4px;">
            <table style="border-collapse: collapse; width:100%;">
                {header}
                {metric_row}
                {opt_rows}
                {fixed_rows}
            </table>
            </div>
            """

            with self.run_bounds_out:
                display(widgets.HTML(table_html))

        finally:
            self._user_selecting = False         # ← rendu la main


# -----------------------------------------------------------------------------#
#  Main widget class                                                           #
# -----------------------------------------------------------------------------#
class OptimizationTab:
    """
    Onglet d’optimisation (widgets + logique de calcul).
    """


    # ------------------------------------------------------------------#
    #  Config selector dédié à l’onglet Optimisation                    #
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #  Config selector (identique à Simulation, sans synchro)           #
    # ------------------------------------------------------------------#
    def _build_opt_config_selector(self):
        # dictionnaires   {cfg_name: Checkbox}
        self.opt_cfg_check = {}
        self.opt_dn_check  = {}

        # --- bouton toggle (ouvre/ferme la liste) ---
        self.opt_toggle_btn = widgets.ToggleButton(
            description="Select Configs & Δn",
            value=True,                      # ouvert par défaut
            icon="caret-up",
            button_style="warning",
            layout=widgets.Layout(width="520px")
        )
        self.opt_toggle_btn.observe(self._toggle_config_list, names="value")

        # --- “Tout sélectionner” ---
        self.opt_select_all_cfg_btn = widgets.Button(
            description="Tout sélectionner Configs", button_style="info",
            layout=widgets.Layout(margin="0 5px 5px 0")
        )
        self.opt_select_all_dn_btn  = widgets.Button(
            description="Tout sélectionner Δn",     button_style="info",
            layout=widgets.Layout(margin="0 0 5px 0")
        )
        self.opt_select_all_cfg_btn.on_click(self._toggle_all_cfg)
        self.opt_select_all_dn_btn.on_click(self._toggle_all_dn)

        # --- lignes Config / Δn ---
        rows = []
        for cfg_name in _load_available_configs():
            chk_cfg = widgets.Checkbox(value=False, description=cfg_name, indent=False)
            chk_dn  = widgets.Checkbox(value=False, description="Δn", indent=False,
                                       layout=widgets.Layout(width="46px"))

            chk_dn.observe(self._update_dn_widgets_state, names="value")


            # callbacks internes
            chk_cfg.observe(self._refresh_parametrization, names="value")
            chk_cfg.observe(self._opt_refresh_custom_modes, names="value")

            self.opt_cfg_check[cfg_name] = chk_cfg
            self.opt_dn_check[cfg_name]  = chk_dn
            rows.append(widgets.HBox([chk_cfg, chk_dn], layout=widgets.Layout(gap="5px")))

        # conteneur scrollable
        visible = min(len(rows), 10)      # 10 lignes max avant scroll
        self.opt_config_list = widgets.VBox(
            [widgets.HBox([self.opt_select_all_cfg_btn, self.opt_select_all_dn_btn],
                          layout=widgets.Layout(gap="10px")),
             *rows],
            layout=widgets.Layout(
                width="500px",
                height=f"{30 + visible*30}px",
                overflow_y="auto",
                border="1px solid lightgray",
                padding="5px",
                display="none"            # affiché par le toggle
            )
        )

        # assembly final
        self.opt_config_selector = widgets.VBox(
            [self.opt_toggle_btn, self.opt_config_list],
            layout=widgets.Layout(padding="5px")
        )

        # Compatibilité : certaines parties du code attendent opt_cfg_box
        self.opt_cfg_box = self.opt_config_selector




    def _rebuild_opt_config_selector(self, *_):
        """Recharge la liste des configurations puis reconstruit les check-boxes
        en conservant les cases déjà cochées quand c’est possible."""
        prev_sel = {n: cb.value  for n, cb in self.opt_cfg_check.items()}
        prev_dn  = {n: cb.value  for n, cb in self.opt_dn_check.items()}
        self._build_opt_config_selector()              # recrée tout

        # restaure l’état (quand les noms existent toujours)
        for name, cb in self.opt_cfg_check.items():
            cb.value = prev_sel.get(name, False)
        for name, cb in self.opt_dn_check.items():
            cb.value = prev_dn.get(name, True)

        # met à jour les panneaux dépendants
        self._refresh_parametrization()
        self._opt_refresh_custom_modes()





    # ------------------------------------------------------------------#
    #  Construction / UI                                                 #
    # ------------------------------------------------------------------#
    def __init__(self, sim_obj: SimulationTab) -> None:
        # ─────────────────────────  saved references  ──────────────────────────
        self.sim                = sim_obj
        self.json_combined_path = str(json_combined_path)

        # runtime-state flags/handles
        self._is_running   = False         
        self._cancelled    = False
        
        self._worker_thread = None

        # background process / queue (may still be used by DE_general)
        ctx                 = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        self._de_process    = None          # type: mp.Process | None
        self._result_queue  = ctx.Queue()

        # ------------------------------------------------------------------ #
        #   STATIC UI — all widgets created **once** and kept forever
        # ------------------------------------------------------------------ #

        # Bounds table container (left column later), scrollable au-delà de 7 lignes
        self.bounds_box = widgets.VBox(
            layout=widgets.Layout(
                border="1px solid #ccc",
                padding="8px",
                gap="5px",
                max_height="210px",   # ~7 lignes × 30px
                overflow_y="auto"
            )
        )



        #  ——————————————————————————————————————————————————————————
        # Contrôles pour bornes communes
        #  ——————————————————————————————————————————————————————————
        self.common_low_w  = widgets.FloatText(
            value=0.0, description="Low all:", layout=widgets.Layout(width="50")
        )
        self.common_up_w   = widgets.FloatText(
            value=1000.0, description="Up all:",  layout=widgets.Layout(width="50")
        )
        self.apply_bounds_btn = widgets.Button(
            description="Apply to all", button_style="primary",
            tooltip="Apply common bounds to all parameters")
        self.apply_bounds_btn.on_click(self._apply_common_bounds)

        self.common_bounds_controls = widgets.HBox(
            [
                widgets.Label("Common bounds:", layout=widgets.Layout(width="200px")),
                self.common_low_w,
                self.common_up_w,
                self.apply_bounds_btn
            ],
            layout=widgets.Layout(gap="50px", align_items="center")
        )
        #  ——————————————————————————————————————————————————————————




        # Families / cost modes selector (small widgets; unchanged code omitted)
        self.family_dd  = widgets.Dropdown(
            options=['multi_layer', 'gap_plasmon_resonator'],
            value='multi_layer', description='Family:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='220px')
        )




        # ------------------------------------------------------------------ #
        #     Sélecteur du mode de coût (indépendant de l’onglet Simulation) #
        # ------------------------------------------------------------------ #
        self.cost_mode = widgets.RadioButtons(
            options=[('Dip (ΔR/Δn)',  'dip'),
                    ('FWHM (half)',  'half'),
                    ('λ₀ fixe',      'fixed_lambda'),
                    ('λ range','range_lambda')],
            value='dip',
            
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )

        self._build_opt_config_selector()
        self._toggle_config_list({'new': True})

        self._cfg_watcher = start_watcher(
            path=str(CONFIG_LIST_JSON),
            callback=lambda *_: self._rebuild_opt_config_selector(),
            extensions=[".json"],
            recursive=False,
        )

        # ─── RCWA modes (Optimisation) ──────────────────────────────────────────
        self.opt_mode_selection = widgets.RadioButtons(
            options=[('Fixe', 'fixed'),
                    ('Custum', 'custom'),
                    ('Auto', 'auto')],
            value='fixed',
            description='RCWA modes (opt)',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='220px')
        )

        self.opt_fixed_n_mod = widgets.IntText(          # visible si 'fixed'
            value=5, min=1,
            description='n_mod',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='120px')
        )

        self.opt_custom_modes_box = widgets.VBox(
            value=5, min=1,
            description='n_mod',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
            )       
        
        self._opt_custom_n_mod_inputs: dict[str, widgets.IntText] = {}


        # ── callbacks ----------------------------------------------------------------
        # 1)  Fixe / Perso / Auto  →   afficher / masquer le IntText « n_mod fixe »
        self.opt_mode_selection.observe(self._toggle_opt_fixed_n_mod,  names='value')
        # 2)  + mise à jour des champs « n_mod personnalisés »
        self.opt_mode_selection.observe(self._opt_refresh_custom_modes, names='value')    
                
        for cb in self.opt_cfg_check.values():           # quand on (dé)coche une config
            cb.observe(self._opt_refresh_custom_modes, names='value')
        
        # états initiaux
        self._toggle_opt_fixed_n_mod()
        self._opt_refresh_custom_modes()


        # Widgets spécifiques au mode 'fixed_lambda' ou 'range_lambda'
        self.lambda0_w = widgets.FloatText(
            value=600, description="λ₀ (nm):", layout=widgets.Layout(width="400px")
        )
        self.band_min_w = widgets.FloatText(
            value=650,
            description="λmin:",
            layout=widgets.Layout(width="400px"),
        )
        self.band_max_w = widgets.FloatText(
            value=750,
            description="λmax:",
            layout=widgets.Layout(width="400px"),
        )
        self.band_box = widgets.HBox(
            [self.band_min_w, self.band_max_w],
            layout=widgets.Layout(gap="200px"),
        )

        # --- copie visuelle du FloatText Δn ---
        sim_dn = self.sim.delta_n_widget        # widget d'origine (Simulation)

        self.delta_n_widget = widgets.FloatText(
            value      = sim_dn.value,
            description= sim_dn.description,
            step       = sim_dn.step,
            layout     = sim_dn.layout  
        )

        # --- copie visuelle du sélecteur de couche(s) ---
        sim_layer = self.sim.layer_selector     # peut être IntText, SelectMultiple, …

        if isinstance(sim_layer, widgets.SelectMultiple):
            self.layer_selector = widgets.SelectMultiple(
                options     = sim_layer.options,
                value       = sim_layer.value,
                description = sim_layer.description,
                layout      = sim_layer.layout    # réutilisation
            )
        else:
            self.layer_selector = type(sim_layer)(
                value       = sim_layer.value,
                min         = getattr(sim_layer, "min", None),
                max         = getattr(sim_layer, "max", None),
                step        = getattr(sim_layer, "step", 1),
                description = sim_layer.description,
                layout      = sim_layer.layout    # idem
            )




        self._toggle_CF_mode_widgets({"new": self.cost_mode.value})      # état initial
        self.cost_mode.observe(self._toggle_CF_mode_widgets, names="value")

        # Widget arborescent pour filtrer les fichiers HDF5
        self.summary_opt_dir = summary_opt_dir

        self.opt_file_arbo = OptimizationFileArboWidget(str(self.summary_opt_dir))
        

        # Watcher pour maj auto de la liste en entière
        self._observer = start_watcher(
            path=str(self.summary_opt_dir),
            callback=lambda *_: self.opt_file_arbo._refresh_families(),
            extensions=[".h5"],
            recursive=True,
        )


        # appel initial pour peupler
        self.opt_file_arbo._refresh_families()

        # Contrôles DE
        self.budget_w = widgets.IntText(value=100, description="Budget")
        self.pop_w = widgets.IntText(value=30, description="Population")
        self.run_btn = widgets.Button(
            description="Run DE", button_style="primary"
        )

        # bouton pour annuler l’optimisation
        self.cancel_btn = widgets.Button(
            description="Cancel",
            button_style="warning"
        )
        self.cancel_btn.disabled = True  # désactivé tant qu'aucune optimisation en cours
        self.cancel_btn.on_click(self._on_cancel)


        self.out = widgets.Output(layout={"border": "1px solid #888"})


        # Bouton de tracé des résultats
        self.plot_btn = widgets.Button(
            description="Plot results", button_style="info"
        )
        self.plot_btn.on_click(self.plot_optimization_results)

        # personnalise la taille du bouton Plot pour qu'il soit plus voyant
        self.plot_btn.layout = widgets.Layout(width="80%", height="40px")

        # crée une 4ᵉ ligne centrée contenant ce bouton
        plot_row = widgets.HBox(
            [self.plot_btn],
            layout=widgets.Layout(
                justify_content="center",
                width="100%",
                padding="10px 0"
            )
        )



        # -------------  PERMANENT runtime widgets (status + progress) -------------
        self._status_html  = widgets.HTML("")          # line that will show the text
        self._progress_bar = widgets.FloatProgress(
            value=0, min=0, max=1, description="0 %",
            bar_style="info",
            layout=widgets.Layout(width="100%", display="none")  # hidden at start
        )
        self.runtime_box   = widgets.VBox(
            [self._status_html, self._progress_bar],
            layout=widgets.Layout(gap="4px")
        )



        # ------------------------------------------------------------------ #
        #   Onglets Parametrization / Queue pane à droite                    #
        # ------------------------------------------------------------------ #
        # A) File de jobs
        self.job_queue = []  # liste de dicts {"config", "bounds", "cf_mode", "lambda", "budget", "pop", "status"}

        # B) Onglet Parametrization (sous-onglets dynamiques)
        self.param_tabs = widgets.Tab()

        # C) Onglet Queue pane
        self.queue_box = widgets.VBox(
            layout=widgets.Layout(
                border="1px solid #ccc",
                overflow_y="auto",   # scrolling vertical quand trop haut
                height="100%"        # occupe tout l’espace parent
            )
        )
        self.run_all_btn    = widgets.Button(description="Run all ▶",    button_style="primary")
        self.cancel_all_btn = widgets.Button(description="Cancel all ⏹", button_style="warning")

        # supprime toute la queue d’un coup -----------------
        self.delete_all_btn = widgets.Button(
            description="Delete all ✖️",
            button_style="danger",
        )

        self.run_all_btn.disabled    = True
        self.delete_all_btn.disabled = True
        self.cancel_all_btn.disabled = True


        # bind des callbacks
        self.run_all_btn.on_click(self._run_all)
        self.cancel_all_btn.on_click(self._cancel_all)        
        self.delete_all_btn.on_click(self._delete_all)



        queue_pane = widgets.VBox(
            [
                widgets.HTML("<b>Job Queue</b>"),
                self.queue_box,
                widgets.HBox(
                    [self.run_all_btn, self.cancel_all_btn, self.delete_all_btn],
                    layout=widgets.Layout(gap="10px")
                ),
            ],
            layout=widgets.Layout(
                padding="10px",
                border="1px solid #ccc",
                height="auto",  # s’adapte à la hauteur de l’écran
            )
        )


        # D) Tab principal (Parametrization + Queue pane)
        self.main_tabs = widgets.Tab(children=[self.param_tabs, queue_pane])
        self.main_tabs.set_title(0, "Parametrization")
        self.main_tabs.set_title(1, "Queue pane")

        # E) Observer les cases à cocher de config pour rafraîchir Parametrization
        for cb in self.opt_cfg_check.values():
            # on enlève l’ancien observer qui appelait update_optimization
            try:
                cb.unobserve(self.update_optimization, names="value")
            except Exception:
                pass
            cb.observe(self._refresh_parametrization, names="value")

        # F) Appel initial pour construire l’onglet Parametrization
        self._refresh_parametrization()
        # on affiche la queue vide au démarrage
        self._refresh_queue()





        # 1) Colonne de gauche (Simulation)
        left_col = widgets.VBox([
            self.opt_config_selector,  # toggle + refresh + cases Config/Δn

            widgets.HTML(value="<b>Spectrum (nm)</b>"),
            widgets.HBox(
                [self.sim.sim_lambda_min, self.sim.sim_lambda_max, self.sim.sim_n_points],
                layout=widgets.Layout(gap='10px')
            ),

            widgets.HTML("<b>RCWA Fourier modes</b>"),
            self.opt_mode_selection,
            widgets.HBox([self.opt_fixed_n_mod]),
            self.opt_custom_modes_box,

            widgets.HTML(value="<b>Δn & Layers</b>"),
            widgets.HBox(
                [self.delta_n_widget, self.layer_selector],
                layout=widgets.Layout(gap='10px')
            ),

        ], layout=widgets.Layout(width='48%', padding='10px'))

        # 2) Colonne de droite (Optimisation sous forme d’onglets)
        self.main_tabs.layout = widgets.Layout(
            width='55%',
            padding='10px'
        )
        right_col = self.main_tabs


        # 3) Bas de page : arborescence puis bouton Plot centré
        plot_row = widgets.HBox(
            [self.plot_btn],
            layout=widgets.Layout(
                justify_content="center",
                width="50%",
                padding="10px 0"
            )
        )

        bottom_controls = widgets.VBox(
            [self.opt_file_arbo.widget, plot_row],
            layout=widgets.Layout(
                width="100%",
                margin="10px 0",
                gap="10px"
            )
        )

        # 4) Zone de sortie pour le plot (plein width)
        plot_area = self.out

        # 5) Assemblage final
        self.ui = widgets.VBox([
            widgets.HBox([left_col, right_col],
                        layout=widgets.Layout(justify_content='space-between')),
            bottom_controls,
            self.runtime_box,
            plot_area
        ], layout=widgets.Layout(padding='10px'))


        # surveille combined_materials.json
        self._json_watcher = start_watcher(
            path=str(json_combined_path),
            callback=lambda *_: self.sim.config_refresh_btn.click(),
            extensions=[".json"],
            recursive=False,
        )





        # s'assurer que l'attribut existe, même si update_optimization retourne tôt
        self.param_widgets: Dict[str, Dict[str, widgets.Widget]] = {}
        # Met à jour la liste de paramètres
        self.update_optimization()






        if self.param_widgets:          # au moins une config cochée pour run DE
            self._update_run_button_state()
        else:
            self.run_btn.disabled = True

        # Callbacks
        self.run_btn.on_click(self._on_run)

        #  observe chaque checkbox
        def _attach_observers():
            for w in self.param_widgets.values():
                w['opt'].observe(lambda _: self._update_run_button_state(),
                                names='value')
        _attach_observers()



        self._update_dn_widgets_state()

    # ------------------------------------------------------------------#
    #  UI helpers                                                       #
    # ------------------------------------------------------------------#
    
    
    def _update_dn_widgets_state(self, *_):
        """
        Active (ou grise) les widgets Δn et layer_selector.

        • Ils ne sont utiles que si :
            – le mode de coût est 'dip' ou 'half',  ET
            – au moins une config a sa case « Δn » cochée.
        """
        uses_dn_mode   = self.cost_mode.value in ("dip", "half")
        any_dn_checked = any(cb.value for cb in self.opt_dn_check.values())
        enable         = uses_dn_mode and any_dn_checked

        self.delta_n_widget.disabled = not enable
        self.layer_selector.disabled = not enable
    





    def _toggle_config_list(self, change):
        show = "block" if change["new"] else "none"
        self.opt_config_list.layout.display = show
        self.opt_toggle_btn.icon = "caret-up" if change["new"] else "caret-down"

    def _toggle_all_cfg(self, _=None):
        all_on = all(cb.value for cb in self.opt_cfg_check.values())
        for cb in self.opt_cfg_check.values():
            cb.value = not all_on     # inverse l’état

    def _toggle_all_dn(self, _=None):
        all_on = all(cb.value for cb in self.opt_dn_check.values())
        for cb in self.opt_dn_check.values():
            cb.value = not all_on



    def _toggle_opt_fixed_n_mod(self, change=None):
        # si le RadioButtons vaut 'fixed'  → on affiche le IntText
        # sinon                            → on le cache
        self.opt_fixed_n_mod.layout.display = (
            ''    if self.opt_mode_selection.value == 'fixed'
            else 'none'
        )



    def _opt_get_n_modes_for(self, cfg_name: str) -> int:
        """
        Nombre de modes RCWA à utiliser pour *cfg_name* selon le choix
        de l’onglet Optimisation (fixed / custom / auto).
        Le mode 'auto' lit directement summary_convergence/convergence_results.json
        pour être indépendant de l’onglet Simulation.
        """
        sel = self.opt_mode_selection.value

        # 1)  Fixed
        if sel == 'fixed':
            return int(self.opt_fixed_n_mod.value)

        # 2)  Custom
        if sel == 'custom':
            it = self._opt_custom_n_mod_inputs.get(cfg_name)
            return int(it.value) if it is not None else int(self.opt_fixed_n_mod.value)

        # 3)  Auto ⇒ on lit summary_convergence
        try:
            conv_json = summary_convergence / "convergence_results.json"
            with open(conv_json, encoding='utf-8') as f:
                master = json.load(f)
            auto_modes = {
                name: r[-1]["optimal_n_mode"]
                for name, r in master.get("configs", {}).items() if r
            }
            return int(auto_modes[cfg_name])
        except (FileNotFoundError, KeyError, ValueError):
            # Fallback : valeur affichée dans le champ n_mod fixe
            return int(self.opt_fixed_n_mod.value)
    
    
    def _opt_refresh_custom_modes(self, *_):
        """Affiche un IntText par config cochée quand 'Personnalisé' est actif."""
        if self.opt_mode_selection.value != 'custom':
            self.opt_custom_modes_box.children = ()
            return

        # quelles configs sont cochées ?
        selected = [name for name, cb in self.opt_cfg_check.items() if cb.value]
        inputs   = []
        for name in selected:
            it = self._opt_custom_n_mod_inputs.get(name)
            if it is None:        # première fois
                it = widgets.IntText(
                    value=self.opt_fixed_n_mod.value,
                    description=name,
                    style={'description_width': 'initial'},
                    layout=widgets.Layout(width='250px')
                )
                self._opt_custom_n_mod_inputs[name] = it
            inputs.append(it)

        self.opt_custom_modes_box.children = tuple(inputs)

        
    
    
    def _on_configs_refreshed(self, _=None):
        """Appelé quand SimulationTab a rechargé les configs."""
        self._rebuild_opt_config_selector()




    def _attach_config_observers(self):
        """(Re)branche update_optimization sur chaque checkbox."""
        for cb in self.opt_cfg_check.values():
            cb.observe(self.update_optimization, names="value")


    def _toggle_CF_mode_widgets(self, change: Dict[str, Any]) -> None:
        """Affiche/masque les widgets selon le mode calcul choisi."""
        m = change["new"]
        self.lambda0_w.layout.display = "" if m == "fixed_lambda" else "none"
        self.band_box.layout.display = "" if m == "range_lambda" else "none"
        uses_dn = (m in ("dip", "half"))             # ← nouveau
        self.delta_n_widget.disabled = not uses_dn

        # Active/désactive les petites cases “Δn” devant chaque config
        for cb in self.opt_dn_check.values():        # existe déjà dans l'objet
            cb.disabled = not uses_dn
            if not uses_dn:
                cb.value = False                     # on décoche proprement

        self._update_dn_widgets_state()

    def close(self) -> None:
        """Explicitly release resources held by the observer."""
        if hasattr(self, "_cfg_watcher") and self._cfg_watcher is not None:
            try:
                self._cfg_watcher.stop()
                self._cfg_watcher.join()
            finally:
                self._cfg_watcher = None



    def __del__(self) -> None:
        self.close()


    def _update_run_button_state(self, *_):
        """
        Active le bouton Run DE si au moins un paramètre est marqué
        'opt', sinon le grise.
        """
        any_selected = any(w['opt'].value for w in self.param_widgets.values())
        self.run_btn.disabled = not any_selected


    def _apply_common_bounds(self, _):
        """
        Applique les valeurs de self.common_low_w / self.common_up_w
        à tous les champs low/up de self.param_widgets.
        """
        low = self.common_low_w.value
        up  = self.common_up_w.value
        for k, widgets_dict in self.param_widgets.items():
            widgets_dict["low"].value = low
            widgets_dict["up"].value  = up


    # ------------------------------------------------------------------#
    #  Callback : lancement DE                                          #
    # ------------------------------------------------------------------#
    def _on_run(self, _):
        """
        Called when the user clicks ‘Run DE’.
        Shows the message first; the progress-bar will be inserted later
        by _check_process when the first result arrives.
        """


        if self._is_running:        # guard against double-click
            return
        self._is_running = True
        self._cancelled  = False
        self.cancel_btn.disabled = False

        # ----------  reset runtime widgets ----------
        self._status_html.value            = (
            "🚀 Optimization is running… (you can Cancel)<br>"
            "The progress-bar will appear after the first evaluation."
        )
        self._status_html.layout.display = ""

        self._progress_bar.layout.display = "none"
        self._progress_bar.value           = 0
        self._progress_bar.description     = "0 %"
        self._progress_bar.bar_style       = "info"

        # ----------  collect UI parameters  ----------
        extra_kwargs = {}

        mode = self.cost_mode.value
        if mode == "fixed_lambda":
            extra_kwargs["fixed_lambda"] = self.lambda0_w.value
        elif mode == "range_lambda":
            extra_kwargs["range_lambda"] = (self.band_min_w.value,
                                            self.band_max_w.value)

        # **même logique que pour la queue** : on capture les valeurs « fixed »
        extra_kwargs["fixed_vals"] = {
            k: w["fixed"].value
            for k, w in self.param_widgets.items()
            if not w["opt"].value
        }

        keys   = [k for k, w in self.param_widgets.items() if w["opt"].value]
        if not keys:
            self._status_html.value = "⚠️ No parameter selected for optimisation."
            self._is_running = False
            self.cancel_btn.disabled = True
            return

        lowers = np.array([self.param_widgets[k]["low"].value for k in keys])
        uppers = np.array([self.param_widgets[k]["up"].value  for k in keys])

        # --- nouvelle sélection de la config ---
        try:
            sel_cfg_name = next(name for name, cb in self.opt_cfg_check.items() if cb.value)
        except StopIteration:
            self._status_html.value = "⚠️ No configuration selected."
            self._is_running = False
            self.cancel_btn.disabled = True
            return

        # ── queue de progression & thread lanceur ──────────
        self._result_queue = q.Queue()
        args = dict(budget=self.budget_w.value,
                    Npop   =self.pop_w.value,
                    lowers =lowers, uppers=uppers,
                    keys   =keys, mode=mode,
                    cfg_name = sel_cfg_name,
                    progress_queue=self._result_queue,
                    **extra_kwargs)



        self._worker_thread = threading.Thread(
            target=self.DE_general, kwargs=args, daemon=True)
        self._worker_thread.start()

        self.cancel_btn.disabled = False
        self._cancelled = False

        # ── boucle de polling sur la queue ─────────────────
        loop = get_ipython().kernel.io_loop
        loop.add_timeout(loop.time() + 0.1, self._check_process)


    @staticmethod
    def _run_de_process(opt_tab: "OptimizationTab",
                        args: dict,
                        queue: mp.Queue) -> None:
        """
        Process fils : exécute DE_general *sans aucun widget/tqdm*,
        en publiant la progression sur `queue`.
        """
        try:
            opt_tab.DE_general(progress_queue=queue, **args)
            # (DE_general se charge d’envoyer un message "DONE")
        except Exception:
            queue.put(("ERROR", traceback.format_exc()))



    def _check_process(self):
        """Poll the queue, update widgets, and reschedule itself."""
        try:
            tag, *payload = self._result_queue.get_nowait()
        except q.Empty:
            tag = None

        # ── first PROG message → on passe en mode “barre visible, texte caché”
        if tag == "PROG" and self._progress_bar.layout.display == "none":
            # masque complètement le HTML
            self._status_html.layout.display = "none"
            # révèle la barre
            self._progress_bar.layout.display = ""

        if tag == "PROG":
            frac = payload[0]
            self._progress_bar.value       = frac
            self._progress_bar.description = f"{int(frac*100)} %"

        elif tag == "DONE":
            conv_best, _, best_final, best_cost = payload
            self._affiche_resultats(conv_best, best_final, best_cost)
            self._is_running = False
            return

        elif tag == "ERROR":
            trace = payload[0]
            self._status_html.value   = f"❌ Optimization aborted:<br><pre>{trace}</pre>"
            self._progress_bar.bar_style = "danger"
            self.cancel_btn.disabled  = True
            self._is_running = False
            return

        # --- keep polling while the thread lives ---
        if self._worker_thread and self._worker_thread.is_alive():
            loop = get_ipython().kernel.io_loop
            loop.add_timeout(loop.time() + 0.1, self._check_process)




    def _affiche_resultats(self, conv_best, best_final, best_cost):
        """Show final results, leave bar green."""
        self.opt_file_arbo._refresh_files()

        vector_txt = np.array2string(best_final, precision=2, separator=', ')
        self._status_html.value = (
            "✅ Optimization ended.<br>"
            f"<b>Best cost&nbsp;:</b> {best_cost:.4g}<br>"
            f"<b>Best vector :</b> {vector_txt}"
        )
        self._progress_bar.value     = 1.0
        self._progress_bar.bar_style = "success"
        self.cancel_btn.disabled     = True






    def _on_cancel(self, _):
        """
        Callback du bouton Cancel :
        – fixe le flag d’annulation
        – termine immédiatement le Pool worker s’il existe
        – affiche le message d’annulation
        """
        if not self._is_running:
            return
        self._cancelled  = True
        self._is_running = False
        self.cancel_btn.disabled = True



        # terminate worker thread (let it die via the flag)
        self._status_html.value   = "❌ Optimization cancelled by user."
        self._progress_bar.bar_style = "danger"



    # ------------------------------------------------------------------#
    #  Génération dynamique des widgets bornes                          #
    # ------------------------------------------------------------------#
    def update_optimization(self, change: Dict[str, Any] | None = None) -> None:
        """
        Reconstruit la table des paramètres (checkbox + bornes) en fonction
        de la **configuration unique** actuellement cochée.
        """
        # ――― Sélectionne seulement les configs dont la check-box existe ―――
        selected_cfgs = [
            cfg for cfg in self.sim.all_configs
            if cfg["config_name"] in self.opt_cfg_check          # évite KeyError
            and self.opt_cfg_check[cfg["config_name"]].value     # case cochée
        ]

        if len(selected_cfgs) != 1:
            self.bounds_box.children = []   # 0 ou >1 config sélectionnée → on vide
            return

        cfg_chosen = selected_cfgs[0]
        geom = cfg_chosen["geometry"]["geometry"]
        
        rows: List[widgets.HBox] = []
        self.param_widgets: Dict[str, Dict[str, widgets.Widget]] = {}

        for k, val in geom.items():
            if val == 0.0:
                continue  # épaisseur nulle → pas optimisé
            low, high = geometry_limits.get(k, (0.0, 0.0))

            chk = widgets.Checkbox(value=True, indent=False, layout={"width": "30px"})
            lbl = widgets.Label(value=k, layout={"width": "150px"})
            lo  = widgets.FloatText(value=low, description="min:", layout={"width": "120px"}, style={"description_width": "40px"})
            hi  = widgets.FloatText(value=high, description="max:", layout={"width": "120px"}, style={"description_width": "40px"})
            fixed = widgets.FloatText(value=val, description="fixed:", layout={"width": "120px"})
            
            # callback pour toggle Low/Up ↔ Fixed
            def _toggle(change, lo=lo, hi=hi, fixed=fixed):
                if change["new"]:
                    lo.layout.display = hi.layout.display = ""
                    fixed.layout.display = "none"
                else:
                    lo.layout.display = hi.layout.display = "none"
                    fixed.layout.display = ""


            chk.observe(_toggle, names="value")
            _toggle({"new": chk.value})

            # on stocke dans param_widgets
            self.param_widgets[k] = {
                "opt":   chk,
                "low":   lo,
                "up":    hi,
                "fixed": fixed
            }
            # on affiche la ligne complète
            rows.append(
                widgets.HBox([chk, lbl, lo, hi, fixed],
                            layout=widgets.Layout(align_items="center", gap="10px"))
            )



            # clear the log Output **only when nothing is running**
            if not self._is_running:
                self.out.clear_output()
                

        self.bounds_box.children = rows
        # branche les observateurs et met à jour -------------
        for w in self.param_widgets.values():               # chaque case 'opt'
            w['opt'].observe(self._update_run_button_state, names='value')

        self._update_run_button_state()                     # recalcul immédiat
        self.out.clear_output()

    # ------------------------------------------------------------------#
    #  Differential Evolution core                                      #
    # ------------------------------------------------------------------#
    def DE_general(
        self,
        *,
        budget: int,
        Npop: int,
        lowers: np.ndarray,
        uppers: np.ndarray,
        keys: List[str],
        cfg_name:str,
        mode: str = "dip",
        fixed_vals: dict[str,float] | None = None,
        n_jobs: int = -1,
        seed: int | None = None,    # Répétabilité
        progress_queue: mp.Queue | None = None,
        cancel_flag: dict | None = None,
        **mode_kw: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Differential Evolution « current-to-best/1/bin » (parallélisé).

        Returns
        -------
        conv_best : np.ndarray
            Meilleure valeur du coût à chaque génération.
        conv_evals : np.ndarray
            Nombre CUMULÉ d’évaluations de la fonction de coût.
        best_final : np.ndarray
            Vecteur optimal après ré-évaluation finale.
        best_cost : float
            Valeur du coût associée à *best_final*.
        cancel_flag : dict | None
            Référence au dict job pour annulation externe.
        """


        # Initialisation RNG & population
        rng = np.random.default_rng(seed)
        if budget < Npop:
            raise ValueError("Le budget doit être ≥ à la taille de la population.")
        Ngen = budget // Npop
        n_params = len(keys)
        pop = lowers + (uppers - lowers) * rng.random((Npop, n_params))
        
        # calcule une seule fois le nombre de modes demandé par l’onglet Opti
        n_modes = self._opt_get_n_modes_for(cfg_name)

        # ------------------------------------------------------------------
        # Pool de process (vrai parallélisme CPU), attaché au cancel_flag
        # ------------------------------------------------------------------
        global _WORKER_SIM
        _WORKER_SIM = self.sim     
        ctx  = mp.get_context("fork" if sys.platform!="win32" else "spawn")
        pool = ctx.Pool()   # un Pool par appel
        
        if cancel_flag is not None:
            cancel_flag["pool"] = pool

        # --- gestion de l’annulation par job ---
        job = cancel_flag
        if job is not None:
            job["pool"] = pool


        if fixed_vals is None:
            fixed_vals = {}

        # ------------------------------------------------------------------
        # Choisit si le mode a besoin d'un Δn (>0) (Dip & Half seulement)
        # ------------------------------------------------------------------
        need_dn = mode in ("dip", "half")        # True ⟹ on utilisera delta_n_val

        if need_dn:
            delta_n_val = self.delta_n_widget.value
            if not self.opt_dn_check[cfg_name].value or delta_n_val <= 0:
                raise ValueError("Δn doit être défini (>0) pour les modes 'dip' et 'half'.")
        else:
            delta_n_val = None                  # explicite : on n’enverra rien




        try:
            # ─────────────────  Évaluation initiale  ──────────────────
            sel_layers_val = (list(self.layer_selector.value)
                            if isinstance(self.layer_selector.value, (list, tuple))
                            else [int(self.layer_selector.value)])
            args0 = [
                (i, pop[i], keys, cfg_name, mode, mode_kw, fixed_vals,
                n_modes, delta_n_val, sel_layers_val)
                for i in range(Npop)
            ]

            cf = np.empty(Npop)
            for idx, val in pool.imap_unordered(cost_worker, args0, chunksize=1):
                if self._cancelled:
                    pool.terminate()
                    raise OptimizationCancelled()
                cf[idx] = val



            conv_best  = np.zeros(Ngen)
            conv_evals = np.arange(1, Ngen+1)*Npop
            best_after_eval: List[float] = []
            F1, F2, cr = 0.9, 0.8, 0.8

            # 3) Boucle DE avec barre de progression et annulation

            for g in range(Ngen):
                # si on a demandé l’annulation sur ce job, on stoppe tout de suite
                if job is not None and job["cancel_flag"]:
                    pool.terminate()
                    raise OptimizationCancelled()
                
                if self._cancelled:
                    pool.terminate()
                    raise OptimizationCancelled()

                # ───────────── mutation / crossover → z_list ─────────────
                z_list: list[tuple[int, np.ndarray]] = []
                for p in range(Npop):
                    a, b, c = pop[rng.choice(Npop, 3, replace=False)]
                    best_ind = pop[np.argmin(cf)]
                    y = c + F1 * (a - b) + F2 * (best_ind - c)
                    mask = rng.random(n_params) < cr
                    if not mask.any():
                        mask[rng.integers(n_params)] = True
                    z = np.where(mask, y, pop[p])
                    z = np.clip(z, lowers, uppers)
                    z_list.append((p, z))          # ← on garde l’index du parent

                # ───────────── évaluation parallèle des enfants ───────────
                sel_layers_val = (list(self.layer_selector.value)
                                if isinstance(self.layer_selector.value, (list, tuple))
                                else [int(self.layer_selector.value)])

                args_child = [
                    (i, z, keys, cfg_name, mode, mode_kw, fixed_vals,
                    n_modes, delta_n_val, sel_layers_val)
                    for (i, z) in z_list
                ]

                cfz = np.empty(Npop)
                for idx, val in pool.imap_unordered(cost_worker, args_child, chunksize=1):
                    if job is not None and job["cancel_flag"]:
                        pool.terminate()
                        raise OptimizationCancelled()
                    if self._cancelled:
                        pool.terminate()
                        raise OptimizationCancelled()
                    cfz[idx] = val

                # ───────────── 3. sélection ─────────────────────────────────
                for (i, z) in z_list:            # i = index du parent
                    if cfz[i] < cf[i]:
                        pop[i], cf[i] = z, cfz[i]


                best_after_eval.append(cf.min())
                conv_best[g] = cf.min()

                if progress_queue is not None:
                            progress_queue.put(("PROG",
                                                (g + 1) / Ngen,      # fraction 0-1
                                                float(cf.min())))     # meilleur coût courant
                            

            # Ré-évaluation finale
            sel_layers_val = (list(self.layer_selector.value)
                            if isinstance(self.layer_selector.value, (list, tuple))
                            else [int(self.layer_selector.value)])

            argsf = [
                (i, pop[i], keys, cfg_name, mode, mode_kw, fixed_vals,
                n_modes, delta_n_val, sel_layers_val)
                for i in range(Npop)
            ]

            cf_final = np.empty(Npop)
            for idx, val in pool.imap_unordered(cost_worker, argsf, chunksize=1):
                cf_final[idx] = val


            best_final = pop[np.argmin(cf_final)]
            best_cost  = cf_final.min()
            


            # Tracé du spectre optimal + sauvegarde HDF5
            lam = np.linspace(self.sim.sim_lambda_min.value,
                              self.sim.sim_lambda_max.value,
                              self.sim.sim_n_points.value)
            
            # on reprend la même config que celle passée au worker
            orig_cfg = next(c for c in self.sim.all_configs
                            if c["config_name"] == cfg_name)
            
            cfg = deepcopy(orig_cfg)          # copie isolée, propre à CE thread            

            # calcul via cost() (même pipeline que dans la pool)
            #R_via_cost = 1.0 - self.sim.cost(best_final, keys, mode=mode, **mode_kw)


            # —–––––––– D'abord, ceux que l'utilisateur a vraiment fixés
            for k, v in (fixed_vals or {}).items():
                # on ne touche qu'aux paramètres *non* optimisés
                if k not in keys:
                    cfg["geometry"]["geometry"][k] = float(v)

            # —–––––––– Ensuite, on écrase (ou confirme) *toujours* les optimisés
            for xi, k in zip(best_final, keys):
                cfg["geometry"]["geometry"][k] = float(xi)


    
            wave  = {"angle": 0, "polarization": 1}

            Rup, Rdown, _ = run_simulation_one_combo(lam, wave, n_modes, cfg, self.json_combined_path)


            # —–––––––– DEBUG
            #R_via_final = float(np.interp(mode_kw.get("fixed_lambda", self.sim.lambda0_in.value), lam, Rup))
            # with self.out:
            #     # on vide l'ancien contenu
            #     self.out.clear_output(wait=True)
            #     # on affiche nos diagnostics
            #     print(" keys optimisés :", keys)
            #     print(" best_final     :", best_final)
            #     print(" fixed_vals     :", fixed_vals)
            #     print("R via cost()    :" , R_via_cost)
            #     print("R via final sim :" , R_via_final)
            #     print("Δ =", R_via_cost - R_via_final)
            #     print(" géométrie finale :")
            #     for k in keys:
            #         print(f"   {k} = {cfg['geometry']['geometry'][k]}")


            Rup, Rdown, _ = run_simulation_one_combo(
                lam, wave, n_modes, cfg, self.json_combined_path
            )
            Rup   = np.asarray(Rup, float)
            Rdown = np.asarray(Rdown, float)

            config_name = cfg_name      # cohérent avec le reste du run            
            fam = 'gap_plasmon_resonator' if mode in ('dip','half') else 'multi_layer'

            fixed_lambda_val = mode_kw.get("fixed_lambda", None)

            # Sauvegarde complète du run d’optimisation dans un fichier HDF5
            save_optimization_hdf5(
                notebook_dir=str(BASE_NOTEBOOKS),   # Dossier racine où créer le .h5
                family=fam,                       # on force ici la bonne famille
                cost_mode=mode,                       # dip / half / fixed_lambda / range_lambda
                # — méta-données pour filtrage futur —
                config_name=config_name,            # Structure optimisée (pour comparer uniquement les runs compatibles)

                # — paramètres DE —
                budget=budget,                      # Budget d’évaluations
                Npop=Npop,                          # Taille de population
                
                wavelength_range=(self.sim.sim_lambda_min.value, self.sim.sim_lambda_max.value),
                n_modes=n_modes,
                fixed_vals=fixed_vals,
                # — espace de recherche —
                keys=keys,                          # Paramètres optimisés
                lowers=lowers,                      # Bornes inf.
                uppers=uppers,                      # Bornes sup.

                # — suivi de la convergence —
                conv_best=conv_best,                # Best cost par génération
                conv_evals=conv_evals,              # Best cost par évaluation

                # — état final de la population —
                cf_final=cf_final,                  # Coûts de la dernière population

                # — meilleurs individus —
                best=pop[np.argmin(cf_final)],      # Meilleur avant dernière sélection
                best_final=best_final,              # Meilleur après rééval finale
                best_cost=best_cost,                # Coût de best_final
                fixed_lambda=fixed_lambda_val,
                best_after_eval=np.asarray(best_after_eval),  # Snapshot du best à t instant

                # — contexte de la métrique —
                mode=mode,                          # 'dip' ou 'half'

                # — spectre du design optimal —
                lam=lam,                            # Grille λ
                Rup=Rup,                            # Spectre R_up
                Rdown=Rdown,                        # Spectre R_down
                geometry=cfg["geometry"]["geometry"],
            )

            # on rafraîchit la liste des runs disponibles
            get_ipython().kernel.io_loop.add_callback(self.opt_file_arbo._refresh_runs)
            
            if progress_queue is not None:
                    progress_queue.put(("DONE",
                                        conv_best, conv_evals, best_final, best_cost))
            return conv_best, conv_evals, best_final, best_cost


        finally:
            # ferme uniquement CE pool
            pool.close()
            pool.join()

    # ------------------------------------------------------------------#
    #  Plot HDF5 results                                                #
    # ------------------------------------------------------------------#
    def plot_optimization_results(self, _=None) -> None:
        """
        Trace : convergence, consistency, bar des paramètres, spectre final.
        """
        self.opt_file_arbo._refresh_files()

        h5file = self.opt_file_arbo.get_selected_file()
        run_key = self.opt_file_arbo.run_dd.value

        if h5file is None:
            raise RuntimeError("Aucun fichier HDF5 sélectionné.")

        # 2) Lecture du run
        data = read_optimization_hdf5(h5file, run_key=run_key)

        ref_keys    = tuple(data['keys'])
        ref_lowers  = np.asarray(data['lowers'])
        ref_uppers  = np.asarray(data['uppers'])
        ref_cfg     = data['config_name']
        ref_budget  = data['budget']
        ref_Npop    = data['Npop']
        ref_mode    = data['mode']

        # On récupère aussi family & cost_mode depuis le chemin
        rel = Path(h5file).relative_to(self.summary_opt_dir)
        ref_family, ref_cost_mode = rel.parts[0], rel.parts[1]

        # 3) Balayage de tous les fichiers + tous les runs
        all_best = []
        for fpath in list_optimization_files(str(self.summary_opt_dir)):
            # n’applique que si même family/cost_mode
            rel = Path(fpath).relative_to(self.summary_opt_dir)
            fam, cost = rel.parts[0], rel.parts[1]
            if fam != ref_family or cost != ref_cost_mode:
                continue

            for rk in list_runs_in_h5(fpath):
                dat = read_optimization_hdf5(fpath, run_key=rk)

                # Compare les métadonnées inchangées
                if (
                    dat['budget']    == ref_budget and
                    dat['Npop']      == ref_Npop   and
                    dat['mode']      == ref_mode   and
                    dat['config_name']== ref_cfg   and
                    tuple(dat['keys']) == ref_keys and
                    np.allclose(dat['lowers'], ref_lowers) and
                    np.allclose(dat['uppers'], ref_uppers)
                ):
                    all_best.append(float(dat['best_cost']))

        all_best = np.sort(all_best)


        keys = data["keys"]
        best_vec = data["best_final"]     # Meilleur structure après réévaluation final
        conv_best = data.get("conv_best", data.get("best_after_eval"))    # Meilleur coût à chaque génération

        if data.get("n_modes") is not None:
            forced_n_modes = int(data["n_modes"])
        else:
            forced_n_modes = self.sim._get_n_modes_for(ref_cfg)


        # --------------------------- FIGURE ---------------------------- #
        # on crée un layout 2×2 qui prend toute la largeur
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
            2, 2, figsize=(15, 8), 
            gridspec_kw={"wspace":0.1, "hspace":0.3},
            constrained_layout=True    # active un ajustement automatique
        )
        # supprimez tout import de matplotlib.gridspec et fig.subplots_adjust

        # 1) Convergence
        ax0.plot(range(1, len(conv_best)+1), conv_best, marker='.')
        ax0.set_title("DE convergence curve")
        ax0.set_xlabel("Iterations")
        ax0.set_ylabel("Cost")
        ax0.grid(True)

        # 2) Consistency
        if len(all_best) >= 2:
            ax1.plot(all_best, marker='o')
            ax1.set_title("Consistency curve (all compatible runs)")
        else:
            ax1.text(0.5, 0.5,
                    " ≥ 2 compatibles runs needed to plot consistency curve",
                    ha='center', va='center', transform=ax1.transAxes)
        ax1.set_xlabel("Best runs (sorted)")
        ax1.set_ylabel("Cost")

        # -----------------------------------------------------------------
        # 3)  Schéma dynamique des couches avec les *best parameters*
        # -----------------------------------------------------------------
        # --- juste après avoir lu le fichier HDF5 ---
        keys      = data["keys"]
        best_vec  = data["best_final"]
        fixed_vals= data.get("fixed", {})

        # fig, ax definitions déjà présents ...
        ax_geom = ax2     # par exemple le 3ᵉ subplot
        
        geo_ref = data.get("geometry", {})
        plot_geometry_static_from_run(
            ax_geom,
            keys,
            best_vec,
            fixed_vals,
            default_geom=geo_ref,
            ax_offset=(-0.18, -0.1)   
        )



        # 4) Spectrum
        lam, Rup, Rdown = None, None, None
        if "spectra" in data:
            lam = data["spectra"]["wavelength"]
            Rup = data["spectra"]["Rup"]
            Rdown = data["spectra"]["Rdown"]

        if lam is not None:
            ax3.plot(lam, Rup, label="Rup")
            lam0 = data.get("fixed_lambda", np.nan)

            if Rdown is not None:
                ax3.plot(lam, Rdown, linestyle='--', label="Rdown")
        ax3.set_title("Best config spectrum")
        ax3.set_xlabel("λ (nm)")
        ax3.set_ylabel("Reflectance")
        ax3.legend()
        ax3.grid(True)

        # ------------------------------------------------------------------
        # repère éventuel λ0 (mode fixed_lambda uniquement)
        # ------------------------------------------------------------------
        lam0 = data.get("fixed_lambda", None)           # float ou None

        if data["mode"] == "fixed_lambda" and lam0 is not None \
        and lam is not None and Rup is not None and np.isfinite(lam0):
            R0 = float(np.interp(lam0, lam, Rup))
            ax3.scatter([lam0], [R0], s=60, color="red", zorder=5)
            ax3.axvline(lam0, ls=":", lw=1, color="red")
            ax3.text(lam0, R0, f"  R(λ₀) = {R0:.3f}", va="bottom", color="red")

        # ------------------------------------------------------------------
        # debug_html  ✨  nouveau rendu modernisé
        # ------------------------------------------------------------------
        if lam0 is not None and lam is not None and np.isfinite(lam0):
            r_up   = float(np.interp(lam0, lam, Rup))
            r_down = float(np.interp(lam0, lam, Rdown)) if Rdown is not None else None
        else:
            r_up = r_down = float("nan")

        geom_items = list(data.get("geometry", {}).items())          # pile simulée
        kv_pairs   = list(zip(data["keys"], data["best_final"]))     # paramètres opti

        debug_html = f"""
        <style>
        .debug-box {{
        font-family: Consolas, monospace;
        font-size: 12px;
        line-height: 1.3;
        }}
        .debug-box h4            {{margin:4px 0 2px; font-size:13px; color:#1565C0;}}
        .debug-box table         {{border-collapse:collapse; width:100%;}}
        .debug-box th, .debug-box td {{
        border:1px solid #ddd; padding:2px 4px; text-align:left; white-space:nowrap;
        }}
        .debug-box tbody tr:nth-child(odd) {{background:#fafafa;}}
        </style>

        <div class="debug-box">

        <h4>Run summary</h4>
        <table>
        <tr><th>run_key</th>          <td>{run_key}</td></tr>
        <tr><th>best_cost</th>        <td>{data['best_cost']:.6f}</td></tr>
        <tr><th>1 - best_cost</th>    <td>{1-data['best_cost']:.6f}</td></tr>
        <tr><th>λ range (nm)</th>     <td>{lam[0]:.1f} – {lam[-1]:.1f} ({len(lam)})</td></tr>
        <tr><th>λ₀</th>               <td>{lam0}</td></tr>
        <tr><th>R_up(λ₀)</th>         <td>{r_up:.6f}</td></tr>
        <tr><th>R_down(λ₀)</th>       <td>{r_down:.6f}</td></tr>
        </table>

        <details open>
        <summary><b>Stack sent to RCWA ({len(geom_items)} layers)</b></summary>
        <div style="overflow-x:auto;">
            <table>
            <thead><tr><th>Layer</th><th>Thickness (nm)</th></tr></thead>
            <tbody>
                {''.join(f'<tr><td>{k}</td><td>{v:.3f}</td></tr>' for k,v in geom_items)}
            </tbody>
            </table>
        </div>
        </details>

        <details>
        <summary><b>Optimised parameters ({len(kv_pairs)})</b></summary>
        <div style="overflow-x:auto;">
            <table>
            <thead><tr><th>Key</th><th>Optimised (nm)</th></tr></thead>
            <tbody>
                {''.join(f'<tr><td>{k}</td><td>{v:.3f}</td></tr>' for k,v in kv_pairs)}
            </tbody>
            </table>
        </div>
        </details>

        </div>
        """



        # ------------------------------------------------------------------
        # Affichage
        # ------------------------------------------------------------------
        with self.out:
            self.out.clear_output(wait=True)
            display(fig)
            display(widgets.HTML(debug_html))
        plt.close(fig)




    # ------------------------------------------------------------------#
    #  Parametrization & Queue methods                                  #
    # ------------------------------------------------------------------#
    def _refresh_parametrization(self, change=None):
        selected = [name for name, cb in self.opt_cfg_check.items() if cb.value]
        panels, titles = [], []

        for cfg_name in selected:
            panels.append(self._make_param_panel(cfg_name))
            titles.append(cfg_name)

        # ── avant d’affecter les enfants, on invalide l’index courant
        if not panels:
            self.param_tabs.selected_index = None   # ← évite le TraitError


        prev = self.param_tabs.selected_index
        self.param_tabs.children = panels           # <-- met à jour les onglets

        if panels:
            # rétablit un index valide (le dernier onglet, par ex.)
            new_idx = prev if prev is not None and prev < len(panels) else len(panels)-1
            self.param_tabs.selected_index = new_idx            
            
            for i, t in enumerate(titles):
                self.param_tabs.set_title(i, t)
            self.param_tabs.selected_index = new_idx




    def _add_copy_panel(self, cfg_name: str):
        # 1) génère un nouveau panel complet (avec ses propres on_click)
        new_panel = self._make_param_panel(cfg_name)

        # 2) ajoute-le à la suite des enfants existants
        existing = list(self.param_tabs.children or ())
        existing.append(new_panel)

        # 3) ré-affecte le tuple d’enfants au Tab
        self.param_tabs.children = tuple(existing)

        # 4) donne-lui un titre
        self.param_tabs.set_title(len(existing) - 1, cfg_name)


    def _remove_param_panel(self, panel: widgets.VBox) -> None:
        """Supprime un panel de parametrization existant."""
        # 1) copie mutable de la liste actuelle
        children = list(self.param_tabs.children)

        if panel not in children:      # rien à faire
            return

        # 2) on enlève l'onglet demandé
        idx_removed = children.index(panel)
        del children[idx_removed]

        # 3) on publie la nouvelle liste **avant** de régler selected_index
        self.param_tabs.children = tuple(children)

        # 4) corrige selected_index pour éviter le TraitError
        if not children:                       # plus aucun onglet
            self.param_tabs.selected_index = None
        else:                                  # au moins un onglet
            # on choisit l'onglet juste avant celui supprimé,
            # ou le dernier si on a supprimé le dernier
            new_idx = min(idx_removed, len(children) - 1)
            self.param_tabs.selected_index = new_idx

        # 5) (optionnel) remettre les titres si besoin
        #    Ipywidgets conserve les titres existants ; la boucle ci-dessous
        #    est seulement utile si vous voulez les recalculer.
        # for i, child in enumerate(children):
        #     self.param_tabs.set_title(i, self.param_tabs.get_title(i))


    
    def _make_param_panel(self, cfg_name: str) -> widgets.VBox:
        """Construit tous les widgets pour la config cfg_name."""
        cfg = next(c for c in self.sim.all_configs if c["config_name"] == cfg_name)
        geom = {k: v for k, v in cfg["geometry"]["geometry"].items() if v != 0.0}
        # Common bounds
        common_low = widgets.FloatText(value=0.0, description="Low all:", layout={"width":"150px"})
        common_up  = widgets.FloatText(value=1.0, description="Up all:", layout={"width":"150px"})
        apply_cb   = widgets.Button(description="Apply to all", button_style="primary")
        cb_controls = widgets.HBox([common_low, common_up, apply_cb], layout=widgets.Layout(gap="10px"))
        # Individual bounds
        rows = []

        for k, val in geom.items():
            lo_val, hi_val = geometry_limits.get(k, (0.0, 0.0))
            chk     = widgets.Checkbox(value=True, description="", indent=False, layout={"width":"30px"})
            lbl     = widgets.Label(value=k, layout={"width":"150px"})
            low_w   = widgets.FloatText(value=lo_val, description="min:", layout={"width":"120px"}, style={"description_width":"40px"})
            up_w    = widgets.FloatText(value=hi_val, description="max:", layout={"width":"120px"}, style={"description_width":"40px"})
            fixed_w = widgets.FloatText(value=val, description="fixed:", layout={"width":"120px"}, style={"description_width":"40px"})
            # callback pour basculer Low/Up ↔ Fixed
            def _toggle(change, lo=low_w, up=up_w, fix=fixed_w):
                if change["new"]:
                    lo.layout.display = up.layout.display = ""
                    fix.layout.display = "none"
                else:
                    lo.layout.display = up.layout.display = "none"
                    fix.layout.display  = ""
            chk.observe(_toggle, names="value")
            _toggle({"new": chk.value})
            rows.append(widgets.HBox(
                [chk, lbl, low_w, up_w, fixed_w],
                layout=widgets.Layout(align_items="center", gap="10px")
            ))


        bounds_box = widgets.VBox(rows, layout=widgets.Layout(border="1px solid #ddd", padding="5px"))
        
        
        def _apply_all(_):
            for row in rows:
                # row.children[2] est le FloatText “min”
                row.children[2].value = common_low.value
                # row.children[3] est le FloatText “max”
                row.children[3].value = common_up.value

        
        apply_cb.on_click(_apply_all)
        # Cost Function mode + λ
        cf_radio = widgets.RadioButtons(
            options=[('Dip','dip'),('FWHM','half'),('λ₀ fixe','fixed_lambda'),('λ range','range_lambda')],
            value='dip', description="CF mode:",
            style={'description_width':'initial'}
        )
        lambda0 = widgets.FloatText(value=600, description="λ₀ :", layout={"width":"150px"})
        lammin  = widgets.FloatText(value=650, description="λmin:", layout={"width":"150px"})
        lammax  = widgets.FloatText(value=750, description="λmax:", layout={"width":"150px"})
        lambda_box = widgets.HBox([lambda0, lammin, lammax], layout=widgets.Layout(gap="10px"))
        
        
        def _toggle(_):
            lambda0.layout.display = "" if cf_radio.value=="fixed_lambda" else "none"
            lammin.layout.display  = "" if cf_radio.value=="range_lambda" else "none"
            lammax.layout.display  = "" if cf_radio.value=="range_lambda" else "none"
        cf_radio.observe(_toggle, names="value"); _toggle(None)
        # DE parameters + actions
        budget_w = widgets.IntText(value=100, description="Budget",     layout={"width":"150px"})
        pop_w    = widgets.IntText(value=30,  description="Population", layout={"width":"150px"})
        add_q    = widgets.Button(description="Add to queue ▶", button_style="success", layout=widgets.Layout(width="100%"))
        add_copy = widgets.Button(description="Add copy", button_style="info", layout=widgets.Layout(width="100%"))
        del_btn  = widgets.Button(description="Delete panel", button_style="danger", layout=widgets.Layout(width="100%"))

        # callback pour supprimer ce panel
        def _on_delete_panel(_):
            self._remove_param_panel(panel)
        del_btn.on_click(_on_delete_panel)

        def _on_add(_):
            param_keys = list(geom.keys())
            # 1) sélection des paramètres à optimiser
            keys = [
                param_keys[i]
                for i, r in enumerate(rows)
                if r.children[0].value  # checkbox cochée
            ]
            # 2) bornes correspondantes
            bounds = [
                (r.children[2].value, r.children[3].value)
                for r in rows
                if r.children[0].value
            ]
            # 3) paramètres laissés fixes
            fixed_vals = {
                param_keys[i]: r.children[4].value
                for i, r in enumerate(rows)
                if not r.children[0].value
            }
            job = {
                "config": cfg_name,
                "keys": keys,                           
                "bounds": bounds,
                "fixed_vals": fixed_vals,
                "cf_mode": cf_radio.value,
                "lambda": (lambda0.value, lammin.value, lammax.value),
                "budget": budget_w.value,
                "pop":    pop_w.value,
                "status": "idle",
                "cancel_flag": False,
                "progress": widgets.FloatProgress(
                    value=0, min=0, max=1, description="",
                    bar_style="info", layout=widgets.Layout(width="90%", height="10px", display="none")
                )
            }
            self.job_queue.append(job)
            self._refresh_queue()
            
        add_q.on_click(_on_add)
        add_copy.on_click(lambda _: self._add_copy_panel(cfg_name))
        # Assemble panel
        panel = widgets.VBox([
            widgets.HTML(f"<b>Bounds for {cfg_name}</b>"),
            cb_controls, bounds_box,
            widgets.HTML("<b>Type of Cost Function</b>"), cf_radio, lambda_box,
            widgets.HTML("<b>DE parameters</b>"),
            widgets.HBox([budget_w, pop_w, add_q, add_copy, del_btn], layout=widgets.Layout(gap="10px"))
        ], layout=widgets.Layout(padding="10px", border="1px solid #bbb", margin="5px"))
        return panel


    # ------------------------------------------------------------------
    # Helper : dit s'il reste des jobs "running"
    # ------------------------------------------------------------------
    def _any_running(self) -> bool:
        return any(job["status"] == "running" for job in self.job_queue)


    def _refresh_queue(self):
        """Met à jour la liste des jobs et leurs barres de progression."""
        rows = []
        for i, job in enumerate(self.job_queue, 1):
            # 1) icône de statut
            status_ico = {
                "idle":    "▶️",  # en attente
                "running": "⌛",  # en cours
                "done":    "✅",  # validé
                "error":   "❌",  # erreur
            }[job["status"]]

            # 2) boutons
            run_b    = widgets.Button(
                description="Run ▶",
                layout=widgets.Layout(width="80px"),
                disabled=(job["status"] == "running")
            )
            cancel_b = widgets.Button(
                description="Cancel ⏹",
                layout=widgets.Layout(width="80px"),
                disabled=(job["status"] not in ("running",))
            )
            delete_b = widgets.Button(
                description="Delete ❌",
                layout=widgets.Layout(width="80px"),
                disabled=(job["status"] == "running")
            )
            
            run_b.on_click(lambda _, idx=i-1: self._run_job(idx))
            cancel_b.on_click(lambda _, idx=i-1: self._cancel_job(idx))
            delete_b.on_click(lambda _, idx=i-1: self._delete_job(idx))

            # 3) ligne d’infos
            info_row = widgets.HBox([
                widgets.Label(str(i),                             layout=widgets.Layout(width="30px")),
                widgets.Label(job["config"],                      layout=widgets.Layout(width="150px")),
                widgets.Label(job["cf_mode"],                     layout=widgets.Layout(width="80px")),
                widgets.Label(f"{job['budget']}/{job['pop']}",    layout=widgets.Layout(width="80px")),
                widgets.Label(status_ico,                         layout=widgets.Layout(width="30px")),
                run_b, cancel_b, delete_b
            ], layout=widgets.Layout(gap="10px"))

            # 4) barre de progression propre à ce job
            prog = job["progress"]
            prog.layout.display = "" if job["status"] == "running" else "none"

            # 5) empile ligne + barre
            rows.append(widgets.VBox([info_row, prog], layout=widgets.Layout(gap="2px")))

        # 6) on met à jour le VBox principal
        self.queue_box.children = rows

        #  état des boutons globaux 
        running = self._any_running()
        self.run_all_btn.disabled    = running or not self.job_queue
        self.delete_all_btn.disabled = running or not self.job_queue
        self.cancel_all_btn.disabled = not running





    def _run_job(self, idx: int, *, refresh_ui: bool = True):
        job = self.job_queue[idx]
        job["status"] = "running"

        # NE PAS reconstruire la file quand on enchaîne plusieurs lancements
        if refresh_ui:
            self._refresh_queue()

        # 1) création de la queue dédiée
        job_queue = q.Queue()
        job["progress_queue"] = job_queue

        # 2) extraction des bornes & clés depuis job["bounds"]
        #    job["bounds"] est une liste de tuples (low, up) et on suppose
        #    que vous avez aussi stocké dans job["keys"] la liste des noms
        lowers = np.array([b[0] for b in job["bounds"]])
        uppers = np.array([b[1] for b in job["bounds"]])
        keys   = job["keys"]

        # 3) préparation des extra_kwargs pour fixed/range lambda
        extra_kwargs = {}

        if job["cf_mode"] == "fixed_lambda":
            extra_kwargs["fixed_lambda"] = job["lambda"][0]
        elif job["cf_mode"] == "range_lambda":
            # on stocke (min, max) dans job["lambda"][1:]
            extra_kwargs["range_lambda"] = tuple(job["lambda"][1:])


        # transférez ici fixed_vals
        if "fixed_vals" in job:
            extra_kwargs["fixed_vals"] = job["fixed_vals"]


        cfg_name = job["config"]

        # 4) constitution du dict args
        args = dict(
            budget=job["budget"],
            Npop=job["pop"],
            lowers=lowers,
            uppers=uppers,
            keys=keys,
            mode=job["cf_mode"],
            cfg_name = cfg_name,
            progress_queue=job_queue,
            cancel_flag=job,
            **extra_kwargs
        )

        # 5) lancement du thread
        t = threading.Thread(target=self.DE_general, kwargs=args, daemon=True)
        job["thread"] = t
        t.start()

        # 6) fonction de polling pour alimenter la barre de progression
        loop = get_ipython().kernel.io_loop
        
        def _poll():
            try:
                tag, *payload = job_queue.get_nowait()
            except q.Empty:
                tag = None

            if tag == "PROG":
                frac = payload[0]
                job["progress"].value = frac   # juste MAJ de la barre, pas de _refresh_queue()

            elif tag == "DONE":
                job["status"] = "done"
                job["progress"].value = 1.0
                self._refresh_queue()         # là on rafraîchit pour passer à ✅

            elif tag == "ERROR":
                job["status"] = "error"
                self._refresh_queue()         # là on rafraîchit pour passer à ❌

            # si on a toujours un thread en cours, on re‐schedule
            if t.is_alive():
                loop = get_ipython().kernel.io_loop
                loop.add_timeout(loop.time() + 0.1, _poll)

        loop = get_ipython().kernel.io_loop
        loop.add_timeout(loop.time() + 0.1, _poll)


    def _cancel_job(self, idx: int):
        job = self.job_queue[idx]
        job["cancel_flag"] = True            # ← signale l’annulation
        # si le pool existe, on le tue immédiatement
        if "pool" in job and job["pool"] is not None:
            try:
                job["pool"].terminate()
                job["pool"].join()
            except Exception:
                pass
        job["status"] = "error"
        self._refresh_queue()


    def _delete_job(self, idx: int):
        """
        Supprime la job d'indice idx de la file et rafraîchit l'affichage.
        """
        # retire l'entrée
        del self.job_queue[idx]
        # met à jour la liste à l’écran
        self._refresh_queue()


    #  Run-all : on lance les jobs sans refresh intermédiaire
    def _run_all(self, _=None):
        if self._any_running():
            return
        # on conserve les indices avant toute mise-à-jour
        for idx in range(len(self.job_queue)):
            self._run_job(idx, refresh_ui=False)

        # un seul redraw une fois tous les jobs lancés
        self._refresh_queue()


    def _cancel_all(self, _=None):
        if not self._any_running():
            return
        for i in range(len(self.job_queue)):
            self._cancel_job(i)


    def _delete_all(self, _=None):
        if self._any_running():
            return
        """Vide entièrement la job‑queue et rafraîchit l’affichage."""
        self.job_queue.clear()
        self._refresh_queue()

        
# -----------------------------------------------------------------------------#
#  Helper                                                                     #
# -----------------------------------------------------------------------------#
def create_optimization_tab(sim_obj: SimulationTab) -> OptimizationTab:
    """Renvoie l’onglet d’optimisation (compatibilité)."""
    return OptimizationTab(sim_obj)
