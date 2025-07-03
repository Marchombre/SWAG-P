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

import threading
from functools import partial     # (pour l’init du Pool)

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from IPython import get_ipython
import ipywidgets as widgets
from tqdm.notebook import trange
import traceback, queue as q

from gap_plasmon_2d import paths
from gap_plasmon_2d.ui.geometry_settings import geometry_limits
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

# data_dir : dossier des données brutes (matériaux, géométries, etc.)
# contient notamment les fichiers de propriétés optiques
data_dir = Path(paths.DATA_DIR)

# json_combined_path : chemin vers le fichier JSON listant les matériaux standards
# et leurs propriétés optiques (indice de réfraction vs λ).
json_combined_path = data_dir / "combined_materials.json"





# -----------------------------------------------------------------------------#
#  Worker-side globals & helpers                                               #
# -----------------------------------------------------------------------------#

# On définit une variable globale `_WORKER_SIM` qui pointe vers l’instance
# `sim_tab` de SimulationTab importée depuis le module simulation.
# Grâce au fork, chaque sous-processus héritera de cette même instance en Copy-On-Write,
# évitant de devoir recharger/configurer le simulateur à chaque appel.
_WORKER_SIM: SimulationTab = sim_tab

# def init_worker(selected_config_name, lam_min, lam_max,
#                 n_points, json_path,
#                 sel_layers, delta_n_value,
#                 mode_sel_value, fixed_n,
#                 custom_map  ):
#     """
#     Initialisateur exécuté dans **chaque** process du pool.

#     On recrée un SimulationTab « sans UI » et on le configure exactement
#     comme dans le process maître.
#     """
#     global _WORKER_SIM
#     _WORKER_SIM = SimulationTab()  # pas d’interface graphique

#     # 1) Sélectionne la config voulue
#     for name, cb in _WORKER_SIM.config_checkboxes.items():
#         cb.value = name == selected_config_name

#     # 2) Propagation des paramètres de simulation
#     _WORKER_SIM.sim_lambda_min.value = lam_min
#     _WORKER_SIM.sim_lambda_max.value = lam_max
#     _WORKER_SIM.sim_n_points.value = n_points

#     # 3) Chemin vers le JSON matériau combiné
#     _WORKER_SIM.json_combined_path = json_path

#     _WORKER_SIM.layer_selector.value = tuple(sel_layers)
#     _WORKER_SIM.delta_n_widget.value = delta_n_value

#    # --- paramètres RCWA repris de l’UI ---
#     _WORKER_SIM.mode_selection.value = mode_sel_value      # 'fixed' / 'custom' / 'auto'
#     _WORKER_SIM.sim_n_mod.value      = fixed_n             # utilisé si 'fixed'
#     _WORKER_SIM.custom_n_mod_inputs  = {}                  # dict de IntText
    
#     for name, val in custom_map.items():                   # seulement si 'custom'
#         _WORKER_SIM.custom_n_mod_inputs[name] = widgets.IntText(value=val)


def cost_worker(args: Tuple[np.ndarray, List[str], str, Dict[str, Any]]) -> float:
    """
    Wrapper minimaliste et picklable pour multiprocessing.Pool.map.

    Il prend en entrée un seul argument `args`, qui est un tuple :
      - x       : vecteur numpy des valeurs de tous les paramètres à optimiser
      - keys    : liste des noms de ces paramètres
      - mode    : nom du mode de coût ('dip', 'fixed_lambda', 'range_lambda', 'half', etc.)
      - mode_kw : dictionnaire d’arguments optionnels pour la méthode cost
    """
    # 1) Dépacke les éléments du tuple
    x, keys, mode, mode_kw = args

    # 2) Appelle la méthode cost de l’instance globale `_WORKER_SIM`,
    #    en lui passant :
    #      • x       : le vecteur de paramètres actuels
    #      • keys    : la correspondance paramètre → valeur
    #      • mode    : le type de métrique à calculer
    #      • **mode_kw : d’éventuels arguments supplémentaires (λ fixée, intervalle, etc.)
    #
    #    La méthode `cost` renvoie un float (1 – sensibilité ou 1 – reflectance,
    #    selon le mode), qui sera utilisé par l’algorithme d’optimisation.
    return _WORKER_SIM.cost(x, keys, mode=mode, **mode_kw)




class OptimizationFileArboWidget:
    """
    Widget 3-niveaux pour parcourir :
      summary_opt_dir/
      ├─ <family>/
      │  ├─ <cost_mode>/
      │  │  └─ .../*.h5
      │  └─ ...
      └─ <family>/
         ├─ <cost_mode>/
         │  └─ .../*.h5
         └─ ...
    """
    def __init__(self, summary_opt_dir: str):
        self.base_dir = Path(summary_opt_dir)
        # Niveau 1 : family (multi_layer, gap_plasmon_resonator, …)
        self.family_dd = widgets.Dropdown(description="Family:")
        # Niveau 2 : cost_mode (lambda_fix, range_lambda, dip, half, …)
        self.cost_mode_dd = widgets.Dropdown(description="Cost mode:")
        # Niveau 3 : liste plate des fichiers .h5 disponibles
        self.file_dd = widgets.Dropdown(description="File:")
        self.run_dd       = widgets.Dropdown(description="Run:")
        
        # UI container
        self.widget = widgets.HBox(
            [self.family_dd, self.cost_mode_dd, self.file_dd, self.run_dd],
            layout=widgets.Layout(gap="10px")
        )

        # 1) quand on change famille ou cost_mode, on met à jour les fichiers
        for dd in (self.family_dd, self.cost_mode_dd):
            dd.observe(lambda _: self._refresh_file_list(), names="value")

        # 2) quand on change de fichier, on liste **seulement** les runs
        self.file_dd.observe(lambda _: self._refresh_run_list(), names="value")

        # premier peuplement
        self._refresh_file_list()
        self._refresh_run_list()
        
    def _list_subdirs(self, path: Path) -> list[str]:
        return sorted(p.name for p in path.iterdir() if p.is_dir())

    def _list_h5_files(self, path: Path) -> list[tuple[str,str]]:
        files = sorted(path.rglob("*.h5"), key=lambda f: f.as_posix())
        return [(f.relative_to(self.base_dir).as_posix(), str(f)) for f in files]

    def _refresh_file_list(self) -> None:
        # 1) Familles
        fams = self._list_subdirs(self.base_dir)
        self.family_dd.options = fams

        # 2) Cost modes (sous-dossier famille)
        if self.family_dd.value:
            self.cost_mode_dd.options = self._list_subdirs(
                self.base_dir / self.family_dd.value
            )
        else:
            self.cost_mode_dd.options = []

        # 3) Fichiers (sous-dossier cost_mode)
        if self.family_dd.value and self.cost_mode_dd.value:
            opts = self._list_h5_files(
                self.base_dir / self.family_dd.value / self.cost_mode_dd.value
            )
        else:
            opts = []
        self.file_dd.options = opts



    def _refresh_run_list(self):
        h5path = self.file_dd.value
        if h5path:
            runs = list_runs_in_h5(h5path)
            self.run_dd.options = runs
            if self.run_dd.value not in runs:
                self.run_dd.value = runs[0] if runs else None
        else:
            self.run_dd.options = []
            self.run_dd.value = None



    def get_selected_file(self) -> str | None:
        return self.file_dd.value
# -----------------------------------------------------------------------------#
#  Main widget class                                                           #
# -----------------------------------------------------------------------------#
class OptimizationTab:
    """
    Onglet d’optimisation (widgets + logique de calcul).
    """

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
        self._pool         = None
        self._worker_thread = None

        # background process / queue (may still be used by DE_general)
        ctx                 = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        self._de_process    = None          # type: mp.Process | None
        self._result_queue  = ctx.Queue()

        # ------------------------------------------------------------------ #
        #   STATIC UI — all widgets created **once** and kept forever
        # ------------------------------------------------------------------ #

        # Bounds table container (left column later)
        self.bounds_box = widgets.VBox(
            layout=widgets.Layout(border="1px solid #ccc", padding="8px", gap="5px")
        )

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
                    ('⟨R⟩ on range','range_lambda')],
            value='dip',
            
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='220px')
        )


        # Widgets spécifiques au mode 'fixed_lambda' ou 'range_lambda'
        self.lambda0_w = widgets.FloatText(
            value=700, description="λ₀ (nm):", layout=widgets.Layout(width="200px")
        )
        self.band_min_w = widgets.FloatText(
            value=650,
            description="λmin:",
            layout=widgets.Layout(width="200px"),
        )
        self.band_max_w = widgets.FloatText(
            value=750,
            description="λmax:",
            layout=widgets.Layout(width="200px"),
        )
        self.band_box = widgets.HBox(
            [self.band_min_w, self.band_max_w],
            layout=widgets.Layout(gap="200px"),
        )



        self._toggle_CF_mode_widgets({"new": self.cost_mode.value})      # état initial
        self.cost_mode.observe(self._toggle_CF_mode_widgets, names="value")




        # Widget arborescent pour filtrer les fichiers HDF5
        self.summary_opt_dir = summary_opt_dir

        self.opt_file_arbo = OptimizationFileArboWidget(str(self.summary_opt_dir))
        

        # Watcher pour maj auto de la liste
        self._observer = start_watcher(
            path=str(self.summary_opt_dir),
            callback=lambda *_: (
                self.opt_file_arbo._refresh_file_list(),
                self.opt_file_arbo._refresh_run_list()
            ),
            extensions=[".h5"],
            recursive=True,
        )

        # appel initial pour peupler
        self.opt_file_arbo._refresh_file_list()

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


        # 1) Colonne de gauche (Simulation)
        left_col = widgets.VBox([
            widgets.HTML(value="<b>Configurations & Δn</b>"),
            self.sim.config_selector,  # toggle + refresh + cases Config/Δn

            widgets.HTML(value="<b>Spectrum (nm)</b>"),
            widgets.HBox(
                [self.sim.sim_lambda_min, self.sim.sim_lambda_max, self.sim.sim_n_points],
                layout=widgets.Layout(gap='10px')
            ),

            widgets.HTML(value="<b>RCWA Fourier modes</b>"),
            self.sim.mode_selection,
            self.sim.sim_n_mod,
            self.sim.custom_modes_box,

            widgets.HTML(value="<b>Δn & Layers</b>"),
            widgets.HBox(
                [self.sim.delta_n_widget, self.sim.layer_selector],
                layout=widgets.Layout(gap='10px')
            ),
        ], layout=widgets.Layout(width='48%', padding='10px'))

        # 2) Colonne de droite (Optimisation)
        right_col = widgets.VBox([
            widgets.HTML(value="<b>Bounds</b>"),
            self.bounds_box,

            widgets.HTML(value="<b>Type of Cost Function</b>"),
            self.cost_mode,
            widgets.HBox([self.lambda0_w, self.band_box],
                        layout=widgets.Layout(gap='10px')),

            widgets.HTML(value="<b>DE parameters</b>"),
            widgets.HBox([self.budget_w, self.pop_w, self.run_btn, self.cancel_btn],
                        layout=widgets.Layout(gap='10px')), self.runtime_box
        ], layout=widgets.Layout(width='48%', padding='10px'))

        # 3) Ligne du bas (full width)
        bottom_controls = widgets.HBox([
            self.opt_file_arbo.widget,
            self.plot_btn,
        ], layout=widgets.Layout(justify_content='space-around', margin='10px'))

        # 4) Zone de sortie pour le plot (plein width)
        plot_area = self.out

        # 5) Assemblage final
        self.ui = widgets.VBox([
            widgets.HBox([left_col, right_col],
                        layout=widgets.Layout(justify_content='space-between')),
            bottom_controls,
            plot_area
        ], layout=widgets.Layout(padding='10px'))



        # branchements initiaux
        self._attach_config_observers()

        # hook sur le bouton Refresh de SimulationTab
        self.sim.config_refresh_btn.on_click(self._on_configs_refreshed)

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


    # ------------------------------------------------------------------#
    #  UI helpers                                                       #
    # ------------------------------------------------------------------#
    def _on_configs_refreshed(self, _):
        """
        Quand SimulationTab recharge ses config_checkboxes,
        on rebranche nos observers et on vide le bounds_box
        pour que l’utilisateur puisse recliquer et repeupler.
        """
        # 1) (re)branche la callback sur TOUTES les cases
        self._attach_config_observers()
        # 2) on vide la zone des bounds (update_optimization la repopulera)
        self.bounds_box.children = []
        self.param_widgets = {}



    def _attach_config_observers(self):
        """(Re)branche update_optimization sur chaque checkbox."""
        for cb in self.sim.config_checkboxes.values():
            cb.observe(self.update_optimization, names="value")


    def _toggle_CF_mode_widgets(self, change: Dict[str, Any]) -> None:
        """Affiche/masque les widgets selon le mode calcul choisi."""
        m = change["new"]
        self.lambda0_w.layout.display = "" if m == "fixed_lambda" else "none"
        self.band_box.layout.display = "" if m == "range_lambda" else "none"



    def close(self) -> None:
        """Explicitly release resources held by the observer."""
        if hasattr(self, "_observer") and self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join()
            except Exception:
                pass
            self._observer = None



    def __del__(self) -> None:
        self.close()


    def _update_run_button_state(self, *_):
        """
        Active le bouton Run DE si au moins un paramètre est marqué
        'opt', sinon le grise.
        """
        any_selected = any(w['opt'].value for w in self.param_widgets.values())
        self.run_btn.disabled = not any_selected




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
        self._progress_bar.value           = 0
        self._progress_bar.description     = "0 %"
        self._progress_bar.bar_style       = "info"
        self._progress_bar.layout.display  = "none"   # stay hidden for now

        # ----------  collect UI parameters (unchanged) ----------
        extra_kwargs = {}
        mode = self.cost_mode.value
        if mode == "fixed_lambda":
            extra_kwargs["fixed_lambda"] = self.lambda0_w.value
        elif mode == "range_lambda":
            extra_kwargs["range_lambda"] = (self.band_min_w.value,
                                            self.band_max_w.value)

        keys   = [k for k, w in self.param_widgets.items() if w["opt"].value]
        if not keys:
            self._status_html.value = "⚠️ No parameter selected for optimisation."
            self._is_running = False
            self.cancel_btn.disabled = True
            return

        lowers = np.array([self.param_widgets[k]["low"].value for k in keys])
        uppers = np.array([self.param_widgets[k]["up"].value  for k in keys])


        # ── queue de progression & thread lanceur ──────────
        self._result_queue = q.Queue()
        args = dict(budget=self.budget_w.value,
                    Npop   =self.pop_w.value,
                    lowers =lowers, uppers=uppers,
                    keys   =keys, mode=mode,
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

        # ── first PROG message → reveal bar, hide text ─────────────────────
        if tag == "PROG" and self._progress_bar.layout.display == "none":
            self._status_html.value       = ""            # hide message
            self._progress_bar.layout.display = ""        # show bar

        if tag == "PROG":
            frac, best = payload
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
        self.opt_file_arbo._refresh_file_list()

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

        # terminate pool if alive
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

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
        # Config cochée
        sels = [
            c
            for c in self.sim.all_configs
            if self.sim.config_checkboxes[c["config_name"]].value
        ]
        if len(sels) != 1:
            self.bounds_box.children = []
            return

        geom = sels[0]["geometry"]["geometry"]
        rows: List[widgets.HBox] = []
        self.param_widgets: Dict[str, Dict[str, widgets.Widget]] = {}

        for k, val in geom.items():
            if val == 0.0:
                continue  # épaisseur nulle → pas optimisé
            low, high = geometry_limits.get(k, (0.0, 0.0))

            chk = widgets.Checkbox(value=True, indent=False, layout={"width": "30px"})
            lbl = widgets.Label(value=k, layout={"width": "150px"})
            lo = widgets.FloatText(
                value=low,
                description="min:",
                layout={"width": "120px"},
                style={"description_width": "40px"},
            )
            hi = widgets.FloatText(
                value=high,
                description="max:",
                layout={"width": "120px"},
                style={"description_width": "40px"},
            )

            self.param_widgets[k] = {"opt": chk, "low": lo, "up": hi}
            rows.append(
                widgets.HBox(
                    [chk, lbl, lo, hi],
                    layout=widgets.Layout(align_items="center", gap="10px"),
                )
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
        mode: str = "dip",
        n_jobs: int = -1,
        seed: int | None = None,    # Répétabilité
        progress_queue: mp.Queue | None = None,
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
        """

        # 1) Initialisation RNG & population
        rng = np.random.default_rng(seed)
        if budget < Npop:
            raise ValueError("Le budget doit être ≥ à la taille de la population.")
        Ngen = budget // Npop
        n_params = len(keys)
        pop = lowers + (uppers - lowers) * rng.random((Npop, n_params))

        # 2) Pool de process (vrai parallélisme CPU)
        global _WORKER_SIM
        _WORKER_SIM = self.sim                              # objet partagé en COW
        ctx   = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        pool  = ctx.Pool()                                  # processes == nb CPU
        self._pool = pool


        try:
            # Évaluation initiale (interruptible)
            args0 = [(pop[i], keys, mode, mode_kw) for i in range(Npop)]
            cf_list: List[float] = []
            for r in pool.imap_unordered(cost_worker, args0, chunksize=1):
                if self._cancelled:
                    pool.terminate()
                    raise OptimizationCancelled()
                cf_list.append(r)
            cf = np.array(cf_list)


            conv_best  = np.zeros(Ngen)
            conv_evals = np.arange(1, Ngen+1)*Npop
            best_after_eval: List[float] = []
            F1, F2, cr = 0.9, 0.8, 0.8

            # 3) Boucle DE avec barre de progression et annulation

            for g in range(Ngen):
                if self._cancelled:
                    pool.terminate()
                    raise OptimizationCancelled()

                # génération des enfants
                z_list: List[Tuple[int,np.ndarray]] = []
                for p in range(Npop):
                    a,b,c = pop[rng.choice(Npop,3,replace=False)]
                    best_ind = pop[np.argmin(cf)]
                    y = c + F1*(a-b) + F2*(best_ind-c)
                    mask = rng.random(n_params) < cr
                    if not mask.any():
                        mask[rng.integers(n_params)] = True
                    z = np.where(mask, y, pop[p])
                    z = np.clip(z, lowers, uppers)
                    z_list.append((p, z))

                # évaluation enfants
                args_child = [(z, keys, mode, mode_kw) for (_, z) in z_list]
                cfz_list: List[float] = []
                for r in pool.imap_unordered(cost_worker, args_child, chunksize=1):
                    if self._cancelled:
                        pool.terminate()
                        raise OptimizationCancelled()
                    cfz_list.append(r)
                cfz = cfz_list

                # sélection
                for (i,z),cval in zip(z_list, cfz):
                    if cval < cf[i]:
                        pop[i], cf[i] = z, cval

                best_after_eval.append(cf.min())
                conv_best[g] = cf.min()

                if progress_queue is not None:
                            progress_queue.put(("PROG",
                                                (g + 1) / Ngen,      # fraction 0-1
                                                float(cf.min())))     # meilleur coût courant
                            

            # 4) Ré-évaluation finale
            argsf       = [(pop[i], keys, mode, mode_kw) for i in range(Npop)]
            cf_final_list: List[float] = []

            for r in pool.imap_unordered(cost_worker, argsf, chunksize=1):
                cf_final_list.append(r)
            cf_final = np.array(cf_final_list)

            best_final = pop[np.argmin(cf_final)]
            best_cost  = cf_final.min()

            # 5) Tracé du spectre optimal + sauvegarde HDF5
            lam = np.linspace(self.sim.sim_lambda_min.value,
                              self.sim.sim_lambda_max.value,
                              self.sim.sim_n_points.value)
            cfg = next(c for c in self.sim.all_configs if self.sim.config_checkboxes[c["config_name"]].value)
            for xi,k in zip(best_final, keys):
                cfg["geometry"]["geometry"][k] = float(xi)

            Rup, Rdown, _ = run_simulation_one_combo(
                lam, {"angle":0,"polarization":1}, self.sim.sim_n_mod.value, cfg, self.json_combined_path
            )
            Rup   = np.asarray(Rup, float)
            Rdown = np.asarray(Rdown, float)

            config_name = next(n for n,cb in self.sim.config_checkboxes.items() if cb.value)
            fam = 'gap_plasmon_resonator' if mode in ('dip','half') else 'multi_layer'

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
                best_after_eval=np.asarray(best_after_eval),  # Snapshot du best à t instant

                # — contexte de la métrique —
                mode=mode,                          # 'dip' ou 'half'

                # — spectre du design optimal —
                lam=lam,                            # Grille λ
                Rup=Rup,                            # Spectre R_up
                Rdown=Rdown,                        # Spectre R_down
            )



            if progress_queue is not None:
                    progress_queue.put(("DONE",
                                        conv_best, conv_evals, best_final, best_cost))
            return conv_best, conv_evals, best_final, best_cost


        finally:
            # Ce bloc s’exécutera systématiquement, succès ou annulation
            if hasattr(self, "_pool") and self._pool is not None:
                self._pool.close()
                self._pool.join()
                self._pool = None

    # ------------------------------------------------------------------#
    #  Plot HDF5 results                                                #
    # ------------------------------------------------------------------#
    def plot_optimization_results(self, _=None) -> None:
        """
        Trace : convergence, consistency, bar des paramètres, spectre final.
        """
        self.opt_file_arbo._refresh_file_list()

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



        # --------------------------- FIGURE ---------------------------- #
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        ax0, ax1, ax2, ax3 = axs.flat

        # Convergence
        ax0.plot(range(1, len(conv_best)+1), conv_best, marker='.')
        ax0.set_title("DE convergence curve")
        ax0.set_xlabel("Itérations")
        ax0.set_ylabel("Cost")
        ax0.grid(True)


        # Tracé consistency (tous les best_cost compatibles)
        if len(all_best) >= 2:
            ax1.plot(all_best, marker='o')
            ax1.set_title("Consistency curve (all compatible runs)")
        else:
            ax1.text(0.5, 0.5,
                    "Il faut ≥ 2 runs compatibles pour la consistency",
                    ha='center', va='center', transform=ax1.transAxes)

        # ------------------------------------------------------------
        # 4) Bar-plot des paramètres optimisés du run courant
        # ------------------------------------------------------------
        ax2.bar(range(len(keys)), best_vec)
        ax2.set_title("Optimized parameters")
        ax2.set_xticks(range(len(keys)))
        ax2.set_xticklabels(keys, rotation=45, ha="right")
        ax2.set_ylabel("Value")
        ax2.grid(True)


        # Spectre
        lam, Rup, Rdown = None, None, None
        if "spectra" in data:
            lam = data["spectra"]["wavelength"]
            Rup = data["spectra"]["Rup"]
            Rdown = data["spectra"]["Rdown"]

        if lam is not None:
            ax3.plot(lam, Rup, label="Rup")
            if Rdown is not None:
                ax3.plot(lam, Rdown, label="Rdown")
        ax3.set_title("Best config spectrum")
        ax3.set_xlabel("λ (nm)")
        ax3.set_ylabel("Reflectance")
        ax3.legend()
        ax3.grid(True)

        # ------------------------------------------------------------------
        # 1)  Mode et coût minimal
        # ------------------------------------------------------------------
        mode       = data["mode"]           # 'dip' | 'half' | 'fixed_lambda' | 'range_lambda'
        best_cost  = float(data["best_cost"])  # Valeur de coût associée à best_final

        # ------------------------------------------------------------------
        # 2)  Conversion 1-CF → métrique
        # ------------------------------------------------------------------
        metric_value = 1.0 - best_cost      # même formule pour tous les modes

        label_map = {
            "dip"          : "Sensitivity S (ΔR/Δn)",
            "half"         : "Sensitivity S½ (ΔR/Δn)",
            "fixed_lambda" : "Reflectance R(λ₀)",
            "range_lambda" : "Mean reflectance ⟨R⟩",
        }
        metric_label = label_map.get(mode, "Metric")


        # Tableau des paramètres
        table_data = [
            [metric_label, f"{metric_value:.3g}"],   # ligne métrique / valeur
            ["Parameter",  "Value"]                 # en-têtes
        ] + [[k, f"{v:.3g}"] for k, v in zip(keys, best_vec)]
        
        table = ax3.table(
            cellText=[[ *row ] for row in table_data[2:]],   # uniquement les vraies données
            colLabels=table_data[1],                         # ["Parameter","Value"]
            cellLoc="center", colLoc="center",
            bbox=[0.0, -0.6, 1.0, 0.4],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        plt.tight_layout()

        with self.out:
            self.out.clear_output(wait=True)   # efface le contenu précédent du widget
            display(fig)                       # affiche la figure dans ce même widget

        plt.close(fig) 


# -----------------------------------------------------------------------------#
#  Helper                                                                     #
# -----------------------------------------------------------------------------#
def create_optimization_tab(sim_obj: SimulationTab) -> OptimizationTab:
    """Renvoie l’onglet d’optimisation (compatibilité)."""
    return OptimizationTab(sim_obj)
