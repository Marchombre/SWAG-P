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

import os
import sys
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import ipywidgets as widgets
from tqdm.notebook import trange

from gap_plasmon_2d import paths
from gap_plasmon_2d.ui.geometry_settings import geometry_limits
from gap_plasmon_2d.simulation.simulation import SimulationTab
from gap_plasmon_2d.simulation.simulate_and_plot import run_simulation_one_combo
from gap_plasmon_2d.utils.saving__functions import save_optimization_hdf5
from gap_plasmon_2d.utils.data_readers import (
    read_optimization_hdf5,
    list_optimization_files,
)
from gap_plasmon_2d.utils.file_watchers import start_watcher

# -----------------------------------------------------------------------------#
#  PATHS & GLOBALS                                                             #
# -----------------------------------------------------------------------------#
module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, module_dir)  # assure la présence du module dans PYTHONPATH

BASE_NOTEBOOKS = (
    Path(__file__).resolve().parent.parent / str(paths.RESULTS_DIR)
)
summary_opt_dir = BASE_NOTEBOOKS / "summary_optimisation"
summary_opt_dir.mkdir(parents=True, exist_ok=True)

data_dir = Path(paths.DATA_DIR)
json_combined_path = data_dir / "combined_materials.json"

# -----------------------------------------------------------------------------#
#  Worker-side globals & helpers                                               #
# -----------------------------------------------------------------------------#
_WORKER_SIM: SimulationTab | None = None


def init_worker(
    selected_config_name: str,
    lam_min: float,
    lam_max: float,
    n_points: int,
    json_path: str,
) -> None:
    """
    Initialisateur exécuté dans **chaque** process du pool.

    On recrée un SimulationTab « sans UI » et on le configure exactement
    comme dans le process maître.
    """
    global _WORKER_SIM
    _WORKER_SIM = SimulationTab()  # pas d’interface graphique

    # 1) Sélectionne la config voulue
    for name, cb in _WORKER_SIM.config_checkboxes.items():
        cb.value = name == selected_config_name

    # 2) Propagation des paramètres de simulation
    _WORKER_SIM.sim_lambda_min.value = lam_min
    _WORKER_SIM.sim_lambda_max.value = lam_max
    _WORKER_SIM.sim_n_points.value = n_points

    # 3) Chemin vers le JSON matériau combiné
    _WORKER_SIM.json_combined_path = json_path


def cost_worker(args: Tuple[np.ndarray, List[str], str, Dict[str, Any]]) -> float:
    """
    Thin, picklable wrapper autour de `SimulationTab.cost`.

    Parameters
    ----------
    args
        (x, keys, mode, mode_kw)
        *x*             -> vecteur paramètre
        *keys*          -> noms des paramètres
        *mode*          -> 'dip', 'refl_one', ...
        *mode_kw*       -> dict d’arguments additionnels
    """
    x, keys, mode, mode_kw = args
    return _WORKER_SIM.cost(x, keys, mode=mode, **mode_kw)  # type: ignore


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
        self.sim = sim_obj
        self.json_combined_path = str(json_combined_path)

        # Conteneur pour les widgets bornes
        self.bounds_box = widgets.VBox(
            layout=widgets.Layout(
                border="1px solid #ccc", padding="8px", gap="5px"
            )
        )





        # ------------------------------------------------------------------ #
        #     Sélecteur du mode de coût (indépendant de l’onglet Simulation) #
        # ------------------------------------------------------------------ #
        self.mode_calc_radio = widgets.RadioButtons(
            options=[('Dip (ΔR/Δn)',  'dip'),
                    ('FWHM (half)',  'half'),
                    ('λ₀ fixe',      'fixed_lambda'),
                    ('⟨R⟩ sur bande','range_lambda')],
            value='dip',
            description='Mode coût :',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='220px')
        )

        # λ₀ et borne de bande (n’apparaissent qu’aux bons modes)
        self.lambda0_w  = widgets.FloatText(
            value=700, description='Custum fixed λ₀ (nm):',
            layout=widgets.Layout(width='130px'),
            style={'description_width':'55px'})
        self.band_min_w = widgets.FloatText(
            value=650, description='λmin:',
            layout=widgets.Layout(width='120px'),
            style={'description_width':'40px'})
        self.band_max_w = widgets.FloatText(
            value=750, description='λmax:',
            layout=widgets.Layout(width='120px'),
            style={'description_width':'40px'})
        self.band_box   = widgets.HBox([self.band_min_w, self.band_max_w],
                                    layout=widgets.Layout(gap='5px'))

        def _toggle_refl_widgets(change):
            m = change['new']
            self.lambda0_w.layout.display = '' if m == 'fixed_lambda' else 'none'
            self.band_box.layout.display  = '' if m == 'range_lambda'   else 'none'

        _toggle_refl_widgets({'new': self.mode_calc_radio.value})      # état initial
        self.mode_calc_radio.observe(_toggle_refl_widgets, names='value')





        # Dropdown listant les fichiers HDF5 d’optimisation
        self.summary_opt_dir = summary_opt_dir
        self.opt_file_dd = widgets.Dropdown(
            options=list_optimization_files(str(self.summary_opt_dir)),
            description="Fichier Opt:",
            layout=widgets.Layout(width="400px"),
        )

        # Watcher pour maj auto de la liste
        self._observer = start_watcher(
            path=str(self.summary_opt_dir),
            callback=self._refresh_file_list,
            extensions=[".h5"],
            recursive=False,
        )

        # Contrôles DE
        self.budget_w = widgets.IntText(value=100, description="Budget")
        self.pop_w = widgets.IntText(value=30, description="Population")
        self.run_btn = widgets.Button(
            description="Run DE", button_style="primary"
        )
        self.out = widgets.Output(layout={"border": "1px solid #888"})

        # Widgets spécifiques au mode 'refl_one' ou 'range_lambda'
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

        # Cachés par défaut
        self.lambda0_w.layout.display = "none"
        self.band_box.layout.display = "none"

        # Bouton de tracé des résultats
        self.plot_btn = widgets.Button(
            description="Tracer résultats", button_style="info"
        )
        self.plot_btn.on_click(self.plot_optimization_results)

        # Assemblage des contrôles supérieurs
        controls = widgets.HBox(
            [ self.budget_w, self.pop_w, self.run_btn,
            self.opt_file_dd, self.plot_btn ],
            layout=widgets.Layout(margin='10px', flex_wrap='wrap',
                                align_items='center')
        )

        self.ui = widgets.VBox(
            [ self.bounds_box,
            self.mode_calc_radio,
            widgets.HBox([self.lambda0_w, self.band_box]),
            controls,
            self.out ],
            layout=widgets.Layout(padding='10px')
        )


        # Observe la sélection de configuration
        for cb in self.sim.config_checkboxes.values():
            cb.observe(self.update_optimization, names="value")

        # Met à jour la liste de paramètres
        self.update_optimization()

        # Callbacks
        self.run_btn.on_click(self._on_run)
        self.sim.mode_calc_radio.observe(
            self._toggle_refl_widgets, names="value"
        )

    # ------------------------------------------------------------------#
    #  UI helpers                                                       #
    # ------------------------------------------------------------------#
    def _refresh_file_list(self) -> None:
        """Met à jour le dropdown lorsqu’un nouveau fichier est créé."""
        files = list_optimization_files(str(self.summary_opt_dir))
        if set(files) != set(self.opt_file_dd.options):
            self.opt_file_dd.options = files

    def _toggle_refl_widgets(self, change: Dict[str, Any]) -> None:
        """Affiche/masque les widgets selon le mode calcul choisi."""
        m = change["new"]
        self.lambda0_w.layout.display = "" if m == "refl_one" else "none"
        self.band_box.layout.display = "" if m == "range_lambda" else "none"

    def __del__(self) -> None:
        self._observer.stop()
        self._observer.join()

    # ------------------------------------------------------------------#
    #  Callback : lancement DE                                          #
    # ------------------------------------------------------------------#
    def _on_run(self, _) -> None:
        """Point d’entrée bouton ‹ Run DE ›."""
        self.out.clear_output()
        
        
        # Arguments additionnels pour certains modes
        extra_kwargs: Dict[str, Any] = {}
        
        
        # Mode choisi dans le radio-bouton et éventuels paramètres supplémentaires
        mode_selected = self.mode_calc_radio.value
        extra_kwargs  = {}
        if mode_selected == 'fixed_lambda':
            extra_kwargs['fixed_lambda'] = self.lambda0_w.value
        elif mode_selected == 'range_lambda':
            extra_kwargs['range_lambda'] = (self.band_min_w.value,
                                        self.band_max_w.value)


        # --- paramètres à optimiser -----------------------------------------
        keys = [k for k,w in self.param_widgets.items() if w['opt'].value]
        if not keys:
            with self.out:
                print("⚠️ Parameters to optimize: none.")
            return

        lowers = np.array([self.param_widgets[k]['low'].value for k in keys])
        uppers = np.array([self.param_widgets[k]['up'].value  for k in keys])

        

        # --- lancement DE ----------------------------------------------------
        with self.out:
            print("🚀 Optimization is running, please wait…")
        conv_best, conv_evals, best_final, best_cost = self.DE_general(
            budget=self.budget_w.value,
            Npop=self.pop_w.value,
            lowers=lowers,
            uppers=uppers,
            keys=keys,
            mode=mode_selected,          # ← maintenant dynamique
            **extra_kwargs
        )

        # --- mise à jour de la liste des fichiers & résumé -------------------
        self._refresh_file_list()
        with self.out:
            print("✅ Optimization ended.")
            print(f"Best cost   : {best_cost:.6g}")
            print("Best vector :", best_final)

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

        self.bounds_box.children = rows
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
        if budget < Npop:
            raise ValueError("Le budget doit être ≥ à la taille de la population.")

        Ngen = budget // Npop
        n_params = len(keys)

        # -------------------- 1) POPULATION INITIALE -------------------- #
        pop = np.random.rand(Npop, n_params)
        pop = lowers + (uppers - lowers) * pop

        # -------------------- 2) CONFIG & POOL WORKERS ------------------- #
        sel_cfgs = [
            name for name, cb in self.sim.config_checkboxes.items() if cb.value
        ]
        if len(sel_cfgs) != 1:
            raise RuntimeError("Sélectionnez exactement une configuration.")
        selected_config = sel_cfgs[0]

        lam_min, lam_max = self.sim.sim_lambda_min.value, self.sim.sim_lambda_max.value
        n_pts = self.sim.sim_n_points.value

        with mp.Pool(
            processes=None,
            initializer=init_worker,
            initargs=(
                selected_config,
                lam_min,
                lam_max,
                n_pts,
                self.json_combined_path,
            ),
        ) as pool:
            # Évaluation initiale
            args0 = [(pop[i], keys, mode, mode_kw) for i in range(Npop)]
            cf = np.array(pool.map(cost_worker, args0))

            conv_best = np.zeros(Ngen)
            conv_evals = np.arange(1, Ngen + 1) * Npop

            # ---------------- 3) BOUCLE DE / CURRENT-TO-BEST ------------- #
            F1, F2, cr = 0.9, 0.8, 0.8

            self.out.clear_output()
            with self.out:
                for g in trange(Ngen, desc="Differential Evolution"):
                    z_list: List[Tuple[int, np.ndarray]] = []

                    for p in range(Npop):
                        idxs = np.random.choice(Npop, 3, replace=False)
                        a, b, c = pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]

                        best_indiv = pop[np.argmin(cf)]
                        y = c + F1 * (a - b) + F2 * (best_indiv - c)

                        mask = np.random.rand(n_params) < cr
                        if not mask.any():
                            mask[np.random.randint(n_params)] = True

                        z = np.where(mask, y, pop[p])
                        z = np.clip(z, lowers, uppers)
                        z_list.append((p, z))

                    # Évaluation enfants
                    args_child = [(z, keys, mode, mode_kw) for (_, z) in z_list]
                    cfz = pool.map(cost_worker, args_child)

                    # Sélection
                    for (i, z), cval in zip(z_list, cfz):
                        if cval < cf[i]:
                            pop[i], cf[i] = z, cval

                    conv_best[g] = cf.min()

                # ------------- 4) RÉ-ÉVALUATION FINALE ------------------- #
                argsf = [(pop[i], keys, mode, mode_kw) for i in range(Npop)]
                cf_final = np.array(pool.map(cost_worker, argsf))

        best_final = pop[np.argmin(cf_final)]
        best_cost = cf_final.min()

        # -------------------- 5) SPECTRE DU MEILLEUR -------------------- #
        lam = np.linspace(
            self.sim.sim_lambda_min.value,
            self.sim.sim_lambda_max.value,
            self.sim.sim_n_points.value,
        )
        cfg = next(
            c
            for c in self.sim.all_configs
            if self.sim.config_checkboxes[c["config_name"]].value
        )
        for xi, k in zip(best_final, keys):
            cfg["geometry"]["geometry"][k] = float(xi)

        Rup, Rdown, _ = run_simulation_one_combo(
            lam,
            {"angle": 0, "polarization": 1},
            self.sim.sim_n_mod.value,
            cfg,
            self.json_combined_path,
        )
        Rup = np.asarray(Rup, float)
        Rdown = np.asarray(Rdown, float)

        # -------------------- 6) SAUVEGARDE HDF5 ------------------------ #
        run_id = f"budget{budget}_pop{Npop}"
        save_optimization_hdf5(
            notebook_dir=str(BASE_NOTEBOOKS),
            run_id=run_id,
            budget=budget,
            Npop=Npop,
            keys=keys,
            lowers=lowers,
            uppers=uppers,
            conv_best=conv_best,
            conv_evals=conv_evals,
            cf_final=cf_final,
            best=pop[np.argmin(cf)],
            best_final=best_final,
            best_cost=best_cost,
            mode=mode,
            lam=lam,
            Rup=Rup,
            Rdown=Rdown,
        )

        return conv_best, conv_evals, best_final, best_cost

    # ------------------------------------------------------------------#
    #  Plot HDF5 results                                                #
    # ------------------------------------------------------------------#
    def plot_optimization_results(self, _=None) -> None:
        """
        Trace : convergence, consistency, bar des paramètres, spectre final.
        """
        self._refresh_file_list()
        h5file = self.opt_file_dd.value
        if h5file is None:
            raise RuntimeError("Aucun fichier HDF5 sélectionné.")

        data = read_optimization_hdf5(str(h5file))

        # Spectres
        lam, Rup, Rdown = None, None, None
        if "spectra" in data:
            lam = data["spectra"]["wavelength"]
            Rup = data["spectra"]["Rup"]
            Rdown = data["spectra"]["Rdown"]

        keys = data["keys"]
        best_vec = data["best_final"]
        conv_best = data["conv_best"]
        evals = data["conv_evals"]
        best_costs = [data["best_cost"]]  # pour consistance multi-runs
        cf_final = data["cf_final"]

        # --------------------------- FIGURE ---------------------------- #
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        ax0, ax1, ax2, ax3 = axs.flat

        # Convergence
        ax0.plot(evals, conv_best, marker="o")
        ax0.set_title("DE convergence curve")
        ax0.set_xlabel("Cost-function evaluations")
        ax0.set_ylabel("Cost")
        ax0.grid(True)

        # Consistency (≥2 runs nécessaires)
        if len(best_costs) >= 2:
            ax1.plot(np.sort(best_costs), marker=".")
            ax1.set_title("Consistency curve")
            ax1.set_xlabel("Run (sorted)")
            ax1.set_ylabel("Final best cost")
            ax1.grid(True)
        else:
            ax1.text(
                0.5,
                0.5,
                "Need ≥ 2 runs\nfor consistency curve",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )

        # Bar des paramètres
        ax2.bar(keys, best_vec)
        ax2.set_title("Optimized parameters")
        ax2.set_xticklabels(keys, rotation=45, ha="right")
        ax2.set_ylabel("Value")
        ax2.grid(True)

        # Spectre
        if lam is not None:
            ax3.plot(lam, Rup, label="Rup")
            if Rdown is not None:
                ax3.plot(lam, Rdown, label="Rdown")
        ax3.set_title("Best config spectrum")
        ax3.set_xlabel("λ (nm)")
        ax3.set_ylabel("Reflectance")
        ax3.legend()
        ax3.grid(True)

        # Tableau des paramètres
        table_data = [[k, f"{v:.3g}"] for k, v in zip(keys, best_vec)]
        table = ax3.table(
            cellText=table_data,
            colLabels=["Parameter", "Value"],
            cellLoc="center",
            colLoc="center",
            bbox=[0.0, -0.6, 1.0, 0.4],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------------#
#  Helper                                                                     #
# -----------------------------------------------------------------------------#
def create_optimization_tab(sim_obj: SimulationTab) -> OptimizationTab:
    """Renvoie l’onglet d’optimisation (compatibilité)."""
    return OptimizationTab(sim_obj)
