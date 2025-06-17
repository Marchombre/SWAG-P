#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimisation.py

Ce module encapsule l’onglet d’optimisation par Differential Evolution
dans une classe `OptimizationTab`, remplaçant les anciennes fonctions DE_general
et create_optimization_tab. 

Usage dans l’application interactive :
    from Optimisation import create_optimization_tab
    opt_tab = create_optimization_tab(sim_obj)
"""
import os
import sys
import numpy as np
from scipy.io import savemat
from joblib import Parallel, delayed
import ipywidgets as widgets
from IPython.display import clear_output
import geometry_settings
from simulation import SimulationTab

# S’assure que le dossier courant est sur le PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)


class OptimizationTab:
    """
    Onglet « Optimisation » : interface et algorithme DE.
    """
    def __init__(self, sim_obj: SimulationTab):
        self.sim = sim_obj

        # 1) prélever les clefs et bornes par paramètre
        self.keys = list(geometry_settings.geometry_limits.keys())
        self.lowers_def = np.array(
            [geometry_settings.geometry_limits[k][0] for k in self.keys],
            dtype=float
        )
        self.uppers_def = np.array(
            [geometry_settings.geometry_limits[k][1] for k in self.keys],
            dtype=float
        )

        # 2) construire les widgets des bornes
        self.lowers_w = [
            widgets.FloatText(value=low, description=k)
            for k, low in zip(self.keys, self.lowers_def)
        ]
        self.uppers_w = [
            widgets.FloatText(value=hi, description=k)
            for k, hi in zip(self.keys, self.uppers_def)
        ]

        # 3) widgets DE
        self.budget_w = widgets.IntText(value=100, description="Budget")
        self.pop_w    = widgets.IntText(value=30,  description="Population")
        self.run_btn  = widgets.Button(
            description="Run DE", button_style='primary'
        )
        self.out      = widgets.Output()

        # 4) mise en page
        self.bounds_box = widgets.HBox([
            widgets.VBox(self.lowers_w, layout=widgets.Layout(flex='1')),
            widgets.VBox(self.uppers_w, layout=widgets.Layout(flex='1')),
        ])
        self.controls = widgets.HBox([
            self.budget_w,
            self.pop_w,
            self.run_btn
        ], layout=widgets.Layout(margin='10px 0'))

        # 5) zone principale
        self.ui = widgets.VBox([
            self.bounds_box,
            self.controls,
            self.out
        ], layout=widgets.Layout(padding='10px'))

        # 6) lier le callback
        self.run_btn.on_click(self._on_run)

    def _on_run(self, _):
        """
        Callback du bouton : lance DE et affiche les résultats.
        """
        self.out.clear_output()
        lowers = np.array([w.value for w in self.lowers_w], dtype=float)
        uppers = np.array([w.value for w in self.uppers_w], dtype=float)

        with self.out:
            print("🚀 Lancement de DE_general…")
        conv, best = self.DE_general(
            budget=self.budget_w.value,
            Npop=self.pop_w.value,
            lowers=lowers,
            uppers=uppers,
            mode="dip"
        )
        with self.out:
            print("✅ Optimisation terminée.")
            print(f"Meilleure valeur (dernier conv) : {conv[-1]:.6g}")
            print("Vecteur optimal :", best)

    def update_optimization(self):
        """
        Méthode appelée lorsque l'onglet devient actif.
        Remet à jour les bornes depuis geometry_settings et vide la sortie.
        """
        self.lowers_def = np.array(
            [geometry_settings.geometry_limits[k][0] for k in self.keys],
            dtype=float
        )
        self.uppers_def = np.array(
            [geometry_settings.geometry_limits[k][1] for k in self.keys],
            dtype=float
        )
        for w, low in zip(self.lowers_w, self.lowers_def):
            w.value = low
        for w, hi in zip(self.uppers_w, self.uppers_def):
            w.value = hi
        self.out.clear_output()

    def DE_general(self, *, budget, Npop, lowers, uppers, mode="dip"):
        """
        Differential Evolution “current-to-best/1/bin”.
        Utilise `self.sim.cost(x, mode)` pour évaluer.
        """
        # nombre de générations
        Ngen = int(budget / Npop)
        # facteurs DE
        F1, F2, cr = 0.9, 0.8, 0.8
        # nombre de paramètres
        n_params = len(lowers)

        # buffers
        cf   = np.zeros(Npop)
        conv = np.zeros(Ngen)

        # initialisation aléatoire dans les bornes
        arr = np.random.rand(Npop, n_params)
        arr = lowers + (uppers - lowers) * arr

        # évaluation initiale
        for i in range(Npop):
            cf[i] = self.sim.cost(arr[i], mode=mode)

        # boucle DE
        for g in range(Ngen):
            for p in range(Npop):
                idxs = np.random.choice(Npop, 3, replace=False)
                a, b, c = arr[idxs]
                best = arr[np.argmin(cf)]
                # mutation current-to-best
                y = c + F1*(a - b) + F2*(best - c)
                # crossover binomial
                mask = np.random.rand(n_params) < cr
                if not np.any(mask):
                    mask[np.random.randint(n_params)] = True
                z = np.where(mask, y, arr[p])
                # remise dans les bornes
                z = np.minimum(np.maximum(z, lowers), uppers)
                # sélection
                cfz = self.sim.cost(z, mode=mode)
                if cfz < cf[p]:
                    arr[p] = z
                    cf[p]  = cfz
            conv[g] = np.min(cf)

        # réévaluation finale
        cf_final   = np.array([
            self.sim.cost(arr[i], mode=mode) for i in range(Npop)
        ])
        best       = arr[np.argmin(cf)]
        best_final = arr[np.argmin(cf_final)]

        # .root si dispo
        try:
            import uproot, awkward as ak
            keys = self.keys
            constraints = {
                f"{k}_min": lowers[i] for i, k in enumerate(keys)
            }
            constraints.update({
                f"{k}_max": uppers[i] for i, k in enumerate(keys)
            })
            constraints.update({"Npop": Npop, "budget": budget})
            with uproot.recreate(f"cost_0.root") as root_file:
                root_file["constraints"] = {
                    k: np.array([v]) for k, v in constraints.items()
                }
                root_file["optimized"] = {
                    "conv":       conv,
                    "cf_final":   cf_final,
                    "best":       best.flatten(),
                    "best_final": best_final.flatten(),
                }
        except ImportError:
            print("uproot/awkward non dispo, .root ignoré")

        return conv, best


def create_optimization_tab(sim_obj):
    """
    Wrapper de compatibilité : renvoie le widget de l’onglet.
    """
    tab = OptimizationTab(sim_obj)
    return tab
