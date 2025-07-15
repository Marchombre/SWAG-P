# -*- coding: utf-8 -*-
"""
cost_function.py  –  moteur d’évaluation du coût
========================================================
Indépendant d’ipywidgets ➜ utilisable directement dans multiprocessing.
"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from copy import deepcopy

from gap_plasmon_2d.simulation.simulate_and_plot import run_simulation_one_combo
from gap_plasmon_2d.analysis.characterization    import find_best_dip


# ------------------------------------------------------------------ #
def compute_cost(
    sim_tab,                   # instance partagée de SimulationTab
    x:       np.ndarray,       # vecteur optimisé
    keys:    List[str],        # noms des variables optimisées
    *,                          # kwargs nommés uniquement
    mode:          str = "dip",
    fixed_lambda:  float | None = None,
    range_lambda:  tuple[float, float] | None = None,
) -> float:
    """
    Renvoie la métrique à *minimiser* (1 – R ou 1 – ΔR/Δn, etc.).
    """
    # 1) configuration active copiée pour ne pas salir l’originale
    cfg = deepcopy(next(
        c for c in sim_tab.all_configs
        if sim_tab.config_checkboxes[c["config_name"]].value
    ))

    # 2) injection des épaisseurs optimisées
    for xi, k in zip(x, keys):
        cfg["geometry"]["geometry"][k] = float(xi)

    # 3) réglages généraux
    lam = np.linspace(sim_tab.sim_lambda_min.value,
                      sim_tab.sim_lambda_max.value,
                      sim_tab.sim_n_points.value)
    wave     = {"angle": 0, "polarization": 1}
    n_modes  = sim_tab._get_n_modes_for(cfg["config_name"])
    sel_layers = list(sim_tab.layer_selector.value)
    delta_n    = max(sim_tab.delta_n_widget.value, 1e-6)

    # 4) spectre de base (Rup seulement)
    Rup0, _, _ = run_simulation_one_combo(
        lam, wave, n_modes, cfg, sim_tab.json_combined_path
    )
    Rup0 = np.asarray(Rup0, float)

    # 5) métriques simples
    fixed_lambda = fixed_lambda or sim_tab.lambda0_in.value
    if mode == "fixed_lambda":
        R = float(np.interp(fixed_lambda, lam, Rup0))
        return 1.0 - R

    if mode == "range_lambda":
        lam_min, lam_max = range_lambda
        msk = (lam >= lam_min) & (lam <= lam_max)
        R_mean = float(np.mean(Rup0[msk]))
        return 1.0 - R_mean

    # 6) métriques dip / half
    best_out, *_ = find_best_dip(
        cfg=cfg, wavelength=lam, reflectance=Rup0,
        wave=wave, n_modes=n_modes,
        sel_layers=sel_layers, delta_n=delta_n,
        json_combined_path=sim_tab.json_combined_path,
        smooth_win=0, polyorder=0,
        dip_prom=1e-2, dip_dist=1, peak_dist=1,
        verbose=False, cfg_name=cfg["config_name"],
        mode=('half' if mode == "half" else 'dip')
    )
    if best_out is None:           # aucun dip valide
        return 1.0

    idx = 13 if mode == "dip" else 15        # index de ΔR/Δn ou ΔR_half
    return 1.0 - float(best_out[idx])
