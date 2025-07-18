# -*- coding: utf-8 -*-
"""
cost_function.py  –  moteur d’évaluation du coût
========================================================
Indépendant d’ipywidgets ➜ utilisable directement dans multiprocessing.
"""
from __future__ import annotations

import logging
logger = logging.getLogger(__name__)


from typing import List, Dict, Any, Mapping
import numpy as np
from copy import deepcopy

from gap_plasmon_2d.simulation.simulate_and_plot import run_simulation_one_combo
from gap_plasmon_2d.analysis.characterization    import find_best_dip


# ------------------------------------------------------------------ #
def compute_cost(
    sim_tab,                       # instance SimulationTab
    x:       np.ndarray,           # vecteur optimisé
    keys:    List[str],            # variables optimisées
    *,                              # kwargs nommés uniquement
    mode:          str  = "dip",
    fixed_lambda:  float | None = None,
    range_lambda:  tuple[float, float] | None = None,
    delta_n: float | None = None,
    sel_layers: list[int] | None = None,
    n_modes:       int   | None = None,          
    selected_cfg:  Mapping | None = None,   
    square_ratio: bool = False,     
) -> float:
    """
    Renvoie la métrique à *minimiser* (1 – R ou 1 – ΔR/Δn, etc.).
    """

    # 0) Δn : si l’appelant ne fournit rien → on prend le widget
    if delta_n is None:
        delta_n = sim_tab.delta_n_widget.value

    if sel_layers is None:
        sel_layers = list(sim_tab.layer_selector.value)        

    if mode in ("dip", "half"):
        if delta_n is None or delta_n <= 0:
            raise ValueError("delta_n doit être >0 pour calculer la sensibilité.")
    else:
        # fixed_lambda ou range_lambda ⇒ on ignore delta_n
        delta_n = 0.0


    # 1) Choix de la configuration
    if selected_cfg is None:       # ⇢ appel depuis l’onglet Simulation
        cfg = deepcopy(next(
            c for c in sim_tab.all_configs
            if sim_tab.config_checkboxes[c["config_name"]].value
        ))
    else:                          # ⇢ appel depuis l’onglet Optimisation
        cfg = deepcopy(selected_cfg)        # déjà la bonne structure

    # 2) injection des épaisseurs optimisées
    for xi, k in zip(x, keys):
        cfg["geometry"]["geometry"][k] = float(xi)

    # En mode square, on n’applique qu’une seule valeur, issue de x[0].
    if square_ratio and ('thick_reso' in keys or 'width_reso' in cfg["geometry"]["geometry"]):
        # Si le carré est activé, x[0] est la valeur commune à appliquer
        val = float(x[0])
        if 'thick_reso' in cfg["geometry"]["geometry"]:
            cfg["geometry"]["geometry"]['thick_reso'] = val
        if 'width_reso' in cfg["geometry"]["geometry"]:
            cfg["geometry"]["geometry"]['width_reso'] = val



    # 3) réglages généraux
    lam = np.linspace(sim_tab.sim_lambda_min.value,
                      sim_tab.sim_lambda_max.value,
                      sim_tab.sim_n_points.value)
    wave     = {"angle": 0, "polarization": 1}
    
    # Nombre de modes RCWA contrôlé par l’appelant
    if n_modes is None:
        n_modes = sim_tab._get_n_modes_for(cfg["config_name"])
    
    

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
    if best_out is None:
        return 1.0

    # Unpack explicite de best_out, dans l'ordre défini par find_best_dip
    (
       lam_left, lam_right,
       fwhm, depth,
       lam_dip, R_dip, ylev,
       lam_max_l, R_max_l,
       lam_max_r, R_max_r,
       lam_sym, R_sym,
       best_dR,    # ← ΔR/Δn au point de mesure (dip ou half)
       best_Slam,  # ← Δλ/Δn
       best_dR_half,
       dip_idx_list,
       dR_over_dn_list,
       dLam_over_dn_list
    ) = best_out

    # On utilise toujours best_dR, qui vaut :
    #  - en mode "dip"  → ΔR(λ₀) / Δn
    #  - en mode "half" → (R_half_dn - R_half_base) / Δn
    sens = best_dR

    return 1.0 - sens
