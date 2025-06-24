from gap_plasmon_2d import paths
#!/usr/bin/env python3
# -*- coding: utf‑8 -*-
"""
Module : Saving_Functions.py

Génération / écriture des résumés de simulation et sauvegarde des figures.
"""

import os, re
from datetime import datetime
import numpy as np
import h5py
from pathlib import Path
from gap_plasmon_2d.utils.data_readers import get_material_str_clean


def sanitize_filename(name):
    """Remplace les caractères non autorisés dans un nom de fichier par un underscore."""
    # Interdit : / \ ? % * : | " < > et supprime les doubles espaces
    import re
    # Remplace slashs et caractères interdits par "_"
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    # Remplace aussi les slashs restants pour Unix
    name = name.replace('/', '_')
    # Optionnel : supprime les espaces multiples consécutifs
    name = re.sub(r'\s+', ' ', name)
    return name.strip()


# --------------------------------------------------------------------------- #
#                        RÉSUMÉ TEXTE DE SIMULATION                            #
# --------------------------------------------------------------------------- #
def save_simulation_summary(
    simulation_details: dict,             # {config_name: details_dict}
    lambda_range,                         # np.ndarray
    wave: dict,                           # {"angle":…, "polarization":…}
    n_mod,                                # int ou list[int]
    summary_dir: str,
    custom_name: str | None = None,
    *,
    fwhm_summaries=None,
    lam_summaries=None,
    delta_lam_over_midLam_summaries=None,
    Q_factor=None,
    best_S_R=None,
    comp_summaries=None
) -> str:
    """
    Écrit un fichier *.txt* résumant chaque configuration simulée.

    Les nouvelles métriques éventuelles (stockées dans
    ``details["extra_metrics"]`` au niveau de **simulation.py**) sont écrites
    automatiquement ; il n’est pas nécessaire de modifier la signature.
    """
    # ---------------- nom de fichier ----------------
    if custom_name and custom_name.strip():
        base = custom_name.strip()
        if not base.startswith("simulation_summary_RCWA_"):
            base = "simulation_summary_RCWA_" + base
    else:
        base = "simulation_summary_RCWA_" + get_material_str_clean(simulation_details)
    safe_base = sanitize_filename(base)
    out_path = os.path.join(summary_dir, f"{safe_base}.txt")


    os.makedirs(summary_dir, exist_ok=True)

    # ---------------- entête ------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = [
        "Simulation Summary - All Geometry/Material Combos",
        f"Timestamp: {stamp}",
        f"Wave parameters: {wave}",
    ]
    multi = isinstance(n_mod, (list, tuple, np.ndarray))
    if not multi:
        lines.append(f"Number of RCWA modes: {n_mod}")
    lines.append("---- COMBINATIONS ----\n")

    # ---------------- corps -------------------------
    sim_names = list(simulation_details.keys())
    for idx, name in enumerate(sim_names):
        det = simulation_details[name]

        lines.append(f"Combo name: {name}")
        if multi:
            lines.append(f"  Number of RCWA modes: {n_mod[idx]}")
        lines.append("Geometry:")
        lines.append(str(det["geometry"]))
        lines.append("Material config (df_config):")
        lines.append(str(det["material_config"]))
        lines.append(f"RI Overrides: {det['ri_overrides']}")

        # ----- métriques classiques -----------------
        if any(x is not None for x in (fwhm_summaries,
                                       lam_summaries,
                                       delta_lam_over_midLam_summaries,
                                       Q_factor,
                                       best_S_R,
                                       comp_summaries)):
            lines.append("Metrics:")

        def _add(label, seq):
            if seq is not None:
                lines.append(f"  {label:<13}: {seq[idx]}")
        _add("FWHM (nm)",         fwhm_summaries)
        _add("Lam_res",           lam_summaries)
        _add("Δλ / λmin or λsym", delta_lam_over_midLam_summaries)
        _add("Q-factor",          Q_factor)
        _add("best_S_R",          best_S_R)
        _add("Score total",       comp_summaries)

        # ----- nouvelles métriques (extra_metrics) --
        for k, v in det.get("extra_metrics", {}).items():
            if v not in ("", None):
                lines.append(f"  {k:<13}: {v}")

        # ----- points de réflectance ---------------
        lines.append("Reflectance points (Rup, Rdown):")
        Rup  = det["Rup"]
        Rdown= det["Rdown"]
        if len(lambda_range) != len(Rup) or len(Rup) != len(Rdown):
            raise ValueError(f"Mismatch de longueurs pour '{name}'")
        for lam, ru, rd in zip(lambda_range, Rup, Rdown):
            lines.append(f"  λ={lam} nm -> Rup={ru}, Rdown={rd}")
        lines.append("-" * 40)
        
        # --- spectre décalé n₀+Δn (facultatif) ----------------------
        if "Rup_dn" in det:
            Rup_dn   = np.asarray(det["Rup_dn"])
            Rdown_dn = np.asarray(det.get("Rdown_dn", []))
            lines.append("Reflectance points (Rup_dn" +
                         (", Rdown_dn" if Rdown_dn.size else "") + "):")
            for i, lam_pt in enumerate(lambda_range):
                up_dn = Rup_dn[i]
                if Rdown_dn.size:
                    dw_dn = Rdown_dn[i]
                    lines.append(f"  λ={lam_pt} nm -> Rup_dn={up_dn}, Rdown_dn={dw_dn}")
                else:
                    lines.append(f"  λ={lam_pt} nm -> Rup_dn={up_dn}")

    # ---------------- écriture disque --------------
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Résumé enregistré : {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
#                               FIGURE PNG                                    #
# --------------------------------------------------------------------------- #
def save_figure(fig, title, figures_dir, material_str_clean: str | None = None):
    """
    Sauvegarde la figure *fig* dans *figures_dir*.
    """
    os.makedirs(figures_dir, exist_ok=True)
    tag = material_str_clean or re.sub(r'[^A-Za-z0-9_]', '', title)
    path = os.path.join(figures_dir, f"Simulation_Reflectance_Spectra_{tag}.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"Figure saved : {path}")
    return path





# --------------------------------------------------------------------------- #
#                               Optimization files                            #
# --------------------------------------------------------------------------- #
def save_optimization_hdf5(
    notebook_dir: str,
    run_id: str,
    budget: int,
    Npop: int,
    keys: list[str],
    lowers: np.ndarray,
    uppers: np.ndarray,
    conv: np.ndarray,
    cf_final: np.ndarray,
    best: np.ndarray,
    best_final: np.ndarray,
    mode: str,
    lam: np.ndarray | None = None,
    Rup: np.ndarray | None = None,
    Rdown: np.ndarray | None = None
) -> str:
    """
    Enregistre les résultats de DE_general dans :
      <notebook_dir>/Summary_Optimization/Opt_<run_id>_<timestamp>.h5

    - notebook_dir   : chemin vers notebooks/ (ex. './notebooks')
    - run_id         : identifiant court pour la passe (ex. 'budget100_pop30')
    - budget, Npop   : paramètres DE
    - keys, lowers, uppers : dimensions optimisées et leurs bornes
    - conv, cf_final : historiques de coût
    - best, best_final : vecteurs optimaux
    - mode           : 'dip' ou 'half'
    """
    summary_opti_dir = Path(notebook_dir) / "summary_optimisation"
    summary_opti_dir.mkdir(parents=True, exist_ok=True)

    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Opt_{run_id}_{stamp}.h5"
    h5path   = summary_opti_dir / filename

    with h5py.File(h5path, "a") as f:
        grp = f.require_group(f"{run_id}_{stamp}")

        # 1) Méta
        grp.attrs.update({
            "date":   datetime.now().isoformat(),
            "run_id": run_id,
            "budget": budget,
            "Npop":   Npop,
            "mode":   mode
        })

        # 2) Paramètres optimisés
        p = grp.require_group("parameters")
        p.create_dataset("keys",   data=np.array(keys, dtype='S'))
        p.create_dataset("lowers", data=lowers)
        p.create_dataset("uppers", data=uppers)

        # 3) Convergence et coûts
        grp.create_dataset("convergence", data=conv,     compression="gzip")
        grp.create_dataset("cf_final",    data=cf_final, compression="gzip")

        # 4) Meilleurs vecteurs
        grp.create_dataset("best",       data=best,       compression="gzip")
        grp.create_dataset("best_final", data=best_final, compression="gzip")
        
        
        # 5) Spectres du meilleur, s’ils sont fournis
        if lam is not None and Rup is not None:
            spec = grp.require_group("spectra")
            spec.create_dataset("wavelength", data=lam,  compression="gzip")
            spec.create_dataset("Rup",        data=Rup,  compression="gzip")
            if Rdown is not None:
                spec.create_dataset("Rdown",   data=Rdown,compression="gzip")        
        

    print(f"[Saving] Optimization saved to {h5path}")
    return str(h5path)

