"""
gap_plasmon_2d.paths
====================

Contient les répertoires et fichiers de référence du projet.
Aucune dépendance vers un autre module du paquet ➜ aucun risque de boucle.
"""

from pathlib import Path

# ───────────────────────── Chemins de base ──────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]        # racine du dépôt

CONFIGS_DIR = ROOT_DIR / "configs"                    # paramètres JSON/YAML
DATA_DIR    = ROOT_DIR / "data"                       # catalogues n,k et autres données
RESULTS_DIR = ROOT_DIR / "results"                    # tout ce que l’on produit

# ───────────────────────── Fichiers catalogues ───────────────────────
CATALOG_NK = DATA_DIR / "catalog_nk.yml"
CATALOG_N2 = DATA_DIR / "catalog-n2.yml"

# ───────────────────────── Helpers optionnels ────────────────────────
def ensure_dirs() -> None:
    """Crée les sous-dossiers standard de results/ si absents."""
    for d in (
        RESULTS_DIR,
        RESULTS_DIR / "figures",
        RESULTS_DIR / "summary_optimisation",
        RESULTS_DIR / "summary_simulation",
        RESULTS_DIR / "summary_convergence",
    ):
        d.mkdir(parents=True, exist_ok=True)

SUMMARY_SIM_DIR  = RESULTS_DIR / "summary_simulation"      # minuscule unique
SUMMARY_OPT_DIR  = RESULTS_DIR / "summary_optimisation"
EXPERIMENTAL_DIR = RESULTS_DIR / "experimental"
H5_RESULTS_DIR   = SUMMARY_SIM_DIR    