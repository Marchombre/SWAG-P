#!/usr/bin/env python3
"""
Phase 1 : restructure le dépôt (déplacement dossiers/fichiers).
Exécuter **une seule fois**.
"""
from pathlib import Path
import datetime, shutil
from utils_struct import ROOT, setup_logger, safe_move, safe_rename, purge_empty_dirs, camel_to_snake

log = setup_logger("01_restructure_log.txt")
from gap_plasmon_2d import paths   # déjà valide avant la migration v2

# -----------------------------------------------------------------------------
# 1. renommer / déplacer les dossiers racine
# -----------------------------------------------------------------------------
mapping_dirs = {
    "CONFIGURATIONS"             : "configs",
    "Convergence"                : "results/summary_convergence",
    "Figures"                    : "results/figures",
    "Images"                     : "docs/assets",
}
for old, new in mapping_dirs.items():
    src = ROOT / old
    if src.exists():
        safe_rename(src, ROOT / new)

# results + sous-dossiers minimum
for sub in ["experimental", "summary_optimisation", "summary_simulation"]:
    (ROOT / "results" / sub).mkdir(parents=True, exist_ok=True)

# .gitignore générique dans results/
if not (ROOT / "results" / ".gitignore").exists():
    (ROOT / "results" / ".gitignore").write_text("*\n!/.gitignore\n")

# -----------------------------------------------------------------------------
# 2. catalogues YAML → /data
# -----------------------------------------------------------------------------
(ROOT / "data").mkdir(exist_ok=True)
for yml in ROOT.glob("catalog*.yml"):
    safe_move(yml, ROOT / "data" / yml.name)

# -----------------------------------------------------------------------------
# 3. Ancien répertoire modules/tools → nouvelle arbo  src/gap_plasmon_2d/*
# -----------------------------------------------------------------------------
def target_path(file_name: str) -> Path:
    pkg = Path("src/gap_plasmon_2d")
    fn = camel_to_snake(file_name)
    fn_low = file_name.lower()
    if file_name == "__init__.py":
        return pkg / "__init__.py"
    if any(k in fn_low for k in ["interactive", "geometry_settings", "material_selector", "widget"]):
        return pkg / "ui" / fn
    if any(k in fn_low for k in ["simulate", "simulation", "difference"]):
        return pkg / "simulation" / fn
    if "optimisation" in fn_low or "optimization" in fn_low:
        return pkg / "optimisation" / fn
    if any(k in fn_low for k in ["characterization", "convergence_analysis"]):
        return pkg / "analysis" / fn
    if any(k in fn_low for k in ["material_", "refractiveindex"]):
        return pkg / "materials" / fn
    if any(k in fn_low for k in ["rcwa", "models"]):
        return pkg / "models" / fn
    return pkg / "utils" / fn

legacy_dump = ROOT / "results/legacy_modules_dump"
for root_name in ["modules", "tools"]:
    root = ROOT / root_name
    if not root.exists():
        continue
    for item in root.rglob("*"):
        if item.is_file():
            if item.suffix == ".py":
                safe_move(item, ROOT / target_path(item.name))
            else:
                safe_move(item, legacy_dump / item.relative_to(root))
    purge_empty_dirs(root)
    if root.exists():
        shutil.rmtree(root)

# -----------------------------------------------------------------------------
# 4. __init__.py manquants
# -----------------------------------------------------------------------------
for pkg in (ROOT / "src" / "gap_plasmon_2d").rglob("*"):
    if pkg.is_dir() and not (pkg / "__init__.py").exists():
        (pkg / "__init__.py").touch()

purge_empty_dirs(ROOT / "src" / "gap_plasmon_2d")
print("Phase 1 terminée ✔  (voir 01_restructure_log.txt)")
