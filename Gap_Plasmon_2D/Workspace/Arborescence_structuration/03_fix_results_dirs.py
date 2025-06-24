#!/usr/bin/env python3
"""
Phase 3 : élimine dossiers fantômes notebooks/, Summary_Simulation/…
          + met à jour les chemins restants vers paths.SUMMARY_SIM_DIR & co.
"""
import shutil, re, logging, datetime
from pathlib import Path
import nbformat
from utils_struct import ROOT, setup_logger

log = setup_logger("03_results_dirs_log.txt")

PKG_NOTEBOOK_DIR = ROOT / "src/gap_plasmon_2d/notebooks"
RES_SUM_CAP      = ROOT / "results/Summary_Simulation"
RES_SUM_LC       = ROOT / "results/summary_simulation"

def move_dir(src: Path, dst: Path):
    if not src.exists(): return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        shutil.move(str(item), dst / item.name)
        logging.info(f"MOVE FILE : {item} → {dst/item.name}")
    shutil.rmtree(src); logging.info(f"DELETE DIR: {src}")

# 1. moves physiques
move_dir(PKG_NOTEBOOK_DIR, ROOT / "results")
move_dir(RES_SUM_CAP,       RES_SUM_LC)

# 2. patch code
REPL = {
    r'"notebooks/Summary_Simulation"'   : 'str(paths.SUMMARY_SIM_DIR)',
    r"'notebooks/Summary_Simulation'"   : 'str(paths.SUMMARY_SIM_DIR)',
    r'"Summary_Simulation"'             : '"summary_simulation"',
    r"'Summary_Simulation'"             : "'summary_simulation'",
}
IMPORT = "from gap_plasmon_2d import paths\n"

def patch_text(txt: str) -> str:
    new, touched = txt, False
    for pat, repl in REPL.items():
        rep = re.sub(pat, repl, new)
        if rep != new: new, touched = rep, True
    if touched and "gap_plasmon_2d import paths" not in new:
        new = IMPORT + new
    return new if touched else txt

def patch_py(p: Path):
    if p.name == "paths.py" and "gap_plasmon_2d" in p.parts: return
    src = p.read_text(encoding="utf-8")
    new = patch_text(src)
    if new != src:
        p.with_suffix(".py.bak").write_text(src, encoding="utf-8")
        p.write_text(new, encoding="utf-8")
        logging.info(f"PATCH PY : {p}")

def patch_ipynb(nb: Path):
    nbk = nbformat.read(nb, as_version=4); changed = False
    for c in nbk.cells:
        if c.cell_type != "code": continue
        ns = patch_text(c.source)
        if ns != c.source: c.source, changed = ns, True
    if changed:
        nb.with_suffix(".ipynb.bak").write_bytes(nbformat.writes(nbk).encode())
        nbformat.write(nbk, nb)
        logging.info(f"PATCH NOTE: {nb}")

for f in ROOT.rglob("*"):
    if f.is_symlink() or f.suffix in {".bak", ".pyc"}: continue
    if f.suffix == ".py": patch_py(f)
    elif f.suffix == ".ipynb": patch_ipynb(f)

logging.info("Phase 3 terminée ✔")
