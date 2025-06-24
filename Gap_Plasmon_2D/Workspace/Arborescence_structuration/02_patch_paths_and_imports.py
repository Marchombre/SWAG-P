#!/usr/bin/env python3
"""
Phase 2 : corrige 100 % des imports & chemins après la restructuration.
• construit le mapping modules à partir de 01_restructure_log.txt
• remplace dans .py & .ipynb :
  – imports obsolètes → nouveaux imports
  – chemins « CONFIGURATIONS », « data », « notebooks/Summary_* », ".h5"…
  – concaténations workspace_dir, module_dir, os.path.join…
• ajoute l'import `from gap_plasmon_2d import paths` si fichier modifié.
"""

import ast, re, shutil, datetime, json, logging
from pathlib import Path
import nbformat
from utils_struct import ROOT, setup_logger, camel_to_snake

log = setup_logger("02_patch_paths_log.txt")

# ----------------------------------------------------------------- 1. mapping
def build_mapping() -> dict[str, str]:
    mapp = {}
    rx = re.compile(r"MOVE:\s+(.+?)\s+→\s+(.+)")
    for line in (Path(__file__).parent / "01_restructure_log.txt").read_text(encoding="utf-8").splitlines():
        m = rx.search(line)
        if not m:
            continue
        old_p, new_p = Path(m.group(1)), Path(m.group(2))
        if old_p.suffix != ".py":
            continue
        def to_mod(p: Path) -> str:
            parts = list(p.with_suffix("").parts)
            while parts and parts[0] not in ("src",):
                parts.pop(0)
            if parts and parts[0] == "src":
                parts.pop(0)
            return ".".join(parts)
        old_m, new_m = to_mod(old_p), to_mod(new_p)
        if old_m and new_m and old_m != new_m:
            mapp[old_m] = new_m
    return mapp

MOD_MAPPING = build_mapping()

# ----------------------------------------------------------------- 2. regex
IMPORT_RX = re.compile(r'^(?P<i>\s*)(?P<kw>from|import)\s+(?P<mod>[A-Za-z0-9_.]+)', re.M)
LIT_REPL = {
    # chemins simples
    r'"CONFIGURATIONS/"'   : 'str(paths.CONFIGS_DIR / "")',
    r"'CONFIGURATIONS/'"   : 'str(paths.CONFIGS_DIR / "")',
    r'"CONFIGURATIONS"'    : 'str(paths.CONFIGS_DIR)',
    r"'CONFIGURATIONS'"    : 'str(paths.CONFIGS_DIR)',

    # notebooks déplacés
    r'"notebooks/Summary_Simulation"'   : 'str(paths.SUMMARY_SIM_DIR)',
    r"'notebooks/Summary_Simulation'"   : 'str(paths.SUMMARY_SIM_DIR)',
    r'"notebooks/Summary_Optimization"' : 'str(paths.SUMMARY_OPT_DIR)',
    r"'notebooks/Summary_Optimization'" : 'str(paths.SUMMARY_OPT_DIR)',
    r'"notebooks/Experimental_Data"'    : 'str(paths.EXPERIMENTAL_DIR)',
    r"'notebooks/Experimental_Data'"    : 'str(paths.EXPERIMENTAL_DIR)',

    # catalogues
    r'"catalog_nk\.yml"'  : 'paths.CATALOG_NK',
    r"'catalog_nk\.yml'"  : 'paths.CATALOG_NK',
    r'"catalog\-n2\.yml"' : 'paths.CATALOG_N2',
    r"'catalog\-n2\.yml'" : 'paths.CATALOG_N2',

    # hdf5
    r'"simulation_results\.h5"' : 'str(paths.SUMMARY_SIM_DIR / "simulation_results.h5")',
    r"'simulation_results\.h5'": 'str(paths.SUMMARY_SIM_DIR / "simulation_results.h5")',
}

PATH_PARENT_RX = re.compile(r'Path\(__file__\)(?:\.resolve\(\))?(?:\.parent(?:\.\w+)*)?\s*/\s*[\'"]data[\'"]', re.I)
JOIN_WS_RX     = re.compile(r'os\.path\.join\(\s*workspace_dir\s*,\s*str\(paths\.(\w+?)\)\s*(,?)', re.I)
WS_LITERAL_RX  = re.compile(r'workspace_dir\s*\+\s*.[\\/]\s*[\'"]([^\'"]+)[\'"]')

IMPORT_INSERT = "from gap_plasmon_2d import paths\n"

def patch_source(code: str, add_import: bool) -> str:
    new, touched = code, False

    # A. imports python
    for m in IMPORT_RX.finditer(code):
        mod = m.group("mod")
        if mod in MOD_MAPPING or any(mod.startswith(k + ".") for k in MOD_MAPPING):
            repl = mod
            for k, v in MOD_MAPPING.items():
                repl = repl.replace(k, v)
            before = m.group(0)
            after  = before.replace(mod, repl, 1)
            new = new.replace(before, after)
            touched = True

    # B. littéraux directs
    for pat, repl in LIT_REPL.items():
        patched = re.sub(pat, repl, new)
        if patched != new:
            new, touched = patched, True

    # C. Path(__file__).parent / "data"
    new2 = PATH_PARENT_RX.sub("paths.DATA_DIR", new)
    if new2 != new:
        new, touched = new2, True

    # D. os.path.join(workspace_dir, str(paths.X_DIR), …)
    new2 = JOIN_WS_RX.sub(r'os.path.join(str(paths.\1)\2', new)
    if new2 != new:
        new, touched = new2, True

    # E. workspace_dir + "/something"
    def _rep_ws(m): return f'str(paths.ROOT_DIR / "{m.group(1)}")'
    new2 = WS_LITERAL_RX.sub(_rep_ws, new)
    if new2 != new:
        new, touched = new2, True

    # F. inject import
    if touched and add_import and "gap_plasmon_2d import paths" not in new:
        new = IMPORT_INSERT + new

    return new if touched else code

# ----------------------------------------------------------------- 3. patchers
def skip_paths_py(p): return p.name == "paths.py" and "gap_plasmon_2d" in p.parts

def patch_py(p: Path):
    if skip_paths_py(p): return
    src = p.read_text(encoding="utf-8")
    new = patch_source(src, add_import=True)
    if new != src:
        p.with_suffix(".py.bak").write_text(src, encoding="utf-8")
        p.write_text(new, encoding="utf-8")
        logging.info(f"PATCH PY   : {p}")

def patch_ipynb(p: Path):
    nb = nbformat.read(p, as_version=4)
    changed = False
    for cell in nb.cells:
        if cell.cell_type != "code": continue
        ns = patch_source(cell.source, add_import=True)
        if ns != cell.source:
            cell.source, changed = ns, True
    if changed:
        p.with_suffix(".ipynb.bak").write_bytes(nbformat.writes(nb).encode())
        nbformat.write(nb, p)
        logging.info(f"PATCH NOTE : {p}")

# ----------------------------------------------------------------- 4. main
def main():
    logging.info(f"Start phase 2 – {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    for f in ROOT.rglob("*"):
        if f.is_symlink() or f.suffix in {".bak", ".pyc"}: continue
        if f.suffix == ".py":      patch_py(f)
        elif f.suffix == ".ipynb": patch_ipynb(f)
    logging.info("Phase 2 terminée ✔")

if __name__ == "__main__":
    main()
