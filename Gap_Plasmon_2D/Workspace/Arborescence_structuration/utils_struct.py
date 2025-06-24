#!/usr/bin/env python3
"""
utils_struct.py – helpers partagés
"""
from pathlib import Path
import shutil, logging, re

ROOT = Path(__file__).resolve()                    \
               .parents[4]      # ← Workspace/ (4 niveaux au-dessus)
LOG_DIR = Path(__file__).parent                    # Arborescence_structuration/

def setup_logger(log_name: str):
    log_path = LOG_DIR / log_name
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("#" * 72)
    return log_path

def safe_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"MOVE: {src}  →  {dst}")
    shutil.move(src, dst)

def safe_rename(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"RENAME: {src}  →  {dst}")
    src.rename(dst)

def purge_empty_dirs(start: Path):
    """Supprime récursivement les dossiers vides + __pycache__."""
    for p in list(start.rglob("*"))[::-1]:
        if p.is_dir() and (not any(p.iterdir()) or p.name == "__pycache__"):
            logging.info(f"DELETE DIR: {p}")
            shutil.rmtree(p, ignore_errors=True)

# Mini convertisseur Camel → snake
def camel_to_snake(name: str) -> str:
    stem, ext = Path(name).stem, Path(name).suffix.lower()
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", stem)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower() + ext
