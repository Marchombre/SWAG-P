# gap_plasmon_2d/utils/logging_setup.py

import logging
from pathlib import Path
from gap_plasmon_2d import paths

def setup_subpackage_loggers() -> None:
    """
    Pour chaque sous-package de gap_plasmon_2d (sauf 'utils' et ceux débutant par '__'),
    crée un dossier log/ et un fichier <subpackage>_log.log, et configure un logger
    qui écrit uniquement dans ce fichier.
    """
    pkg_dir = paths.PACKAGE_DIR  # chemin vers gap_plasmon_2d/
    for sub in pkg_dir.iterdir():
        # on ne traite que les vrais dossiers de code
        if not sub.is_dir() or sub.name.startswith("__") or sub.name == "utils":
            continue

        # 1) création du dossier gap_plasmon_2d/<sub>/log/
        log_dir = sub / "log"
        log_dir.mkdir(parents=True, exist_ok=True)

        # 2) création du fichier gap_plasmon_2d/<sub>/log/<sub>_log.log
        log_file = log_dir / f"{sub.name}_log.log"
        log_file.touch(exist_ok=True)

        # 3) configuration du logger
        logger_name = f"gap_plasmon_2d.{sub.name}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        # n’ajoute pas plusieurs handlers si on rappelle la fonction
        if any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
               for h in logger.handlers):
            continue

        handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fmt = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.propagate = False
