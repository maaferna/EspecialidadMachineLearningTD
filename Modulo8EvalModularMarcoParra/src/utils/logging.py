"""Configuración de logging para el proyecto."""
from __future__ import annotations
import logging, sys
from pathlib import Path




def setup_logging(log_dir: str | Path, level: str = "INFO") -> None:
    """
    Configura el logging para que imprima en consola y guarde en archivo.
    Parameters
    ----------
    log_dir : str | Path
        Carpeta donde se guardará el archivo de log.
    level : str
        Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    Returns
    -------
    None
    -------
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(log_dir) / "run.log", encoding="utf-8"),
            ],
)