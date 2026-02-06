from __future__ import annotations

import logging
from pathlib import Path


def init_logger(log_file: str | Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("cdcircuit")
    logger.handlers = []
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(Path(log_file), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
