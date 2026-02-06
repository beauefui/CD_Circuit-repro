from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ConfigDict = dict[str, Any]


def _deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: str | Path) -> ConfigDict:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    inherits = cfg.pop("inherits", None)
    if inherits:
        parent_path = (config_path.parent / inherits).resolve()
        parent_cfg = load_config(parent_path)
        cfg = _deep_merge(parent_cfg, cfg)

    cfg["_config_path"] = str(config_path)
    return cfg
