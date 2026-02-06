from __future__ import annotations

from pathlib import Path

from cdcircuit.io.config import load_config
from cdcircuit.io.logging import init_logger
from cdcircuit.io.paths import assert_runtime_constraints, create_run_dirs
from cdcircuit.io.save import save_json, save_yaml


def run_greater_than_pipeline(config_path: str | Path) -> Path:
    cfg = load_config(config_path)

    output_root = cfg.get("output_root")
    output_root = assert_runtime_constraints(output_root=output_root)

    task = "greater_than"
    model_name = cfg.get("model", {}).get("name", "gpt2-small")
    run_paths = create_run_dirs(task=task, model_name=model_name, output_root=output_root)

    logger = init_logger(run_paths.logs_dir / "run.log", cfg.get("log_level", "INFO"))
    logger.info("Run started: %s", run_paths.run_id)
    logger.info("Config loaded from: %s", cfg.get("_config_path"))

    save_yaml(cfg, run_paths.run_dir / "config_snapshot.yaml")
    save_json(
        {
            "status": "initialized",
            "task": task,
            "model": model_name,
            "run_id": run_paths.run_id,
        },
        run_paths.results_dir / "metrics.json",
    )

    logger.info("Run initialized at: %s", run_paths.run_dir)
    return run_paths.run_dir
