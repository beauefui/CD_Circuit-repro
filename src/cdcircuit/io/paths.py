from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REQUIRED_CONDA_ENV = "cdcircuit"
REQUIRED_HF_HOME = Path("/mnt/nfs/zijie/huggingface_cache")
REQUIRED_OUTPUT_ROOT = Path("/mnt/nfs/zijie/cd_circuit_output")


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_dir: Path
    logs_dir: Path
    results_dir: Path
    figures_dir: Path
    artifacts_dir: Path


def _assert_path_under(path: Path, required_root: Path, name: str) -> None:
    resolved = path.expanduser().resolve()
    root = required_root.expanduser().resolve()
    if resolved != root:
        raise RuntimeError(f"{name} must be exactly {root}, got {resolved}")


def assert_runtime_constraints(output_root: str | Path | None = None) -> Path:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env != REQUIRED_CONDA_ENV:
        raise RuntimeError(
            f"All operations must run in conda env '{REQUIRED_CONDA_ENV}', got '{conda_env}'"
        )

    hf_home = Path(os.environ.get("HF_HOME", str(REQUIRED_HF_HOME)))
    _assert_path_under(hf_home, REQUIRED_HF_HOME, "HF_HOME")

    os.environ.setdefault("HF_HOME", str(REQUIRED_HF_HOME))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(REQUIRED_HF_HOME / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(REQUIRED_HF_HOME / "transformers"))

    if output_root is None:
        output_root = os.environ.get("CDC_OUTPUT_ROOT", str(REQUIRED_OUTPUT_ROOT))
    output_root_path = Path(output_root)
    _assert_path_under(output_root_path, REQUIRED_OUTPUT_ROOT, "CDC_OUTPUT_ROOT/output_root")

    os.environ.setdefault("CDC_OUTPUT_ROOT", str(REQUIRED_OUTPUT_ROOT))
    return output_root_path


def create_run_dirs(task: str, model_name: str, output_root: str | Path) -> RunPaths:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "-")
    run_id = f"{ts}_{task}_{safe_model}"
    run_dir = Path(output_root) / run_id

    logs_dir = run_dir / "logs"
    results_dir = run_dir / "results"
    figures_dir = run_dir / "figures"
    artifacts_dir = run_dir / "artifacts"

    for p in (run_dir, logs_dir, results_dir, figures_dir, artifacts_dir):
        p.mkdir(parents=True, exist_ok=False)

    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        logs_dir=logs_dir,
        results_dir=results_dir,
        figures_dir=figures_dir,
        artifacts_dir=artifacts_dir,
    )
