from pathlib import Path

import pytest

from cdcircuit.io.paths import (
    REQUIRED_HF_HOME,
    REQUIRED_OUTPUT_ROOT,
    assert_runtime_constraints,
    create_run_dirs,
)


def test_assert_runtime_constraints_success(monkeypatch):
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "cdcircuit")
    monkeypatch.setenv("HF_HOME", str(REQUIRED_HF_HOME))
    out = assert_runtime_constraints(str(REQUIRED_OUTPUT_ROOT))
    assert out == REQUIRED_OUTPUT_ROOT


def test_assert_runtime_constraints_fail_env(monkeypatch):
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
    monkeypatch.setenv("HF_HOME", str(REQUIRED_HF_HOME))
    with pytest.raises(RuntimeError):
        assert_runtime_constraints(str(REQUIRED_OUTPUT_ROOT))


def test_create_run_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "cdcircuit")
    monkeypatch.setenv("HF_HOME", str(REQUIRED_HF_HOME))
    root = tmp_path
    p = create_run_dirs(task="greater_than", model_name="gpt2-small", output_root=root)
    assert p.run_dir.exists()
    assert (p.run_dir / "logs").exists()
    assert (p.run_dir / "results").exists()
    assert (p.run_dir / "figures").exists()
    assert (p.run_dir / "artifacts").exists()
