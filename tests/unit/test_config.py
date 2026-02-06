from pathlib import Path

from cdcircuit.io.config import load_config


def test_load_config_with_inherits(tmp_path: Path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"

    base.write_text("a: 1\nb:\n  c: 2\n", encoding="utf-8")
    child.write_text("inherits: base.yaml\nb:\n  d: 3\n", encoding="utf-8")

    cfg = load_config(child)
    assert cfg["a"] == 1
    assert cfg["b"]["c"] == 2
    assert cfg["b"]["d"] == 3
