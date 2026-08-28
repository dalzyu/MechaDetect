from pathlib import Path

from aigc_detector.config import load_config


def test_load_config_expands_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EXAMPLE_ROOT", "E:/example")
    path = tmp_path / "config.yaml"
    path.write_text("path: ${EXAMPLE_ROOT}/data\nvalue: 42\n", encoding="utf-8")
    assert load_config(path) == {"path": "E:/example/data", "value": 42}
