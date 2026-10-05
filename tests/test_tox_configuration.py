import configparser
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_tox_installs_declared_dev_extra():
    project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(project_file.read_text(encoding="utf-8"))
    tox_config = configparser.ConfigParser(interpolation=None)
    tox_config.read_string(project["tool"]["tox"]["legacy_tox_ini"])

    extras = tox_config["testenv"]["extras"].split()
    assert "dev" in extras
    assert "dev" in project["project"]["optional-dependencies"]
