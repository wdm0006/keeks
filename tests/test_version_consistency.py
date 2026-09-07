"""Pin the package version sources together.

``version.py`` feeds the Sphinx build (``docs/source/conf.py``) while
``pyproject.toml`` drives packaging and PyPI; the two have drifted before.
"""

import importlib.metadata

from version import __version__


def test_version_module_matches_installed_package_metadata():
    assert importlib.metadata.version("keeks") == __version__
