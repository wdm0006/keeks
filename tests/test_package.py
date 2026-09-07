"""Pin the package root's public surface: __version__ and documented re-exports.

version.py is the single version source (the Sphinx build imports it and
hatchling's dynamic versioning consumes it), so the runtime __version__ must
match it exactly. The re-export set is the documented public API: the classes
autodoc'd across docs/source/*.rst plus the names the README and CLAUDE.md
reference, so a rename that documentation does not follow fails here.
"""

import ast
from pathlib import Path

import keeks
import keeks.bankroll
import keeks.binary_strategies
import keeks.binary_strategies.base
import keeks.simulators
import keeks.utils

REPO_ROOT = Path(__file__).resolve().parent.parent


def _version_py_version():
    """Parse version.py without importing it (the root is not a package)."""
    tree = ast.parse((REPO_ROOT / "version.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("version.py does not define __version__")


def test_version_matches_single_source():
    assert keeks.__version__ == _version_py_version()


def test_version_matches_installed_distribution():
    from importlib.metadata import version

    assert keeks.__version__ == version("keeks")


def test_reexports_are_the_documented_classes():
    assert keeks.BankRoll is keeks.bankroll.BankRoll
    assert keeks.BaseStrategy is keeks.binary_strategies.base.BaseStrategy
    assert keeks.KellyCriterion is keeks.binary_strategies.KellyCriterion
    assert keeks.RepeatedBinarySimulator is keeks.simulators.RepeatedBinarySimulator
    assert keeks.RuinError is keeks.utils.RuinError
    assert keeks.crra_utility is keeks.utils.crra_utility


def test_all_names_resolve():
    for name in keeks.__all__:
        assert getattr(keeks, name) is not None, name
