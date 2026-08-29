"""Execute the strategy snippets published in README.md and the Sphinx docs.

The blocks are read out of the real files, so a constructor signature change that
the documentation does not follow fails here instead of reaching a new user.
"""

import ast
import re
import textwrap
from pathlib import Path

import pytest

from keeks.bankroll import BankRoll
from keeks.binary_strategies import (
    CPPIStrategy,
    DrawdownAdjustedKelly,
    DynamicBankrollManagement,
    FixedFractionStrategy,
    FractionalKellyCriterion,
    KellyCriterion,
    MertonShare,
    NaiveStrategy,
    OptimalF,
)
from keeks.simulators.random_binary import RandomBinarySimulator
from keeks.simulators.random_uncertain_binary import RandomUncertainBinarySimulator
from keeks.simulators.repeated_binary import RepeatedBinarySimulator
from keeks.utils import crra_utility, expected_utility, find_indifference_price

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
GETTING_STARTED = REPO_ROOT / "docs" / "source" / "getting_started.rst"

MARKDOWN_BLOCK = re.compile(
    r"^(?P<indent>[ \t]*)```python\n(?P<body>.*?)^(?P=indent)```",
    re.DOTALL | re.MULTILINE,
)

# Docs are written for a reader who already has the library imported and a live
# bankroll value in hand; supply both so snippets can stay as short as they read.
NAMESPACE_SEED = {
    "BankRoll": BankRoll,
    "CPPIStrategy": CPPIStrategy,
    "DrawdownAdjustedKelly": DrawdownAdjustedKelly,
    "DynamicBankrollManagement": DynamicBankrollManagement,
    "FixedFractionStrategy": FixedFractionStrategy,
    "FractionalKellyCriterion": FractionalKellyCriterion,
    "KellyCriterion": KellyCriterion,
    "MertonShare": MertonShare,
    "NaiveStrategy": NaiveStrategy,
    "OptimalF": OptimalF,
    "RandomBinarySimulator": RandomBinarySimulator,
    "RandomUncertainBinarySimulator": RandomUncertainBinarySimulator,
    "RepeatedBinarySimulator": RepeatedBinarySimulator,
    "crra_utility": crra_utility,
    "expected_utility": expected_utility,
    "find_indifference_price": find_indifference_price,
    "current_bankroll": 1000.0,
}

# Calls that make a block unsuitable to execute here. The failures this file
# guards against are all construction-time, so nothing is lost by skipping them.
UNRUNNABLE_CALLS = {
    "evaluate_strategy": "runs a full simulation (slow and random)",
    "plot_history": "opens a matplotlib window",
}

# Blocks below this count mean the extraction broke rather than the docs shrinking.
MINIMUM_EXECUTED = {"README.md": 1, "getting_started.rst": 7}


def _markdown_blocks(text):
    for match in MARKDOWN_BLOCK.finditer(text):
        yield textwrap.dedent(match.group("body"))


def _rst_blocks(text):
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped != ".. code-block:: python":
            continue
        directive_indent = len(line) - len(stripped)
        body = []
        for candidate in lines[index + 1 :]:
            if not candidate.strip():
                body.append("")
                continue
            indent = len(candidate) - len(candidate.lstrip())
            if indent <= directive_indent:
                break
            body.append(candidate)
        yield textwrap.dedent("\n".join(body))


def _collect(path, extractor):
    blocks = []
    text = path.read_text()
    for number, code in enumerate(extractor(text), start=1):
        blocks.append(pytest.param(code, id=f"{path.name}-block{number}"))
    return blocks


DOCUMENTED_BLOCKS = _collect(README, _markdown_blocks) + _collect(
    GETTING_STARTED, _rst_blocks
)


def _skip_reason(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "not Python (a shell command in a python-tagged block)"
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in UNRUNNABLE_CALLS:
            return UNRUNNABLE_CALLS[node.attr]
    return None


@pytest.mark.parametrize("code", DOCUMENTED_BLOCKS)
def test_documented_block_executes(code):
    reason = _skip_reason(code)
    if reason is not None:
        pytest.skip(reason)
    exec(compile(code, "<docs>", "exec"), dict(NAMESPACE_SEED))


@pytest.mark.parametrize(
    ("path", "extractor"),
    [(README, _markdown_blocks), (GETTING_STARTED, _rst_blocks)],
    ids=["README.md", "getting_started.rst"],
)
def test_extraction_still_finds_the_documented_blocks(path, extractor):
    executed = [
        code for code in extractor(path.read_text()) if _skip_reason(code) is None
    ]
    assert len(executed) >= MINIMUM_EXECUTED[path.name]
