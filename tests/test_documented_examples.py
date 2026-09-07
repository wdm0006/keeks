"""Execute the strategy snippets published in README.md and the Sphinx docs.

The blocks are read out of the real files, so a constructor signature change that
the documentation does not follow fails here instead of reaching a new user.
"""

import ast
import re
import textwrap
from pathlib import Path

import matplotlib
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

# Headless: the flagship README block renders a plot via BankRoll.plot_history.
matplotlib.use("Agg")

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


class _DownscaledRepeatedBinary(RepeatedBinarySimulator):
    """Stand-in simulator that runs the README's flagship block down-scaled.

    The published narrative simulates 1,000 trials; executing that verbatim
    in the suite would be slow and unseeded. The demo clamps to a fixed
    50-trial, seeded run so the exact block the README ships executes
    quickly and deterministically.
    """

    DEMO_TRIALS = 50
    DEMO_SEED = 42

    def __init__(self, *args, **kwargs):
        kwargs["trials"] = self.DEMO_TRIALS
        kwargs["seed"] = self.DEMO_SEED
        super().__init__(*args, **kwargs)


def _readme_flagship_block():
    for code in _markdown_blocks(README.read_text()):
        if "evaluate_strategy" in code:
            return code
    raise AssertionError("README flagship simulation block went missing")


def test_readme_flagship_simulation_runs_downscaled(tmp_path, monkeypatch, capsys):
    """Run the README's flagship simulation as a seeded, down-scaled demo.

    The generic harness skips blocks calling evaluate_strategy/plot_history;
    this is the flagship block's dedicated runner. Same code, read from
    README.md (so constructor drift still fails here), with the simulator
    clamped to 50 seeded trials and the plot rendered headlessly into a
    throwaway directory.
    """
    monkeypatch.chdir(tmp_path)
    # The block executes verbatim, imports included — patch the module attribute
    # so its own `from keeks.simulators... import RepeatedBinarySimulator` binds
    # the down-scaled stand-in instead of the 1,000-trial real class.
    monkeypatch.setattr(
        "keeks.simulators.repeated_binary.RepeatedBinarySimulator",
        _DownscaledRepeatedBinary,
    )
    code = _readme_flagship_block()
    namespace = dict(NAMESPACE_SEED)
    exec(compile(code, "<README flagship>", "exec"), namespace)

    bankroll = namespace["bankroll"]
    assert bankroll.history[0] == 1_000.0
    assert len(bankroll.history) > 1, "no settlement was recorded"
    assert capsys.readouterr().out.startswith("Final bankroll: $")
    assert (tmp_path / "bankroll-history.png").is_file()

    # Seeded demo: rerunning the same block replays the identical bankroll.
    replay = dict(NAMESPACE_SEED)
    exec(compile(code, "<README flagship>", "exec"), replay)
    assert replay["bankroll"].history == bankroll.history
