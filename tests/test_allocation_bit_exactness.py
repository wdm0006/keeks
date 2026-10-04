"""Bit-exactness oracle for the allocation simulator.

Freezes the exact public outputs of the simulator — the float bits of
``BankRoll.history`` — for the solver-free allocators (RiskBudgeting,
HierarchicalRiskParity, FixedWeights, ExponentialGradient) across every
scenario x seed x trial-count combination. It is the allocation analogue of
``tests/test_portfolio_bit_exactness.py`` and the acceptance oracle for
output-identical performance work on this class: any change that alters a
single settlement bit fails this module.

``MeanVariance`` is the one grid member whose ``evaluate`` routes through
scipy's SLSQP solver at call time. Its converged weights are only
tolerance-stable: last-ulp solver differences across runner hardware
compound over long runs until one history element flips a bit, so exact-bit
freezing is not portable (observed on CI: only 400-trial cases drift). Its
36 cases are frozen instead as (final funds, history length) references
checked at 1e-5 relative tolerance — roughly eight orders of magnitude
above the solver noise and far below any real regression signal. The four
solver-free allocators prove the settlement machinery bit-exactly.

Regenerate on purpose only (then format)::

    uv run python tests/test_allocation_bit_exactness.py
    ruff format tests/test_allocation_bit_exactness.py

which rewrites the GOLDEN blocks between the markers below from the current
code.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from keeks.allocation import (
    AllocationSimulator,
    ExponentialGradient,
    FixedWeights,
    HierarchicalRiskParity,
    MeanVariance,
    RiskBudgeting,
    scenario_model,
    scenarios_to_moments,
)
from keeks.bankroll import BankRoll

INITIAL_FUNDS = 1000.0

# One deterministic joint-return universe shared by every scenario axis:
# six realizations over three options, mixed-sign and small enough that the
# fee-heavy axis bleeds rather than explodes.
MATRIX = (
    (0.02, -0.01, 0.015),
    (-0.005, 0.012, -0.008),
    (0.018, 0.004, 0.011),
    (-0.012, -0.006, 0.009),
    (0.007, 0.015, -0.004),
    (0.001, -0.002, 0.003),
)

# Scenario axes: a standard equally-likely universe with a small fee, a
# cash-heavy universe (75% of every trial's probability mass is residual
# cash, so most periods settle nothing), and a fee-heavy universe whose
# flat per-period cost visibly bites the compounding.
SCENARIOS = {
    "standard": {
        "probabilities": None,
        "fee": 0.01,
        "max_transaction_loss": None,
    },
    "cash-heavy": {
        "probabilities": (0.05, 0.05, 0.05, 0.05, 0.02, 0.03),
        "fee": 0.01,
        "max_transaction_loss": None,
    },
    "fee-heavy": {
        "probabilities": None,
        "fee": 0.50,
        "max_transaction_loss": None,
    },
}

# Mirrors the other oracles' discipline: stateful allocators must be rebuilt
# per run because they carry state across evaluate() calls. Moment-based
# allocators estimate their inputs once from the shared universe.
MEAN, COV = scenarios_to_moments(np.asarray(MATRIX, dtype=float))

ALLOCATOR_FACTORIES = {
    "MeanVariance": lambda: MeanVariance(MEAN, COV, risk_aversion=1.0),
    "RiskBudgeting": lambda: RiskBudgeting(COV),
    "HierarchicalRiskParity": lambda: HierarchicalRiskParity(COV),
    "FixedWeights": lambda: FixedWeights((0.3, 0.3, 0.3)),
    "ExponentialGradient": lambda: ExponentialGradient(3),
}

# Grid of seeds and trial counts. Short runs catch early stops and the
# cash-period path; the 400-trial runs compound.
SEEDS = (0, 1, 42, 20260803)
TRIAL_COUNTS = (7, 53, 400)


def run_case(allocator_name, scenario_name, seed, trials):
    """Run one grid cell and return the full bankroll history."""
    scenario = SCENARIOS[scenario_name]
    allocator = ALLOCATOR_FACTORIES[allocator_name]()
    bankroll = BankRoll(
        initial_funds=INITIAL_FUNDS,
        max_transaction_loss=scenario["max_transaction_loss"],
    )
    simulator = AllocationSimulator(
        scenario_model(np.asarray(MATRIX, dtype=float)),
        probabilities=scenario["probabilities"],
        fee_per_bet=scenario["fee"],
        trials=trials,
        seed=seed,
    )
    simulator.evaluate_strategy(allocator, bankroll)
    return bankroll.history


def _digest(history):
    """SHA-256 over the hex repr of every history float (exact bits)."""
    joined = " ".join(value.hex() for value in history)
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def generate_golden():
    """Digest every cell of the grid against the current code."""
    golden = {}
    references = {}
    for allocator_name in ALLOCATOR_FACTORIES:
        for scenario_name in SCENARIOS:
            for seed in SEEDS:
                for trials in TRIAL_COUNTS:
                    key = (allocator_name, scenario_name, seed, trials)
                    history = run_case(*key)
                    if allocator_name == "MeanVariance":
                        references[key] = (float(history[-1]), len(history))
                    else:
                        golden[key] = _digest(history)
    return golden, references


_BEGIN_MARKER = "# === GOLDEN FIXTURE BEGIN (regenerate: uv run python tests/test_allocation_bit_exactness.py) ===\n"  # noqa: E501
_END_MARKER = "# === GOLDEN FIXTURE END ===\n"

# === GOLDEN FIXTURE BEGIN (regenerate: uv run python tests/test_allocation_bit_exactness.py) ===

GOLDEN = {
    (
        "ExponentialGradient",
        "cash-heavy",
        0,
        7,
    ): "cde670e72e6e9a758f45985425b6da4a9cfe16e7edef146ad176f998b099e45c",
    (
        "ExponentialGradient",
        "cash-heavy",
        0,
        53,
    ): "ca97ed149ba8605a3ac13449ce3ebfd443d69ff436ef5e73d7883f624b17de30",
    (
        "ExponentialGradient",
        "cash-heavy",
        0,
        400,
    ): "a7c0ea638a04af62ad5fdb6095a2cc0d6007cee88929c1dfa05276a649afca54",
    (
        "ExponentialGradient",
        "cash-heavy",
        1,
        7,
    ): "2fea87ad3138b5238a5a6215392dcbc5570bea58c8b0b1c4779a260c2eb4b5d1",
    (
        "ExponentialGradient",
        "cash-heavy",
        1,
        53,
    ): "1f77f69745ac1a899421b1e66d7abb87acea05acf7ea383394242fb9a1648211",
    (
        "ExponentialGradient",
        "cash-heavy",
        1,
        400,
    ): "9f26a5e8177dda54435686fd8be05d727b7ed94c086c56e96d77b64f47764bd7",
    (
        "ExponentialGradient",
        "cash-heavy",
        42,
        7,
    ): "dee71ddfac2afbef355704b80ff7c958cbc40886a687131a8bf8bcff26a8ee09",
    (
        "ExponentialGradient",
        "cash-heavy",
        42,
        53,
    ): "7c74535df10e9b73916c56ec90602a73ca7f27c7a51c1fdd8a3ee7798029179d",
    (
        "ExponentialGradient",
        "cash-heavy",
        42,
        400,
    ): "43ee96a8028daf3ff072ce92a4f0aa7242601395e9fb36b83e2ade10728f64f8",
    (
        "ExponentialGradient",
        "cash-heavy",
        20260803,
        7,
    ): "8a8851564aa1c3069b5cfd09f79e8fd8f7716768836ba34ae68336d144dceabc",
    (
        "ExponentialGradient",
        "cash-heavy",
        20260803,
        53,
    ): "e14899743f24629a7c609778eb7f61e3f34a914bf48043fc18fbb6975b4afefe",
    (
        "ExponentialGradient",
        "cash-heavy",
        20260803,
        400,
    ): "36d9fe845295fb2cef15e99c247aee694f943b1446e1f0af8feb97ce57aff7ef",
    (
        "ExponentialGradient",
        "fee-heavy",
        0,
        7,
    ): "00914ff2eaa1f73508ae936a36750e5abd643a46d2f55e42bd252988f2e2ef90",
    (
        "ExponentialGradient",
        "fee-heavy",
        0,
        53,
    ): "19c3456cd8394e1ee912e1cb8190029f0d5080861ea3787143027cec2dc6d57f",
    (
        "ExponentialGradient",
        "fee-heavy",
        0,
        400,
    ): "5751e7c3dfc91cd09aeed02457f14dbd4ee75418a68fec1dbbdbdb2ab6b95a73",
    (
        "ExponentialGradient",
        "fee-heavy",
        1,
        7,
    ): "ccc5d981bdff2ddf8b4e26fcf8f83f1906167b9d3007cc39a8e14d1f3a8953bd",
    (
        "ExponentialGradient",
        "fee-heavy",
        1,
        53,
    ): "15a382de195a5147b3e43719419461c46f68ead5195801904eca83440e3186b7",
    (
        "ExponentialGradient",
        "fee-heavy",
        1,
        400,
    ): "8d061cde2b208e21a35e7f502ee5b1226fef93955ede39f9dcd0fba5281adb2a",
    (
        "ExponentialGradient",
        "fee-heavy",
        42,
        7,
    ): "d42cbaa8d5a9abf1ce2af80e7bbcf1d6acdc331eb99b4fb37a77975847b8037b",
    (
        "ExponentialGradient",
        "fee-heavy",
        42,
        53,
    ): "8aea3f00d75b23cc3505f733dee033ab1b5482dc0aeda840d9a9582999a02d84",
    (
        "ExponentialGradient",
        "fee-heavy",
        42,
        400,
    ): "93e868fc1da022cc48cb1e055b215b51a1e1d945802dac5f0e3619efc6fbcd22",
    (
        "ExponentialGradient",
        "fee-heavy",
        20260803,
        7,
    ): "3d8e700576464c819e80ccc3040403b0a55e677acb1b8dd5a2f977392b373029",
    (
        "ExponentialGradient",
        "fee-heavy",
        20260803,
        53,
    ): "8cfedfe42b39f416fa2afb0d458c84577aa949bbeb34550f66e236c77632b5f3",
    (
        "ExponentialGradient",
        "fee-heavy",
        20260803,
        400,
    ): "f06c61a4f4fe22f5213e03f3f742583a2a8599971ead2313fadb5a593e8193c6",
    (
        "ExponentialGradient",
        "standard",
        0,
        7,
    ): "be44cccef4fc97034c8188c85b38c99fd631b3364e5fe4c0e290c0b955dc489b",
    (
        "ExponentialGradient",
        "standard",
        0,
        53,
    ): "523f5062a123fcfd41384afcbbbe351a328d5ee7bad70225d166e76dd310df73",
    (
        "ExponentialGradient",
        "standard",
        0,
        400,
    ): "54ec955d77b72f4f32854f49b79bac95d3a9b506bff44d2ea878c894c43c3e94",
    (
        "ExponentialGradient",
        "standard",
        1,
        7,
    ): "815034ca01a193a9ac545e95724d29b84ff0b0bea17dd4630daf89d1b9f26c4d",
    (
        "ExponentialGradient",
        "standard",
        1,
        53,
    ): "e2f314c9453fdff050b46cc1a719af78c0a238737b80da423e8a9be8d7ca727c",
    (
        "ExponentialGradient",
        "standard",
        1,
        400,
    ): "93c977d58e0f5f59493585779a8d4e46b98753f79f6137fa7499d65fbc19d952",
    (
        "ExponentialGradient",
        "standard",
        42,
        7,
    ): "908d69898b1a7b954f73d53c93f5624b8fc3d21f7cc92a7612db7d2935216bb7",
    (
        "ExponentialGradient",
        "standard",
        42,
        53,
    ): "530714880d17790a6598dab208d9a3694992ac72fb028ca63c99fa4a129f305d",
    (
        "ExponentialGradient",
        "standard",
        42,
        400,
    ): "e935d85d5194674cabb8ff06f1c5d405f39c72ac45ae0dbfda826def73cbb7e5",
    (
        "ExponentialGradient",
        "standard",
        20260803,
        7,
    ): "f7687774e61f1afd04e15a574980ccb0f2b4d01589d3c36d60eaec0261de46f4",
    (
        "ExponentialGradient",
        "standard",
        20260803,
        53,
    ): "39605e491514669633d766ed9c07038dfe0e37cef79e09f795ad4b8b0e803e4d",
    (
        "ExponentialGradient",
        "standard",
        20260803,
        400,
    ): "dcaac959fc36190971fe028314316f248d94d18eb127d2e600b0ece3f58dd07d",
    (
        "FixedWeights",
        "cash-heavy",
        0,
        7,
    ): "7e827cbe33392e8a443e1e74844fed551b13a1d50db81df6fda22a9bbe79999b",
    (
        "FixedWeights",
        "cash-heavy",
        0,
        53,
    ): "b20a613fd44e92b337db740662597eaea9fe208a22cc01365c330f327abc574f",
    (
        "FixedWeights",
        "cash-heavy",
        0,
        400,
    ): "234fb6b562aea6559a10f3b07ebf15c6fe0aa754e0b87e0cab0b71becc6b09cc",
    (
        "FixedWeights",
        "cash-heavy",
        1,
        7,
    ): "879e73d36f24776d53df66b86203468a35ada330983335f942845ad63cb0a53b",
    (
        "FixedWeights",
        "cash-heavy",
        1,
        53,
    ): "08d29f4d2489693903e59acd53491ab5349a4c9b77b289d8a4e5426a27f19591",
    (
        "FixedWeights",
        "cash-heavy",
        1,
        400,
    ): "4c4088fa7d09fe73a600474442ee7ab93c8b4d02a1c6657aa6843c2eaf718371",
    (
        "FixedWeights",
        "cash-heavy",
        42,
        7,
    ): "a32d770ab2e944ae9245b753573eafb53826bf2de40fa126530a2d105ba9563f",
    (
        "FixedWeights",
        "cash-heavy",
        42,
        53,
    ): "c9e8e71998248188026ecdc97ac6943940730877db3e8ec81e0860490922ea6b",
    (
        "FixedWeights",
        "cash-heavy",
        42,
        400,
    ): "0b6766cc265adbd37d2a78d4f5591a74ce7c6996d353dbbdecc21a651406990b",
    (
        "FixedWeights",
        "cash-heavy",
        20260803,
        7,
    ): "6cd92ce897286cb915e183a716bd98ddf3c31c03088020f46a6c0546760cc0a0",
    (
        "FixedWeights",
        "cash-heavy",
        20260803,
        53,
    ): "959b0758971442051ef66d62f70dc50b47fe2381a63fc093dfb324ab76753d89",
    (
        "FixedWeights",
        "cash-heavy",
        20260803,
        400,
    ): "8e87c01b332a8d333b4e501edec84cd471dbef42e3dae3090eb66db3825a0c69",
    (
        "FixedWeights",
        "fee-heavy",
        0,
        7,
    ): "b571d817985d19a7005940f585788da3daed951302bf915aae4a88ca84649e6f",
    (
        "FixedWeights",
        "fee-heavy",
        0,
        53,
    ): "840a8ae751c22a50d4312981f507e405756f3ab7fcf4101293215ccce2be34a8",
    (
        "FixedWeights",
        "fee-heavy",
        0,
        400,
    ): "5d8da9cf82a8609349e86805aa1500f198e9f7ec13f2bfe4aff01b661eb0425c",
    (
        "FixedWeights",
        "fee-heavy",
        1,
        7,
    ): "90443294d24d009a58b3cab263f529b39b94e551e8aa9af614c3695a21871dad",
    (
        "FixedWeights",
        "fee-heavy",
        1,
        53,
    ): "164fa29ed0fb88f36c6dcd39338b5cface1ec92a3ed34066fa10145222c55b45",
    (
        "FixedWeights",
        "fee-heavy",
        1,
        400,
    ): "f8a61afd85019255cae69170ccc040b07ad6358e44452765794cef8707c0a369",
    (
        "FixedWeights",
        "fee-heavy",
        42,
        7,
    ): "8b5c7f93e2162b7853755beba113aee8476588496c8340ab8b091f2e56a20d9f",
    (
        "FixedWeights",
        "fee-heavy",
        42,
        53,
    ): "dfa8239f31b71a975d9de6d692c6091da3104e2599a56bb492d2d82e4e2e45b3",
    (
        "FixedWeights",
        "fee-heavy",
        42,
        400,
    ): "018b48c9a05375dc63fa3981d1912034fcf6937cc17cf6d1ffedd56581b4441b",
    (
        "FixedWeights",
        "fee-heavy",
        20260803,
        7,
    ): "7d25fba07322bd2104634e6f6b1819febd54bca4981eb441732086fbb46000f9",
    (
        "FixedWeights",
        "fee-heavy",
        20260803,
        53,
    ): "f660a8ca26082514ecfc5a42d331ae1622ba5913869e8ff7aece4d37070022ae",
    (
        "FixedWeights",
        "fee-heavy",
        20260803,
        400,
    ): "0657a1cade33e7500056406e563906d33866813cf63f5cdefec6667a523092fe",
    (
        "FixedWeights",
        "standard",
        0,
        7,
    ): "15a55928fdcfdaf0e374a7dcf6192590d818de044902ad05afc2b04a58b51b86",
    (
        "FixedWeights",
        "standard",
        0,
        53,
    ): "c66ff2b3dfce8b9f325746424df2a1e95e66a84eb95538e285c7eaa1a3598174",
    (
        "FixedWeights",
        "standard",
        0,
        400,
    ): "cd87495b6703fdfb5f55f72a67c535a9cdf9476981e26e6189c85d0118faa23a",
    (
        "FixedWeights",
        "standard",
        1,
        7,
    ): "7ea1669958b42f56957e6ad9900efce9e1456fa79768b2fdddb763fd2296e536",
    (
        "FixedWeights",
        "standard",
        1,
        53,
    ): "53fdf88300dac9c06c416490b0f563ef7613d50d5077dfa4dcf170b4785ecb8c",
    (
        "FixedWeights",
        "standard",
        1,
        400,
    ): "70ff43673617e6bf60dc3ea8b0be50b1affc532c4bc4fcdc200fd91bd51725c7",
    (
        "FixedWeights",
        "standard",
        42,
        7,
    ): "26fc4d10bdc92bb035ba8fd54b7f0a39788c1da09e2fc94b3b6e284d1fd9227e",
    (
        "FixedWeights",
        "standard",
        42,
        53,
    ): "17b179d83d28ee9a119e2f0d4336ba9617b4b3462ad11e20fd43e553993a9abc",
    (
        "FixedWeights",
        "standard",
        42,
        400,
    ): "f04460bbf8fbf64ac417ee1b68103b91862f5ef7d9255faae3278d008fedf17d",
    (
        "FixedWeights",
        "standard",
        20260803,
        7,
    ): "e27e9b0334eacd8e643b19866b066c2e4d066f58ee12ae2be550167908336cec",
    (
        "FixedWeights",
        "standard",
        20260803,
        53,
    ): "0d93f80a87db950f2232aa6806abad2e88f5832f230c5635d9ffcc0a381770cf",
    (
        "FixedWeights",
        "standard",
        20260803,
        400,
    ): "11514e61470431a9f04d5c6bab07cda076111ac91fdf66c652de44ab9e23c066",
    (
        "HierarchicalRiskParity",
        "cash-heavy",
        0,
        7,
    ): "338564951065a90d7c45738d54ad4954bd3db66501da71c498d72acb2bd04d62",
    (
        "HierarchicalRiskParity",
        "cash-heavy",
        0,
        53,
    ): "1c9fdd40f515e370e8990fe341bb8a53999eb7a6110902d0fad0544d604f33f9",
    (
        "HierarchicalRiskParity",
        "cash-heavy",
        0,
        400,
    ): "7b481f25cdeaf926c667945c16990c8b48b60af661fa18784a9710602041bca0",
    (
        "HierarchicalRiskParity",
        "cash-heavy",
        1,
        7,
    ): "d91845145e113bd69dab9d3db55af4448a719e00c653415cab7a642ce33a3fa7",
    (
        "HierarchicalRiskParity",
        "cash-heavy",
        1,
        53,
    ): "d87f2381cc4e60d609b7ac79f4274438a4d831428f4e87027d37c68f8bc2e8a2",
    (
        "HierarchicalRiskParity",
        "cash-heavy",
        1,
        400,
    ): "10748a4dd082f83c26f9ce124c2b94418fefadc85bd387ee22b5463214b89609",
    (
        "HierarchicalRiskParity",
        "cash-heavy",
        42,
        7,
    ): "dc9b8459b0adbe46745f5564a9fcefa40930e2c78efb40730ae7f4c943b63687",
    (
        "HierarchicalRiskParity",
        "cash-heavy",
        42,
        53,
    ): "d01030081d651acdc6f91d6bab3efac0a38afa315a28b90cd99ca8024127784e",
    (
        "HierarchicalRiskParity",
        "cash-heavy",
        42,
        400,
    ): "24ba61dde694336a09cd4744bf2cf13b24bfc5aead902cda00f1fada026cc59c",
    (
        "HierarchicalRiskParity",
        "cash-heavy",
        20260803,
        7,
    ): "597c9c7c6354ee341e4d4d3835621643868e0e133f501ce0d225c4b7682fc637",
    (
        "HierarchicalRiskParity",
        "cash-heavy",
        20260803,
        53,
    ): "c830b20c52a7cf325a4a7415ea9f0d1ce9380b7c35edcfc66f04b7782d76c90d",
    (
        "HierarchicalRiskParity",
        "cash-heavy",
        20260803,
        400,
    ): "f66d7a7f71f26c2dc9e4127cb619be582b0e9a2ccde3542bf5ce4d17ac77ffd7",
    (
        "HierarchicalRiskParity",
        "fee-heavy",
        0,
        7,
    ): "72224ef17cdf310711b75bd74ccb75feb6a9db21b05a83ed3a2658487aea7ab6",
    (
        "HierarchicalRiskParity",
        "fee-heavy",
        0,
        53,
    ): "0015efc45352af6750173d3ae61907a501be02992afb2ff1d17b4e74b8ede6ea",
    (
        "HierarchicalRiskParity",
        "fee-heavy",
        0,
        400,
    ): "c08fd2a24374334faf33573832af5b363a15a2725f9e7ac5a6378bea0b3de032",
    (
        "HierarchicalRiskParity",
        "fee-heavy",
        1,
        7,
    ): "cb25ba5b40083f0ae6f582c35350b2719f9dcfb60769f876ff4a87bfc5aeae3d",
    (
        "HierarchicalRiskParity",
        "fee-heavy",
        1,
        53,
    ): "a78486625a910a21325e98a8362370b708b616f3e92c1a1fbcd6576e647a6e8a",
    (
        "HierarchicalRiskParity",
        "fee-heavy",
        1,
        400,
    ): "95ee9b5cae8f9970902bd042d1ba7aa584749e9c9a3344fbda0fd8078597bdd4",
    (
        "HierarchicalRiskParity",
        "fee-heavy",
        42,
        7,
    ): "2f46891b99790ba2e7e4d07a78f3f61c8016ca17a0425d49ea40ba89e031eabe",
    (
        "HierarchicalRiskParity",
        "fee-heavy",
        42,
        53,
    ): "57f88412631f543b49f4203d3a666861a31d49f225d14bc5c2db2f3ec176f67e",
    (
        "HierarchicalRiskParity",
        "fee-heavy",
        42,
        400,
    ): "538f47cb86af0153282dd3327bf4618650b031905ef55111f6d595b6373290b9",
    (
        "HierarchicalRiskParity",
        "fee-heavy",
        20260803,
        7,
    ): "1186b6c25de2afc63c55ebf6bd84508d771def604a4dff8d838071946fb169b6",
    (
        "HierarchicalRiskParity",
        "fee-heavy",
        20260803,
        53,
    ): "e46343fbb0113645a6069d25ee4cfa4a29440e76dd01d5744ba6d594437f01c1",
    (
        "HierarchicalRiskParity",
        "fee-heavy",
        20260803,
        400,
    ): "41ca5514ccf059134af58628b5fbc93ad8b800d40fb6914c58f359d8db549549",
    (
        "HierarchicalRiskParity",
        "standard",
        0,
        7,
    ): "14219431953237f352bffbed8d7bc87e40dd264eb354adddb55116d732c435ef",
    (
        "HierarchicalRiskParity",
        "standard",
        0,
        53,
    ): "826bae931af5d38a1f764b560eb5e8e8d93c110a9489fb0c6d9fe7c659c3dba6",
    (
        "HierarchicalRiskParity",
        "standard",
        0,
        400,
    ): "487deacfdd253df2c9c9ea0164ed6d985ec6494b4bf27490168464ae17ab3b7b",
    (
        "HierarchicalRiskParity",
        "standard",
        1,
        7,
    ): "ab42eae47c5749eb8a133137a1ddabe392d153369f8a086d615b4094a7622f6f",
    (
        "HierarchicalRiskParity",
        "standard",
        1,
        53,
    ): "e876398aea60b328f8f32671132aaba627d9761ee20665a6b09cb06db87bfc99",
    (
        "HierarchicalRiskParity",
        "standard",
        1,
        400,
    ): "a54c1fb2fa093b8acfc24854a5ef9e0d70fec5264841bc3fa7c37b5bc8a73391",
    (
        "HierarchicalRiskParity",
        "standard",
        42,
        7,
    ): "4dc42b30f642f7cc1a84c9158705665b985ef7c00c46809e33a0b7cc23592f9e",
    (
        "HierarchicalRiskParity",
        "standard",
        42,
        53,
    ): "83495633523a4dd1c06f9cd4775be8cfc05df2072ea9747de043aedef66111e3",
    (
        "HierarchicalRiskParity",
        "standard",
        42,
        400,
    ): "99e9f1ebfacf9b22da8556267ee9bc620574b65efc31b1e8752561ea18063004",
    (
        "HierarchicalRiskParity",
        "standard",
        20260803,
        7,
    ): "acd3419dd3cf406501fcee6372cf3b53b0e918656cad906e58ffb6f9f0d0e803",
    (
        "HierarchicalRiskParity",
        "standard",
        20260803,
        53,
    ): "0305f312d2f89062c7a6a4981e656dbc55ae2d06ba6a6cc656508cffbeacf96f",
    (
        "HierarchicalRiskParity",
        "standard",
        20260803,
        400,
    ): "906b30c72ece5f19a52287afd26a26a71a5a957ed7818d7eebf97f85902d0fb0",
    (
        "RiskBudgeting",
        "cash-heavy",
        0,
        7,
    ): "21d415c70c3d88b1a3fb6b77e3330913bb7c206deff32c7493083049997eb50f",
    (
        "RiskBudgeting",
        "cash-heavy",
        0,
        53,
    ): "ec8b12da586734ab57a68f3031c905e6256541fd4650329ab5f6c6eb1f1128d4",
    (
        "RiskBudgeting",
        "cash-heavy",
        0,
        400,
    ): "e8211112469c94e504d7798c31571b30e91b6743ec8635954a286a1b6a279c24",
    (
        "RiskBudgeting",
        "cash-heavy",
        1,
        7,
    ): "b1ef78d7c891671fbdf6bad3e55fb1328d6ffd9f5b5b72e34a07cc57e53e6949",
    (
        "RiskBudgeting",
        "cash-heavy",
        1,
        53,
    ): "09296658f984642fcb59713ae00c21e7095b5fe4f53675f9dbb4723d257f716c",
    (
        "RiskBudgeting",
        "cash-heavy",
        1,
        400,
    ): "b6fa6db9fd2e3183c39e3085c89824f0233b851585c189dc52a0de8b5963d7c8",
    (
        "RiskBudgeting",
        "cash-heavy",
        42,
        7,
    ): "43fe0f430c80e32fcab6483c8393bcd86723b7aa39106e74cd4b37e8b0bfd7ed",
    (
        "RiskBudgeting",
        "cash-heavy",
        42,
        53,
    ): "f5eaf397c5c2549ad602302786af8d8ad8eac3682c34b17028d73d338d23bae4",
    (
        "RiskBudgeting",
        "cash-heavy",
        42,
        400,
    ): "9c95f22dfbcbf2387fd93d09fddc752e50518f2d03f02ced47f5a786a56d81e1",
    (
        "RiskBudgeting",
        "cash-heavy",
        20260803,
        7,
    ): "ee4c47d2203471951166a5499f7c2d6b4b240353581d4522996df4bac6f960f2",
    (
        "RiskBudgeting",
        "cash-heavy",
        20260803,
        53,
    ): "d9981d8ddadd440d08f7680975c02914db8a99d3ed151c5e4ce6f67a6e31d3ce",
    (
        "RiskBudgeting",
        "cash-heavy",
        20260803,
        400,
    ): "470d5d3c21b3566657a3f88b9f1c08c023bfc462666d9b307797046067065b18",
    (
        "RiskBudgeting",
        "fee-heavy",
        0,
        7,
    ): "92b54b7d79021c6de63e67aaef7d7d1cea605219326a483814c6aaa66467ec89",
    (
        "RiskBudgeting",
        "fee-heavy",
        0,
        53,
    ): "737a6ead3c5c69a2232567ea944be2535830fd4efa728d8b37833293991b22eb",
    (
        "RiskBudgeting",
        "fee-heavy",
        0,
        400,
    ): "f35d7ab8f1a3d048a5aefdd3c37a6d7a519ddef991460308029737edcf4d4565",
    (
        "RiskBudgeting",
        "fee-heavy",
        1,
        7,
    ): "b7e73b0a34683826dad8adbe9893f3457b765fa9135694c3fdddd9a405de6152",
    (
        "RiskBudgeting",
        "fee-heavy",
        1,
        53,
    ): "16f088af452608a5bdbb0f975b5f2958689a92ff15a9a30d1954fe4bcd6a2dd4",
    (
        "RiskBudgeting",
        "fee-heavy",
        1,
        400,
    ): "44027632bb911908c5c86b5de76e0a77cd3dcb7ec65c156f61a0cf39e12ef8c7",
    (
        "RiskBudgeting",
        "fee-heavy",
        42,
        7,
    ): "0864e814d01fb0368511c5381371c784813fb1724282b15247fcf6b0b7d1e18c",
    (
        "RiskBudgeting",
        "fee-heavy",
        42,
        53,
    ): "a6227959ac59a93d0197bc204850563b6b20e77ac7a82ceb1cd158a0a5b33198",
    (
        "RiskBudgeting",
        "fee-heavy",
        42,
        400,
    ): "cdf374b1ad250845dde47ce43eb1394bf3c6f780daee81f29b126d3b1867225c",
    (
        "RiskBudgeting",
        "fee-heavy",
        20260803,
        7,
    ): "d4cb83dbdb736ab130236f8d2b67cee9719ca310928fc1c8624bad320b1e80c9",
    (
        "RiskBudgeting",
        "fee-heavy",
        20260803,
        53,
    ): "588cedb41b1e688c7922706fafc196a363263407f1d5b63d3d0c4e00574fc1b9",
    (
        "RiskBudgeting",
        "fee-heavy",
        20260803,
        400,
    ): "f0ca0fcbbd404c762bf1daa37b57152ca4fb91a7c9c4305cee127c6c2aec99c9",
    (
        "RiskBudgeting",
        "standard",
        0,
        7,
    ): "2aeb31d9c43c1d8b5dc288bf42cf9610e771a3d322253608c7b2d3deca05fa13",
    (
        "RiskBudgeting",
        "standard",
        0,
        53,
    ): "71c3729b9deecbf20b027569bb902f7d6c2ab386c3678b4492107c2f009ff54d",
    (
        "RiskBudgeting",
        "standard",
        0,
        400,
    ): "85e4c4d4f220825b52524cfa29e500e2b664e07cc9d4ba707a45b6affe9dfbd6",
    (
        "RiskBudgeting",
        "standard",
        1,
        7,
    ): "b34261f82e4bc98bb0321b5970c3e645bf64e018fcaab709fd0ab8cc31e9ebc8",
    (
        "RiskBudgeting",
        "standard",
        1,
        53,
    ): "b515a947f67b37a5aa3862c99c7e7f205755a208daf8289aaaa5fd13a139b143",
    (
        "RiskBudgeting",
        "standard",
        1,
        400,
    ): "a53d573cbdbbbedddd6a511458b8326fa333a23e8a18b3f97c9bbc6ac42f76ed",
    (
        "RiskBudgeting",
        "standard",
        42,
        7,
    ): "4e4b38efba8a4683e06277857ae3ccfb7b30be9a2eb069b0b20781b40b32d593",
    (
        "RiskBudgeting",
        "standard",
        42,
        53,
    ): "6143e09a66b24b8be0d6fad43c3d0b8874ada6036ad4a5eb0551cd284575bbaf",
    (
        "RiskBudgeting",
        "standard",
        42,
        400,
    ): "c7d56f1ae4a3b1a391b5b7c308bb8bc9ae72a437c305fa1a8bf3d8f592615721",
    (
        "RiskBudgeting",
        "standard",
        20260803,
        7,
    ): "e6a1d00cef806b6b8c55512f35a5795e8c93d1abc24ae923e1974ba261a63c74",
    (
        "RiskBudgeting",
        "standard",
        20260803,
        53,
    ): "5158216a7604265a96ad1d0c96de8b4632544b05b03655d90c92f2f07003e570",
    (
        "RiskBudgeting",
        "standard",
        20260803,
        400,
    ): "6217d982a714108a806912197b9ffc89dc2d4f85a4d7869774edc4a087eb79e8",
}

MEAN_VARIANCE_REFERENCES = {
    ("MeanVariance", "cash-heavy", 0, 7): (1021.0, 3),
    ("MeanVariance", "cash-heavy", 0, 53): (1084.67, 15),
    ("MeanVariance", "cash-heavy", 0, 400): (1926.9, 109),
    ("MeanVariance", "cash-heavy", 1, 7): (1001.95, 3),
    ("MeanVariance", "cash-heavy", 1, 53): (1046.27, 11),
    ("MeanVariance", "cash-heavy", 1, 400): (1643.85, 101),
    ("MeanVariance", "cash-heavy", 42, 7): (1001.98, 3),
    ("MeanVariance", "cash-heavy", 42, 53): (1035.82, 18),
    ("MeanVariance", "cash-heavy", 42, 400): (1157.08, 107),
    ("MeanVariance", "cash-heavy", 20260803, 7): (983.04, 3),
    ("MeanVariance", "cash-heavy", 20260803, 53): (1104.76, 12),
    ("MeanVariance", "cash-heavy", 20260803, 400): (1441.81, 100),
    ("MeanVariance", "fee-heavy", 0, 7): (1028.68, 8),
    ("MeanVariance", "fee-heavy", 0, 53): (1355.37, 54),
    ("MeanVariance", "fee-heavy", 0, 400): (5358.38, 401),
    ("MeanVariance", "fee-heavy", 1, 7): (1008.19, 8),
    ("MeanVariance", "fee-heavy", 1, 53): (1370.37, 54),
    ("MeanVariance", "fee-heavy", 1, 400): (9342.73, 401),
    ("MeanVariance", "fee-heavy", 42, 7): (1018.29, 8),
    ("MeanVariance", "fee-heavy", 42, 53): (1154.11, 54),
    ("MeanVariance", "fee-heavy", 42, 400): (4196.01, 401),
    ("MeanVariance", "fee-heavy", 20260803, 7): (1019.16, 8),
    ("MeanVariance", "fee-heavy", 20260803, 53): (1103.27, 54),
    ("MeanVariance", "fee-heavy", 20260803, 400): (5402.65, 401),
    ("MeanVariance", "standard", 0, 7): (1032.14, 8),
    ("MeanVariance", "standard", 0, 53): (1386.02, 54),
    ("MeanVariance", "standard", 0, 400): (5914.13, 401),
    ("MeanVariance", "standard", 1, 7): (1011.57, 8),
    ("MeanVariance", "standard", 1, 53): (1400.95, 54),
    ("MeanVariance", "standard", 1, 400): (10098.07, 401),
    ("MeanVariance", "standard", 42, 7): (1021.76, 8),
    ("MeanVariance", "standard", 42, 53): (1182.94, 54),
    ("MeanVariance", "standard", 42, 400): (4688.78, 401),
    ("MeanVariance", "standard", 20260803, 7): (1022.62, 8),
    ("MeanVariance", "standard", 20260803, 53): (1129.94, 54),
    ("MeanVariance", "standard", 20260803, 400): (5950.33, 401),
}
# === GOLDEN FIXTURE END ===

_CASES = sorted(GOLDEN) + sorted(MEAN_VARIANCE_REFERENCES)

_MEAN_VARIANCE_TOLERANCE = 1e-5


@pytest.mark.parametrize(
    ("allocator_name", "scenario_name", "seed", "trials"),
    _CASES,
    ids=[f"{case[0]}|{case[1]}|{case[2]}|{case[3]}" for case in _CASES],
)
def test_seeded_outputs_are_bit_exact(allocator_name, scenario_name, seed, trials):
    history = run_case(allocator_name, scenario_name, seed, trials)
    key = (allocator_name, scenario_name, seed, trials)
    if allocator_name == "MeanVariance":
        final_funds, length = MEAN_VARIANCE_REFERENCES[key]
        assert len(history) == length, (
            f"History length changed for allocation/{allocator_name}/{scenario_name} "
            f"seed={seed} trials={trials}: {len(history)} vs {length}"
        )
        assert abs(float(history[-1]) - final_funds) <= (
            _MEAN_VARIANCE_TOLERANCE * abs(final_funds)
        ), (
            f"Final funds changed for allocation/{allocator_name}/{scenario_name} "
            f"seed={seed} trials={trials}: {float(history[-1])!r} vs {final_funds!r}"
        )
        return
    expected = GOLDEN[key]
    actual = _digest(history)
    assert actual == expected, (
        f"Output bits changed for allocation/{allocator_name}/{scenario_name} "
        f"seed={seed} trials={trials}: final funds {history[-1]!r}, "
        f"history length {len(history)}"
    )


if __name__ == "__main__":
    _path = Path(__file__)
    _text = _path.read_text()
    _start = _text.index(_BEGIN_MARKER) + len(_BEGIN_MARKER)
    _end = _text.index(_END_MARKER)
    _golden, _references = generate_golden()
    _entries = "\n".join(
        f"    {key!r}: {value!r}," for key, value in sorted(_golden.items())
    )
    _ref_entries = "\n".join(
        f"    {key!r}: {value!r}," for key, value in sorted(_references.items())
    )
    _path.write_text(
        _text[:_start]
        + "\nGOLDEN = {\n"
        + _entries
        + "\n}\n\nMEAN_VARIANCE_REFERENCES = {\n"
        + _ref_entries
        + "\n}\n"
        + _text[_end:]
    )
    print(
        f"Regenerated {len(_golden)} exact digests and "
        f"{len(_references)} MeanVariance references"
    )
