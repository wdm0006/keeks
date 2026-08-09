import keeks.simulators as simulators
from keeks.simulators.random_binary import RandomBinarySimulator
from keeks.simulators.random_uncertain_binary import RandomUncertainBinarySimulator
from keeks.simulators.repeated_binary import RepeatedBinarySimulator


def test_simulator_exports_are_leaf_module_classes():
    assert simulators.__all__ == [
        "RandomBinarySimulator",
        "RandomUncertainBinarySimulator",
        "RepeatedBinarySimulator",
    ]
    assert simulators.RandomBinarySimulator is RandomBinarySimulator
    assert simulators.RandomUncertainBinarySimulator is RandomUncertainBinarySimulator
    assert simulators.RepeatedBinarySimulator is RepeatedBinarySimulator
