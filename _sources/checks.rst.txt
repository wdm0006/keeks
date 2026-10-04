Contract Checks
===============

The ``keeks.checks`` module is the contract harness for contributors, on the
``scikit-learn`` ``check_estimator`` pattern: each check runs a contributor's
implementation against the documented contract of its base class - the bets a
binary strategy sizes, the weights an allocator returns, the draws a
joint-return model samples - and fails with the first violation it meets,
so an implementation can be verified without running a simulator.

.. automodule:: keeks.checks
    :members:
    :undoc-members:
    :show-inheritance:
