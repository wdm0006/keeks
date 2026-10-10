Utilities
=========

The utils module provides utility functions and classes used throughout the keeks package.

.. automodule:: keeks.utils
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: RuinError

Exceptions
----------

.. autoexception:: keeks.utils.RuinError
    :members:
    :undoc-members:
    :show-inheritance:

Checking stated probabilities
-----------------------------

Every strategy sizes from a stated probability, so before replaying a recorded
log it is worth asking whether those probabilities were calibrated.
:func:`keeks.calibration.calibration_report` scores them with the Brier score
and log loss and compares stated and observed frequencies in equal-width bins
(empty bins are omitted). It is an educational diagnostic, not advice. Inputs
are validated first: probabilities must be finite in ``[0, 1]``, outcomes must
be ``bool`` or ``0``/``1`` (not ``1.0``), and at least one observation is
required.

.. code-block:: python

    from keeks import calibration_report

    report = calibration_report([0.2, 0.2, 0.8, 0.8], [0, 1, 1, 1])
    print(round(report.brier_score, 6))  # 0.19
    for b in report.bins:
        print(b.lower, b.mean_predicted, b.observed_frequency)

.. autofunction:: keeks.calibration.calibration_report

.. autoclass:: keeks.calibration.CalibrationReport
    :no-members:

.. autoclass:: keeks.calibration.CalibrationBin
    :no-members:
