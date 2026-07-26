import numpy as np
import pytest

from heavyedge_features.plateau import _segreg


def test_segreg_finds_breakpoint():
    x = np.linspace(0, 10, 101)
    expected_psi = 6.0
    Y = 2.0 + 0.2 * x + 1.5 * np.maximum(x - expected_psi, 0)

    params, reached_max = _segreg(x, Y, psi0=4.0)

    assert not reached_max
    assert params[-1] == pytest.approx(expected_psi, abs=1e-3)


def test_segreg_backtracking_limit_terminates():
    x = np.linspace(0, 10, 101)
    Y = 2.0 + 0.2 * x + 1.5 * np.maximum(x - 6.0, 0)

    params, reached_max = _segreg(
        x,
        Y,
        # The full update leaves the domain, and a single backtrack is not
        # enough to find an admissible step.
        psi0=9.9,
        max_backtracks=1,
    )

    assert reached_max
    assert np.all(np.isfinite(params))
