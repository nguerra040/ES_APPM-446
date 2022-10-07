import pytest
import numpy as np
import finite

order_range = [2, 4, 6, 8]

error_bound_1 = [0.0011052024716421555, 1.3849193579502928e-05, 3.784245780277806e-07, 1.7638375468531642e-08]

@pytest.mark.parametrize('order_index', np.arange(4))
def test_UFD(order_index):
    order = order_range[order_index]
    grid = finite.UniformPeriodicGrid(100, 2*np.pi)
    x = grid.values

    f = 1/(2+np.sin(x)**2)

    d = finite.DifferenceUniformGrid(1, order, grid)

    df = d @ f
    df0 = -((2*np.cos(x)*np.sin(x))/(2 + np.sin(x)**2)**2)

    error = np.max(np.abs(df - df0))
    error_est = error_bound_1[order_index]

    assert error < error_est

error_bound_2 = [0.002460084948765484, 2.8916785483446006e-05, 7.619883177767406e-07, 3.4954780403495533e-08]

@pytest.mark.parametrize('order_index', np.arange(4))
def test_UFD_2(order_index):
    order = order_range[order_index]
    grid = finite.UniformPeriodicGrid(100, 2*np.pi)
    x = grid.values

    f = 1/(2+np.sin(x)**2)

    d2 = finite.DifferenceUniformGrid(2, order, grid)

    d2f = d2 @ f
    d2f0 = (8*np.cos(x)**2*np.sin(x)**2)/(2 + np.sin(x)**2)**3 - (2*np.cos(x)**2)/(2 + np.sin(x)**2)**2 + (2*np.sin(x)**2)/(2 + np.sin(x)**2)**2

    error = np.max(np.abs(d2f - d2f0))
    error_est = error_bound_2[order_index]

    assert error < error_est

error_bound_3 = [0.026372498378231357, 0.0007845734749516997, 3.823640687972141e-05, 2.7946235658937724e-06]

@pytest.mark.parametrize('order_index', np.arange(4))
def test_UFD_3(order_index):
    order = order_range[order_index]
    grid = finite.UniformPeriodicGrid(100, 2*np.pi)
    x = grid.values

    f = 1/(2+np.sin(x)**2)

    d3 = finite.DifferenceUniformGrid(3, order, grid)

    d3f = d3 @ f
    d3f0 = (2*np.cos(x)**3 * (37*np.sin(x) + np.sin(3*x)))/(2 + np.sin(x)**2)**4

    error = np.max(np.abs(d3f - d3f0))
    error_est = error_bound_3[order_index]

    assert error < error_est

