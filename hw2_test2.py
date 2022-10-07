
import pytest
import numpy as np
import finite

order_range = [2, 4, 6, 8]

error_bound_1 = [0.003487359345263997, 7.2859015893244896e-06, 1.626790011920982e-08, 3.753470375790787e-11]

@pytest.mark.parametrize('order_index', np.arange(4))
def test_FD(order_index):
    order = order_range[order_index]
    x = np.linspace(0, 1, 100, endpoint=False)
    y = 2*np.pi*(x + 0.1*np.sin(2*np.pi*x))
    grid = finite.NonUniformPeriodicGrid(y, 2*np.pi)

    f = np.sin(y)

    d = finite.DifferenceNonUniformGrid(1, order, grid)

    df = d @ f
    df0 = np.cos(y)

    error = np.max(np.abs(df - df0))
    error_est = error_bound_1[order_index]

    assert error < error_est

error_bound_2 = [0.0016531241262078115, 3.953150377056438e-06, 1.01879280197617e-08, 2.6460656709051168e-11]

order_range_odd = [1, 3, 5, 7]

@pytest.mark.parametrize('order_index', np.arange(4))
def test_FD_2(order_index):
    order = order_range_odd[order_index]
    x = np.linspace(0, 1, 100, endpoint=False)
    y = 2*np.pi*(x + 0.1*np.sin(2*np.pi*x))
    grid = finite.NonUniformPeriodicGrid(y, 2*np.pi)

    f = np.sin(y)

    d2 = finite.DifferenceNonUniformGrid(2, order, grid)

    d2f = d2 @ f
    d2f0 = -np.sin(y)

    error = np.max(np.abs(d2f - d2f0))
    error_est = error_bound_2[order_index]

    assert error < error_est

error_bound_3 = [0.005224671143882285, 1.2706445107820117e-05, 3.067306559578247e-08, 7.409826640410579e-11]

@pytest.mark.parametrize('order_index', np.arange(4))
def test_FD_3(order_index):
    order = order_range[order_index]
    x = np.linspace(0, 1, 100, endpoint=False)
    y = 2*np.pi*(x + 0.1*np.sin(2*np.pi*x))
    grid = finite.NonUniformPeriodicGrid(y, 2*np.pi)

    f = np.sin(y)

    d3 = finite.DifferenceNonUniformGrid(3, order, grid)

    d3f = d3 @ f
    d3f0 = -np.cos(y)

    error = np.max(np.abs(d3f - d3f0))
    error_est = error_bound_3[order_index]

    assert error < error_est

