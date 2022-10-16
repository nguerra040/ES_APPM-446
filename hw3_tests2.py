
import pytest
import numpy as np
import finite
import timesteppers

resolution_list = [100, 200, 400]

error_AB_2_4 = {100:0.1, 200:0.05, 400:0.01}
@pytest.mark.parametrize('resolution', resolution_list)
def test_AB_2_4(resolution):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values
    IC = np.exp(-(x-np.pi)**2*8)

    target = np.exp(-(x-np.pi+2*np.pi*0.2)**2*8)

    d = finite.CenteredFiniteDifference4(grid)
    f = lambda u: d @ u

    alpha = 0.3
    ts = timesteppers.AdamsBashforth(IC, f, 2, alpha*grid.dx)

    num_periods = 1.2
    ts.evolve(alpha*grid.dx, 2*np.pi*num_periods)

    error = np.max(np.abs( ts.u - target))
    error_est = error_AB_2_4[resolution]

    assert error < error_est

error_AB_3_2 = {100:0.5, 200:0.2, 400:0.05}
@pytest.mark.parametrize('resolution', resolution_list)
def test_AB_3_2(resolution):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values
    IC = np.exp(-(x-np.pi)**2*8)

    target = np.exp(-(x-np.pi-2*np.pi*0.2)**2*8)

    d = finite.CenteredFiniteDifference(grid)
    f = lambda u: d @ u

    alpha = 0.3
    ts = timesteppers.AdamsBashforth(IC, f, 2, alpha*grid.dx)

    num_periods = 1.8
    ts.evolve(alpha*grid.dx, 2*np.pi*num_periods)

    error = np.max(np.abs( ts.u - target))
    error_est = error_AB_3_2[resolution]

    assert error < error_est


error_AB_5_4 = {100:0.05, 200:0.003, 400:2e-4}
@pytest.mark.parametrize('resolution', resolution_list)
def test_AB_5_4(resolution):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values
    IC = np.exp(-(x-np.pi)**2*8)

    target = np.exp(-(x-np.pi-2*np.pi*0.2)**2*8)

    d = finite.CenteredFiniteDifference4(grid)
    f = lambda u: d @ u

    alpha = 0.1
    ts = timesteppers.AdamsBashforth(IC, f, 5, alpha*grid.dx)

    num_periods = 1.8
    ts.evolve(alpha*grid.dx, 2*np.pi*num_periods)

    error = np.max(np.abs( ts.u - target))
    error_est = error_AB_2_4[resolution]

    assert error < error_est

error_AB_6_2 = {100:0.5, 200:0.3, 400:0.06}
@pytest.mark.parametrize('resolution', resolution_list)
def test_AB_6_2(resolution):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    IC = np.exp(-(x-np.pi)**2*8)
    target = np.exp(-(x-np.pi+2*np.pi*0.2)**2*8)

    d = finite.CenteredFiniteDifference(grid)
    f = lambda u: d @ u

    alpha = 0.05
    ts = timesteppers.AdamsBashforth(IC, f, 6, alpha*grid.dx)

    num_periods = 1.2
    ts.evolve(alpha*grid.dx, 2*np.pi*num_periods)

    error = np.max(np.abs( ts.u - target))
    error_est = error_AB_6_2[resolution]

    assert error < error_est

error_AB_6_4 = {100:0.04, 200:0.003, 400:2e-4}
@pytest.mark.parametrize('resolution', resolution_list)
def test_AB_6_4(resolution):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    IC = np.exp(-(x-np.pi)**2*8)
    target = np.exp(-(x-np.pi+2*np.pi*0.2)**2*8)

    d = finite.CenteredFiniteDifference4(grid)
    f = lambda u: d @ u

    alpha = 0.05
    ts = timesteppers.AdamsBashforth(IC, f, 6, alpha*grid.dx)

    num_periods = 1.2
    ts.evolve(alpha*grid.dx, 2*np.pi*num_periods)

    error = np.max(np.abs( ts.u - target))
    error_est = error_AB_6_4[resolution]

    assert error < error_est


