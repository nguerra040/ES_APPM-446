
import pytest
import numpy as np
import finite
import timesteppers

resolution_list = [100, 200, 400]

error_BDF2_wave = {(100,2):0.5, (200,2):0.3, (400,2):0.08, (100,4):0.4, (200,4):0.1, (400,4):0.02}
@pytest.mark.parametrize('resolution', resolution_list)
@pytest.mark.parametrize('spatial_order', [2, 4])
def test_BDF2_wave(resolution, spatial_order):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    IC = np.exp(-(x-np.pi)**2*8)
    target = np.exp(-(x-np.pi-2*np.pi*0.2)**2*8)

    d = finite.DifferenceUniformGrid(1, spatial_order, grid)
    ts = timesteppers.BackwardDifferentiationFormula(IC, d, 2)

    num_periods = 1.8
    alpha1 = 0.01
    t1 = 2*np.pi*0.01
    alpha2 = 0.1
    t2 = 2*np.pi*0.2
    alpha3 = 0.5
    t3 = 2*np.pi*num_periods
    
    ts.evolve(alpha1*grid.dx, t1)
    ts.evolve(alpha2*grid.dx, t2)
    ts.evolve(alpha3*grid.dx, t3)

    error = np.max(np.abs( ts.u - target))
    error_est = error_BDF2_wave[(resolution,spatial_order)]

    assert error < error_est

error_BDF2_diff = {(100,2):2e-3, (200,2):5e-4, (400,2):1e-4, (100,4):5e-5, (200,4):2e-5, (400,4):5e-6}
@pytest.mark.parametrize('resolution', resolution_list)
@pytest.mark.parametrize('spatial_order', [2, 4])
def test_BDF2_diff(resolution, spatial_order):
    grid = finite.UniformPeriodicGrid(resolution, 50)
    x = grid.values

    IC = np.exp(-(x-20)**2/4)
    target = 1/np.sqrt(5)*np.exp(-(x-20)**2/20)

    d = finite.DifferenceUniformGrid(2, spatial_order, grid)
    ts = timesteppers.BackwardDifferentiationFormula(IC, d, 2)

    alpha1 = 0.01
    t1 = 0.1
    alpha2 = 0.05
    t2 = 2
    alpha3 = 0.2
    t3 = 4

    ts.evolve(alpha1*grid.dx, t1)
    ts.evolve(alpha2*grid.dx, t2)
    ts.evolve(alpha3*grid.dx, t3)

    error = np.max(np.abs( ts.u - target))
    error_est = error_BDF2_diff[(resolution,spatial_order)]

    assert error < error_est

error_BDF3_diff = {(100,2):2e-3, (200,2):5e-4, (400,2):1.5e-4, (100,4):4e-5, (200,4):3e-6, (400,4):2e-7}
@pytest.mark.parametrize('resolution', resolution_list)
@pytest.mark.parametrize('spatial_order', [2, 4])
def test_BDF3_diff(resolution, spatial_order):
    grid = finite.UniformPeriodicGrid(resolution, 50)
    x = grid.values

    IC = np.exp(-(x-30)**2/4)
    target = 1/np.sqrt(5)*np.exp(-(x-30)**2/20)

    d = finite.DifferenceUniformGrid(2, spatial_order, grid)
    ts = timesteppers.BackwardDifferentiationFormula(IC, d, 3)

    alpha1 = 0.01
    t1 = 0.1
    alpha2 = 0.05
    t2 = 2
    alpha3 = 0.2
    t3 = 4

    ts.evolve(alpha1*grid.dx, t1)
    ts.evolve(alpha2*grid.dx, t2)
    ts.evolve(alpha3*grid.dx, t3)

    error = np.max(np.abs( ts.u - target))
    error_est = error_BDF3_diff[(resolution,spatial_order)]

    assert error < error_est

error_BDF4_diff = {(100,4):4e-5, (200,4):2e-6, (400,4):1.5e-7, (100,6):1e-6, (200,6):1e-7, (400,6):2e-8}
@pytest.mark.parametrize('resolution', resolution_list)
@pytest.mark.parametrize('spatial_order', [4, 6])
def test_BDF4_diff(resolution, spatial_order):
    grid = finite.UniformPeriodicGrid(resolution, 50)
    x = grid.values

    IC = np.exp(-(x-25)**2/4)
    target = 1/np.sqrt(5)*np.exp(-(x-25)**2/20)

    d = finite.DifferenceUniformGrid(2, spatial_order, grid)
    ts = timesteppers.BackwardDifferentiationFormula(IC, d, 4)

    alpha1 = 0.01
    t1 = 0.1
    alpha2 = 0.05
    t2 = 2
    alpha3 = 0.2
    t3 = 4

    ts.evolve(alpha1*grid.dx, t1)
    ts.evolve(alpha2*grid.dx, t2)
    ts.evolve(alpha3*grid.dx, t3)

    error = np.max(np.abs( ts.u - target))
    error_est = error_BDF4_diff[(resolution,spatial_order)]

    assert error < error_est

error_BDF5_wave = {(100,4):0.04, (200,4):0.002, (400,4):2e-4, (100,6):0.003, (200,6):1e-4, (400,6):4e-6}
@pytest.mark.parametrize('resolution', resolution_list)
@pytest.mark.parametrize('spatial_order', [4, 6])
def test_BDF5_wave(resolution, spatial_order):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    IC = np.exp(-(x-np.pi)**2*8)
    target = np.exp(-(x-np.pi+2*np.pi*0.2)**2*8)

    d = finite.DifferenceUniformGrid(1, spatial_order, grid)
    ts = timesteppers.BackwardDifferentiationFormula(IC, d, 5)

    num_periods = 1.2
    alpha1 = 0.01
    t1 = 2*np.pi*0.01
    alpha2 = 0.1
    t2 = 2*np.pi*0.2
    alpha3 = 0.5
    t3 = 2*np.pi*num_periods

    ts.evolve(alpha1*grid.dx, t1)
    ts.evolve(alpha2*grid.dx, t2)
    ts.evolve(alpha3*grid.dx, t3)

    error = np.max(np.abs( ts.u - target))
    error_est = error_BDF5_wave[(resolution,spatial_order)]

    assert error < error_est

error_BDF6_wave = {100:0.004, 200:6e-5, 400:1.e-6}
@pytest.mark.parametrize('resolution', resolution_list)
def test_BDF6_wave(resolution):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    IC = np.exp(-(x-np.pi)**2*8)
    target = np.exp(-(x-np.pi+2*np.pi*0.2)**2*8)

    d = finite.DifferenceUniformGrid(1, 6, grid)
    ts = timesteppers.BackwardDifferentiationFormula(IC, d, 6)

    num_periods = 1.2
    alpha1 = 0.01
    t1 = 2*np.pi*0.01
    alpha2 = 0.1
    t2 = 2*np.pi*0.2
    alpha3 = 0.5
    t3 = 2*np.pi*num_periods

    ts.evolve(alpha1*grid.dx, t1)
    ts.evolve(alpha2*grid.dx, t2)
    ts.evolve(alpha3*grid.dx, t3)

    error = np.max(np.abs( ts.u - target))
    error_est = error_BDF6_wave[resolution]

    assert error < error_est

