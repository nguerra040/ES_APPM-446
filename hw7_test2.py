import pytest
import numpy as np
import finite
import timesteppers
import equations

error_derivative_2 = {(50, 2): 0.2, (100, 2): 0.1, (200, 2): 0.05, (50, 4): 0.0015, (100, 4): 2e-4, (200, 4): 2.5e-5, (50, 6): 1e-5, (100, 6): 4e-7, (200, 6): 1e-8}
@pytest.mark.parametrize('resolution', [50, 100, 200])
@pytest.mark.parametrize('convergence_order', [2, 4, 6])
def test_derivative_2(resolution, convergence_order):
    grid = finite.UniformNonPeriodicGrid(resolution,(0, 5))
    x = grid.values
    u = np.sin(x)

    d = finite.DifferenceUniformGrid(2, convergence_order, grid)
    du = d @ u

    error = np.max(np.abs(du + np.sin(x)))
    error_est = error_derivative_2[(resolution, convergence_order)]

    assert error < error_est

error_derivative_3 = {(50, 2): 0.04, (100, 2): 0.008, (200, 2): 0.002, (50, 4): 4e-4, (100, 4): 2e-5, (200, 4): 1.5e-6, (50, 6): 3e-6, (100, 6): 6e-8, (200, 6): 4.5e-9}
@pytest.mark.parametrize('resolution', [50, 100, 200])
@pytest.mark.parametrize('convergence_order', [2, 4, 6])
def test_derivative_3(resolution, convergence_order):
    grid = finite.UniformNonPeriodicGrid(resolution,(0, 5))
    x = grid.values
    u = np.sin(x)

    d = finite.DifferenceUniformGrid(3, convergence_order, grid)
    du = d @ u

    error = np.max(np.abs(du + np.cos(x)))
    error_est = error_derivative_3[(resolution, convergence_order)]

    assert error < error_est

error_derivative_bump = {(50, 2): 0.15, (100, 2): 0.03, (200, 2): 0.006, (50, 4): 0.04, (100, 4): 0.001, (200, 4): 2.5e-5, (50, 6): 0.01, (100, 6): 9e-5, (200, 6): 3e-7}
@pytest.mark.parametrize('resolution', [50, 100, 200])
@pytest.mark.parametrize('convergence_order', [2, 4, 6])
def test_derivative_bump(resolution, convergence_order):
    grid = finite.UniformNonPeriodicGrid(resolution,(0, 5))
    x = grid.values
    u = np.exp(-(x-2.5)**2*4)

    d = finite.DifferenceUniformGrid(1, convergence_order, grid)
    du = d @ u
    du_target = -8*np.exp(-4*(x-2.5)**2)*(x-2.5)

    error = np.max(np.abs(du - du_target))
    error_est = error_derivative_bump[(resolution, convergence_order)]

    assert error < error_est


error_wave = {(100, 0.4, 2): 0.15, (100, 0.4, 4): 0.08, (100, 0.2, 2): 0.16, (100, 0.2, 4): 0.02,
              (200, 0.4, 2): 0.08, (200, 0.4, 4): 0.02, (200, 0.2, 2): 0.08, (200, 0.2, 4): 0.004}
@pytest.mark.parametrize('resolution', [100, 200])
@pytest.mark.parametrize('alpha', [0.4, 0.2])
@pytest.mark.parametrize('spatial_order', [2, 4])
def test_wave_equation(resolution, alpha, spatial_order):
    grid_x = finite.UniformNonPeriodicGrid(resolution,(0,2*np.pi))
    grid_y = finite.UniformPeriodicGrid(resolution,2*np.pi)
    domain = finite.Domain([grid_x, grid_y])
    x, y = domain.values()

    u = np.zeros(domain.shape)
    v = np.zeros(domain.shape)
    p = np.zeros(domain.shape)

    wave2DBC = equations.Wave2DBC(u, v, p, spatial_order, domain)

    r = np.sqrt((x-3*np.pi/4)**2 + (y-np.pi/2)**2)
    IC = np.exp(-r**2*16)
    p[:] = IC

    ts = timesteppers.RK22(wave2DBC)
    dt = alpha*grid_x.dx

    while ts.t < 2*np.pi - 1e-5:
        ts.step(dt)

    p_target = np.loadtxt('solutions/p_%i.dat' %resolution)
    error = np.max(np.abs(p - p_target))
    error_est = error_wave[(resolution, alpha, spatial_order)]

    assert error < error_est

