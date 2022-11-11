import pytest
import numpy as np
import finite
import timesteppers
import equations

error_derivative_1 = {(50, 2): 0.005, (100, 2): 0.0015, (200, 2): 4e-4, (50, 4): 4e-5, (100, 4): 2e-6, (200, 4): 1.5e-7, (50, 6): 3e-7, (100, 6): 4e-9, (200, 6): 7e-11}
@pytest.mark.parametrize('resolution', [50, 100, 200])
@pytest.mark.parametrize('convergence_order', [2, 4, 6])
def test_derivative_1(resolution, convergence_order):
    grid = finite.UniformNonPeriodicGrid(resolution,(0, 5))
    x = grid.values
    u = np.sin(x)

    d = finite.DifferenceUniformGrid(1, convergence_order, grid)
    du = d @ u

    error = np.max(np.abs(du - np.cos(x)))
    error_est = error_derivative_1[(resolution, convergence_order)]

    assert error < error_est

error_diffusion = {(100, 0.5, 2): 2.5e-6, (100, 0.5, 4): 7e-7,   (100, 0.25, 2): 2.5e-6, (100, 0.25, 4): 1.5e-7,
                   (200, 0.5, 2): 6e-7,   (200, 0.5, 4): 1.5e-7, (200, 0.25, 2): 6e-6,   (200, 0.25, 4): 4e-8}
@pytest.mark.parametrize('resolution', [100, 200])
@pytest.mark.parametrize('alpha', [0.5, 0.25])
@pytest.mark.parametrize('spatial_order', [2, 4])
def test_diffusion_equation(resolution, alpha, spatial_order):
    grid_x = finite.UniformNonPeriodicGrid(resolution,(0,2*np.pi))
    grid_y = finite.UniformPeriodicGrid(resolution,2*np.pi)
    domain = finite.Domain([grid_x, grid_y])
    x, y = domain.values()

    c = np.zeros(domain.shape)
    D = 1

    r = np.sqrt((x-3*np.pi/4)**2 + (y-np.pi/2)**2)
    IC = np.exp(-r**2*16)
    c[:] = IC

    diff = equations.DiffusionBC(c, D, spatial_order, domain)

    dt = alpha*grid_y.dx

    while diff.t < 3*np.pi/4 - 1e-5:
        diff.step(dt)

    c_target = np.loadtxt('solutions/c_%i.dat' %resolution)
    error = np.max(np.abs(c - c_target))
    error_est = error_diffusion[(resolution, alpha, spatial_order)]

    assert error < error_est

