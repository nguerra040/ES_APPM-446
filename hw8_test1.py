
import pytest
import numpy as np
import finite
import timesteppers
import equations


error_burgers = {(100, 0.5): 0.03, (100, 1): 0.04, (100, 2): 0.05, (100, 4): 0.15,
                 (200, 0.5): 0.0015, (200, 1): 0.002, (200, 2): 0.005, (200, 4): 0.02,
                 (400, 0.5): 0.0003, (400, 1): 0.0003, (400, 2): 0.001, (400, 4): 0.004}
error_CN = {(100, 0.25): 0.02, (100, 0.5): 0.1, (100, 1): 0.4,
            (200, 0.25): 0.02, (200, 0.5): 0.1, (200, 1): 0.4}
@pytest.mark.parametrize('resolution', [100, 200])
@pytest.mark.parametrize('dt', [0.25, 0.5, 1])
def test_CN(resolution, dt):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    c = np.zeros(resolution)
    D = 1e-3
    rd = equations.ReactionDiffusionFI(c, D, 6, grid)
    ts = timesteppers.CrankNicolsonFI(rd)

    c[:] = 0.1*np.exp(-(x-np.pi)**2*8)

    ts.evolve(dt, 10-1e-5)

    c_target = np.loadtxt('solutions/c_const_%i.dat' %resolution)
    error = np.max(np.abs(c.data - c_target))
    error_est = error_CN[(resolution, dt)]

    assert error < error_est


error_burgers = {(100, 0.5): 0.03, (100, 1): 0.04, (100, 2): 0.05, (100, 4): 0.15,
                 (200, 0.5): 0.0015, (200, 1): 0.002, (200, 2): 0.005, (200, 4): 0.02,
                 (400, 0.5): 0.0003, (400, 1): 0.0003, (400, 2): 0.001, (400, 4): 0.004}
@pytest.mark.parametrize('resolution', [100, 200, 400])
@pytest.mark.parametrize('alpha', [0.5, 1, 2, 4])
def test_burgers(resolution, alpha):
    grid = finite.UniformPeriodicGrid(resolution, 5)
    x = grid.values

    u = np.zeros(resolution)
    nu = 1e-2
    burgers = equations.BurgersFI(u, nu, 6, grid)
    ts = timesteppers.CrankNicolsonFI(burgers)

    IC = 0*x
    for i, xx in enumerate(x):
        if xx > 1 and xx <= 2:
            IC[i] = (xx-1)
        elif xx > 2 and xx < 3:
            IC[i] = (3-xx)
    
    u[:] = IC
    dt = alpha*grid.dx

    ts.evolve(dt, 3-1e-5)

    u_target = np.loadtxt('solutions/u_burgers_%i.dat' %resolution)
    error = np.max(np.abs(u - u_target))
    error_est = error_burgers[(resolution, alpha)]

    assert error < error_est

