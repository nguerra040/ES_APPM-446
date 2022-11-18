import pytest
import numpy as np
import finite
import timesteppers
import equations

error_c1 = {(100, 0.1, 10): 0.005, (100, 0.2, 10): 0.015, (100, 0.5, 10): 0.07,
            (200, 0.1, 10): 0.004, (200, 0.2, 10): 0.015, (200, 0.5, 10): 0.07,
            (100, 0.1, 100): 0.001, (100, 0.2, 100): 0.006, (100, 0.5, 100): 0.04,
            (200, 0.1, 100): 0.001, (200, 0.2, 100): 0.006, (200, 0.5, 100): 0.04, 
            (100, 0.1, 1000): 0.0015, (100, 0.2, 1000): 0.006, (100, 0.5, 1000): 0.04,
            (200, 0.1, 1000): 0.0015, (200, 0.2, 1000): 0.006, (200, 0.5, 1000): 0.04}

error_c2 = {(100, 0.1, 10): 0.02,  (100, 0.2, 10): 0.1,   (100, 0.5, 10): 0.4,
            (200, 0.1, 10): 0.02,  (200, 0.2, 10): 0.1,   (200, 0.5, 10): 0.4,
            (100, 0.1, 100): 0.005, (100, 0.2, 100): 0.015, (100, 0.5, 100): 0.07,
            (200, 0.1, 100): 0.005, (200, 0.2, 100): 0.015, (200, 0.5, 100): 0.07,
            (100, 0.1, 1000): 0.0015, (100, 0.2, 1000): 0.006, (100, 0.5, 1000): 0.04,
            (200, 0.1, 1000): 0.0015, (200, 0.2, 1000): 0.006, (200, 0.5, 1000): 0.04}

@pytest.mark.parametrize('resolution', [100, 200])
@pytest.mark.parametrize('dt', [0.1, 0.2, 0.5])
@pytest.mark.parametrize('r', [10, 100, 1000])
def test_reactiontwospecies(resolution, r, dt):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    c1 = np.zeros(resolution)
    c2 = np.zeros(resolution)
    X = timesteppers.StateVector([c1, c2])

    D = 1e-3
    rd = equations.ReactionTwoSpeciesDiffusion(X, D, r, 4, grid)

    IC = 0.1*np.exp(-(x-np.pi)**2*2)
    c1[:] = IC
    c2[:] = 0.5*IC

    ts = timesteppers.CrankNicolsonFI(rd)

    ts.evolve(dt, 10-1e-5)

    c1_target = np.loadtxt('solutions/c1_%i_%i.dat' %(r, resolution))
    c2_target = np.loadtxt('solutions/c2_%i_%i.dat' %(r, resolution))
    error1 = np.max(np.abs(c1 - c1_target))
    error2 = np.max(np.abs(c2 - c2_target))

    error_est1 = error_c1[(resolution, dt, r)]
    error_est2 = error_c2[(resolution, dt, r)]

    assert (error1 < error_est1) and (error2 < error_est2)

