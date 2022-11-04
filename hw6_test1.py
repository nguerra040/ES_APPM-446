import pytest
import numpy as np
import finite
import timesteppers
import equations

error_RD = {(50,0.5):3e-3, (50,0.25):2.5e-3, (50,0.125):2.5e-3,(100,0.5):4e-4, (100,0.25):2e-4, (100,0.125):1e-4, (200,0.5):8e-5, (200,0.25):2e-5, (200,0.125):5e-6}
@pytest.mark.parametrize('resolution', [50, 100, 200])
@pytest.mark.parametrize('alpha', [0.5, 0.25, 0.125])
def test_reaction_diffusion(resolution, alpha):
    grid_x = finite.UniformPeriodicGrid(resolution, 20)
    grid_y = finite.UniformPeriodicGrid(resolution, 20)
    domain = finite.Domain((grid_x, grid_y))
    x, y = domain.values()

    IC = np.exp(-(x+(y-10)**2-14)**2/8)*np.exp(-((x-10)**2+(y-10)**2)/10)

    c = np.zeros(domain.shape)
    c[:] = IC
    D = 1e-2

    dx2 = finite.DifferenceUniformGrid(2, 8, grid_x, 0)
    dy2 = finite.DifferenceUniformGrid(2, 8, grid_y, 1)

    rd_problem = equations.ReactionDiffusion2D(c, D, dx2, dy2)

    dt = alpha*grid_x.dx

    while rd_problem.t < 1-1e-5:
        rd_problem.step(dt)

    solution = np.loadtxt('solutions/c_%i.dat' %resolution)
    error = np.max(np.abs(solution - c))

    error_est = error_RD[(resolution,alpha)]

    assert error < error_est


