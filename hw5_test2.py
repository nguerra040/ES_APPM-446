import pytest
import numpy as np
import finite
import timesteppers
import equations

resolution_list = [100, 200, 400]

error_wave_variable = {(100,2):0.25, (200,2):0.25, (400,2):0.1, (100,4):0.2, (200,4):0.02, (400,4):0.002}
@pytest.mark.parametrize('resolution', resolution_list)
@pytest.mark.parametrize('spatial_order', [2, 4])
def test_wave_variable(resolution, spatial_order):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    u = np.exp(-(x-np.pi)**2*8)
    p = 0*x

    d = finite.DifferenceUniformGrid(1, spatial_order, grid)

    rho0 = 1+0.8*np.sin(x+np.pi/3)
    gamma_p0 = 1

    soundwave_problem = equations.SoundWave(u, p, d, rho0, gamma_p0)
    ts = timesteppers.CNAB(soundwave_problem)

    alpha = 0.2
    dt = alpha*grid.dx
    ts.evolve(dt, np.pi)

    solution = np.loadtxt('solutions/u_0p8_%i.dat' %resolution)
    error = np.max(np.abs(solution - u))
    error_est = error_wave_variable[(resolution,spatial_order)]
    assert error < error_est

error_RD = {(100,2):0.02, (200,2):0.006, (400,2):0.002, (100,4):0.001, (200,4):0.001, (400,4):0.001}
@pytest.mark.parametrize('resolution', resolution_list)
@pytest.mark.parametrize('spatial_order', [2, 4])
def test_reaction_diffusion(resolution, spatial_order):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    c = 0.1*np.exp(-(x-np.pi)**2*8)

    c_target = 1
    D = 1e-3

    d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)

    reaction_diffusion_problem = equations.ReactionDiffusion(c, d2, c_target, D)
    ts = timesteppers.CNAB(reaction_diffusion_problem)

    alpha = 0.2
    dt = alpha*grid.dx
    ts.evolve(dt, 10)

    solution = np.loadtxt('solutions/c_const_%i.dat' %resolution)
    error = np.max(np.abs(solution - c))
    error_est = error_RD[(resolution,spatial_order)]
    assert error < error_est

error_RD_variable = {(100,2):0.02, (200,2):0.006, (400,2):0.003, (100,4):0.0015, (200,4):0.0015, (400,4):0.002}
@pytest.mark.parametrize('resolution', resolution_list)
@pytest.mark.parametrize('spatial_order', [2, 4])
def test_reaction_diffusion_variable(resolution, spatial_order):
    grid = finite.UniformPeriodicGrid(resolution, 2*np.pi)
    x = grid.values

    c = 0.1*np.exp(-(x-np.pi)**2*8)

    c_target = 1-0.5*np.sin(3*x)
    D = 1e-3

    d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)

    reaction_diffusion_problem = equations.ReactionDiffusion(c, d2, c_target, D)
    ts = timesteppers.CNAB(reaction_diffusion_problem)

    alpha = 0.2
    dt = alpha*grid.dx
    ts.evolve(dt, 10)

    solution = np.loadtxt('solutions/c_sin3x_%i.dat' %resolution)
    error = np.max(np.abs(solution - c))
    error_est = error_RD_variable[(resolution,spatial_order)]
    assert error < error_est

