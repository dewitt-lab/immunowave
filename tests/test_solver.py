r"""Test :py:mod:`immunowave.solver`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx
import diffrax as dx

from immunowave import model, spatial, state, solver

# Configure JAX to use 64-bit precision
jax.config.update("jax_enable_x64", True)


# Define a simple test state and model for testing the solver
class TestState(state.State):
    u: spatial.ScalarField


class TestModel(model.Model):
    """Simple diffusion model for testing."""

    D: float = eqx.field(static=True, default=1.0)

    def __call__(self, t, state, args=None):
        return TestState(self.D * state.u.laplacian())


def test_spatial_pde_term():
    """Test the SpatialPDETerm class."""
    # Create simple model and state
    shape, lb, h = (20,), [0], 0.1
    test_state = TestState(
        spatial.ScalarField(shape, lb, h, fn=lambda x: jnp.exp(-5 * (x - 1.0) ** 2))
    )
    test_model = TestModel(D=0.5)

    # Create the PDE term
    pde_term = solver.SpatialPDETerm(test_model)

    # Test that it has the right attributes
    assert hasattr(pde_term, "vector_field")
    assert pde_term.vector_field is test_model

    # Test calling the term's vf method directly
    t = 0.0
    args = None
    y = test_state

    # Use the vf method (not the term directly)
    dy = pde_term.vf(t, y, args)

    # Verify the result is the same as calling the model directly
    expected_dy = test_model(t, y, args)

    # Check that derivatives match
    assert isinstance(dy, TestState)
    assert dy.u.values.shape == expected_dy.u.values.shape
    assert jnp.allclose(dy.u.values, expected_dy.u.values)


def test_crank_nicolson():
    """Test the CrankNicolson solver."""
    # Create the solver
    rtol, atol = 1e-4, 1e-4
    cn_solver = solver.CrankNicolson(rtol, atol)

    # Verify it has the right attributes
    assert cn_solver.rtol == rtol
    assert cn_solver.atol == atol

    # Check that it's a valid diffrax solver
    assert isinstance(cn_solver, dx.AbstractSolver)


@pytest.mark.parametrize("shape, h", [((50,), 0.1), ((20, 20), 0.1)])
def test_diffusion_integration(shape, h):
    """Test integration of a diffusion equation with the CrankNicolson solver."""
    # Create a simple diffusion problem
    ndim = len(shape)
    lb = [0] * ndim

    # Gaussian initial condition
    midpoint = [s * h / 2 for s in shape]
    sigma = 0.1

    if ndim == 1:
        init_fn = lambda x: jnp.exp(-(((x - midpoint[0]) / sigma) ** 2))
    else:  # 2D
        init_fn = lambda x, y: jnp.exp(
            -(((x - midpoint[0]) / sigma) ** 2 + ((y - midpoint[1]) / sigma) ** 2)
        )

    test_state = TestState(spatial.ScalarField(shape, lb, h, fn=init_fn))

    # Analytical diffusion coefficient
    D = 0.1
    test_model = TestModel(D=D)

    # Integration parameters
    t0, t1 = 0.0, 0.01  # Short integration time for testing
    dt0 = 1e-4

    # Create the solver and PDE term
    cn_solver = solver.CrankNicolson(rtol=1e-4, atol=1e-4)
    pde_term = solver.SpatialPDETerm(test_model)

    # Solve the diffusion equation
    stepsize_controller = dx.PIDController(rtol=1e-4, atol=1e-4)
    solution = dx.diffeqsolve(
        pde_term,
        cn_solver,
        t0,
        t1,
        dt0,
        test_state,
        stepsize_controller=stepsize_controller,
        saveat=dx.SaveAt(t1=True, dense=True),
    )

    # Extract the final state
    final_state = solution.evaluate(t1)

    # Check that the solution diffused (variance increased)
    init_var = spatial_variance(test_state.u)
    final_var = spatial_variance(final_state.u)

    # Variance should increase with diffusion
    assert final_var > init_var

    # Conservation of mass
    init_mass = test_state.u.integral()
    final_mass = final_state.u.integral()
    assert jnp.abs(final_mass - init_mass) < 1e-3 * jnp.abs(init_mass)

    # Check that maximum value decreased (diffusion spreads the peak)
    assert jnp.max(final_state.u.values) < jnp.max(test_state.u.values)


def spatial_variance(field):
    """Calculate spatial variance of a scalar field."""
    # Get the grid coordinates
    coords = field.domain

    # Convert to meshgrid
    if field.ndim == 1:
        X = coords[0]
        weights = field.values / jnp.sum(field.values)
        mean_x = jnp.sum(weights * X)
        var_x = jnp.sum(weights * (X - mean_x) ** 2)
        return var_x
    elif field.ndim == 2:
        X, Y = jnp.meshgrid(*coords, indexing="ij")
        weights = field.values / jnp.sum(field.values)
        mean_x = jnp.sum(weights * X)
        mean_y = jnp.sum(weights * Y)
        var_x = jnp.sum(weights * (X - mean_x) ** 2)
        var_y = jnp.sum(weights * (Y - mean_y) ** 2)
        return var_x + var_y
    else:
        raise NotImplementedError("Variance calculation for ndim > 2 not implemented")


def test_conservation_properties_crank_nicolson():
    """Test that the Crank-Nicolson solver preserves conservation laws."""

    # Create a conservative transport model
    class TransportState(state.State):
        density: spatial.ScalarField

    class TransportModel(model.Model):
        velocity: float = eqx.field(static=True, default=1.0)

        def __call__(self, t, state, args=None):
            # For testing conservation, we'll use a simple advection equation
            # with periodic boundary conditions (where mass should be conserved)
            density = state.density.values
            dx = state.density.h

            # Simple upwind scheme with periodic boundary
            # Not super accurate but good enough for testing conservation
            density_shifted = jnp.roll(density, -1)

            # Derivative term for advection equation: du/dt = -v * du/dx
            d_density_dt = -self.velocity * (density_shifted - density) / dx

            # Wrap in ScalarField to return proper state
            return TransportState(
                spatial.ScalarField(
                    density.shape,
                    state.density.lb,
                    state.density.h,
                    values=d_density_dt,
                )
            )

    # Create a Gaussian pulse as the initial condition
    shape, lb, h = (100,), [0], 0.1
    field = spatial.ScalarField(
        shape, lb, h, fn=lambda x: jnp.exp(-100 * (x - lb[0] - (shape[0] * h) / 2) ** 2)
    )
    initial_state = TransportState(field)

    # Create the model with a specific velocity
    test_model = TransportModel(velocity=1.0)

    # Create solver components
    cn_solver = solver.CrankNicolson(rtol=1e-6, atol=1e-6)
    pde_term = solver.SpatialPDETerm(test_model)

    # Integrate for a short time
    t0, t1 = 0.0, 0.1
    dt0 = 1e-3

    solution = dx.diffeqsolve(
        pde_term,
        cn_solver,
        t0,
        t1,
        dt0,
        initial_state,
        stepsize_controller=dx.PIDController(rtol=1e-6, atol=1e-6),
        saveat=dx.SaveAt(t1=True, dense=True),
    )

    # Check mass conservation
    initial_mass = initial_state.density.integral()
    final_state = solution.evaluate(t1)
    final_mass = final_state.density.integral()

    # The mass should be very well conserved with Crank-Nicolson
    relative_error = jnp.abs(final_mass - initial_mass) / jnp.abs(initial_mass)
    assert relative_error < 1e-5  # Crank-Nicolson should conserve mass well
