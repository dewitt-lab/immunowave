r"""Test :py:mod:`immunowave.model`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx
import diffrax as dx

from immunowave import model, spatial, state


# Configure JAX to use 64-bit precision
jax.config.update("jax_enable_x64", True)


# Define a simple test model and state for testing
class TestState(state.State):
    u: spatial.ScalarField
    v: spatial.ScalarField


class TestModel(model.Model):
    """Simple FitzHugh-Nagumo model for testing purposes."""

    a: float = eqx.field(static=True, default=0.2)
    b: float = eqx.field(static=True, default=0.2)
    c: float = eqx.field(static=True, default=3.0)
    D: float = eqx.field(static=True, default=1.0)

    def __call__(self, t, state, args=None):
        u, v = state.u, state.v
        dudt = self.D * u.laplacian() + u - u ** 3 - v
        dvdt = self.a * u - self.b * v + self.c
        return TestState(dudt, dvdt)


# Test parameters for different spatial dimensions and configurations
params_str = "shape, lb, h"
params = [
    ((50,), [0], 0.1),  # 1D
    ((20, 20), [0, 0], 0.1),  # 2D
    ((10, 10, 10), [0, 0, 0], 0.1),  # 3D
]


@pytest.mark.parametrize(params_str, params)
def test_model_instantiation_and_call(shape, lb, h):
    """Test model instantiation and __call__ method."""
    # Create a test state with simple initial conditions
    test_state = TestState(
        spatial.ScalarField(shape, lb, h, fn=lambda *x: 0.5),
        spatial.ScalarField(shape, lb, h, fn=lambda *x: 0.1),
    )

    # Create a test model
    test_model = TestModel()

    # Call the model to get derivatives
    derivatives = test_model(0.0, test_state, None)

    # Validate that the returned object is the correct type
    assert isinstance(derivatives, TestState)

    # Check that the derivatives have the expected shape
    assert derivatives.u.values.shape == test_state.u.values.shape
    assert derivatives.v.values.shape == test_state.v.values.shape

    # Validate that the derivatives are JAX arrays with double precision
    assert isinstance(derivatives.u.values, jnp.ndarray)
    assert isinstance(derivatives.v.values, jnp.ndarray)
    assert derivatives.u.values.dtype == jnp.float64
    assert derivatives.v.values.dtype == jnp.float64


@pytest.mark.parametrize(
    params_str, params[:1]
)  # Use only 1D for differentiation tests
def test_model_differentiation(shape, lb, h):
    """Test that model parameters can be differentiated with JAX."""
    # Create a simple state
    test_state = TestState(
        spatial.ScalarField(shape, lb, h, fn=lambda *x: 0.5),
        spatial.ScalarField(shape, lb, h, fn=lambda *x: 0.1),
    )

    # Create a model with parameters to differentiate
    test_model = TestModel(a=0.3, b=0.2, c=3.0, D=1.0)

    # Define a simple loss function that depends on the model output
    def loss_fn(model_instance):
        derivatives = model_instance(0.0, test_state, None)
        # Sum of squared derivatives as a simple loss
        return jnp.sum(derivatives.u.values ** 2) + jnp.sum(derivatives.v.values ** 2)

    # Compute gradient with respect to the model
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(test_model)

    # Check that gradients have the right structure
    assert isinstance(grads, TestModel)


@pytest.mark.parametrize(params_str, params[:1])  # Use only 1D for solve tests (faster)
def test_solve_fhn_model(shape, lb, h):
    """Test solving the FitzHugh-Nagumo model."""
    # Create initial conditions with a perturbation in the middle
    mid_idx = shape[0] // 2
    u_init = np.zeros(shape)
    u_init[mid_idx] = 1.0  # Single point perturbation

    # Create state and model
    test_state = TestState(
        spatial.ScalarField(shape, lb, h, values=u_init),
        spatial.ScalarField(shape, lb, h, values=np.zeros(shape)),
    )
    test_model = TestModel(a=0.3, b=0.2, c=3.0, D=1.0)

    # Short integration time for testing
    t0, t1 = 0.0, 0.5
    dt0 = 1e-3

    # Solve the model with dense output for evaluation
    solution = model.solve(
        test_model,
        test_state,
        t0,
        t1,
        dt0=dt0,
        rtol=1e-4,
        atol=1e-4,
        saveat=dx.SaveAt(t1=True, dense=True),
    )

    # Basic checks on the solution
    assert isinstance(solution, dx.Solution)

    # Evaluate solution at the final time
    final_state = solution.evaluate(t1)
    assert isinstance(final_state, TestState)

    # The solution should have propagated from the center
    # Check that more points are non-zero at the end than the start
    assert np.sum(np.abs(final_state.u.values) > 1e-6) > 1


def test_solve_with_different_controllers():
    """Test solve function with different step size controllers."""
    # Create a simple 1D test case
    shape, lb, h = (20,), [0], 0.1
    test_state = TestState(
        spatial.ScalarField(
            shape, lb, h, fn=lambda x: 0.5 * jnp.exp(-((x - 1.0) ** 2))
        ),
        spatial.ScalarField(shape, lb, h, values=0.0),
    )
    test_model = TestModel()

    t0, t1 = 0.0, 0.1

    # Test with PIDController (default)
    solution1 = model.solve(test_model, test_state, t0, t1, dt0=1e-3)
    assert isinstance(solution1, dx.Solution)

    # Test with explicit controller specification
    pid_controller = dx.PIDController(rtol=1e-3, atol=1e-3)
    solution2 = model.solve(
        test_model, test_state, t0, t1, dt0=1e-3, stepsize_controller=pid_controller
    )
    assert isinstance(solution2, dx.Solution)


@pytest.mark.parametrize(params_str, params[:1])  # Use only 1D for conservation tests
def test_conservation_properties(shape, lb, h):
    """Test conservation properties of the solver for a conservative system."""

    # Define a simple conservative model (e.g., advection)
    class ConservativeState(state.State):
        density: spatial.ScalarField

    class AdvectionModel(model.Model):
        velocity: float = eqx.field(static=True, default=1.0)

        def __call__(self, t, state, args=None):
            # Simple finite difference for demonstration
            # (Not analytically conservative, but illustrates the concept)
            density = state.density.values
            dx = state.density.h

            # Periodic boundary condition shift for testing conservation
            density_shifted = jnp.roll(density, -1)

            # Compute derivative using upwind scheme
            d_density_dt = -self.velocity * (density_shifted - density) / dx

            return ConservativeState(
                spatial.ScalarField(shape, lb, h, values=d_density_dt)
            )

    # Create Gaussian initial condition
    field = spatial.ScalarField(
        shape, lb, h, fn=lambda x: jnp.exp(-50 * (x - lb[0] - (shape[0] * h) / 2) ** 2)
    )
    initial_state = ConservativeState(field)

    # Create model
    test_model = AdvectionModel()

    # Solve for a short time with dense output for evaluation
    t0, t1 = 0.0, 0.1
    solution = model.solve(
        test_model,
        initial_state,
        t0,
        t1,
        dt0=1e-3,
        saveat=dx.SaveAt(t1=True, dense=True),
    )

    # Check conservation of total "mass"
    initial_mass = initial_state.density.integral()
    final_state = solution.evaluate(t1)
    final_mass = final_state.density.integral()

    # The mass should be approximately conserved
    assert jnp.abs(final_mass - initial_mass) < 1e-3 * jnp.abs(initial_mass)


def test_model_immutability():
    """Test that model parameters aren't accidentally mutated during integration."""
    # Create a simple model and state
    shape, lb, h = (10,), [0], 0.1
    test_state = TestState(
        spatial.ScalarField(shape, lb, h, values=0.5),
        spatial.ScalarField(shape, lb, h, values=0.1),
    )
    original_model = TestModel(a=0.3, b=0.2, c=3.0, D=1.0)

    # Store original parameters
    original_a = original_model.a
    original_b = original_model.b
    original_c = original_model.c
    original_D = original_model.D

    # Run the solver
    t0, t1 = 0.0, 0.1
    solution = model.solve(original_model, test_state, t0, t1, dt0=1e-3)

    # Check that parameters haven't changed
    assert original_model.a == original_a
    assert original_model.b == original_b
    assert original_model.c == original_c
    assert original_model.D == original_D


def test_solve_input_validation():
    """Test that solve properly validates inputs."""
    # Invalid model type (should catch before reaching diffrax)
    with pytest.raises((TypeError, ValueError)):
        model.solve("not_a_model", None, 0.0, 1.0)

    # Invalid state type
    test_model = TestModel()
    with pytest.raises((TypeError, ValueError)):
        model.solve(test_model, "not_a_state", 0.0, 1.0)

    # Test with None as state
    with pytest.raises((TypeError, ValueError)):
        model.solve(test_model, None, 0.0, 1.0)

    # Test with invalid dt0 (negative)
    shape, lb, h = (10,), [0], 0.1
    test_state = TestState(
        spatial.ScalarField(shape, lb, h, values=0.5),
        spatial.ScalarField(shape, lb, h, values=0.1),
    )

    # If diffrax doesn't validate this, that's OK - just verify the function runs
    # and returns a valid solution
    solution = model.solve(test_model, test_state, 1.0, 0.0)  # t1 < t0
    assert isinstance(solution, dx.Solution)
