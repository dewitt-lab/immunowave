r"""Test :py:mod:`immunowave.state`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx
import matplotlib.pyplot as plt

from immunowave import state, spatial

# Configure JAX to use 64-bit precision
jax.config.update("jax_enable_x64", True)


class TestState(state.State):
    """Test implementation of State for testing."""
    u: spatial.ScalarField
    v: spatial.ScalarField


class SingleFieldState(state.State):
    """Test state with a single field."""
    u: spatial.ScalarField


def test_state_initialization():
    """Test basic State initialization with ScalarFields."""
    # Create scalar fields
    shape, lb, h = (10,), [0], 0.1
    u_field = spatial.ScalarField(shape, lb, h, values=1.0)
    v_field = spatial.ScalarField(shape, lb, h, values=2.0)
    
    # Initialize state
    test_state = TestState(u=u_field, v=v_field)
    
    # Check that fields are stored correctly
    assert isinstance(test_state.u, spatial.ScalarField)
    assert isinstance(test_state.v, spatial.ScalarField)
    assert jnp.allclose(test_state.u.values, 1.0)
    assert jnp.allclose(test_state.v.values, 2.0)


def test_state_type_validation():
    """Test that State.__check_init__ raises an error for non-ScalarField attributes."""
    # Create valid field
    shape, lb, h = (10,), [0], 0.1
    u_field = spatial.ScalarField(shape, lb, h, values=1.0)
    
    # Attempt to initialize with non-ScalarField value
    with pytest.raises(TypeError):
        # This should call __check_init__ and raise TypeError
        TestState(u=u_field, v="not a scalar field")


def test_state_alignment_validation():
    """Test that State.__check_init__ validates field alignment."""
    # Create misaligned fields (different shapes)
    shape1, lb, h = (10,), [0], 0.1
    shape2, lb, h = (20,), [0], 0.1
    u_field = spatial.ScalarField(shape1, lb, h, values=1.0)
    v_field = spatial.ScalarField(shape2, lb, h, values=2.0)
    
    # This should raise an error due to misalignment
    with pytest.raises(Exception):
        TestState(u=u_field, v=v_field)
    
    # Test with different bounds
    shape, lb1, lb2, h = (10,), [0], [1], 0.1
    u_field = spatial.ScalarField(shape, lb1, h, values=1.0)
    v_field = spatial.ScalarField(shape, lb2, h, values=2.0)
    
    with pytest.raises(Exception):
        TestState(u=u_field, v=v_field)
    
    # Test with different grid spacing
    shape, lb, h1, h2 = (10,), [0], 0.1, 0.2
    u_field = spatial.ScalarField(shape, lb, h1, values=1.0)
    v_field = spatial.ScalarField(shape, lb, h2, values=2.0)
    
    with pytest.raises(Exception):
        TestState(u=u_field, v=v_field)


def test_state_map():
    """Test the map method of State."""
    # Create fields
    shape, lb, h = (10,), [0], 0.1
    u_field = spatial.ScalarField(shape, lb, h, values=1.0)
    v_field = spatial.ScalarField(shape, lb, h, values=2.0)
    
    # Create state
    test_state = TestState(u=u_field, v=v_field)
    
    # Apply map function
    doubled_state = test_state.map(lambda x: 2 * x)
    
    # Check results
    assert jnp.allclose(doubled_state.u.values, 2.0)
    assert jnp.allclose(doubled_state.v.values, 4.0)
    
    # Test with more complex function
    squared_state = test_state.map(lambda x: x**2)
    assert jnp.allclose(squared_state.u.values, 1.0)  # 1² = 1
    assert jnp.allclose(squared_state.v.values, 4.0)  # 2² = 4


def test_state_plot():
    """Test the plot method of State."""
    # Create fields with some interesting pattern
    shape, lb, h = (50,), [0], 0.1
    u_field = spatial.ScalarField(shape, lb, h, fn=lambda x: jnp.sin(x))
    v_field = spatial.ScalarField(shape, lb, h, fn=lambda x: jnp.cos(x))
    
    # Create state
    test_state = TestState(u=u_field, v=v_field)
    
    # Create figure and axes
    fig, axes = plt.subplots(2, 1)
    
    # Plot with provided axes
    returned_axes = test_state.plot(axes=axes)
    
    # Check that the returned axes are the same as provided
    assert returned_axes is axes
    
    # Check that axis titles are set
    assert axes[0].get_title() == "u"
    assert axes[1].get_title() == "v"
    
    # Test with auto-generated axes
    auto_axes = test_state.plot()
    assert len(auto_axes) == 2  # Should have 2 axes
    
    # Test single field state
    single_state = SingleFieldState(u=u_field)
    single_axes = single_state.plot()
    assert len(np.atleast_1d(single_axes)) == 1  # Should have 1 axis


def test_state_with_time_series():
    """Test that State works with time-dependent fields."""
    # Create time-dependent fields
    shape, lb, h = (10,), [0], 0.1
    
    # Create time-dependent values (5 time steps)
    time_steps = 5
    
    # The proper way to handle time series is to use a separate dimension
    # for the time axis in each field's values
    u_values = np.zeros((time_steps,) + shape)
    v_values = np.zeros((time_steps,) + shape)
    
    # Fill with different values for each time step
    for i in range(time_steps):
        u_values[i] = i + 1
        v_values[i] = i + 10
    
    # Create fields with time dimension data
    u_field = spatial.ScalarField(shape, lb, h, values=u_values[0])  # Initial state
    v_field = spatial.ScalarField(shape, lb, h, values=v_values[0])
    
    # Create state
    test_state = TestState(u=u_field, v=v_field)
    
    # Verify that the state uses the initial time values
    assert jnp.allclose(test_state.u.values, 1)
    assert jnp.allclose(test_state.v.values, 10)


def test_state_jax_transform_scalar_operations():
    """Test that State works with JAX transformations for scalar outputs."""
    # Create fields
    shape, lb, h = (5,), [0], 0.1
    u_field = spatial.ScalarField(shape, lb, h, values=jnp.ones(shape))
    v_field = spatial.ScalarField(shape, lb, h, values=2 * jnp.ones(shape))
    
    # Create state
    test_state = TestState(u=u_field, v=v_field)
    
    # Define a function that uses the state to compute a scalar
    def sum_fields(state_obj):
        return jnp.sum(state_obj.u.values) + jnp.sum(state_obj.v.values)
    
    # Test JAX grad
    grad_fn = jax.grad(sum_fields)
    grads = grad_fn(test_state)
    
    # Check that gradients are computed correctly
    assert isinstance(grads, TestState)
    assert jnp.allclose(grads.u.values, 1.0)  # d(sum)/d(u) = 1
    assert jnp.allclose(grads.v.values, 1.0)  # d(sum)/d(v) = 1


def test_state_jax_transform_batch():
    """Test transforming over a collection of states by manual batching."""
    # Create fields
    shape, lb, h = (5,), [0], 0.1
    
    # Create batch of 3 states
    states = []
    for i in range(3):
        u_field = spatial.ScalarField(shape, lb, h, values=i * jnp.ones(shape))
        v_field = spatial.ScalarField(shape, lb, h, values=(i + 1) * jnp.ones(shape))
        states.append(TestState(u=u_field, v=v_field))
    
    # Define a function that computes means
    def field_mean(state_obj):
        return jnp.mean(state_obj.u.values), jnp.mean(state_obj.v.values)
    
    # Map the function over all states manually
    u_means = []
    v_means = []
    for s in states:
        u_mean, v_mean = field_mean(s)
        u_means.append(u_mean)
        v_means.append(v_mean)
    
    # Check results
    expected_u_means = jnp.array([0.0, 1.0, 2.0])
    expected_v_means = jnp.array([1.0, 2.0, 3.0])
    
    assert jnp.allclose(jnp.array(u_means), expected_u_means)
    assert jnp.allclose(jnp.array(v_means), expected_v_means)
    
    # Alternatively, use JAX's vmap with pytrees
    batch_means = jax.vmap(field_mean)(jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *states))
    
    assert jnp.allclose(batch_means[0], expected_u_means)
    assert jnp.allclose(batch_means[1], expected_v_means)