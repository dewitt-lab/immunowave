r"""Test :mod:`immunowave.spatial`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from immunowave import spatial

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "d, shape, lb, h",
    [(1, 10, 0, 0.1), (2, (10, 10), 0, 0.1), (3, (10, 10, 10), 0, 0.1)],
)
def test_scalar_field(d, shape, lb, h):
    r"""Test :class:`immunowave.spatial.ScalarField`."""
    values = np.random.standard_normal(shape)
    field = spatial.ScalarField(shape, lb, h, values=values)
    assert (field.lb.squeeze() == lb).all()
    assert field.h == h
    np.broadcast_shapes(field.values.shape, shape)
    assert field.values.dtype == jnp.float64
    assert field.ndim == d


@pytest.mark.parametrize(
    "d, shape, lb, h",
    [(1, 10, 0, 0.1), (2, (10, 10), 0, 0.1), (3, (10, 10, 10), 0, 0.1)],
)
def test_scalar_field_from_fn(d, shape, lb, h):
    r"""Test :class:`immunowave.spatial.ScalarField`."""
    fn = lambda *x: sum(xi**2 for xi in x)
    field = spatial.ScalarField(shape, lb, h, fn=fn)
    assert (field.lb.squeeze() == lb).all()
    assert field.h == h
    np.broadcast_shapes(field.values.shape, shape)
    assert field.values.dtype == jnp.float64
    assert field.ndim == d
    assert np.allclose(field.laplacian().values[*[slice(1, -1)] * d], 2 * d)
