r"""PDE term for Diffrax."""

from typing import Callable
import jax.numpy as jnp
from jaxtyping import PyTree, Scalar
import jax.tree_util as jtu
import diffrax as dx

from immunowave import spatial


class PDETerm(dx.AbstractTerm):
    r"""A term representing :math:`f(t, f(t), args) \mathrm{d}t`. That is to say, the term
    appearing on the right hand side of an PDE, in which the control is time.

    Args:
        vector_field: Should return some PyTree, with the same structure as the initial
    state ``y0``.
    """
    vector_field: Callable[
        [Scalar, PyTree[spatial.ScalarField, " n_components"], PyTree],
        PyTree[spatial.ScalarField, " n_components"],
    ]

    def vf(
        self, t: Scalar, f: PyTree[spatial.ScalarField, " n_components"], args: PyTree
    ) -> PyTree[spatial.ScalarField, " n_components"]:
        out = self.vector_field(t, f, args)
        if spatial._field_structure(out) != spatial._field_structure(f):
            raise ValueError(
                "The vector field inside `PDETerm` must return a pytree with the "
                "same structure as `y0`."
            )
        return out

    @staticmethod
    def contr(t0: Scalar, t1: Scalar) -> Scalar:
        return t1 - t0

    @staticmethod
    def prod(vf: PyTree, control: Scalar) -> PyTree:
        return jtu.tree_map(lambda v: control * v, vf)
