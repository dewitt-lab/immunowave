r"""Dynamical models."""

import numpy as np
import jax
from numpy.typing import NDArray
import diffrax as dx
import equinox as eqx
import abc
from typing import Any
from jaxtyping import Float

from immunowave import spatial, state, solver

jax.config.update("jax_enable_x64", True)


class Model(eqx.Module, abc.ABC):
    r"""Abstract base class for dynamical models. Subclass this to implement your
    model.
    """

    @abc.abstractmethod
    @jax.jit
    def __call__(self, t: float, state: state.State, args: Any) -> state.State:
        r"""Evaluate the right-hand side of the dynamical equation.

        Args:
            t: Time.
            state: State of the system, as :py:class:`equinox.Module` of :py:class:`immunowave.spatial.ScalarField` fields.
            args: Additional arguments.

        Returns:
            Time derivative of the state.
        """


@eqx.filter_jit
def solve(
    model: Model,
    state: state.State,
    t0: float,
    t1: float,
    dt0: float | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    **kwargs: Any,
) -> dx.Solution:
    r"""Solve the dynamical system.

    Args:
        model: Immune respone model.
        state: Initial condition, as :py:class:`equinox.Module` of :py:class:`immunowave.spatial.ScalarField` fields.
        t0: The start of the region of integration.
        t1: The end of the region of integration.
        dt0: Initial step size. If ``None``, the step size is chosen automatically.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        **kwargs: Additional keyword arguments to pass to ``diffrax.diffeqsolve``.

    Returns:
        Solution.
    """
    if "stepsize_controller" not in kwargs:
        kwargs["stepsize_controller"] = dx.PIDController(
            pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol, dtmax=0.001
        )
    return dx.diffeqsolve(
        solver.SpatialPDETerm(model),
        solver.CrankNicolson(rtol, atol),
        t0,
        t1,
        dt0,
        state,
        **kwargs,
    )
