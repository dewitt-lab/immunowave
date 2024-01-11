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
            state: State of the system, as :py:class:`eqx.Module` of :py:class:`spatial.ScalarField` fields.
            args: Additional arguments.

        Returns:
            Time derivative of the state.
        """

    # @staticmethod
    # @abc.abstractmethod
    # def boundary_metric(state: state.State) -> float:
    #     r"""Compute a metric on the boundary of the domain.

    #     Args:
    #         state: State of the system, as :py:class:`eqx.Module` of :py:class:`spatial.ScalarField` fields.
    #     """


# @jax.jit
def solve(
    model: Model,
    state: state.State,
    t0: float,
    t1: float,
    dt0: float | None = None,
    t: Float[np.ndarray, " k"] | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    boundary_threshold: float = 1e-3,
    **kwargs: Any,
) -> dx.Solution:
    r"""Solve the dynamical system.

    Args:
        model: Immune respone model.
        state: Initial condition, as :py:class:`eqx.Module` of :py:class:`spatial.ScalarField` fields.
        t0: The start of the region of integration.
        t1: The end of the region of integration.
        dt0: Initial step size. If ``None``, the step size is chosen automatically.
        t: Time points at which to save the solution. If ``None``, dense output is
           returned.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        boundary_threshold: Threshold for the boundary metric.
        **kwargs: Additional keyword arguments to pass to ``diffrax.diffeqsolve``.

    Returns:
        Solution. If ``t`` is ``None``, the solution can evaluated densely via the
        ``evaluate()`` method. Otherwise, the solution at the time points ``t`` is
        accessible via the ``ys`` attribute.
    """
    return dx.diffeqsolve(
        solver.SpatialPDETerm(model),
        # dx.ODETerm(model),
        solver.CrankNicolson(rtol, atol),
        # dx.Tsit5(),
        t0,
        t1,
        dt0,
        state,
        saveat=dx.SaveAt(ts=t) if t is not None else dx.SaveAt(dense=True),
        # stepsize_controller=dx.PIDController(
        #     pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol, dtmax=0.001
        # ),
        # discrete_terminating_event=dx.DiscreteTerminatingEvent(
        #     lambda t, y, args: model.boundary_metric(y) > boundary_threshold
        # ),
        **kwargs,
    )


# # diffrax.MultiTerm  <--- combines terms
# # diffrax events <--- maybe good for convergence issues
