r"""Dynamical models."""

import abc
from typing import Any, Optional
from jaxtyping import PyTree, Float, Array
import diffrax as dx
import equinox as eqx
import jax.numpy as jnp
from jax.config import config

from immunowave import spatial, solvers

config.update("jax_enable_x64", True)


class Model(eqx.Module, abc.ABC):
    r"""Abstract base class for dynamical models."""

    @abc.abstractmethod
    def __call__(
        self, t: float, state: PyTree[spatial.GridFn, "N"], args: Any
    ) -> PyTree[spatial.GridFn, "N"]:
        r"""Evaluate the right-hand side of the dynamical equation.

        Args:
            t: Time.
            state: State of the system.
            args: Additional arguments.

        Returns:
            Time derivative of the state vector.
        """

    @staticmethod
    @abc.abstractmethod
    def boundary_metric(state: PyTree[spatial.GridFn, "N"]) -> float:
        r"""Compute a metric on the boundary of the domain.

        Args:
            state: State of the system.
        """

    # @jax.jit
    def solve(
        self,
        state: PyTree[spatial.GridFn, "N"],
        t: Float[Array, " k"],
        rtol: float = 1e-8,
        atol: float = 1e-8,
        **kwargs: Any,
    ) -> dx.Solution:
        r"""Solve the dynamical system.

        Args:
            model: Immune respone model.
            state: Initial condition Pytree.
            t: Times to evaluate the solution at.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            **kwargs: Additional keyword arguments to pass to ``diffrax.diffeqsolve``.

        Returns:
            Solution. If ``t`` is ``None``, the solution can evaluated densely via the ``evaluate()`` method.
            Otherwise, the solution at the time points ``t`` is accessible via the ``ys`` attribute.
        """
        stepsize_controller = dx.PIDController(
            pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol, dtmax=0.001
        )
        solver = solvers.CrankNicolson(rtol=rtol, atol=atol)
        discrete_terminating_event = dx.DiscreteTerminatingEvent(
            lambda t, y, args: self.boundary_metric(y) > 1e-3
        )
        return dx.diffeqsolve(
            dx.ODETerm(self),
            solver,
            t[0],
            t[-1],
            discrete_terminating_event=discrete_terminating_event,
            y0=state,
            **kwargs,
            saveat=dx.SaveAt(ts=t),
            stepsize_controller=stepsize_controller,
        )


class FHNB(Model):
    r"""FitzHugh-Nagumo + Bacteria. The dynamical variables :math:`(A, R, B)` denote
    the concentration of antimicrobial peptide, repression, and bacteria, respectively,
    and satisfy

    .. math::
        \begin{align*}
        \partial_t A &=& \nabla^2 A + A (A - \theta) (1 - A) + \eta B - \rho R \\
        \partial_t R &=& \epsilon (A - R) \\
        \partial_t B &=& \xi \nabla^2 B + \lambda B (1-B) - \mu A B
        \end{align*}

    """
    α: float
    θ: float
    η: float
    ρ: float
    ε: float
    ξ: float
    λ: float
    μ: float

    def __call__(
        self,
        t: float,
        state: jnp.ndarray,  # NOTE: bad typehint
        args,
    ) -> jnp.ndarray:  # NOTE: bad typehint
        r"""Evaluate the right-hand side of the dynamical equation.

        Args:
            t: Time (ignored).
            state: Spatial fields :math:`(A, R, B)`.
            args: Additional arguments (ignored).

        Returns:
            :math:`\partial_t A`, :math:`\partial_t R`, :math:`\partial_t B`.
        """
        A, R, B = state
        dAdt = (
            self.α * A.laplacian(bc="neumann")
            + A * (A - self.θ) * (1 - A)
            + self.η * B
            - self.ρ * R
        )
        dRdt = self.ε * (A - R)
        dBdt = (
            self.ξ * B.laplacian(bc="neumann") + self.λ * B * (1 - B) - self.μ * A * B
        )

        return dAdt, dRdt, dBdt

    @staticmethod
    def boundary_metric(state: jnp.ndarray):  # NOTE: bad typehint
        r"""Maximum antimicrobial peptide concentration on the boundary.

        Args:
            state: Spatial fields :math:`(A, R, B)`.
        """
        A, R, B = state
        return max(A.values[0], A.values[-1])


# def plot(
#     model: ImmuneResponse,
#     state: Dict[str, ArrayLike],
#     axes,
#     *args,
#     **kwargs,
# ) -> None:
#     r"""Plot the state of the system."""
#     n_components = len(state)
#     for i, name in enumerate(state):
#         axes[i].plot(model.x, state[name], *args, **kwargs)
#         axes[i].set_ylabel(name)
#         axes[i].set_xlabel("x")


# # diffrax.MultiTerm  <--- combines terms
# # diffrax events <--- maybe good for convergence issues
