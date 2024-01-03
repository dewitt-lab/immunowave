r"""Dynamical models."""

import numpy as np
import diffrax as dx
import equinox as eqx
import abc
from typing import Any
from jaxtyping import PyTree, Float
import jax.tree_util as jtu
from jax.config import config
import matplotlib.pyplot as plt
import dataclasses as dc

from immunowave import spatial, solvers

config.update("jax_enable_x64", True)


class Model(eqx.Module, abc.ABC):
    r"""Abstract base class for dynamical models."""

    @abc.abstractmethod
    def __call__(
        self, t: float, state: PyTree[spatial.ScalarField, " n_components"], args: Any
    ) -> PyTree[spatial.ScalarField, " n_components"]:
        r"""Evaluate the right-hand side of the dynamical equation.

        Args:
            t: Time.
            state: State of the system, as :py:class:`eqx.Module` of :py:class:`spatial.ScalarField` fields.
            args: Additional arguments.

        Returns:
            Time derivative of the state vector.
        """

    @staticmethod
    @abc.abstractmethod
    def boundary_metric(state: PyTree[spatial.ScalarField, " n_components"]) -> float:
        r"""Compute a metric on the boundary of the domain.

        Args:
            state: State of the system, as :py:class:`eqx.Module` of :py:class:`spatial.ScalarField` fields.
        """

    # @jax.jit
    def solve(
        self,
        state: PyTree[spatial.ScalarField, " n_components"],
        t: Float[np.ndarray, " k"],
        rtol: float = 1e-8,
        atol: float = 1e-8,
        boundary_threshold: float = 1e-3,
        dt0: float | None = None,
        **kwargs: Any,
    ) -> dx.Solution:
        r"""Solve the dynamical system.

        Args:
            model: Immune respone model.
            state: Initial condition, as :py:class:`eqx.Module` of :py:class:`spatial.ScalarField` fields.
            t: Times to evaluate the solution at.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            boundary_threshold: Threshold for the boundary metric.
            dt0: Initial step size. If ``None``, the step size is chosen automatically.
            **kwargs: Additional keyword arguments to pass to ``diffrax.diffeqsolve``.

        Returns:
            Solution. If ``t`` is ``None``, the solution can evaluated densely via the
            ``evaluate()`` method. Otherwise, the solution at the time points ``t`` is
            accessible via the ``ys`` attribute.
        """
        stepsize_controller = dx.PIDController(
            pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol, dtmax=0.001
        )
        solver = solvers.CrankNicolson(rtol=rtol, atol=atol)
        # discrete_terminating_event = dx.DiscreteTerminatingEvent(
        #     lambda t, y, args: self.boundary_metric(y) > boundary_threshold
        # )
        return dx.diffeqsolve(
            dx.ODETerm(self),
            solver,
            t[0],
            t[-1],
            # discrete_terminating_event=discrete_terminating_event,
            y0=state,
            dt0=dt0,
            **kwargs,
            saveat=dx.SaveAt(ts=t),
            stepsize_controller=stepsize_controller,
        )

    def plot(
        self, state: PyTree[spatial.ScalarField] | dx.Solution, file: str | None = None
    ) -> None:
        r"""Plot the state of the system. If the state is a Pytree of fields, plot each
        field in a separate subplot. If state is a :py:mod:`dx.Solution` object (i.e.,
        the result of :py:meth:`Model.solve`), plot the solution at each time point.

        Args:
            state: State of the system. Either a :py:class:`eqx.Module` of
                   :py:class:`spatial.ScalarField` fields or a :py:mod:`dx.Solution` object.
            file: If not ``None``, save the figure to this file.
        """
        if isinstance(state, dx.Solution):
            state = state.ys
            time_series = True
        elif isinstance(state, PyTree[spatial.ScalarField]):
            time_series = False
        else:
            raise RuntimeError(
                f"state must be a Pytree of fields or a dx.Solution object, got {state=}"
            )
        labels, fields = zip(
            *[(field.name, getattr(state, field.name)) for field in dc.fields(state)]
        )
        if not all(field.ndim == 1 for field in fields):
            raise NotImplementedError(
                "Can only plot 1D fields, got {field.ndim} dimensions"
            )
        fig, axes = plt.subplots(len(fields), 1, figsize=(6, 2 * len(fields)))
        if time_series:
            for axis, label, field in zip(axes, labels, fields):
                lines = field.values.T
                colors = plt.cm.viridis(np.linspace(0, 1, lines.shape[1]))
                axis.set_prop_cycle("color", colors)
                axis.plot(field.domain[0], lines)
                axis.set_ylabel(label)
        else:
            for axis, label, field in zip(axes, labels, fields):
                axis.plot(field.domain[0], field.values)
                axis.set_ylabel(label)
        if file is not None:
            plt.savefig(file)
        plt.show()


# # diffrax.MultiTerm  <--- combines terms
# # diffrax events <--- maybe good for convergence issues
