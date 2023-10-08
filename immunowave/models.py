import abc
from typing import Any, Optional, Dict
import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax.experimental import sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from jax.config import config

config.update("jax_enable_x64", True)


def discrete_laplacian_1d(n: int, δx: float) -> jnp.ndarray:
    r"""Discrete Laplacian operator in 1D.

    Args:
        n: Number of grid points.
        δx: Grid spacing.
    """
    Δ2 = (
        jnp.diag(-2 * jnp.ones(n, dtype=int))
        + jnp.diag(jnp.ones(n - 1, dtype=int), k=-1)
        + jnp.diag(jnp.ones(n - 1, dtype=int), k=1)
    )
    Δ2 = Δ2.at[0, :4].set([2, -5, 4, -1])
    Δ2 = Δ2.at[-1, -4:].set([-1, 4, -5, 2])

    Δ2 = sparse.BCOO.fromdense(Δ2)

    return Δ2 / δx**2


class ImmuneResponse(eqx.Module, abc.ABC):
    r"""Abstract class for immune response dynamical models."""
    domain_length: float
    n: int
    δx: float
    x: jnp.ndarray

    def __init__(self, domain_length=1.0, n=100) -> None:
        self.domain_length = domain_length
        self.n = n
        self.δx = domain_length / n
        self.x = jnp.linspace(-domain_length / 2, domain_length / 2, n)

    @abc.abstractmethod
    def init_fields(self, Any) -> Dict[str, ArrayLike]:
        r"""Initial state generator."""

    @abc.abstractmethod
    def __call__(
        self, t: float, state: Dict[str, ArrayLike], params: Dict[str, Any]
    ) -> ArrayLike:
        r"""Right-hand side of the dynmamical equation."""


class ImmuneWave(ImmuneResponse):
    r"""Fitz-Hugh Nagumo + Bacteria.

    .. math::
        \begin{align*}
        \partial_t A &=& \nabla_x A + A (A - \theta) (1 - A) + \eta B - \rho R \\
        \partial_t R &=& \epsilon (A - R) \\
        \partial_t B &=& \xi \nabla_x B + \lambda B (1-B) - \mu A B
        \end{align*}
    """
    Δ2: jnp.ndarray

    def __init__(self, domain_length=1.0, n=100) -> None:
        super().__init__(domain_length, n)
        self.Δ2 = discrete_laplacian_1d(self.n, self.δx)

    def init_fields(
        self, B0_height: float = 1.0, B0_width: float = 1.0
    ) -> Dict[str, ArrayLike]:
        r"""Initial state generator.

        .. math::
            A(x, 0) = 0, \quad R(x, 0) = 0, \quad B(x, 0) = B_0 \delta(x)

        Args:
            B0: Initial concentration of bacteria at the center of the domain.
        """
        return dict(
            A=jnp.zeros(self.n),
            R=jnp.zeros(self.n),
            B=jnp.where(jnp.abs(self.x) <= B0_width / 2, B0_height, 0.0),
        )

    @jax.jit
    def __call__(
        self, t: float, state: Dict[str, ArrayLike], params: Dict[str, float]
    ) -> ArrayLike:
        r"""Right-hand side of the dynmamical equation.

        Args:
            t: Time.
            state: Dictionary of state variable :math:`A`, :math:`R`, and :math:`B`.
            params: Additional arguments (ignored).
        """
        A = state["A"]
        R = state["R"]
        B = state["B"]

        θ = params["θ"]
        η = params["η"]
        ρ = params["ρ"]
        ε = params["ε"]
        ξ = params["ξ"]
        λ = params["λ"]
        μ = params["μ"]

        dAdt = self.Δ2 @ A + A * (A - θ) * (1 - A) + η * B - ρ * R
        dRdt = ε * (A - R)
        dBdt = ξ * self.Δ2 @ B + λ * B * (1 - B) - μ * A * B

        return dict(A=dAdt, R=dRdt, B=dBdt)


@jax.jit
def solve(
    model: ImmuneResponse,
    params: Dict[str, float],
    state: ArrayLike,
    t0: float,
    t1: float,
    dt0: Optional[float] = None,
    t: Optional[ArrayLike] = None,
    solver: dx.AbstractSolver = dx.Kvaerno5(),
    stepsize_controller: dx.AbstractStepSizeController = dx.PIDController(
        rtol=1e-6, atol=1e-6
    ),
    **kwargs,
) -> dx.Solution:
    r"""Solve the dynamical system.

    Args:
        model: Immune respone model.
        params: Dictionary of parameters.
        state0: Initial condition.
        t0: Start of integration region.
        t1: End of integration region.
        dt0: Initial step size.
                If ``None``, the step size is chosen automatically.
        t: Time points to evaluate the solution at.
            If ``None``, the solution is evaluated densely.
        solver: A ``diffrax`` solver object.
                The default is the Kvaerno's 5/4 method (stiffly accurate).
        stepsize_controller: A ``diffrax`` step size controller object.
        **kwargs: Additional keyword arguments to pass to ``diffrax.diffeqsolve``.

    Returns:
        Solution. If ``t`` is ``None``, the solution can evaluated densely via the ``evaluate()`` method.
        Otherwise, the solution at the time points ``t`` is accessible via the ``ys`` attribute.
    """
    return dx.diffeqsolve(
        dx.ODETerm(model),
        solver,
        t0,
        t1,
        dt0,
        y0=state,
        args=params,
        **kwargs,
        saveat=dx.SaveAt(dense=True) if t is None else dx.SaveAt(ts=t),
        stepsize_controller=stepsize_controller,
    )


def plot(
    model: ImmuneResponse,
    state: Dict[str, ArrayLike],
    axes,
    *args,
    **kwargs,
) -> None:
    r"""Plot the state of the system."""
    n_components = len(state)
    for i, name in enumerate(state):
        axes[i].plot(model.x, state[name], *args, **kwargs)
        axes[i].set_ylabel(name)
        axes[i].set_xlabel("x")


# diffrax.MultiTerm  <--- combines terms
# diffrax events <--- maybe good for convergence issues
