r"""Discretization of a spatial domain.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from typing import Callable, Self, Literal

jax.config.update("jax_enable_x64", True)


class GridFn(eqx.Module):
    r"""Represent a scalar function :math:`\mathbb{R}^d\to\mathbb{R}` on a
    :math:`d`-dimensional spatial domain :math:`\mathcal{D}\subset\mathbb{R}^d`
    discretized into a regular grid.

    Args:
        lb: Lower bounds of domain.
        ub: Upper bounds of domain.
        values: Values of the discretized function at each grid point.
    """
    ndim: int = eqx.field(static=True)
    lb: Float[Array, "ndim"]
    ub: Float[Array, "ndim"]
    shape: Int[Array, "ndim"]
    spacing: Float[Array, "ndim"]
    values: Float[Array, " n"] | Float[Array, "m n"] | Float[Array, "l m n"]

    def __init__(
        self,
        lb: Float[Array, " d"],
        ub: Float[Array, " d"],
        values: Float[Array, " n"] | Float[Array, "m n"] | Float[Array, "l m n"],
    ) -> None:
        lb = jnp.atleast_1d(lb)
        ub = jnp.atleast_1d(ub)
        if lb.shape != ub.shape:
            raise ValueError(f"Mismatched bounds dimensions: {len(lb)} and {len(ub)}")
        if values.ndim != len(lb):
            raise ValueError(
                "Mismatched bounds and values dimensions: "
                f"{len(lb)} and {len(values.shape)}"
            )
        self.lb = lb
        """Lower bounds of domain."""
        self.ub = ub
        """Upper bounds of domain."""
        self.values = values
        """Values of the discretized function at each grid point."""
        self.ndim = values.ndim
        """Number of dimensions."""
        self.shape = jnp.array(values.shape, dtype=int)
        """Grid shape."""
        self.spacing = (ub - lb) / (self.shape - 1)
        """Grid spacing."""

    @classmethod
    def discretize_fn(
        cls,
        f: Callable[[Float[Array, " d"]], Float],
        lb: Float[Array, " d"],
        ub: Float[Array, " d"],
        shape: Int[Array, " d"],
    ) -> Self:
        r"""Discretize function :math:`f:\mathbb{R}^d\to\mathbb{R}`.

        Args:
            f: Function to discretize.
            lb: Lower bounds of domain.
            ub: Upper bounds of domain.
            shape: Grid dimensions.

        Returns:
            Discretized function.
        """
        if any(shape < 2):
            raise ValueError(
                f"Must discretize each dimension into at least two points, got {shape=}"
            )
        values = jax.vmap(f)(
            *jnp.meshgrid(
                jnp.linspace(lbi, ubi, dim)
                for lbi, ubi, dim in zip(lb, ub, shape, strict=True)
            )
        )
        return cls(lb, ub, values)

    def binop(
        self,
        other: Self | Float[Array, " n"],
        f: Callable[[Float[Array, " n"], Float[Array, " n"]], Float[Array, " n"]],
    ):
        r"""Pointwise binary operation with another discretized function.

        Args:
            other: Another spatially discretized function.
            f: Pointwise binary operation.

        Returns:
            Discretized function.
        """
        if isinstance(other, GridFn):
            if not (
                jnp.array_equal(self.lb, other.lb)
                and jnp.array_equal(self.ub, other.ub)
                and jnp.array_equal(self.shape, other.shape)
                and jnp.array_equal(self.spacing, other.spacing)
            ):
                raise ValueError("Mismatched spatial discretizations")
            other = other.values
        return GridFn(self.lb, self.ub, f(self.values, other))

    def __add__(self, other):
        return self.binop(other, lambda x, y: x + y)

    def __mul__(self, other):
        return self.binop(other, lambda x, y: x * y)

    def __radd__(self, other):
        return self.binop(other, lambda x, y: y + x)

    def __rmul__(self, other):
        return self.binop(other, lambda x, y: y * x)

    def __sub__(self, other):
        return self.binop(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self.binop(other, lambda x, y: y - x)

    def integral(self):
        return jnp.sum(self.values * self.spacing.prod())

    def laplacian(self, bc: Literal["dirichlet", "neumann"] = "dirichlet") -> Self:
        r"""Laplacian :math:`\nabla^2 f`.

        Args:
            bc: Zero boundary condition (either ``"dirichlet"`` or ``"neumann"``).

        Returns:
            Discretized Laplacian field :math:`\nabla^2 f`.
        """
        if self.ndim != 1:
            raise NotImplementedError(
                f"Laplacian not implemented for {self.ndim}D domain"
            )
        # Handle boundary conditions
        if bc == "dirichlet":
            # Extend domain with zeros for Dirichlet boundary conditions
            f_extended = jnp.pad(self.values, (1, 1), mode="constant")
        elif bc == "neumann":
            # Extend domain with edge values for Neumann boundary conditions
            f_extended = jnp.pad(self.values, (1, 1), mode="edge")
        else:
            raise ValueError(f"Unknown boundary condition: {bc}")

        # Compute the Laplacian
        laplacian = (
            jnp.roll(f_extended, -1) - 2 * f_extended + jnp.roll(f_extended, 1)
        ) / self.spacing[0] ** 2

        # Return the interior points to exclude the boundary condition padding
        return GridFn(self.lb, self.ub, laplacian[1:-1])
