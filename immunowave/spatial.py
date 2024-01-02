r"""Discretization of a spatial domain.
"""

import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import ArrayLike, Array, Float
from typing import Callable, Self, Literal
from collections.abc import Sequence

jax.config.update("jax_enable_x64", True)


laplacian_kernel_1d = np.array([1.0, -2.0, 1.0])
"""1D Laplacian kernel."""

laplacian_kernel_2d = np.array([[1, 2, 1], [2, -12, 2], [1, 2, 1]]) / 4
"""2D Laplacian kernel (isotropic 9-point stencil of Oono and Puri)."""

laplacian_kernel_3d = (
    np.array(
        [
            [
                [2, 3, 2],
                [3, 6, 3],
                [2, 3, 2],
            ],
            [
                [3, 6, 3],
                [6, -88, 6],
                [3, 6, 3],
            ],
            [
                [2, 3, 2],
                [3, 6, 3],
                [2, 3, 2],
            ],
        ]
    )
    / 26
)
"""3D Laplacian kernel (isotropic 27-point stencil of O'Reilly and Beck)."""

_laplacian_kernels = [
    laplacian_kernel_1d,
    laplacian_kernel_2d,
    laplacian_kernel_3d,
]
"""List of Laplacian kernels for 1, 2, and 3 dimensions."""

NDFn = (
    Callable[[float], float]
    | Callable[[float, float], float]
    | Callable[[float, float, float], float]
)
"""Scalar function of 1, 2, or 3 variables."""


def _is_scalar_field(node):
    return isinstance(node, ScalarField)


def _field_map(fn, field, *rest):
    return jtu.tree_map(fn, field, *rest, is_leaf=_is_scalar_field)


def _field_structure(field):
    return jtu.tree_structure(field, is_leaf=_is_scalar_field)


def _field_leaves(field):
    return jtu.tree_leaves(field, is_leaf=_is_scalar_field)


class ScalarField(eqx.Module):
    r"""A scalar field :math:`f:\mathbb{R}^d\to\mathbb{R}` on a :math:`d`-dimensional
    spatial domain :math:`\mathcal{D}\subset\mathbb{R}^d` discretized into a regular
    grid with spacing :math:`h`.

    Args:
        shape: Grid shape.
        lb: Lower bounds of domain. Must be consistent via broadcasting with the
            dimension implied by ``shape``.
        h: grid spacing.
        values: Values of the discretized function at each grid point. Must be
                broadcastable to ``shape``.
        fn: Function :math:`f:\mathbb{R}^d\to\mathbb{R}` to discretize (overrides
            ``values``). Callable must accept :py:attr:`ScalarField.ndim`
            ``float`` arguments and return a scalar ``float``.
    """
    ndim: int = eqx.field(static=True)
    lb: Float[np.ndarray, "ndim"] = eqx.field(static=True)
    ub: Float[np.ndarray, "ndim"] = eqx.field(static=True)
    h: float = eqx.field(static=True)
    values: Float[Array, "#l #m n"]

    def __init__(
        self,
        shape: Sequence[int],
        lb: Sequence[float],
        h: float,
        values: ArrayLike = 0.0,
        # NOTE: would be cleaner if the signature was
        #       Callable[[Float[Array, " ndim"], ...], float]
        fn: NDFn | None = None,
    ) -> None:
        self.lb = np.array(lb, dtype=float)
        """Lower bounds of domain."""
        self.ndim = len(self.lb)
        """Number of dimensions."""
        if self.ndim not in (1, 2, 3):
            raise NotImplementedError(
                f"ScalarField only supports 1D, 2D, and 3D fields, got {self.ndim=}"
            )
        self.h = h
        """Grid spacing."""
        self.ub = self.lb + self.h * (np.array(shape, dtype=float) - 1)
        """Upper bounds of domain."""
        self.values = jnp.full(shape, values, dtype=float)
        """Values of the discretized function at each grid point."""
        if fn is not None:
            fn_vmap = jax.vmap(fn, in_axes=tuple(range(self.ndim)))
            domain_meshgrid = jnp.meshgrid(*self.domain, indexing="ij")
            self.values = fn_vmap(*domain_meshgrid)

    @property
    def domain(self) -> tuple[Float[Array, "..."], ...]:
        r"""Discretized spatial domain :math:`\mathcal{D}\subset\mathbb{R}^d`."""
        return tuple(
            jnp.linspace(self.lb[i], self.ub[i], self.values.shape[-self.ndim + i])
            for i in range(self.ndim)
        )

    def check_aligned(self, other: Self) -> None:
        r"""Check if another field is spatially aligned with this one (i.e., it has the
        same domain parameters).

        Args:
            other: Other field.

        Raises:
            eqx.EquinoxTracetimeError: If ``other`` is not aligned.
        """
        eqx.error_if(
            other,
            jnp.logical_not(jnp.array_equal(self.lb, other.lb)),
            f"Mismatched bounds {self.lb} and {other.lb}",
        )
        eqx.error_if(
            other,
            jnp.logical_not(jnp.array_equal(self.ub, other.ub)),
            f"Mismatched bounds {self.ub} and {other.ub}",
        )
        eqx.error_if(
            other,
            jnp.logical_not(jnp.array_equal(self.h, other.h)),
            f"Mismatched spacings {self.h} and {self.h}",
        )

    def __pos__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        return ScalarField(
            self.values.shape[-self.ndim :], self.lb, self.h, values=-self.values
        )

    def __abs__(self) -> Self:
        return ScalarField(
            self.values.shape[-self.ndim :],
            self.lb,
            self.h,
            values=jnp.abs(self.values),
        )

    def binop(
        self,
        other: Self | Float[Array, "#l #m n"],
        f: Callable[
            [Float[Array, "#l #m n"], Float[Array, "#l #m n"]], Float[Array, "#l #m n"]
        ],
    ) -> Self:
        r"""Pointwise binary operation with another discretized function.

        Args:
            other: Another spatially discretized function.
            f: Pointwise binary operation.

        Returns:
            Discretized function.
        """
        if isinstance(other, ScalarField):
            self.check_aligned(other)
            other = other.values
        return ScalarField(
            self.values.shape[-self.ndim :],
            self.lb,
            self.h,
            values=f(self.values, other),
        )

    def __add__(self, other):
        return self.binop(other, jnp.add)

    def __mul__(self, other):
        return self.binop(other, jnp.multiply)

    def __radd__(self, other):
        return self.binop(other, jnp.add)

    def __rmul__(self, other):
        return self.binop(other, jnp.multiply)

    def __sub__(self, other):
        return self.binop(other, jnp.subtract)

    def __rsub__(self, other):
        return self.binop(other, lambda x, y: jnp.subtract(y, x))

    def integral(self) -> ArrayLike:
        result = self.values
        for _ in range(self.ndim):
            result = jnp.trapz(result, dx=self.h)
        return result

    def laplacian(self, bc: Literal["dirichlet", "neumann"] = "dirichlet") -> Self:
        r"""Laplacian :math:`\nabla^2 f`.

        Args:
            bc: Zero boundary condition (either ``"dirichlet"`` or ``"neumann"``).

        Returns:
            Discretized Laplacian field :math:`\nabla^2 f`.
        """
        # Handle boundary conditions
        if bc == "dirichlet":
            # Extend domain with zeros for Dirichlet boundary conditions
            f_extended = jnp.pad(self.values, (1, 1), mode="constant")
        elif bc == "neumann":
            # Extend domain with edge values for Neumann boundary conditions
            f_extended = jnp.pad(self.values, (1, 1), mode="edge")
        else:
            raise ValueError(f"Unknown boundary condition: {bc}")

        # Compute the Laplacian via convolution
        laplacian = jax.scipy.signal.convolve(
            f_extended, _laplacian_kernels[self.ndim - 1], mode="valid", method="auto"
        )
        laplacian /= self.h**2

        return ScalarField(
            self.values.shape[-self.ndim :], self.lb, self.h, values=laplacian
        )
