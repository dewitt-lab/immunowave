r"""Discretization of a spatial domain.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental import checkify
import jax.tree_util as jtu
from jaxtyping import ArrayLike, Array, Float, Int
from typing import Callable, Self, Literal
from collections.abc import Sequence

jax.config.update("jax_enable_x64", True)


laplacian_kernel_1d = jnp.array([1.0, -2.0, 1.0])
"""1D Laplacian kernel."""

laplacian_kernel_2d = jnp.array([[1, 2, 1], [2, -12, 2], [1, 2, 1]]) / 4
"""2D Laplacian kernel (isotropic 9-point stencil of Oono and Puri)."""

laplacian_kernel_3d = (
    jnp.array(
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

_laplacian_kernels = {
    1: laplacian_kernel_1d,
    2: laplacian_kernel_2d,
    3: laplacian_kernel_3d,
}
"""Dict of Laplacian kernels for different dimensions."""


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
        lb: Lower bounds of domain. Must be consistent via broadcasting with the dimension
               implied by ``shape``.
        h: grid spacing.
        values: Values of the discretized function at each grid point. Must be
                broadcastable to ``shape``.
        fn: Function :math:`f:\mathbb{R}^d\to\mathbb{R}` to discretize
            (overrides``values``). Callable must accept :py:attr:`ScalarField.ndim`
            ``float`` arguments and return a scalar ``float``.
    """
    ndim: Int = eqx.field(static=True)
    lb: Float[Array, "ndim"] = eqx.field(static=True)
    ub: Float[Array, "ndim"] = eqx.field(static=True)
    h: Float = eqx.field(static=True)
    laplacian_kernel: Float[Array, "#3 #3 3"] = eqx.field(static=True)
    values: Float[Array, "#l #m n"]

    def __init__(
        self,
        shape: Int | Sequence[Int],
        lb: Float | Sequence[Float],
        h: Float,
        values: ArrayLike = 0.0,
        # NOTE: would be cleaner if the signature was Callable[[Float[Array, " ndim"], ...], float]
        fn: Callable[..., float] | None = None,
    ) -> None:
        # if jnp.atleast_1d(jnp.asarray(shape)).min() < 2:
        #     raise ValueError(
        #         f"Must discretize each dimension into at least two points, got {shape=}"
        #     )
        self.lb = jnp.full_like(
            jnp.atleast_1d(jnp.asarray(shape)), jnp.asarray(lb), dtype=float
        )
        """Lower bounds of domain."""
        self.h = h
        """Grid spacing."""
        self.ndim = len(self.lb)
        """Number of dimensions."""
        self.ub = self.lb + self.h * (jnp.asarray(shape, dtype=float) - 1)
        """Upper bounds of domain."""
        self.values = jnp.full(shape, values, dtype=float)
        """Values of the discretized function at each grid point."""
        if fn is not None:
            self.values = jax.vmap(fn)(*self.domain)
        if self.ndim not in _laplacian_kernels:
            raise NotImplementedError(
                f"Laplacian not implemented for {self.ndim}D domain"
            )
        self.laplacian_kernel = _laplacian_kernels[self.ndim]
        """Laplacian kernel."""

    @checkify.checkify
    def check_aligned(self, other: Self) -> None:
        r"""Check if two fields are spatially aligned.

        Args:
            other: Another field.

        Raises:
            ValueError: If the fields are not aligned.
        """
        checkify.check(
            jnp.array_equal(self.lb, other.lb),
            "Mismatched bounds {lb1} and {lb2}",
            lb1=self.lb,
            lb2=other.lb,
        )
        checkify.check(
            jnp.array_equal(self.ub, other.ub),
            "Mismatched bounds {ub1} and {ub2}",
            lb1=self.ub,
            lb2=other.ub,
        )
        checkify.check(
            self.h == other.h,
            "Mismatched spacings {h1} and {h2}",
            h1=jnp.asarray(self.h),
            h2=jnp.asarray(other.h),
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
            err, _ = self.check_aligned(other)
            err.throw()
            other = other.values
        return ScalarField(
            self.values.shape, self.lb, self.h, values=f(self.values, other)
        )

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

    def abs(self) -> Self:
        r"""Pointwise absolute value."""
        return ScalarField(
            self.values.shape, self.lb, self.h, values=jnp.abs(self.values)
        )

    def integral(self) -> float:
        result = self.values
        for _ in range(self.ndim):
            result = jnp.trapz(result, dx=self.h)
        return result

    @property
    def domain(self) -> tuple[Array, ...]:
        r"""Spatial domain :math:`\mathcal{D}\subset\mathbb{R}^d` in :py:func`jnp.meshgrid` format."""
        return jnp.meshgrid(
            *[
                jnp.linspace(self.lb[i], self.ub[i], self.values.shape[i])
                for i in range(self.ndim)
            ],
            indexing="ij",
        )

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
            f_extended, self.laplacian_kernel, mode="valid", method="auto"
        )
        laplacian /= self.h**2

        # Return the interior points to exclude the boundary condition padding
        return ScalarField(laplacian.shape, self.lb, self.h, values=laplacian)


class VectorField(eqx.Module):
    r"""A vector field :math:`\mathbf{f}:\mathbb{R}^d\to\mathbb{R}^c` on a
    :math:`d`-dimensional spatial domain :math:`\mathcal{D}\subset\mathbb{R}^d`
    discretized into a regular grid with spacing :math:`h`.

    Args:
        components: Sequence of scalar fields :math:`f_i:\mathbb{R}^d\to\mathbb{R}`.
    """
    components: Sequence[ScalarField]

    def __post_init__(self):
        if not len(self.components):
            raise ValueError("Vector field must have at least one component")
        for component in self.components[1:]:
            err, _ = self.components[0].check_aligned(component)
            err.throw()
        self.ndim = self.components[0].ndim
