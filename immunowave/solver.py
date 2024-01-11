r"""Spatial PDE solvers."""

import diffrax as dx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from immunowave import spatial, state
from jaxtyping import PyTree, Scalar
from typing import Callable

jax.config.update("jax_enable_x64", True)


class SpatialPDETerm(dx.AbstractTerm):
    r"""A term representing :math:`f(t, y(x, t), args) \mathrm{d}t`. That is to say,
    the term appearing on the right hand side of a PDE, in which the control is time.

    ``vector_field`` should return some PyTree, with the same structure as the initial
    state ``y0``, and with every leaf broadcastable to the equivalent leaf in ``y0``.
    """
    vector_field: Callable[[Scalar, state.State, PyTree], state.State]

    def vf(self, t: Scalar, y: state.State, args: PyTree) -> state.State:
        out = self.vector_field(t, y, args)
        if spatial._field_structure(out) != spatial._field_structure(y):
            raise ValueError(
                f"Vector field output structure {spatial._field_structure(out)} "
                f"does not match input structure {spatial._field_structure(y)}"
            )
        return out
        # return jtu.tree_map(lambda o, yi: jnp.broadcast_to(o, jnp.shape(yi)), out, y)

    @staticmethod
    def contr(t0: Scalar, t1: Scalar) -> Scalar:
        return t1 - t0

    @staticmethod
    def prod(vf: state.State, control: Scalar) -> PyTree:
        return jtu.tree_map(lambda v: control * v, vf)


class CrankNicolson(dx.AbstractSolver):
    r"""Crank-Nicolson solver for spatial PDEs, adapted from `Diffrax docs
    <https://docs.kidger.site/diffrax/examples/nonlinear_heat_pde/>`_.

    Args:
        rtol: Relative tolerance.
        atol: Absolute tolerance.
    """

    rtol: float
    atol: float

    term_structure = SpatialPDETerm
    interpolation_cls = dx.LocalLinearInterpolation

    def order(self, terms):
        return 2

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        δt = t1 - t0
        f0 = terms.vf(t0, y0, args)

        def keep_iterating(val):
            _, not_converged = val
            return not_converged

        def fixed_point_iteration(val):
            y1, _ = val
            new_y1 = spatial._field_map(
                lambda y0, f0, flow1: y0 + 0.5 * δt * (f0 + flow1),
                y0,
                f0,
                terms.vf(t1, y1, args),
            )
            # diff = spatial._field_map(lambda new_y1, y1: abs(new_y1 - y1), new_y1, y1)
            diff = jnp.stack(
                [
                    jnp.abs((new_y1_field - y1_field).values)
                    for new_y1_field, y1_field in zip(
                        spatial._field_leaves(new_y1), spatial._field_leaves(y1)
                    )
                ]
            )
            # max_y1 = spatial._field_map(
            #     lambda y1, new_y1: abs(y1).binop(abs(new_y1), jnp.maximum),
            #     y1,
            #     new_y1,
            # )
            max_y1 = jnp.maximum(
                jnp.stack(
                    [jnp.abs(y1_field.values) for y1_field in spatial._field_leaves(y1)]
                ),
                jnp.stack(
                    [
                        jnp.abs(new_y1_field.values)
                        for new_y1_field in spatial._field_leaves(new_y1)
                    ]
                ),
            )
            # scale = spatial._field_map(
            #     lambda max_y1: self.atol + self.rtol * max_y1, max_y1
            # )
            scale = self.atol + self.rtol * max_y1
            # not_converged = jnp.any(
            #     jnp.asarray(
            #         jtu.tree_leaves(
            #             spatial._field_map(
            #                 lambda diff, scale: jnp.all(diff.values > scale.values),
            #                 diff,
            #                 scale,
            #             )
            #         )
            #     )
            # )
            not_converged = jnp.any(diff > scale)
            return new_y1, not_converged

        euler_y1 = spatial._field_map(lambda y0, f0: y0 + δt * f0, y0, f0)
        y1, _ = jax.lax.while_loop(
            keep_iterating, fixed_point_iteration, (euler_y1, False)
        )

        y_error = spatial._field_map(lambda x, y: x - y, y1, euler_y1)
        dense_info = dict(y0=y0, y1=y1)

        solver_state = None
        result = dx.RESULTS.successful
        return y1, y_error, dense_info, solver_state, result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
