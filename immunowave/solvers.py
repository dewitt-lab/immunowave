r"""PDE solvers."""

import equinox as eqx
import diffrax as dx
import jax
import jax.numpy as jnp


class CrankNicolson(dx.AbstractSolver):
    r"""Crank-Nicolson solver for spatial PDEs, adapted from `Diffrax docs
    <https://docs.kidger.site/diffrax/examples/nonlinear_heat_pde/>`_.

    Args:
        rtol: Relative tolerance.
        atol: Absolute tolerance.
    """

    rtol: float
    atol: float

    term_structure = dx.ODETerm
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
            new_y1 = y0 + 0.5 * δt * (f0 + terms.vf(t1, y1, args))
            diff = jnp.abs((new_y1 - y1).values)
            max_y1 = jnp.maximum(jnp.abs(y1.values), jnp.abs(new_y1.values))
            scale = self.atol + self.rtol * max_y1
            not_converged = jnp.any(diff > scale)
            return new_y1, not_converged

        euler_y1 = y0 + δt * f0
        y1, _ = jax.lax.while_loop(
            keep_iterating, fixed_point_iteration, (euler_y1, False)
        )

        y_error = y1 - euler_y1
        dense_info = dict(y0=y0, y1=y1)

        solver_state = None
        result = dx.RESULTS.successful
        return y1, y_error, dense_info, solver_state, result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
