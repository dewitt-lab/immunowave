import immunowave as iw
import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dx

from multiprocessing import Pool
from functools import partial


class State4(iw.State):
    A: iw.ScalarField
    B: iw.ScalarField


class HillModel(iw.Model):
    KD: float
    n: int
    η: float
    ξ: float
    λ: float
    μ: float
    D: float = 1.0
    gamma: float = 1.0

    @jax.jit
    def __call__(self, t, state, args=None):
        # unpack field variables
        A, B = state.A, state.B
        # unpack parameters
        KD, n, η, ξ, λ, μ, D, gamma = (
            self.KD,
            self.n,
            self.η,
            self.ξ,
            self.λ,
            self.μ,
            self.D,
            self.gamma,
        )
        # define PDE
        An = A.binop(n, jnp.power)
        tmp = An + KD**n
        tmp = tmp.binop(-1, jnp.power)
        hill_term = An * tmp
        dAdt = D * A.laplacian(bc="neumann") + hill_term - gamma * A + η * B
        dBdt = ξ * B.laplacian(bc="neumann") + λ * B * (1 - B) - μ * A * B
        return State4(dAdt, dBdt)


# @jax.jit
def response(B0, KD, t_max, hill_coefficient=2):
    model = HillModel(KD=KD, n=hill_coefficient, η=1.0, ξ=0.0, λ=0.0, μ=0.0)
    state = State4(
        A=iw.ScalarField(shape, lb, h, 0),
        B=iw.ScalarField(
            shape,
            lb,
            h,
            fn=lambda x: B0 * jax.scipy.stats.norm.pdf(x, loc=L / 2, scale=1),
        ),
    )
    solution = iw.solve(
        model,
        state,
        t0=0,
        t1=t_max,
        t=jnp.array([t_max]),
        **kwargs,
    )
    return np.sum(solution.ys.A.values[-1] * solution.ys.A.h)
    # return solution.evaluate(t_final).A.integral() / L


def find_B0(final_mean_A, B0s):
    this_id = np.where(np.diff(final_mean_A) == np.max(np.diff(final_mean_A)))[0][0]
    B0c = 0.5 * (B0s[this_id] + B0s[this_id + 1])
    uncertainty = 0.5 * (B0s[this_id + 1] - B0s[this_id])
    return B0c, uncertainty


def tissue_response(
    B0,
    KD,
    t_max,
    shape,
    lb,
    h,
    L,
    hill_coefficient=2,
    D=1,
    gamma=1,
    scale=0.5,
    **kwargs,
):
    model = HillModel(
        KD=KD, n=hill_coefficient, η=1.0, ξ=0.0, λ=0.0, μ=0.0, D=D, gamma=gamma
    )
    state = State4(
        A=iw.ScalarField(shape, lb, h, 0),
        B=iw.ScalarField(
            shape,
            lb,
            h,
            fn=lambda x: B0 * jax.scipy.stats.norm.pdf(x, loc=L / 2, scale=scale),
        ),
        # B=iw.ScalarField(
        #    shape, lb, h, fn=lambda x: B0 * jax.scipy.stats.norm.pdf(x, loc=0, scale=0.5)
        # ),
    )
    solution = iw.solve(
        model,
        state,
        t0=0,
        t1=t_max,
        **kwargs,
    )

    tissue_response = np.sum(solution.ys.A.values[-1] * solution.ys.A.h)

    return tissue_response


def single_cell_response(
    B0, KD, t_max, shape, lb, h, L, hill_coefficient=2, gamma=1.0, **kwargs
):
    """single cell response"""
    model = HillModel(
        KD=KD, n=hill_coefficient, η=1.0, ξ=0.0, λ=0.0, μ=0.0, D=0.0, gamma=gamma
    )
    state = State4(
        A=iw.ScalarField((1,), lb, 1, 0),
        B=iw.ScalarField(
            (1,), lb, 1, fn=lambda x: B0 * jnp.array(x == 0).astype("float")
        ),
    )
    solution = iw.solve(
        model,
        state,
        t0=0,
        t1=t_max,
        **kwargs,
    )

    single_cell_response = solution.ys.A.values[-1, int(L // 2)]

    return single_cell_response


def compute_wave_threshold(
    B0_grid,
    KD,
    t_max,
    shape,
    lb,
    h,
    L,
    hill_coefficient=2,
    D=1,
    gamma=1,
    n_iters=5,
    scale=0.5,
    **kwargs,
):
    """do a two-step grid search. First on user-input B0_grid (typically logarithmically spaced).
    Then, find the grid points that bound the threshold and do a second search on a linear grid
    using the same number of data points as in the original B0_grid."""
    with Pool(processes=10) as pool:
        func = partial(
            tissue_response,
            KD=KD,
            t_max=t_max,
            hill_coefficient=hill_coefficient,
            D=D,
            gamma=gamma,
            shape=shape,
            lb=lb,
            h=h,
            L=L,
            scale=scale,
            **kwargs,
        )
        res = pool.map(func, B0_grid)
    final_integrated_A = np.array(res)
    this_id = np.where(
        np.diff(final_integrated_A) == np.max(np.diff(final_integrated_A))
    )[0][0]
    if this_id < len(B0_grid) - 1:
        linear_B0_grid = np.linspace(
            B0_grid[this_id], B0_grid[this_id + 1], len(B0_grid)
        )

    for i in range(n_iters):
        with Pool(processes=10) as pool:
            func = partial(
                tissue_response,
                KD=KD,
                t_max=t_max,
                hill_coefficient=hill_coefficient,
                gamma=gamma,
                shape=shape,
                lb=lb,
                h=h,
                L=L,
                scale=scale,
                **kwargs,
            )
            res = pool.map(func, linear_B0_grid)
        final_integrated_A = np.array(res)
        this_id = np.where(
            np.diff(final_integrated_A) == np.max(np.diff(final_integrated_A))
        )[0][0]
        if this_id < len(B0_grid) - 1:
            linear_B0_grid = np.linspace(
                linear_B0_grid[this_id],
                linear_B0_grid[this_id + 1],
                len(linear_B0_grid),
            )
        else:
            # initial grid does not contain true threshold, abort
            raise ValueError("initial B0_grid lower bound too high")
    wave_threshold, wave_threshold_uncertainty = find_B0(
        final_integrated_A, linear_B0_grid
    )

    return wave_threshold, wave_threshold_uncertainty


def compute_cell_threshold(
    B0_grid,
    KD,
    t_max,
    shape,
    lb,
    h,
    L,
    hill_coefficient=2,
    D=1,
    gamma=1,
    n_iters=10,
    **kwargs,
):
    """do a two-step grid search. First on user-input B0_grid (typically logarithmically spaced).
    Then, find the grid points that bound the threshold and do a second search on a linear grid
    using the same number of data points as in the original B0_grid."""
    with Pool(processes=10) as pool:
        func = partial(
            single_cell_response,
            KD=KD,
            t_max=t_max,
            hill_coefficient=hill_coefficient,
            gamma=gamma,
            shape=shape,
            lb=lb,
            h=h,
            L=L,
            **kwargs,
        )
        res = pool.map(func, B0_grid)
    final_cell_A = np.array(res)
    this_id = np.where(np.diff(final_cell_A) == np.max(np.diff(final_cell_A)))[0][0]
    if this_id < len(B0_grid) - 1:
        linear_B0_grid = np.linspace(
            B0_grid[this_id], B0_grid[this_id + 1], len(B0_grid)
        )
    else:
        # initial grid does not contain true threshold, abort
        raise ValueError("initial B0_grid lower bound too high")

    for i in range(n_iters):
        with Pool(processes=10) as pool:
            func = partial(
                single_cell_response,
                KD=KD,
                t_max=t_max,
                hill_coefficient=hill_coefficient,
                gamma=gamma,
                shape=shape,
                lb=lb,
                h=h,
                L=L,
                **kwargs,
            )
            res = pool.map(func, linear_B0_grid)
        final_cell_A = np.array(res)
        this_id = np.where(np.diff(final_cell_A) == np.max(np.diff(final_cell_A)))[0][0]
        if this_id < len(B0_grid) - 1:
            linear_B0_grid = np.linspace(
                linear_B0_grid[this_id],
                linear_B0_grid[this_id + 1],
                len(linear_B0_grid),
            )
        else:
            # initial grid does not contain true threshold, abort
            raise ValueError("initial B0_grid lower bound too high")
    cell_threshold, cell_threshold_uncertainty = find_B0(final_cell_A, linear_B0_grid)

    return cell_threshold, cell_threshold_uncertainty
