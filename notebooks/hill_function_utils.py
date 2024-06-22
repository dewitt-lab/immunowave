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
    D: float=1.0
    gamma: float=1.0

    @jax.jit
    def __call__(self, t, state, args=None):
        # unpack field variables
        A, B = state.A, state.B
        # unpack parameters
        KD, n, η, ξ, λ, μ, D, gamma = self.KD, self.n, self.η, self.ξ, self.λ, self.μ, self.D, self.gamma
        # define PDE
        An = A.binop(n, jnp.power)
        tmp = An + KD ** n
        tmp = tmp.binop(-1, jnp.power)
        hill_term = An * tmp
        dAdt = D * A.laplacian(bc="neumann") + hill_term - gamma * A + η * B
        dBdt = ξ * B.laplacian(bc="neumann") + λ * B * (1 - B) - μ * A * B
        return State4(dAdt, dBdt)
    
    
#@jax.jit
def response(B0, KD, t_max, hill_coefficient=2):
    model = HillModel(KD=KD, n=hill_coefficient, η=1.0, ξ=0.0, λ=0.0, μ=0.0)
    state = State4(
        A=iw.ScalarField(shape, lb, h, 0),
        B=iw.ScalarField(
            shape, lb, h, fn=lambda x: B0 * jax.scipy.stats.norm.pdf(x, loc=L / 2, scale=1)
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
    uncertainty = 0.5 * (B0s[this_id+1] - B0s[this_id])
    return B0c, uncertainty


def tissue_response(B0, KD, t_max, shape, lb, h, L, hill_coefficient=2, D=1, gamma=1, **kwargs):
    model = HillModel(KD=KD, n=hill_coefficient, η=1.0, ξ=0.0, λ=0.0, μ=0.0, D=D, gamma=gamma)
    state = State4(
        A=iw.ScalarField(shape, lb, h, 0),
        B=iw.ScalarField(
            shape, lb, h, fn=lambda x: B0 * jax.scipy.stats.norm.pdf(x, loc=L / 2, scale=1)
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
    
    tissue_response = np.sum(solution.ys.A.values[-1] * solution.ys.A.h)
    
    return tissue_response

    
def single_cell_response(B0, KD, t_max, shape, lb, h, L, hill_coefficient=2, gamma=1.0, **kwargs):
    """single cell response"""
    model = HillModel(KD=KD, n=hill_coefficient, η=1.0, ξ=0.0, λ=0.0, μ=0.0, D=1.0, gamma=gamma)
    state = State4(
        A=iw.ScalarField((1,), lb, 1, 0),
        B=iw.ScalarField(
            (1,), lb, 1, fn=lambda x: B0 * jnp.array(x == 0).astype('float')
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
        
    single_cell_response = solution.ys.A.values[-1, int(L // 2)]

    return single_cell_response
