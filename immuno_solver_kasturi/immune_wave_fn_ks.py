### Written by Kasturi Shah, December 16-18 2023
# numerical solver for immuno wave, A, B and R equation

import numpy as np

# construct d^2/dx^2 matrix
def D2x(Nx, dx):
    S_D2x = np.diag(-2 * np.ones(Nx)) + np.diag(np.ones(Nx - 1), k=1) + np.diag(np.ones(Nx - 1), k=-1)
    S_D2x = S_D2x / dx**2
    S_D2x[0, :3] = [1 / dx**2, -2 / dx**2, 1 / dx**2] # left most point (forwards central difference)
    S_D2x[-1, -3:] = [1 / dx**2, -2 / dx**2, 1 / dx**2] # right most point (backwards central difference)
    
    return S_D2x

def construct_rhs_A(AB_withR,kappa,S_D2x,A_old,theta,eta,B_old,rho,R_old):
    # if not AB_withR: # just A and B equations
    #     rhs_A = kappa * (S_D2x @ A_old) + A_old * (A_old - theta) * (1 - A_old) + eta * B_old 
    # else: # A, B and R equations
    rhs_A = kappa * (S_D2x @ A_old) + A_old * (A_old - theta) * (1 - A_old) + eta * B_old - (rho*R_old)
    rhs_A[0] = rhs_A[1] # Neumann BC on LHS
    rhs_A[-1] = rhs_A[-2] # Neumann BC on RHS

    return rhs_A

def construct_rhs_B(xi,S_D2x,B_old,lmbd,mu,A_old):
    rhs_B = xi * (S_D2x @ B_old) + lmbd * B_old * (1 - B_old) - mu * A_old * B_old # RHS of B equation
    rhs_B[0] = rhs_B[1] # Neumann BC on LHS
    rhs_B[-1] = rhs_B[-2] # Neumann BC on RHS

    return rhs_B

def construct_rhs_R(epsilon,A_old,R_old):
    rhs_R = epsilon*(A_old - R_old) # RHS of R equation
    rhs_R[0] = rhs_R[1] # Neumann BC on LHS
    rhs_R[-1] = rhs_R[-2] # Neumann BC on RHS

    return rhs_R