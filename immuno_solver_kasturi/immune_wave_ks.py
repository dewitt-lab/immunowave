### Written by Kasturi Shah, December 16-18 2023
# numerical solver for immuno wave, A, B and R equation

# can toggle variables on and off to see which equations to solve:
#       A_only = True --> only solve A equation with B impulse at t=0
#       A_only = False --> only solve A equation with B impulse at t=0
#       AB_withR = False --> just A and B equations
#       AB_withR = True --> A, B and R equations

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from immune_wave_fn_ks import D2x,construct_rhs_A,construct_rhs_B,construct_rhs_R

np.random.seed(42)

# Parameters
theta = 0.1             # unstable fixed point
kappa = 1               # A diffusion coefficient
xi = 0.1                # B diffusion coefficient
lmbd = 0.9              # strength of influence of B on itself
mu = 0.1                # strength of reduction of B by A
eta = 10                # strength of increase of A by B
epsilon = 0.1           # for R RHS
rho = 0.5               # influence of R on A

# construct x
Nx = 1000
x = np.linspace(0, 20, Nx)
dx = x[1] - x[0]

# construct t
deltat = 1e-4
tictoc = np.arange(0, 30 + deltat, deltat)
Nt = len(tictoc)

# construct A array + initialise
A = np.full((Nx, Nt), np.nan)
A[:, 0] = np.zeros(Nx)

# construct B array + initialise
B = np.full((Nx, Nt), np.nan)
x0 = x[-1] / 2
xs = 0.1
As = 1
B[:, 0] = As * np.exp(-(x - x0) ** 2 / (2 * xs ** 2))

# only solve A equation with B impulse at t=0? (False = no (i.e., both A and B vary), True = yes)
A_only = False

# with R? (False = no (i.e., just A and B equations), True = yes (i.e., A, B and R equations))
AB_withR = False

if AB_withR: # if with R, then construct R + initialise
    R = np.full((Nx, Nt), np.nan)
    R[:, 0] = np.zeros(Nx) # initialise R with zeros

# # construct d^2/dx^2 matrix
S_D2x = D2x(Nx, dx)

for dt in range(1, Nt): # looping through time
    if A_only: # if B not varying in time and B0 just provides an impulse at first time step
        if dt == 1:
            B0 = B[:, 0]
        else:
            B0 = np.zeros(Nx)
        B_old = B0
    else: # if A and B both varying in time
        B_old = B[:, dt - 1]
    
    # save previous slice of R
    if AB_withR: # if A, B and R
        R_old = R[:, dt - 1]

    A_old = A[:, dt - 1] # saving as separate variable for clarity
    # RHS of A equation
    if not AB_withR: # just A and B equations
        rhs_A = construct_rhs_A(AB_withR,kappa,S_D2x,A_old,theta,eta,B_old,rho,np.zeros(Nx))
    else: # A, B and R equations
        rhs_A = construct_rhs_A(AB_withR,kappa,S_D2x,A_old,theta,eta,B_old,rho,R_old)
    
    # integrate A in time
    A[:, dt] = odeint(lambda A, t: rhs_A, A_old, [tictoc[dt - 1], tictoc[dt]])[-1]

    # RHS of B equation (if A_only == False)
    if not A_only: # i.e., if A and B are both varying in time
        rhs_B = construct_rhs_B(xi,S_D2x,B_old,lmbd,mu,A_old) # RHS of B equation
    
        # integrate B in time
        B[:, dt] = odeint(lambda B, t: rhs_B, B_old, [tictoc[dt - 1], tictoc[dt]])[-1] 
    
    if AB_withR:
        rhs_R = construct_rhs_R(epsilon,A_old,R_old)

        # integrate R in time
        R[:, dt] = odeint(lambda R, t: rhs_R, R_old, [tictoc[dt - 1], tictoc[dt]])[-1] 

# plotting every nth slice
plot_slice = np.arange(0, Nt, 10000) 
plt.rcParams['text.usetex'] = True # latex interpreter for figure
if not AB_withR: # just A and B
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.title('$A$', fontsize=14)
    plt.subplot(2, 1, 2)
    plt.title('$B$', fontsize=14)
    plt.xlabel('$x$', fontsize=14)

    for dp in range(len(plot_slice)):
        plt.subplot(2, 1, 1)
        plt.plot(x, A[:, plot_slice[dp]], linewidth=2, color=plt.cm.Reds((dp+2) / len(plot_slice)))

        if not A_only:
            plt.subplot(2, 1, 2)
            plt.plot(x, B[:, plot_slice[dp]], linewidth=2, color=plt.cm.Blues((dp+2) / len(plot_slice)))
    
    plt.savefig('/Users/kasturishah/Downloads/AB_KS_Python_v2.jpg', format='jpg', dpi=500, bbox_inches='tight')
else:
    plt.figure(figsize=(12, 9))
    plt.subplot(3, 1, 1)
    plt.title('$A$', fontsize=10)
    plt.subplot(3, 1, 2)
    plt.title('$B$', fontsize=10)
    plt.subplot(3, 1, 3)
    plt.title('$R$', fontsize=10)
    plt.xlabel('$x$', fontsize=10)

    for dp in range(len(plot_slice)):
        plt.subplot(3, 1, 1)
        plt.plot(x, A[:, plot_slice[dp]], linewidth=2, color=plt.cm.Reds((dp+2) / len(plot_slice)))

        if not A_only:
            plt.subplot(3, 1, 2)
            plt.plot(x, B[:, plot_slice[dp]], linewidth=2, color=plt.cm.Blues((dp+2) / len(plot_slice)))
        
        plt.subplot(3, 1, 3)
        plt.plot(x, R[:, plot_slice[dp]], linewidth=2, color=plt.cm.Greens((dp+2) / len(plot_slice)))

    plt.savefig('/Users/kasturishah/Downloads/ABR_KS_Python_v2.jpg', format='jpg', dpi=500, bbox_inches='tight')

plt.show()