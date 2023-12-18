## Solver for Toy Model for Immune Response
Written by Kasturi Shah

First upload: 18 December 2023
Last updated: 18 December 2023

My Python code solves model #4 (equations 10) and model #5 (equations 11) in outline.tex in our shared Overleaf. The boolean variable AB_withR allows a user to toggle between model #4 and #5 by simply changing it to True to run #5 instead of #4. Boundary conditions are Neumann on LHS and RHS. 

I have used quite simple numerical solving techniques: second order finite difference and Runge-Kutta time integration. It probably isn't as fast as jax, though!

I have uploaded .jpg files for test cases run on both MATLAB and Python versions of my code. For these test cases, my parameter value choices, timestep size, dx size, initial conditions, etc are all in the script I have put onto the Github. Just want to highlight that my initial condition is a thin Gaussian. My reasoning is to first test the code on an initial condition that doesn't have very very steep gradients in x, and then try the comparison for the delta function. 

Solutions in the attached images are plotted for t=1,2,3,...,30. The colour scheme is such that increasing red (A), blue (B ) and green (C) indicate solutions at later times. As you can see, there is no 'witch's hat' shape in the solutions (though this might return if I use a delta-function rather than Gaussian initial condition). 
