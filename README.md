# computational_econ_coursework
Numerical Analyses: Root Finding, Eigen Decomps, SVD, Anderson-Moore

## PS1
Fixed-Point, Bisection, and Newton algorithms to find roots ("zeros") of three functions with specified initial values.

## PS2
Eigenvalues, Eigenvectors, and Power Iteration for Dominant Eigenvalue-Eigenvector Pair

## PS3
Eigendecomposition and Simultaneous Iteration

## PS4
Singular Value Decomposition and Image Mappings

## PS5
Anderson-Moore algorithm for solving a forward-looking macro model. 

Simulates the effects of a one percentage point shock to the long-term interest rate, setting it above its steady state level in period $t-1$. The nonlinear model is based on a simple, four-equation New Keynesian model, drawn in part from Fuhrer-Madigan (1997). The model includes a nonlinear equation for the term structure of interest rates. The term structure equation is the equation for the real long-term interest rate, the real rate on a consol. Linearization is required for this nonlinearity.

### Variables: 
$$i$$ the short-term (one-period) nominal interest rate
$$\pi$$ the one-period inflation rate
$$\rho$$ the long-term real interest rate (the real rate on a consol)
$$y$$ the GDP gap (zero in the steady state)

### Model Equations

$$i = 0.021 + \pi + 0.5(\pi_{t-1} - 0.02) + 0.5 y_{t-1}$$

$$\pi = 0.5(\pi_{t-1} + \pi_{t+1}) + 0.1 y$$

$$\rho - \frac{1}{\rho_{t+1}}(\rho_{t+1} - \rho) = i - \pi_{t+1}$$

$$y = 0.016758 + 1.254 y_{t-1} - 0.415 y_{t-2} - 0.798 \rho_{t-1}$$

Note: maim.py includes the core Anderson-Moore algorithm, created by Professor Brian Madigan.
