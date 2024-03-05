# Graph Neural Network-Based Preconditioners for Solving Large-Scale Linear Systems via GMRES

Some comments about SciPy's gmres implementation scipy.sparse.linalg.gmres(A, b, x0 = None, restart = None, maxiter = None)

- if restart == None then gmres does not restart.
- if restart is omitted, restart == min(20, n).

- if maxiter != None and restart != None then the total number of iterations is restart*maxiter.
- if maxiter != None and restart == None then the total number of iterations is maxiter
- if maxiter == None --> maxiter == 10*n (At least always plots this) for restart == None and for restart == value

However, the only way to use gmres without restart is to set: 'maxiter = 1' and 'restart = val'.

Stopping criteria:
- Our implementation: relative error |r^k|/|r^0|.
- Scipy: rtol (tol before version 1.12.0): norm(b - A @ x) <= max(rtol*norm(b), atol). Default rtol is 1e-5 and default atol is 0.

TODO: 
    # - (FIXED) Arnoldi algorithm only provides one teration at a time so we do not build every time the whole basis from scratch in the GMRES algorithm
    # - torch?
    # - Instead of A, callable function that calculates Ax for any input vector x? A = lambda x: a.dot(x)
    # - Change back_substitution function to numpy.linalg.lstsq for better numerical stability and correctness?
    # - Should we not count the time it takes to form the inverse of M and calculate it outside the function and add only M as input variable and not precondition in case M = None.
    # - Every time we precondition the residual we solve a system with it. Should we just calculate the inverse of M at the beginning. If we restart many times we will have to calculate many inverses.