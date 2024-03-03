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