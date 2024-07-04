import numpy as np
from numpy.linalg import qr
from krylov.utils import arnoldi_one_iter, back_substitution

def precon_GMRES_restarted(prec, A, b, true_sol, x0=None, k_max=None, restart=None, epsilon=1e-6):
    """
    Implements the Preconditioned Generalized Minimal Residual (GMRES) method with restarts.
    
    Parameters:
    - prec: A function that applies the preconditioner to a vector.
    - A: The matrix of the linear system to solve.
    - b: The right-hand side vector of the linear system.
    - true_sol: The true solution of the system for error comparison.
    - x0: Initial guess for the solution. Defaults to a zero vector of the same size as b.
    - k_max: Maximum number of iterations. Defaults to the size of A if not provided or if larger than the size of A.
    - restart: Number of iterations before restarting the GMRES process. If None, no restart is performed.
    - epsilon: Convergence tolerance. The algorithm stops if the residual norm is below this value.
    
    Returns:
    - xk: The approximate solution of the linear system.
    - res_list: List of residual norms at each iteration.
    - error_list: List of residuals of the preconditioned system at each iteration.
    - true_sol_list: List of errors with respect to the true solution at each iteration.
    - total_k: Total number of iterations performed.
    
    Implementation:
    ---------------
    1. Initialization:
        - If x0 is not provided, it is initialized to a zero vector.
        - The maximum number of iterations (k_max) is set to the size of A if not provided or larger than the size of A.
        - Compute the initial residual r0 = b - A * x0 and apply the preconditioner.
        - Normalize the initial residual to get the first basis vector of the Krylov subspace.
    
    2. Arnoldi Iteration:
        - Expand the Krylov subspace by computing new orthonormal basis vectors using the Arnoldi process.
        - Update the Hessenberg matrix H with the new orthogonal vectors.
        - Perform QR factorization of H and solve the least squares problem to get the new approximate solution xk.
    
    3. Residual and Error Calculation:
        - Compute the residual norm and append it to the res_list.
        - Compute the error with respect to the preconditioned system and the true solution, and append them to error_list and true_sol_list, respectively.
    
    4. Restart Mechanism:
        - If the restart parameter is provided and the current iteration count reaches the restart value, reinitialize the solution x0, residual r0, and Krylov subspace basis.
    
    5. Stopping Criteria:
        - The algorithm stops if the residual norm is below the convergence tolerance epsilon.
        - The algorithm also stops if the total number of iterations reaches k_max.
    
    6. Return:
        - The final approximate solution xk, the list of residual norms, the list of errors, the list of true solution errors, and the total number of iterations performed.
    
    """
    
    x0 = x0 if x0 is not None else np.zeros_like(b)
    
    n = A.shape[0]
    
    if k_max is None or k_max > n:
        k_max = n
    
    r0 = b - A @ x0
    
    # Apply initial preconditioning to the residual
    r0 = prec(r0)

    p0 = np.linalg.norm(r0)
    beta = p0
    pk = p0
    k = 0
    total_k = 0

    # Save list of errors at each iteration
    res_list = [pk]
    error_list = [pk]
    a = np.linalg.norm(x0 - true_sol)
    true_sol_list = [a]
    
    # Initialize the V basis of the Krylov subspace (concatenate as iteration continues). May terminate early.
    V = np.zeros((n, 1))
    V[:, 0] = r0 / beta
    
    # Hessenberg matrix
    H = np.zeros((n + 1, 1))        
    
    while pk > epsilon and total_k < k_max:

        # Arnoldi iteration
        V = np.concatenate((V, np.zeros((n, 1))), axis=1)
        H = np.concatenate((H, np.zeros((n + 1, 1))), axis=1)
        
        # Minv_A will be A if precondition is None
        H[:k + 2, k], v_new = arnoldi_one_iter(prec, A, V, k)

        if v_new is None:
            # print("ENCOUNTER EXACT SOLUTION")
            # Append 0 for plots...
            res_list.append(0)
            Q, R = qr(H[:k + 2, :k + 1], mode='complete')
        
            # pk = abs(beta * Q[0, k])  # Compute norm of residual vector
            pk = np.linalg.norm(b - A @ xk)
            res_list.append(pk)  # Add new residual at current iteration       
        
            yk = back_substitution(R[:-1, :], beta * Q[0][:-1])
            xk = x0 + V[:, :k + 1] @ yk  # Compute the new approximation x0 + V_{k}y
            error_list.append(np.linalg.norm(b - A @ xk))
            true_sol_list.append(np.linalg.norm(xk - true_sol))
            break
        
        else:
            V[:, k + 1] = v_new
        
        Q, R = qr(H[:k + 2, :k + 1], mode='complete')
        
        # pk = abs(beta * Q[0, k])  # Compute norm of residual vector
        res_list.append(pk)  # Add new residual at current iteration       
        
        yk = back_substitution(R[:-1, :], beta * Q[0][:-1])
        xk = x0 + V[:, :k + 1] @ yk  # Compute the new approximation x0 + V_{k}y
        pk = np.linalg.norm(b - A @ xk)
        error_list.append(np.linalg.norm(b - A @ xk))
        true_sol_list.append(np.linalg.norm(xk - true_sol))
        k += 1
        total_k += 1
        
        if restart is not None and k == restart:
            x0 = xk
            r0 = b - A @ x0
            
            r0 = prec(r0)
            
            p0 = np.linalg.norm(r0)
            beta = p0
            pk = p0
            k = 0
            
            V = np.zeros((n, 1))
            V[:, 0] = r0 / beta
            H = np.zeros((n + 1, 1))
  
    return xk, res_list, error_list, true_sol_list, total_k
