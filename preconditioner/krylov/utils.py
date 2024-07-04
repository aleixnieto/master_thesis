import numpy as np

def arnoldi_one_iter(prec, A, V, k, tol = 1e-12):
    """
    Computes the new vectors of the Arnoldi iteration for both V_{k+1} and H_{k + 1, k}

    Input parameters:
    -----------------
    A: array_like
         An (n x n) array.
          
    V: array_like
        An (n x (k + 1)) array. The current Krylov orthonormal basis.
      
    k: int
        One less than the step we are obtaining in the Arnoldi's algorrithm to increase
        the dimension of the Krylov subspace. Must be >= 0.
    
    precondition: PreconditionEnum or None
        An enumeration representing the preconditioning method to be applied.
        
    M: scipy.sparse matrix or None
        The preconditioning matrix if applicable, otherwise None.    
      
    epsilon : float, optional
        Tolerance for convergence.
    
    Output:
    -------
      h_k: 
          
      v_new:
          
    """
    # Note that to obtain the first column of H ((k + 1) x k) we need 2 vectors in V. Later in the GMRES algorithm
    # we will use the notation H[: k + 2, : k + 1] as k starts at 0 and we select the first two rows and first column.
    
    # Here h_k respresents the column k + 1 in H. (k stars at 0)
    
    # Inialize k + 2 nonzero elements of H along column k. (k starts at 0)
    h_k = np.zeros((k + 2, ))

    v_new = prec(A @ V[:, k])
    
    # Calculate first k elements of the kth Hessenberg column
    for j in range(k + 1): # Here k is from 0 to k 
        h_k[j] = v_new @ V[:, j]
        v_new = v_new - h_k[j] * V[:, j]
    
    # Add the k+1 element
    h_k[k + 1] = np.linalg.norm(v_new)

    if h_k[k + 1] <= tol:
        # None for v to check in gmres (early termination with EXACT SOLUTION)
        return h_k, None
    
    else:
        # Find the new orthogonal vector in the basis of the Krylov subspace
        v_new = v_new / h_k[k + 1]

    return h_k, v_new
    
def back_substitution(A, b):
    """
    Solve a linear system using back substitution.
    
    Args:
    ----------
        A: list of lists
            Coefficient matrix (must be upper triangular).
        
        b: list
            Column vector of constants.
    
    Returns:
    --------
        list: Solution vector.
        
    Raises:
        ValueError: If the matrix A is not square or if its dimensions are incompatible with the vector b.
    """
    
    n = len(b)
    
    # Check if A is a square matrix
    if len(A) != n or any(len(row) != n for row in A):
        raise ValueError("Matrix A must be square.")
    
    # Check if dimensions of A and b are compatible
    if len(A) != len(b):
        raise ValueError("Dimensions of A and b are incompatible.")
    
    x = [0] * n
    
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
    
    return x