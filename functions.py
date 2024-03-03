import numpy as np
from time import time

# from preconditioner import iter_precondition
from preconditioner import iter_precondition

def arnoldi_og(A, b, x0, m, tol=1e-12):
    """
    This function computes an orthonormal basis V_m = {v_1,...,v_{m+1}} of 
    K_{m+1}(A, r^{(0)}) = span{r^{(0)}, Ar^{(0)}, ..., A^{m}r^{(0)}}.

    Input parameters:
    -----------------
      A: array_like
          An (n x n) array.
      
      b: array_like
          Initial vector of length n
      
      m: int
          One less than the dimension of the Krylov subspace. Must be > 0.
      
      x0: array_like 
          Initial approximation of the solution (length n)
      
      tol: 
          Tolerance for convergence

    Output:
    -------
      Q: numpy.array 
          n x m array, the columns are an orthonormal basis of the Krylov subspace.
      
      H: numpy.array
          An (m + 1) x m array. It is the matrix A on basis Q. It is upper Hessenberg.
    """
    # Check inputs
    n = A.shape[0]
    assert A.shape == (n, n) and b.shape == (n,) and x0.shape == (n,), "Matrix and vector dimensions don not match"
    assert isinstance(m, int) and m > 0, "m must be a positive integer"
    
    m = min(m, n)
    
    # Initialize matrices
    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))
    
    # Normalize input vector and use for Krylov vector
    r0 = b - A @ x0
    beta = np.linalg.norm(r0)
    V[:, 0] = r0 / beta

    for k in range(1, m + 1):
        # Generate a new candidate vector
        w = A @ V[:, k - 1]
        
        # Orthogonalization
        for j in range(k):
            H[j, k - 1] = V[:, j] @ w
            w -= H[j, k - 1] * V[:, j]
        
        H[k, k - 1] = np.linalg.norm(w)

        # Check convergence
        if H[k, k - 1] <= tol:
#             print(f"Converged in {k} iterations.")
            return V, H
        
        # Normalize and store the new basis vector
        V[:, k] = w / H[k, k - 1]
    
    return V, H


def arnoldi(A, b, r0, m, precondition = None, M = None, tol = 1e-12):
    """
    This function computes an orthonormal basis V_m = {v_1,...,v_{m+1}} of 
    K_{m+1}(A, r^{(0)}) = span{r^{(0)}, Ar^{(0)}, ..., A^{m}r^{(0)}}.

    Input parameters:
    -----------------
      A: array_like
          An (n x n) array.
      
      b: array_like
          Initial vector of length n
      
      m: int
          One less than the dimension of the Krylov subspace. Must be > 0.
      
      x0: array_like 
          Initial approximation of the solution (length n)
      
      tol: 
          Tolerance for convergence

    Output:
    -------
      Q: numpy.array 
          n x m array, the columns are an orthonormal basis of the Krylov subspace.
      
      H: numpy.array
          An (m + 1) x m array. It is the matrix A on basis Q. It is upper Hessenberg.
    """
    
    # TODO: Now we calculate precondition time at each iteration. The idea is that arnoldi algorithm only provides one 
    # - iteration at a time so we don't build every time the whole basis from scratch in the GMRES algorithm
    # - torch?
    # - instead of A, callable function that calculates Ax for any input vector x?
    # Check inputs
    n = A.shape[0]
    assert A.shape == (n, n) and b.shape == (n,) and r0.shape == (n,), "Matrix and vector dimensions don not match"
    assert isinstance(m, int) and m > 0, "m must be a positive integer"
    
    m = min(m, n)
    
    # Initialize matrices
    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))
    
    # Normalize input vector and use for Krylov vector
    
    beta = np.linalg.norm(r0)
    V[:, 0] = r0 / beta

    for k in range(1, m + 1):
        # Generate a new candidate vector
        w, _= iter_precondition(A, M, V, k, precondition)
        
        # Orthogonalization
        for j in range(k):
            H[j, k - 1] = V[:, j] @ w
            w -= H[j, k - 1] * V[:, j]
        
        H[k, k - 1] = np.linalg.norm(w)

        # Check convergence
        if H[k, k - 1] <= tol:
            print(f"Converged in {k} iterations.")
            return V, H
        
        # Normalize and store the new basis vector
        V[:, k] = w / H[k, k - 1]
    
    return V, H
    
def arnoldi_one_iter(A, b, r0, m, precondition = False, M = None, tol = 1e-12):
    """
    This function computes an orthonormal basis V_m = {v_1,...,v_{m+1}} of 
    K_{m+1}(A, r^{(0)}) = span{r^{(0)}, Ar^{(0)}, ..., A^{m}r^{(0)}}.

    Input parameters:
    -----------------
      A: array_like
          An (n x n) array.
      
      b: array_like
          Initial vector of length n
      
      m: int
          One less than the dimension of the Krylov subspace. Must be > 0.
      
      x0: array_like 
          Initial approximation of the solution (length n)
      
      tol: 
          Tolerance for convergence

    Output:
    -------
      Q: numpy.array 
          n x m array, the columns are an orthonormal basis of the Krylov subspace.
      
      H: numpy.array
          An (m + 1) x m array. It is the matrix A on basis Q. It is upper Hessenberg.
    """
    
    # TODO: Now we calculate precondition time at each iteration. The idea is that arnoldi algorithm only provides one 
    # - iteration at a time so we don't build every time the whole basis from scratch in the GMRES algorithm
    # - torch?
    # - instead of A, callable function that calculates Ax for any input vector x?
    # Check inputs
    n = A.shape[0]
    assert A.shape == (n, n) and b.shape == (n,) and r0.shape == (n,), "Matrix and vector dimensions don not match"
    assert isinstance(m, int) and m > 0, "m must be a positive integer"
    
    m = min(m, n)
    
    # Initialize matrices
    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))
    
    # Normalize input vector and use for Krylov vector
    
    beta = np.linalg.norm(r0)
    V[:, 0] = r0 / beta

    for k in range(1, m + 1):
        # Generate a new candidate vector
        w, _= iter_precondition(A, M, V, k, precondition)
        
        # Orthogonalization
        for j in range(k):
            H[j, k - 1] = V[:, j] @ w
            w -= H[j, k - 1] * V[:, j]
        
        H[k, k - 1] = np.linalg.norm(w)

        # Check convergence
        if H[k, k - 1] <= tol:
            print(f"Converged in {k} iterations.")
            return V, H
        
        # Normalize and store the new basis vector
        V[:, k] = w / H[k, k - 1]
    
    return V, H
    
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