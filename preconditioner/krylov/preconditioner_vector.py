
import time
import numpy as np
import ilupp
from scipy.sparse.linalg import spsolve_triangular as triang
from scipy.sparse import tril, triu, diags, eye
from scipy.sparse.linalg import spsolve, inv
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

def fb_solve(L, U, r):
    y = triang(L, r, lower=True)
    z = triang(U, y, lower=False)
    return z

def f_solve(C, r):
    y = triang(C, r, lower=True)
    return y

time_function = lambda: time.perf_counter()

def preconditioner_vector(method, A, model=None, data=None):
    n = A.shape[0]
    
    # Ensure A is in float64 format
    A = A.astype(np.float64)

    if method == "baseline":
        p_start, p_stop = 0, 0
        prec = lambda x: x
    
    elif method == "jacobi":
        p_start = time_function()
        
        diag = A.diagonal()
        diag_inv = csr_matrix((1/diag, (range(n), range(n))), shape=(n,n))

        p_stop = time_function()

        prec = lambda x: diag_inv @ x

    elif method == "GS":
        p_start = time_function()

        # Extract strictly lower triangular part of A
        L = tril(A, format="csr")

        p_stop = time_function()
        
        prec = lambda x: f_solve(L, x)

    elif method == "sym. GS":
        p_start = time_function()
        
        # Extract diagonal part of A and compute its inverse
        D = diags(A.diagonal())
        
        # Extract strictly lower triangular part of A
        L = tril(A, k=-1, format="csr")
        
        # Extract strictly upper triangular part of A
        U = triu(A, k=1, format="csr")
        
        # Construct the forward Gauss-Seidel preconditioner
        M_forward = D + L
        
        # Construct the backward Gauss-Seidel preconditioner
        M_backward = D + U
        
        p_stop = time_function()

        # Define the preconditioner as a lambda function
        prec = lambda x: fb_solve(M_forward, M_backward, x)

    elif method == "ILU(0)":
        p_start = time_function()

        B = ilupp.ILU0Preconditioner(A)
        L, U = B.factors()
        
        p_stop = time_function()

        prec = lambda x: fb_solve(L, U, x)

    elif method == "learned":
        data = data.to("cpu")
        p_start = time_function()

        # matrices obtained from forward pass through the model
        L, U, _ = model(data)
        
        p_stop = time_function()

        L = L.coalesce()
        data = L.values().numpy()
        indices = L.indices().numpy()
        shape = L.size()
        L = coo_matrix((data, indices), shape=shape).tocsr()

        U = U.coalesce()
        data = U.values().numpy()
        indices = U.indices().numpy()
        shape = U.size()
        U = coo_matrix((data, indices), shape=shape).tocsr()

        prec = lambda x: fb_solve(L, U, x)

    else:
        raise NotImplementedError(f"Preconditioner {method} not implemented!")       
        
    total_time = p_stop - p_start

    return prec, total_time
