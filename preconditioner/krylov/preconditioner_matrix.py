import numpy as np
import time
import ilupp
from scipy.sparse.linalg import spsolve_triangular as triang
from scipy.sparse import tril, triu, diags
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy import sparse

time_function = lambda: time.perf_counter()

def preconditioner_matrix(method, A, model = None, data = None):
    
    # Ensure A is in float64 format
    A = A.astype(np.float64)

    n = A.shape[0]

    if method == "baseline":

        prec = lambda x: x
    
    elif method == "jacobi":
        
        D = A.diagonal()
        D_inv = csr_matrix((1/D, (range(n), range(n))), shape=(n,n))

        prec = lambda x: D_inv@x

    elif method == "GS":

        # Extract lower triangular part of A
        L = tril(A, format = "csr")

        # Construct the Gauss-Seidel preconditioner
        M = triang(L, np.eye(L.shape[0]))
        
        M = sparse.csr_matrix(M)
        
        prec = lambda x: M@x

    elif method == "sym. GS":

        # Extract diagonal part of A
        D = diags(A.diagonal())

        # Extract strictly lower triangular part of A
        L = tril(A, k=-1, format="csr")
        
        # Extract strictly upper triangular part of A
        U = triu(A, k=1, format="csr")

        # Construct the forward Gauss-Seidel preconditioner
        M_forward = D + L
        
        # Construct the backward Gauss-Seidel preconditioner
        M_backward = D + U
        
        # Convert to csr_matrix
        M_forward = csr_matrix(M_forward)
        M_backward = csr_matrix(M_backward)

        prec = lambda x: M_backward@M_forward@x
        
    elif method == "ILU(0)":

        # https://ilupp.readthedocs.io/en/latest/

        A = A.astype(float)  # Convert to double precision
        B = ilupp.ILU0Preconditioner(A)
        L, U = B.factors()

        L_inv = triang(L, np.eye(L.shape[0]), lower = True)
        U_inv = triang(U, np.eye(U.shape[0]), lower = False)

        M = U_inv @ L_inv

        M = sparse.csr_matrix(M)
        
        prec = lambda x: M@x

    elif method == "learned":
        
        data = data.to("cpu")

        # matrices obtained from forward pass through the model
        L, U, _ = model(data)

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

        L_inv = triang(L, np.eye(L.shape[0]), lower = True)
        U_inv = triang(U, np.eye(U.shape[0]), lower = False)

        M = L_inv @ U_inv

        M = sparse.csr_matrix(M)

        prec = lambda x: M@x


    else:
        raise NotImplementedError(f"Preconditioner {method} not implemented!")       
        

    return prec
