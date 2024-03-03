from enum import Enum, auto

from time import time

from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import spsolve

class PreconditionEnum(Enum):
    JACOBI = auto()
    SYMMETRIC_GAUSS_SEIDEL = auto()

def initial_precondition(A, precondition, r0):
    # total precondition time
    total_precondition_time = 0
    precondition_start_time = time()
    match precondition:
        case PreconditionEnum.JACOBI:
            M = sp_diags(A.diagonal(), offsets=0, format="csr")
            r = spsolve(M, r0)
          
        case PreconditionEnum.SYMMETRIC_GAUSS_SEIDEL:
            D_vector = A.diagonal()
            D_inv_vector = 1 / D_vector
            D_inv = sp_diags(D_inv_vector, offsets=0, format="csr")

            L = sp_eye(m) + sp_tril(A, k=-1, format="csr") @ D_inv
            U = sp_triu(A, format="csr")
            M = [L, U]

            z = spsolve(L, r_original)
            r = spsolve(U, z)
            
        # default case (None)
        case _:
            M = None
            r = r0

    precondition_end_time = time()
    total_precondition_time += (precondition_end_time - precondition_start_time)

    return M, r, total_precondition_time

def iter_precondition(A, M, V, k, precondition):
    
    # Apply the preconditioning at each iteration step
    
    precondition_start_time = time()

    match precondition:
        
#         case PreconditionEnum.JACOBI | PreconditionEnum.GAUSS_SEIDEL:
        case PreconditionEnum.JACOBI:
            w = spsolve(M, A @ V[:, k - 1])
            
        case PreconditionEnum.SYMMETRIC_GAUSS_SEIDEL:
            L, U = M
            z = spsolve(L, A @ V[:, k])
            w = spsolve(U, z)

        case _:
            w = A @ V[:, k - 1]

    precondition_end_time = time()
    
    iter_precondition_time = precondition_end_time - precondition_start_time
    
    return w, iter_precondition_time