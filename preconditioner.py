from enum import Enum, auto

from time import time

from scipy.sparse import diags as sp_diags, tril as sp_tril, eye as sp_eye, triu as sp_triu
from scipy.sparse.linalg import spsolve, inv

class PreconditionEnum(Enum):
    JACOBI = auto(),
    GAUSS_SEIDEL = auto(),
    SYMMETRIC_GAUSS_SEIDEL = auto(),

def preconditioner(A, precondition):
    
    n = A.shape[0]
    
    precondition_start_time = time()
    
    match precondition:
        
        case PreconditionEnum.JACOBI:
            M = sp_diags(A.diagonal(), offsets = 0, format = "csr")
          
        case PreconditionEnum.GAUSS_SEIDEL:
            M = sp_tril(A, format = "csr")
            
        case PreconditionEnum.SYMMETRIC_GAUSS_SEIDEL:
            D_vector = A.diagonal()
            D_inv_vector = 1 / D_vector
            D_inv = sp_diags(D_inv_vector, offsets = 0, format="csr")

            L = sp_eye(n) + sp_tril(A, k = -1, format = "csr") @ D_inv
            U = sp_triu(A, format = "csr")
            M = [L, U]
            
        case _:
            M = None
            
    precondition_end_time = time()
    precondition_time = (precondition_end_time - precondition_start_time)

    return M, precondition_time

def MinvA(A, M, precondition):
    
    n = A.shape[0]
    
    precondition_start_time = time()

    match precondition:
        
        case PreconditionEnum.JACOBI | PreconditionEnum.GAUSS_SEIDEL:
            w = spsolve(M, A) # Optimize this inverse???
            
        case PreconditionEnum.SYMMETRIC_GAUSS_SEIDEL:
            L, U = M # Optimize with forward and backward subst??
            z = spsolve(L, A)
            w = spsolve(U, z)

        case _:
            w = A

    precondition_end_time = time()   
    precondition_time = precondition_end_time - precondition_start_time
    
    return w, precondition_time

def residual_precondition(M, precondition, r0):

    precondition_start_time = time()
    
    match precondition:
        
        case PreconditionEnum.JACOBI | PreconditionEnum.GAUSS_SEIDEL:
            
            r0 = spsolve(M, r0)
          
        case PreconditionEnum.SYMMETRIC_GAUSS_SEIDEL:

            L, U = M
            
            z = spsolve(L, r0)
            r0 = spsolve(U, z)
            
        # default case (None)
        case _:
            r0 = r0

    precondition_end_time = time()
    precondition_time = (precondition_end_time - precondition_start_time)

    return r0, precondition_time
