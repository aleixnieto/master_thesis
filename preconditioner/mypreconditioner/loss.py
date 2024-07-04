import warnings
import torch

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ddtype = torch.float32
torch.set_default_dtype(ddtype)

warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

def loss_function(L, U, A):
    # Convert sparse tensors L and U to dense tensors
    L_dense = L.to_dense()
    U_dense = U.to_dense()

    # Convert sparse tensor A to dense tensor
    A_dense = A.to_dense()

    r = L_dense @ U_dense - A_dense
    return torch.norm(r, p='fro')  # Frobenius norm

def loss_function_sketched(L, U, A):
    z = torch.randn((A.shape[0], 1), device=device, dtype=ddtype)
    A = A.to(device).to(ddtype)
    est = L @ (U @ z) - A @ z
    return torch.linalg.vector_norm(est, ord=2)  # vector norm

# FUNCTIONS WHEN WE CONSTRUCT A FULL INVERSE M

# def loss_function(M, A):
#     M_dense = M.to_dense()

#     # Convert sparse tensor A to dense tensor
#     A_dense = A.to_dense()

#     r = M_dense - A_dense
#     return torch.norm(r, p = 'fro') # Frobenius norm

# def loss_function_sketched(M, A):
#     z = torch.randn((A.shape[0], 1), device=device)

#     est = (M@z) - A@z
#     return torch.linalg.vector_norm(est, ord = 2) # vector norm

# def loss_function(M, A,):
#     I = torch.eye(A.shape[0], device=A.device)
#     r = I - M@A
#     return torch.norm(r, p = 'fro') # Frobenius norm

# def loss_function_sketched(M, A):
#     z = torch.randn((A.shape[0], 1), device=device)

#     I = torch.eye(A.shape[0], device=A.device)

#     est = I@z - A@(M@z)
#     return torch.linalg.vector_norm(est, ord = 2) # vector norm