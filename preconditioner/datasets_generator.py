import os
import numpy as np
import torch
import pyamg
import scipy
from scipy.sparse import coo_matrix, csr_matrix, rand
from torch_geometric.data import Data

# Without this it does not run, I have no idea what this is :)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# The function matrix_to_graph_sparse takes a sparse matrix A and extracts its non-zero elements and their corresponding 
# row and column indices. It then constructs a graph representation where each non-zero element becomes an edge, 
# and each row and column index pair becomes an edge index. 
def matrix_to_graph_sparse(A, b):
    """
    Convert a sparse matrix representation into a graph representation.

    Parameters:
        A (scipy.sparse.coo_matrix): The sparse matrix.
        b (numpy.ndarray): Vector of node features.

    Returns:
        torch_geometric.data.Data: A Data object representing the graph.
    """

    edge_index = torch.tensor(list(map(lambda x: [x[0], x[1]], zip(A.row, A.col))), dtype = torch.long)
    edge_features = torch.tensor(list(map(lambda x: [x], A.data)), dtype = torch.float)
    node_features = torch.tensor(list(map(lambda x: [x], b)), dtype = torch.float) # This is actually to have something, because this
                                                                                # will be later updated with augment_features. It is helpful to read b from graph in testing.
    
    # Embed the information into graph data object
    data = Data(x = node_features, edge_index = edge_index.t().contiguous(), edge_attr = edge_features)
    return data


def matrix_to_graph(A, b):
    return matrix_to_graph_sparse(coo_matrix(A), b)

def generate_sparse_random(n, random_state = 0, sol = False):
    """
    Generate an arbitrary sparse random matrix along with a corresponding right-hand side vector.

    Parameters:
        - n (int): The size of the square matrix.
        - random_state (int, optional): A random seed to ensure reproducibility. Defaults to 0.
        - sol (bool, optional): Whether to compute the solution or not. Defaults to False.

    Returns:
        tuple: A tuple containing the sparse matrix, solution vector (if computed), and right-hand side vector.
    """

    rng = np.random.RandomState(random_state)
    
    if n == 1000:
        zero_prob = rng.uniform(0.995, 0.999)
    elif n == 100:
        zero_prob = rng.uniform(0.98, 0.99)
    else:
        raise NotImplementedError(f"Can\'t generate sparse matrix for n={n}")
    
    nnz = int((1 - zero_prob) * n ** 2)
    rows = [rng.randint(0, n) for _ in range(nnz)]
    cols = [rng.randint(0, n) for _ in range(nnz)]
    
    uniques = set(zip(rows, cols)) # Ensure we do not have repeated edges
    rows, cols = zip(*uniques)
    
    # generate values
    vals = np.array([rng.normal(0, 1) for _ in cols]) # Random values are generated for the non-zero elements from a normal distribution.
    A = coo_matrix((vals, (rows, cols)), shape=(n, n))
    
    # Set diagonal elements to a random number different from 0
    diag_indices = np.arange(n)
    diag_vals = rng.normal(-1, 1, size=n)  # Random values for the diagonal elements
    diag_vals[diag_vals == 0] = rng.uniform(0.01, 1)  # Replace 0s with a different random number
    A = A + coo_matrix((diag_vals, (diag_indices, diag_indices)), shape=(n, n))
    
    A = coo_matrix(A)

    # right hand side is uniform
    b = rng.uniform(0, 1, size=n)

    # We want a high-accuracy solution, so we use a direct sparse solver here.
    # only produce when in test mode
    if sol:
        # spsolve requires A be CSC or CSR matrix format
        A_csr = A.tocsr()  # Convert to CSR format
        x = scipy.sparse.linalg.spsolve(A_csr, b)
    else:
        x = None
    
    return A, x, b

def generate_poisson_perturbed(n, random_state=0, sol=False):
    """
    Generate a perturbed Poisson problem matrix and corresponding right-hand side vector.

    Parameters:
        - n (int): The size of the problem grid (n x n).
        - random_state (int, optional): A random seed to ensure reproducibility. Defaults to 0.
        - sol (bool, optional): Whether to compute the solution or not. Defaults to False.

    Returns:
        tuple: A tuple containing the perturbed sparse matrix, solution vector (if computed), and right-hand side vector.
    """
    
    n = int(np.sqrt(n))

    rng = np.random.RandomState(random_state)

    A = pyamg.gallery.poisson((n, n))

    A = csr_matrix(A)

    # Find the non-zero elements in the matrix
    non_zero_indices = A.nonzero()

    # Generate random values for the non-zero elements using rng
    random_values = rng.normal(0, 1, size=len(non_zero_indices[0]))

    # Add the random values to the non-zero elements of the matrix
    A.data += random_values

    # Define the source term function f(x, y)
    def f(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    # Generate the grid points
    x = np.linspace(0, 1, n + 2)  # include boundary points
    y = np.linspace(0, 1, n + 2)
    X, Y = np.meshgrid(x, y)

    # Evaluate f at interior grid points (ignoring boundaries)
    b = f(X[1:-1, 1:-1], Y[1:-1, 1:-1])

    # Flatten the array to create the vector b
    b = b.flatten()
    
    # We want a high-accuracy solution, so we use a direct sparse solver here.
    # only produce when in test mode
    if sol:
        # spsolve requires A be CSC or CSR matrix format

        # The function does indeed assume that the matrix A is invertible, or non-singular. This is because
        # the LU factorization method involves decomposing A into the product of a lower triangular matrix
        # L and an upper triangular matrix U. If A is not invertible, then this decomposition is not possible, 
        # and the method cannot proceed.

        # Check rank and if determinant is non-zero
        A = A.todense()
        
        rank_A = np.linalg.matrix_rank(A)
        det_A = np.linalg.det(A)
        
        # Check if rank equals the number of rows or columns
        if rank_A == min(A.shape):
            pass
        else:
            print("Matrix is singular")

        # Check if determinant is non-zero
        if np.abs(det_A) > 1e-1:  # Adjust the threshold as needed
            pass
        else:
            print("The determinant is 0")

        A = csr_matrix(A)

        x = scipy.sparse.linalg.spsolve(A, b)
    else:
        x = None
    
    return A, x, b

def generate_poisson_perturbed2(n, random_state = 0, sol = False):
    """
    Generate a more perturbed Poisson problem matrix with additional random sparse matrix perturbations.

    Parameters:
        - n (int): The size of the problem grid (n x n).
        - random_state (int, optional): A random seed to ensure reproducibility. Defaults to 0.
        - sol (bool, optional): Whether to compute the solution or not. Defaults to False.

    Returns:
        tuple: A tuple containing the perturbed sparse matrix, solution vector (if computed), and right-hand side vector.
    """

    n = int(np.sqrt(n))

    rng = np.random.RandomState(random_state)

    A = pyamg.gallery.poisson((n, n))
    A = csr_matrix(A)

    # Find the non-zero elements in the matrix
    non_zero_indices = A.nonzero()
    num_non_zero_elements = len(non_zero_indices[0])

    # Determine the number of elements to perturb
    perturb_percentage = rng.uniform(0.01, 0.99)
    num_elements_to_perturb = int(perturb_percentage * num_non_zero_elements)

    # Select a random subset of the non-zero elements
    selected_indices = rng.choice(num_non_zero_elements, num_elements_to_perturb, replace=False)

    # Generate random values for the selected non-zero elements
    perturbation_values = rng.normal(0, 1, size=num_elements_to_perturb)

    # Perturb the selected non-zero elements with different random values
    A.data[selected_indices] += perturbation_values

    # Generate a random sparse matrix with a density between 0.5% and 2.5%
    density = rng.uniform(0.005, 0.025)
    random_sparse_matrix = scipy.sparse.random(A.shape[0], A.shape[1], density=density, format='csr', random_state=random_state)

    # Add the random sparse matrix to A
    A += random_sparse_matrix

    # Define the source term function f(x, y)
    def f(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    # Generate the grid points
    x = np.linspace(0, 1, n+2)  # include boundary points
    y = np.linspace(0, 1, n+2)
    X, Y = np.meshgrid(x, y)

    # Evaluate f at interior grid points (ignoring boundaries)
    b = f(X[1:-1, 1:-1], Y[1:-1, 1:-1])

    # Flatten the array to create the vector b
    b = b.flatten()

    # We want a high-accuracy solution, so we use a direct sparse solver here.
    # only produce when in test mode
    if sol:

        # Check rank and if determinant is non-zero
        A = A.todense()
        
        rank_A = np.linalg.matrix_rank(A)
        det_A = np.linalg.det(A)
        
        # Check if rank equals the number of rows or columns
        if rank_A == min(A.shape):
            pass
        else:
            print("Matrix is singular")

        # Check if determinant is non-zero
        if np.abs(det_A) > 1e-1:  # Adjust the threshold as needed
            pass
        else:
            print("The determinant is 0")

        A = csr_matrix(A)
        x = scipy.sparse.linalg.spsolve(A, b)
    
    else:
        x = None
    
    return A, x, b

def create_dataset(n, samples, graph=True, rs=0, mode='train', type = "random"):
    """
    Generate datasets for training, validation, or testing.

    Parameters:
        - n (int): The size of the square matrix.
        - samples (int): The number of samples to generate.
        - graph (bool, optional): Whether to represent the data as graphs (True) or matrices (False). Defaults to True.
        - rs (int, optional): A random seed to ensure reproducibility. Defaults to 0.
        - mode (str, optional): The mode of the dataset ('train', 'val', or 'test'). Defaults to 'train'.
    """
       
    if mode != 'train':
        assert rs != 0, 'rs must be set for test and val to avoid overlap'
    
    for sam in range(samples):
        # generate solution only for test
        
        if type == "random":
            # Generate sparse matrix, solution, and right-hand side vector
            A, x, b = generate_sparse_random(n, random_state=(rs + sam), sol=(mode == 'test'))
            # ALL TEST MATRICES ARE INVERTIBLE!
        elif type == "poisson1":
            A, x, b = generate_poisson_perturbed(n, random_state=(rs + sam), sol=(mode == 'test'))
        
        elif type == "poisson2":
            A, x, b = generate_poisson_perturbed2(n, random_state=(rs + sam), sol=(mode == 'test'))

        if graph:
            # Convert matrix to graph representation
            graph = matrix_to_graph(A, b)
            if x is not None:
                graph.s = torch.tensor(x, dtype=torch.float)
            graph.n = n
            # Save graph as a PyTorch file
            if type == "random":
                torch.save(graph, f'./data_random/{mode}/{n}_{sam}.pt')
            elif type == "poisson1":
                torch.save(graph, f'./data_poisson1/{mode}/{n}_{sam}.pt')
            elif type == "poisson2":
                torch.save(graph, f'./data_poisson2/{mode}/{n}_{sam}.pt')
        else:
            # Save matrix, right-hand side vector, and solution vector (if provided) as a NumPy archive file
            A = coo_matrix(A)
            if type == "random":
                np.savez(f'./data_random/{mode}/{n}_{sam}.npz', A = A, b = b, x = x)
            
            elif type == "poisson1":
                np.savez(f'./data_poisson1/{mode}/{n}_{sam}.npz', A = A, b = b, x = x)

            elif type == "poisson1":
                np.savez(f'./data_poisson2/{mode}/{n}_{sam}.npz', A = A, b = b, x = x)


if __name__ == '__main__':
    # samples = args.samples
    np.random.seed(0)
    
    # logging
    # print(f"Creating random dataset with {samples} samples for n={n}")
    
    # create the folders and subfolders where the data is stored
    os.makedirs(f'./data/data_random/train', exist_ok=True)
    os.makedirs(f'./data/data_random/val', exist_ok=True)
    os.makedirs(f'./data/data_random/test', exist_ok=True)
    
    os.makedirs(f'./data/data_poisson1/train', exist_ok=True)
    os.makedirs(f'./data/data_poisson1/val', exist_ok=True)
    os.makedirs(f'./data/data_poisson1/test', exist_ok=True)

    os.makedirs(f'./data/data_poisson2/train', exist_ok=True)
    os.makedirs(f'./data/data_poisson2/val', exist_ok=True)
    os.makedirs(f'./data/data_poisson2/test', exist_ok=True)

    # create all datasets
    n = 2500
    
    # create_dataset(n, 1000, mode='train', rs = 0, graph = True, type = "random")
    # create_dataset(n, 100, mode='val', rs = 10000, graph = True, type = "random")
    # create_dataset(n, 100, mode='test', rs = 103600, graph = True, type = "random")

    # create_dataset(n, 5000, mode='train', rs = 0, graph = True, type = "poisson1")
    # create_dataset(n, 1000, mode='val', rs = 10000, graph = True, type = "poisson1")
    # create_dataset(n, 100, mode='test', rs = 103600, graph = True, type = "poisson1")

    # create_dataset(n, 5000, mode='train', rs = 0, graph = True, type = "poisson2")
    # create_dataset(n, 1000, mode='val', rs = 10000, graph = True, type = "poisson2")
    # create_dataset(n, 1000, mode='test', rs = 103600, graph = True, type = "poisson2")