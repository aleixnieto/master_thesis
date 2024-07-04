import os
import re
import time
import json
import torch
import scipy
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import gmres as gm
from torch_geometric.data import DataLoader
from krylov.gmres import precon_GMRES_restarted
from krylov.preconditioner_vector import preconditioner_vector
from krylov.preconditioner_matrix import preconditioner_matrix
from utils import FolderDataset, save_plot_eig, save_plot_res
from mypreconditioner.model import MYPRECONDITIONER
import warnings

warnings.filterwarnings("ignore")

# Required for some environments to run properly
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class GMRESCounter(object):
    """
    Class to count the iterations and residuals for the scipy GMRES implementation.
    """
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.callbacks = []
        self.internal_list = []

    def append(self, elem):
        self.internal_list.append(elem)

    def getList(self):
        return self.internal_list

    def __call__(self, rk=None):
        self.callbacks.append(rk)
        self.internal_list.append(rk)
        self.niter += 1

def load_best_model(config_path, checkpoint_path):
    """
    Load the best model from the specified configuration and checkpoint paths.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = MYPRECONDITIONER(**config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))

    return model

@torch.inference_mode()
def test_preconditioner(model, test_loader, dataset, gmres_mode, average_diff=False, cond=False):
    """
    Test the preconditioner using the provided model and test data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is not None:
        model.eval()
        model = model.to("cpu")

    print(f"Test performance: {len(test_loader.dataset)} samples\n")

    if model is None:
        methods = ["baseline", "jacobi", "GS", "sym. GS", "ILU(0)"]
    else:
        methods = ["learned"]

    for method in methods:
        print(f"\033[1mTesting {method} preconditioner\033[0m\n")

        total_gmres_time, total_residual, total_prectime = 0, 0, 0
        total_iterations, total_diff, total_diff2 = 0, 0, 0
        total_A_con, total_preconA_con = 0, 0
        num_samples, k = len(test_loader), 0

        for data in test_loader:
            d = data.edge_attr.squeeze().numpy().astype(np.float64)
            i, j = data.edge_index.numpy()
            A = coo_matrix((d, (i, j)), dtype=np.float64)
            A = csr_matrix(A, dtype=np.float64)

            b = data.x[:, 0].squeeze().numpy().astype(np.float64)

            data = data.to(device)
 
            prec_vector, prectime = preconditioner_vector(method, A, model, data)
            prec_matrix = preconditioner_matrix(method, A, model, data)

            restart, maxiter, epsilon = None, 1000, 1e-6
            I = np.eye(A.shape[0], dtype=np.float64)

            r = prec_matrix(A) - I
            total_diff += np.linalg.norm(r, ord="fro")

            r = prec_matrix(I) - A
            total_diff2 += np.linalg.norm(r, ord="fro")

            solution = data.s.cpu().numpy()

            if gmres_mode == "mine":
                start_gmres = time.perf_counter()
                x, residuals, residuals, errors, iterations = precon_GMRES_restarted(
                    prec_vector, A, b, solution, k_max=maxiter, restart=restart, epsilon=epsilon)
                stop_gmres = time.perf_counter()
            else:
                M = csr_matrix(prec_matrix(I))
                counter = GMRESCounter()
                start_gmres = time.perf_counter()
                x, iterations = gm(A, b, M=M, callback=counter, restart=restart, maxiter=maxiter, rtol=1e-6)
                stop_gmres = time.perf_counter()
                residuals = counter.getList()

            gmres_time = stop_gmres - start_gmres
            residual = np.linalg.norm(A @ x - b)

            total_A_con += np.linalg.cond(A.toarray())
            total_preconA_con += np.linalg.cond(prec_matrix(A).toarray())

            if residual < 1e-5:
                k += 1
                total_gmres_time += gmres_time
                total_residual += residual
                total_prectime += prectime
                total_iterations += iterations

        avg_gmres_time = total_gmres_time / num_samples
        avg_residual = total_residual / num_samples
        avg_prectime = total_prectime / num_samples
        avg_iterations = total_iterations / num_samples
        avg_diff = total_diff / num_samples
        avg_diff2 = total_diff2 / num_samples
        avg_cond = total_preconA_con / total_A_con
        avg_converged_samples = k / num_samples

        print("Average GMRES Time:", avg_gmres_time)
        print("Average Preconditioner Time:", avg_prectime)
        print("Total Time:", avg_gmres_time + avg_prectime)
        print("Average Residual:", avg_residual)
        print("Average Iterations:", avg_iterations)
        print("Percentage of Condition Number Reduction:", avg_cond)
        print("Percentage of Converged Samples:", avg_converged_samples)

        save_plot_res(method, range(iterations + 1), residuals, errors, dataset)

        if average_diff:
            print("Average Norm Diff between Preconditioned A and I:", avg_diff)
            print("Average Norm Diff between Preconditioner and A:", avg_diff2)
        
        if cond:
            print("\nAnalyzing Preconditioner Effects for a Single Sampled Matrix A \n")
            cond_number_original = np.linalg.cond(A.toarray())
            eigvals_preconditioned, _ = np.linalg.eig(prec_matrix(A).toarray())
            cond_number_preconditioned = np.linalg.cond(prec_matrix(A).toarray())

            print("Original Matrix Condition Number:", cond_number_original)
            print("Preconditioned Matrix Condition Number:", cond_number_preconditioned, "\n")
            save_plot_eig(method, eigvals_preconditioned, dataset)

def main():
    """
    Main function to run the testing of preconditioners.
    """
    torch.set_default_dtype(torch.float)
    torch.set_num_threads(1)

    print("Using GPU" if torch.cuda.is_available() else "Using CPU")

    model_type = "learned"

    if model_type == "learned":
        print("Using model: learned")
        results_dir = "./results/"
        dirs = os.listdir(results_dir)
        dirs.sort(reverse=True)

        best_model = "specific_case"
        if best_model == "last":
            for dir_name in dirs:
                dir_path = os.path.join(results_dir, dir_name)
                if os.path.isdir(dir_path):
                    if "config.json" in os.listdir(dir_path) and "final_model.pt" in os.listdir(dir_path):
                        config_path = os.path.join(dir_path, "config.json")
                        checkpoint_path = os.path.join(dir_path, "final_model.pt")
                        break
        elif best_model == "specific_case":
            for dir_name in dirs:
                if dir_name == "random_size1000_train1000_2024-05-23_01-19-37_config0_model2":
                    dir_path = os.path.join(results_dir, dir_name)
                    if os.path.isdir(dir_path):
                        if "config.json" in os.listdir(dir_path) and "best_model.pt" in os.listdir(dir_path):
                            config_path = os.path.join(dir_path, "config.json")
                            checkpoint_path = os.path.join(dir_path, "best_model.pt")
                            break

        if "config_path" not in locals() or "checkpoint_path" not in locals():
            print("Error: No directory with both config.json and best_model.pt found.")
            return

        model = load_best_model(config_path, checkpoint_path)
        print("Best model loaded successfully.")

    elif model_type == "None":
        print("Running non-data-driven baselines")
        model = None

    else:
        print(f"Model type '{model_type}' not recognized.")
        return

    batch_size = 1
    dataset = "random"
    n = 100
    size_test = 100
    graph = True

    if dataset == "random":
        test_data = FolderDataset("./data/data_random/test/", n, graph, size_test)
    elif dataset == "poisson1":
        test_data = FolderDataset("./data/data_poisson1/test/", n, graph, size_test)
    elif dataset == "poisson2":
        test_data = FolderDataset("./data/data_poisson2/test/", n, graph, size_test)
    else:
        raise NotImplementedError("Dataset not implemented. Available: random, poisson1, poisson2")

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    test_preconditioner(model, test_loader, dataset, gmres_mode="mine", average_diff=True, cond=True)

if __name__ == "__main__":
    main()
