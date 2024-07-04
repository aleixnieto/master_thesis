import os
import matplotlib.pyplot as plt
import torch
import glob

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set default data type to float32 for tensors
ddtype = torch.float32
torch.set_default_dtype(ddtype)

class FolderDataset(torch.utils.data.Dataset):
    """
    Custom dataset class to load data from a folder.
    
    Args:
        folder (str): Path to the folder containing data files.
        n (int): Identifier to filter files.
        graph (bool): Whether the data files are graphs or matrices.
        size (int, optional): Number of files to load. Defaults to None.
    """
    def __init__(self, folder, n, graph=True, size=None) -> None:
        super().__init__()
        self.folder_path = folder
        self.graph = graph

        print(f"Initializing FolderDataset with folder: {folder}, n: {n}, graph: {graph}, size: {size}")

        file_extension = 'pt' if graph else 'npz'
        
        if n != 0:
            files = glob.glob(os.path.join(self.folder_path, f'*.{file_extension}'))
            files = [file.replace("\\", "/") for file in files]
            self.files = list(filter(lambda x: x.split("/")[-1].split('_')[0] == str(n), files))
            self.files = [file.replace("\\", "/") for file in self.files]
        else:
            self.files = list(glob.glob(os.path.join(folder, f'*.{file_extension}')))
        
        if size is not None:
            assert len(self.files) >= size, f"Only {len(self.files)} files found in {folder} with n = {n}"
            self.files = self.files[:size]

        if len(self.files) == 0:
            raise FileNotFoundError(f"No files found in {folder} with n = {n}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.graph:
            g = torch.load(self.files[idx])
            g.x = g.x.to(torch.double)
            if hasattr(g, "s"):
                g.s = g.s.to(torch.double)
            g.edge_attr = g.edge_attr.to(torch.double)
            return g
        else:
            raise NotImplementedError("Matrix to graph conversion not implemented")


def save_plot_eig(method, eigvals_preconditioned, dataset):
    """
    Save a scatter plot of eigenvalues.

    Args:
        method (str): Name of the method.
        eigvals_preconditioned (torch.Tensor): Eigenvalues of the preconditioned matrix.
        dataset (str): Name of the dataset.
    """
    main_folder = f"plots_{dataset}/eigenvalues"
    os.makedirs(main_folder, exist_ok=True)

    tex_fonts = {
        "text.usetex": True,
        "font.family": "serif"
    }
    plt.rcParams.update(tex_fonts)

    folder_name = os.path.join(main_folder, f"{method}")
    os.makedirs(folder_name, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(eigvals_preconditioned.real, eigvals_preconditioned.imag, 'o', color='cornflowerblue', markersize=3)
    plt.xlabel('Real part', fontsize=28, labelpad=10)
    plt.ylabel('Imaginary part', fontsize=28, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    method_legend = plt.Line2D([0], [0], color='w', markerfacecolor='w', markersize=0, label=r'\textbf{' + method + '}')
    plt.legend(handles=[method_legend], loc='upper right', fontsize=22, frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, "eigenvalue_distribution.pdf"), dpi=600, bbox_inches='tight', pad_inches=0.1, format='pdf')
    plt.close()


def save_plot_res(method, iterations, residuals, errors, dataset):
    """
    Save a plot of residuals and errors over iterations.

    Args:
        method (str): Name of the method.
        iterations (list): List of iteration numbers.
        residuals (list): List of residuals.
        errors (list): List of errors.
        dataset (str): Name of the dataset.
    """
    main_folder = f"plots_{dataset}/residuals"
    os.makedirs(main_folder, exist_ok=True)

    tex_fonts = {
        "text.usetex": True,
        "font.family": "serif"
    }
    plt.rcParams.update(tex_fonts)

    folder_name = os.path.join(main_folder, f"{method}")
    os.makedirs(folder_name, exist_ok=True)

    plt.figure(figsize=(7, 6))
    residuals_line, = plt.plot(iterations, residuals, linestyle='-', color='lightcoral', label='residual')
    errors_line, = plt.plot(iterations, errors, linestyle='-', color='cornflowerblue', label=r'error $||x_k - x_*||$')
    plt.yscale('log')
    plt.xlabel('Iterations', fontsize=28, labelpad=10)
    plt.ylabel('Residual norm (log10)', fontsize=28, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    first_legend = plt.legend(handles=[residuals_line, errors_line], fontsize=17, loc='lower left')

    method_legend = plt.Line2D([0], [0], color='w', markerfacecolor='w', markersize=0, label=r'\textbf{' + method + '}')
    plt.legend(handles=[method_legend], loc='lower left', fontsize=22, bbox_to_anchor=(0.43, 0.00), frameon=False)
    plt.gca().add_artist(first_legend)

    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, "residualsvsiterations.pdf"), dpi=600, bbox_inches='tight', pad_inches=0.1, format='pdf')
    plt.close()
    