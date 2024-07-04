import torch
import torch_geometric
import torch.nn.utils as nn_utils
import time
import json
import datetime
import os
import csv
import itertools # For the grid search
from utils import FolderDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from mypreconditioner.loss import loss_function, loss_function_sketched
from mypreconditioner.model import MYPRECONDITIONER
from torch_geometric.data import Batch

# Without this it does not run, I have no idea what this is :)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.cuda.is_available() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ddtype = torch.float32

# Counts the number of trainable parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Save a dictionary to a JSON file
def save_dict_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        file.write(json.dumps(dictionary))

def sum_eigenvalues(L, U, A):
    # Convert the sparse tensors to dense format for computation
    L = L.to_dense()
    U = U.to_dense()
    A = A.to_dense()

    # Compute the product of L and U
    LU = torch.mm(L, U)

    # Compute the inverse of LU
    LU_inv = torch.linalg.inv(LU)

    # Compute the product of LU_inv and A
    LU_inv_A = torch.mm(LU_inv, A)

    # Compute the eigenvalues of LU_inv_A
    eigenvalues = torch.linalg.eigvals(LU_inv_A)

    # Compute the sum of the absolute values of the eigenvalues
    sum_abs_eigenvalues = torch.sum(torch.abs(eigenvalues))

    return sum_abs_eigenvalues

@torch.no_grad()
def validate(model, validation_loader):
    model.eval()
    total_loss = 0.0
    for data in validation_loader:
        data = data.to(device)
        L, U, _ = model(data)
        A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr.squeeze(), requires_grad=False)
        l = loss_function(L, U, A)
        total_loss += l.item()
    average_loss = total_loss / len(validation_loader)
    return average_loss

@torch.no_grad()
def initial_train_loss(model, train_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        L, U, _ = model(data)
        A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr.squeeze(), requires_grad=False)
        l = loss_fn(L, U, A)
        total_loss += l.item()
    average_loss = total_loss / len(train_loader)
    return average_loss

def train_model(size_train, size_val, number_of_epochs, loss_fn, model_idx):
    torch_geometric.seed_everything(42)
    
    dataset = "poisson2"
    n = 2500 # size of the matrix
    graph = True

    if dataset == "random":
        train_data = FolderDataset("./data/data_random/train/", n, graph, size_train)
        validation_data = FolderDataset("./data/data_random/val/", n, graph, size_val)
        
    elif dataset == "poisson1":
        train_data = FolderDataset("./data/data_poisson1/train/", n, graph, size_train)
        validation_data = FolderDataset("./data/data_poisson1/val/", n, graph, size_val)

    elif dataset == "poisson2":
        train_data = FolderDataset("./data/data_poisson2/train/", n, graph, size_train)
        validation_data = FolderDataset("./data/data_poisson2/val/", n, graph, size_val)
    
    else:
        raise NotImplementedError("Dataset not implemented, Available: random, pde, ipm")

    hidden_sizes_node, hidden_sizes_edge, message_passing_steps, augment_nodes, augment_edges, patiences, factors = [8], [14], [7], [True], [True], [10], [1]

    hyperparameter_combinations = list(itertools.product(hidden_sizes_node, hidden_sizes_edge, message_passing_steps, augment_nodes, patiences, factors))
    
    all_train_losses = []
    all_val_losses = []
    
    for i, (hidden_size_node, hidden_size_edge, message_passing_steps, augment_nodes, patience, factor) in enumerate(hyperparameter_combinations):
        print(f"Training model {model_idx} - {i+1}/{len(hyperparameter_combinations)}")

        args = {"hidden_size_node": hidden_size_node, "hidden_size_edge": hidden_size_edge,  "message_passing_steps": message_passing_steps, 
                "skip_connections": True, "edge_connections": False, "augment_nodes": augment_nodes, 
                "augment_edges": augment_edges, "global_features": 0}
                
        model = MYPRECONDITIONER(**args)
        model.to(device)
        print(f"Number params in model: {count_parameters(model)}")
        print()
        
        parameters = count_parameters(model)

        def custom_collate(batch):
            return Batch.from_data_list(batch)

        train_loader = DataLoader(train_data, batch_size = 1, shuffle = True, collate_fn = custom_collate)
        validation_loader = DataLoader(validation_data, batch_size=1, shuffle=False, collate_fn = custom_collate)
        
        best_val = float("inf")
        best_model_state = None
        
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)
                
        number_of_epochs = number_of_epochs
        
        total_time = 0

        train_losses = []
        val_losses = []

        initial_train = initial_train_loss(model, train_loader, loss_fn)
        initial_val = validate(model, validation_loader)
        train_losses.append(initial_train)
        val_losses.append(initial_val)

        # print(f"Initial train loss: {initial_train:.4f} \t Initial validation loss: {initial_val:.4f}")

        for epoch in range(number_of_epochs):
            running_loss = 0.0
            start_epoch = time.perf_counter()
            for data in train_loader:
                model.train()
                data = data.to(device)
                optimizer.zero_grad()
                L, U, reg = model(data)
                A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr.squeeze(), requires_grad=False)
                l = loss_fn(L, U, A)
                lr = l + 0*reg
                lr.backward()

                # Gradient clipping
                nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                running_loss += lr.item()
                optimizer.step()
            
            train_loss = running_loss / len(train_loader)
            val_loss = validate(model, validation_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_model_state = model.state_dict().copy()

            epoch_time = time.perf_counter() - start_epoch
            total_time += epoch_time

            if (epoch+1) in [50, 100, 200]:
                print(f"validation loss: {best_val:.4f} \t time: {total_time:.4f}")

            print(f"Epoch {epoch+1} \t train loss: {train_loss:.4f} \t validation loss: {val_loss:.4f} \t time: {epoch_time:.4f}")
        
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        folder = "results/" + f"{dataset}_" + f"size{n}_" + f"train{size_train }_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + f"_config{i}_model{model_idx}"
        os.makedirs(folder, exist_ok=True)
        save_dict_to_file(args, os.path.join(folder, "config.json"))
        
        if best_model_state is not None:
            torch.save(best_model_state, f"{folder}/best_model.pt")

        csv_filename = "all_results.csv"
        fieldnames = ["Approach", "Parameters", "Epochs", "Last train loss", 
                      "Best validation loss", "Time", "Train Batch Size",
                       "Validation Batch Size", "Patience", "Factor"] + list(args.keys())

        csv_exists = os.path.exists(csv_filename)

        with open(csv_filename, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if not csv_exists:
                writer.writeheader()

            writer.writerow({
                "Approach": "LU",
                "Parameters": parameters,
                "Epochs": number_of_epochs,
                "Last train loss": train_loss,
                "Best validation loss": best_val,
                "Time": total_time,
                "Train Batch Size": train_loader.batch_size,
                "Validation Batch Size": validation_loader.batch_size,
                "Patience": patience,
                "Factor": factor,
                **args,
            })

        print(f"Best validation loss: {best_val}. Configuration {i+1} complete.")
        print()

    return all_train_losses, all_val_losses

if __name__ == "__main__":
    number_of_epochs = 5
    configurations = [(100, 10)]

    # all_train_losses_1, all_val_losses_1 = train_model(configurations[0][0], configurations[0][1], number_of_epochs, loss_function, 1)
    all_train_losses_2, all_val_losses_2 = train_model(configurations[0][0], configurations[0][1], number_of_epochs, loss_function_sketched, 2)

    tex_fonts = {
    "text.usetex": True,
    "font.family": "serif"}

    plt.rcParams.update(tex_fonts)

    epochs = range(0, number_of_epochs + 1) 

    # Plotting the training losses
    plt.figure(figsize=(8, 6))
    # plt.plot(epochs, all_train_losses_1[0], label='Frobenius', linestyle='-')
    plt.plot(epochs, all_train_losses_2[0], label='Stochastic', linestyle='-')
    plt.xlabel('Epoch', fontsize=16, labelpad=10)
    plt.ylabel('Training loss',  fontsize=16, labelpad=10)
    plt.legend(fontsize=14)  
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(range(0, number_of_epochs+1 , 50), fontsize=16)
    plt.yticks(fontsize=16)
    
    # plt.ylim(bottom=22)
    # plt.ylim(top= 40)
    plt.tight_layout(pad=4)

    folder_name = ".\plots\generalplots"
    plt.savefig(os.path.join(folder_name, "train_losses_comparison.pdf"), dpi=600, bbox_inches='tight', pad_inches=0.1, format='pdf')
    plt.close()

    # Plotting the validation losses
    plt.figure(figsize=(8, 6))
    # plt.plot(epochs, all_val_losses_1[0], label='Frobenius', linestyle='-')
    plt.plot(epochs, all_val_losses_2[0], label='Stochastic', linestyle='-')
    plt.xlabel('Epoch', fontsize=16, labelpad=10)
    plt.ylabel('Validation loss',  fontsize=16, labelpad=10)
    plt.legend(fontsize=14)  
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(range(1, number_of_epochs+1 , 50), fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.ylim(bottom=22)
    plt.ylim(top= 40)
    plt.tight_layout(pad=4)

    plt.savefig(os.path.join(folder_name, "val_losses_comparison.pdf"), dpi=600, bbox_inches='tight', pad_inches=0.1, format='pdf')
    plt.close()
