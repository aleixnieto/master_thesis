import torch
import torch_geometric
import torch.sparse as sparse
import torch.nn.utils as nn_utils
import time
import json
import datetime
import os
import csv
import itertools
from utils import FolderDataset
from torch.utils.data import DataLoader
from mypreconditioner.loss import loss_function, loss_function_sketched
from mypreconditioner.model import MYPRECONDITIONER
from torch_geometric.data import Batch

torch.cuda.is_available() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_dict_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        file.write(json.dumps(dictionary))

@torch.no_grad()
def validate(model, data_loader):
    model.eval()
    total_loss = 0.0
    for data in data_loader:
        data = data.to(device)
        L, U, _ = model(data)
        A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr.squeeze(), requires_grad=False).to(device).float()
        l = loss_function(L, U, A)
        total_loss += l.item()
    average_loss = total_loss / len(data_loader)
    return average_loss

@torch.no_grad()
def initial_train_loss(model, train_loader):
    model.eval()
    total_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        L, U, _ = model(data)
        A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr.squeeze(), requires_grad=False).to(device).float()
        l = loss_function_sketched(L, U, A)
        total_loss += l.item()
    average_loss = total_loss / len(train_loader)
    return average_loss

def main():
    torch_geometric.seed_everything(42)
    
    batch_size = 1
    dataset = "random"
    n = 1000
    size_train = 1000
    size_val = 100
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
        raise NotImplementedError("Dataset not implemented, Available: random, poisson1, poisson2")

    hidden_sizes_node, hidden_sizes_edge, message_passing_steps, augment_nodes, augment_edges, patiences, factors = [8], [14], [7], [True], [True], [10], [1]

    hyperparameter_combinations = list(itertools.product(hidden_sizes_node, hidden_sizes_edge, message_passing_steps, augment_nodes, patiences, factors))
    
    for i, (hidden_size_node, hidden_size_edge, message_passing_steps, augment_nodes, patience, factor) in enumerate(hyperparameter_combinations):
        print(f"Training model {i+1}/{len(hyperparameter_combinations)}")
        
        args = {"hidden_size_node": hidden_size_node, "hidden_size_edge": hidden_size_edge,  "message_passing_steps": message_passing_steps, 
                "skip_connections": True, "edge_connections": False, "augment_nodes": augment_nodes, 
                "augment_edges": augment_edges, "global_features": 0}
                
        model = MYPRECONDITIONER(**args)
        model.to(device)
        print(f"Number of params in model: {count_parameters(model)}")
        print()
        
        parameters = count_parameters(model)

        def custom_collate(batch):
            return Batch.from_data_list(batch)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

        initial_train = initial_train_loss(model, train_loader)
        initial_val = validate(model, validation_loader)
        print(f"Initial train loss: {initial_train:.4f} \t Initial validation loss: {initial_val:.4f}")
        
        best_val = float("inf")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        
        number_of_epochs = 150
        total_time = 0

        for epoch in range(number_of_epochs):
            running_loss = 0.0
            start_epoch = time.perf_counter()
            for data in train_loader:
                model.train()
                data = data.to(device)
                optimizer.zero_grad()
                L, U, reg = model(data)
                A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr.squeeze(), requires_grad=False)
                l = loss_function_sketched(L, U, A)
                l.backward()

                # Gradient clipping
                nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                running_loss += l.item()
                optimizer.step()
            
            train_loss = running_loss / len(train_loader)
            val_loss = validate(model, validation_loader)

            if val_loss < best_val:
                best_val = val_loss
                best_model_state = model.state_dict().copy()

            epoch_time = time.perf_counter() - start_epoch
            total_time += epoch_time

            if (epoch+1) in [50, 100]:
                print(f"validation loss: {best_val:.4f} \t time: {total_time:.4f}")

            print(f"Epoch {epoch+1} \t train loss: {train_loss:.4f} \t validation loss: {val_loss:.4f} \t time: {epoch_time:.4f}")

        folder = "results/" + f"{dataset}_" + f"size{n}_" + f"train{size_train }_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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
                "Last train loss": running_loss / len(train_loader),
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

if __name__ == "__main__":
    main()
