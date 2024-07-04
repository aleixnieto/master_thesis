import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import aggr

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the default data type for torch tensors
ddtype = torch.float32
torch.set_default_dtype(ddtype)

############################
#          Layers          #
############################

class MLP(nn.Module):
    def __init__(self, width, layer_norm=False, activation="sigmoid", activate_final=False):
        """
        Initialize the MLP class.
        
        Parameters:
            width (list): List specifying the number of neurons in each layer.
            layer_norm (bool): Whether to apply layer normalization.
            activation (str): Activation function to use.
            activate_final (bool): Whether to apply activation to the final layer.
        """
        super().__init__()

        width = list(filter(lambda x: x > 0, width))
        assert len(width) >= 2, "Need at least one layer in the network!"

        layers = []
        for k in range(len(width) - 1):
            layers.append(nn.Linear(width[k], width[k + 1], bias=True))

            if k != (len(width) - 2) or activate_final:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "leakyrelu":
                    layers.append(nn.LeakyReLU())
                elif activation == "sigmoid":
                    layers.append(nn.Sigmoid())
                elif activation == "exp":
                    layers.append(nn.ELU())
                else:
                    raise NotImplementedError(f"Activation '{activation}' not implemented")
        
        if layer_norm:
            layers.append(nn.LayerNorm(width[-1]))

        self.m = nn.Sequential(*layers)

    def forward(self, x):
        return self.m(x)


class GraphNet(nn.Module):
    def __init__(self, node_features, edge_features, edge_2_features=0, global_features=0,
                 layer_norm=False, hidden_size_node=0, hidden_size_edge=0, aggregate="mean", activation="relu",
                 skip_connection=False, edge_connection=False):
        """
        Initialize the GraphNet class.

        Parameters:
            node_features (int): Number of node features.
            edge_features (int): Number of edge features.
            edge_2_features (int): Number of additional edge features.
            global_features (int): Number of global features.
            layer_norm (bool): Whether to apply layer normalization.
            hidden_size_node (int): Number of hidden units in node MLP.
            hidden_size_edge (int): Number of hidden units in edge MLP.
            aggregate (str): Aggregation method ('sum', 'mean', 'max').
            activation (str): Activation function.
            skip_connection (bool): Whether to use skip connections.
            edge_connection (bool): Whether to use edge connections.
        """
        super().__init__()

        # Different aggregation functions
        if aggregate == "sum":
            self.aggregate = aggr.SumAggregation()
        elif aggregate == "mean":
            self.aggregate = aggr.MeanAggregation()
        elif aggregate == "max":
            self.aggregate = aggr.MaxAggregation()
        else:
            raise NotImplementedError(f"Aggregation '{aggregate}' not implemented")

        self.global_aggregate = aggr.MeanAggregation()

        add_edge_fs = 1 if skip_connection else 0
        add_edge = 1 if edge_connection else 0

        self.edge_block = MLP([global_features + edge_features + 2 * node_features + add_edge_fs + add_edge,
                               hidden_size_edge, edge_features], layer_norm=layer_norm, activation=activation)

        if edge_2_features > 0:
            self.mp_edge_block = MLP([global_features + edge_2_features + 2 * node_features + add_edge_fs + add_edge,
                                      hidden_size_edge, edge_2_features], layer_norm=layer_norm, activation=activation)
        else:
            self.mp_edge_block = None

        self.node_block = MLP([global_features + edge_features + edge_2_features + node_features,
                               hidden_size_node, node_features], layer_norm=layer_norm, activation=activation)

        self.global_block = None
        if global_features > 0:
            self.global_block = MLP([edge_features + edge_2_features + node_features + global_features,
                                     hidden_size_node, global_features], layer_norm=layer_norm, activation=activation)

    def forward(self, x, edge_index, edge_attr, g=None, edge_index_2=None, edge_attr_2=None):
        row, col = edge_index

        x = x.float()
        edge_attr = edge_attr.float()

        if self.global_block is not None:
            assert g is not None, "Need global features for global block"

            edge_embedding = self.edge_block(torch.cat([torch.ones(x[row].shape[0], 1, device=x.device) * g,
                                                        x[row], x[col], edge_attr], dim=1))
            aggregation = self.aggregate(edge_embedding, row)

            if edge_index_2 is not None:
                mp = self.mp_edge_block(torch.cat([torch.ones(x[row].shape[0], 1, device=device) * g, x[row], x[col], edge_attr_2], dim=1))
                agg_features = torch.cat([torch.ones(x.shape[0], 1, device=device) * g, x, aggregation, self.aggregate(mp, row)], dim=1)
                mp_global_aggr = torch.cat([g, self.aggregate(mp)], dim=1)
            else:
                agg_features = torch.cat([torch.ones(x.shape[0], 1, device=x.device) * g, x, aggregation], dim=1)
                mp_global_aggr = g

            node_embeddings = self.node_block(agg_features)

            edge_aggregation_global = self.global_aggregate(edge_embedding)
            node_aggregation_global = self.global_aggregate(node_embeddings)

            global_embeddings = self.global_block(torch.cat([node_aggregation_global, edge_aggregation_global, mp_global_aggr], dim=1))

            return edge_embedding, node_embeddings, global_embeddings

        else:
            edge_embedding = self.edge_block(torch.cat([x[row], x[col], edge_attr], dim=1))
            aggregation = self.aggregate(edge_embedding, row)

            if edge_index_2 is not None:
                mp = self.mp_edge_block(torch.cat([x[row], x[col], edge_attr_2], dim=1))
                mp_aggregation = self.aggregate(mp, row)
                agg_features = torch.cat([x, aggregation, mp_aggregation], dim=1)
            else:
                agg_features = torch.cat([x, aggregation], dim=1)

            node_embeddings = self.node_block(agg_features)

            return edge_embedding, node_embeddings, None


class GraphLayer(nn.Module):
    def __init__(self, skip_connections, edge_connections, edge_features, node_features, global_features, hidden_size_node, hidden_size_edge, **kwargs):
        """
        Initialize the GraphLayer class.

        Parameters:
            skip_connections (bool): Whether to use skip connections.
            edge_connections (bool): Whether to use edge connections.
            edge_features (int): Number of edge features.
            node_features (int): Number of node features.
            global_features (int): Number of global features.
            hidden_size_node (int): Number of hidden units in node MLP.
            hidden_size_edge (int): Number of hidden units in edge MLP.
        """
        super().__init__()

        self.l = GraphNet(node_features=node_features, edge_features=edge_features, global_features=global_features,
                          hidden_size_node=hidden_size_node, hidden_size_edge=hidden_size_edge,
                          skip_connection=skip_connections, edge_connection=edge_connections, aggregate="sum")

    def forward(self, x, edge_index, edge_attr, global_features):
        edge_embedding, node_embeddings, global_features = self.l(x, edge_index, edge_attr, g=global_features)
        return edge_embedding, node_embeddings, global_features


############################
#     MYPRECONDITIONER     #
############################

class MYPRECONDITIONER(nn.Module):
    def __init__(self, **kwargs):
        """
        Initialize the MYPRECONDITIONER class.

        Parameters:
            kwargs (dict): Keyword arguments containing configuration parameters.
        """
        super().__init__()

        self.global_features = kwargs["global_features"]
        self.hidden_size_node = kwargs["hidden_size_node"]
        self.hidden_size_edge = kwargs["hidden_size_edge"]

        self.augment_node_features = kwargs["augment_nodes"]
        self.augment_edge_features = kwargs["augment_edges"]

        num_node_features = 8 if self.augment_node_features else 1
        num_node_edges = 2 if self.augment_edge_features else 1
        message_passing_steps = kwargs["message_passing_steps"]

        self.skip_connections = kwargs["skip_connections"]
        self.edge_connections = kwargs["edge_connections"]

        self.mps = nn.ModuleList()
        for l in range(message_passing_steps):
            self.mps.append(GraphLayer(skip_connections=(l != 0 and self.skip_connections),
                                       edge_connections=(l != 0 and self.edge_connections),
                                       edge_features=num_node_edges, node_features=num_node_features,
                                       global_features=self.global_features, hidden_size_node=self.hidden_size_node,
                                       hidden_size_edge=self.hidden_size_edge))

    def forward(self, data):
        a_edges = data.edge_attr.clone()

        if self.augment_node_features:
            data = augment_features(data)

        if self.augment_edge_features:
            data = augment_edge_features(data)

        edge_embedding = data.edge_attr
        node_embedding = data.x
        l_index = data.edge_index

        row, col = data.edge_index
        lower_mask = row > col
        upper_mask = row < col

        additional_edge_feature = torch.zeros_like(a_edges)
        additional_edge_feature[lower_mask] = -1
        additional_edge_feature[upper_mask] = 1

        if self.global_features > 0:
            global_features = torch.zeros((1, self.global_features), device=data.x.device, requires_grad=False)
        else:
            global_features = None

        for i, layer in enumerate(self.mps):
            if i != 0 and self.skip_connections:
                edge_embedding = torch.cat([edge_embedding, a_edges], dim=1)

            if i != 0 and self.edge_connections:
                edge_embedding = torch.cat([edge_embedding, additional_edge_feature], dim=1)

            edge_embedding, node_embedding, global_features = layer(node_embedding, l_index, edge_embedding, global_features)

        return self.transform_output_matrixLU(a_edges, node_embedding, l_index, edge_embedding)

    def transform_output_matrixLU(self, a_edges, node_x, edge_index, edge_values, tolerance=1e-10):
        """
        Transform the output into L and U matrices.

        Parameters:
            a_edges (Tensor): Original edge attributes.
            node_x (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            edge_values (Tensor): Edge values.
            tolerance (float): Tolerance for small values.

        Returns:
            tuple: Lower and upper matrices, and L1 norm.
        """
        a_edges = a_edges.float()
        edge_values = edge_values.float()
        tolerance = torch.tensor(tolerance, dtype=torch.float, device=edge_index.device)

        lower_mask = edge_index[0] >= edge_index[1]
        upper_mask = edge_index[0] <= edge_index[1]
        diag_mask = edge_index[0] == edge_index[1]

        lower_indices = edge_index[:, lower_mask]
        lower_values = edge_values[lower_mask][:, 0].squeeze()

        upper_indices = edge_index[:, upper_mask]
        upper_values = edge_values[upper_mask][:, 1].squeeze()

        lower_values[diag_mask[lower_mask]] = torch.where(lower_values[diag_mask[lower_mask]] < tolerance.abs(), 
                                                          a_edges[diag_mask].squeeze(), 
                                                          lower_values[diag_mask[lower_mask]].squeeze())

        upper_values[diag_mask[upper_mask]] = torch.where(upper_values[diag_mask[upper_mask]] < tolerance.abs(), 
                                                          a_edges[diag_mask].squeeze(), 
                                                          upper_values[diag_mask[upper_mask]].squeeze())

        lower_matrix = torch.sparse_coo_tensor(lower_indices, lower_values.squeeze(),
                                               size=(node_x.size()[0], node_x.size()[0]))
        upper_matrix = torch.sparse_coo_tensor(upper_indices, upper_values.squeeze(),
                                               size=(node_x.size()[0], node_x.size()[0]))

        l1_l = torch.sum(torch.abs(lower_values)) / len(lower_values)
        l1_u = torch.sum(torch.abs(upper_values)) / len(upper_values)
        l1 = (l1_l + l1_u) / 2

        return lower_matrix, upper_matrix, l1


############################
#         HELPERS          #
############################

def augment_features(data):
    """
    Augment node features with additional information.

    Parameters:
        data (Data): Input data.

    Returns:
        Data: Data with augmented node features.
    """
    data.x = torch.arange(data.x.size()[0], dtype=ddtype, device=data.x.device).unsqueeze(1)

    data = torch_geometric.transforms.LocalDegreeProfile()(data)
    data.x = data.x.to(ddtype)

    row, col = data.edge_index
    diag = (row == col)
    diag_elem = torch.abs(data.edge_attr[diag])
    non_diag_elem = data.edge_attr.clone()
    non_diag_elem[diag] = 0

    row_sums = aggr.SumAggregation()(torch.abs(non_diag_elem), row)
    alpha = diag_elem / row_sums
    row_dominance_feature = alpha / (alpha + 1)
    row_dominance_feature = torch.nan_to_num(row_dominance_feature, nan=1.0)

    row_max = aggr.MaxAggregation()(torch.abs(non_diag_elem), row)
    alpha = diag_elem / row_max
    row_decay_feature = alpha / (alpha + 1)
    row_decay_feature = torch.nan_to_num(row_decay_feature, nan=1.0)

    data.x = torch.cat([data.x, row_dominance_feature, row_decay_feature], dim=1)

    return data


def augment_edge_features(data):
    """
    Augment edge features with additional information.

    Parameters:
        data (Data): Input data.

    Returns:
        Data: Data with augmented edge features.
    """
    existing_edge_feature = data.edge_attr.clone()
    augmented_edge_features = torch.cat([data.edge_attr, existing_edge_feature], dim=1)
    data.edge_attr = augmented_edge_features

    return data
