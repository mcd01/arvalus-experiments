import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import InstanceNorm
from torch_geometric.utils import dropout_adj, add_remaining_self_loops
from torch_sparse import SparseTensor
import logging


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size=None, sublayer=None, act=None):
        super(SublayerConnection, self).__init__()

        if size is None:
            raise ValueError("Size must be a positive integer.")

        if sublayer is None:
            raise ValueError("Sublayer must be defined.")

        self.sublayer = sublayer
        self.size = size
        
        self.norm = InstanceNorm(self.size, affine=False)

        self.act = act

        self.skip = nn.Identity()

        logging.info(
            f"SLC: act={self.act.__class__.__name__}, size={self.size}, norm={self.norm.__class__.__name__}")

    def forward(self, x, batch, adj=None, dropout_adj_prob: float = None):
        "Apply residual connection to any sublayer with the same size."
        residual = self.skip(x)
        
        x_det = x

        layer_out = None

        if adj is not None:
            adj = adj if dropout_adj_prob is None else self.__dropout_adj__(adj, dropout_adj_prob)
            layer_out = self.sublayer(x_det, adj)
        else:
            layer_out = self.sublayer(x_det)

        layer_out = self.act(layer_out) if self.act is not None else layer_out

        return self.norm(residual + layer_out, batch)

    def __dropout_adj__(self, sparse_adj: SparseTensor, dropout_adj_prob: float):
        # number of nodes
        N = sparse_adj.size(0)
        # sparse adj matrix to dense adj matrix
        row, col, edge_attr = sparse_adj.coo()
        edge_index = torch.stack([row, col], dim=0)
        # dropout adjacency matrix -> generalization
        edge_index, edge_attr = dropout_adj(edge_index, 
                                            edge_attr=edge_attr, 
                                            p=dropout_adj_prob,
                                            force_undirected=True,
                                            training=self.training)
        # because dropout removes self-loops (due to force_undirected=True), make sure to add them back again
        edge_index, edge_attr = add_remaining_self_loops(edge_index,
                                                         edge_weight=edge_attr,
                                                         fill_value=0.00,
                                                         num_nodes=N)
        # dense adj matrix to sparse adj matrix
        sparse_adj = SparseTensor.from_edge_index(edge_index, 
                                                  edge_attr=edge_attr, 
                                                  sparse_sizes=(N, N))

        return sparse_adj
