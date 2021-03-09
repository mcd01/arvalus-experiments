from typing import Optional
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
import logging


from typing import Optional
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr
from torch_geometric.utils.num_nodes import maybe_num_nodes

def softmax(src: Tensor, index: Optional[Tensor], ptr: Optional[Tensor] = None,
            num_nodes: Optional[int] = None) -> Tensor:
    out = src
    if src.numel() > 0:
        out = out - src.max()
    out = out.exp()

    if ptr is not None:
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
    else:
        raise NotImplementedError

    return out / (out_sum + 1e-16)


class GNNConvCustom(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, num_edge_types: int = 0, bias = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GNNConvCustom, self).__init__(**kwargs)
        
        logging.info(f"Using custom {self.__class__.__name__}")

        self.num_edge_types = num_edge_types

        self.att_l = Parameter(torch.Tensor(1, out_channels))
        self.att_m = Parameter(torch.Tensor(1, self.num_edge_types))
        self.att_r = Parameter(torch.Tensor(1, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_m)
        glorot(self.att_r)


    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """"""
        row, col, edge_attr = edge_index.coo()
        edge_index = torch.stack([row, col], dim=0)
                
        # calculate edge weights [edge_attr -> edge_weight]
        encodings = F.one_hot(torch.flatten(edge_attr).long(), num_classes=self.num_edge_types).double()     
        alpha_m = (encodings * self.att_m).sum(dim=-1).reshape(-1, 1)
        
        alpha_l = (x * self.att_l).sum(dim=-1).reshape(-1, 1)
        alpha_r = (x * self.att_r).sum(dim=-1).reshape(-1, 1)

        # propagate_type: (x: Tensor, x_norm: Tensor)
        return self.propagate(edge_index, 
                              x=x,
                              alpha=(alpha_l, alpha_r),
                              alpha_m=alpha_m,
                              size=None)


    def message(self, x_j, alpha_i, alpha_j, alpha_m, index, ptr, size_i) -> Tensor:
        
        edge_weight = alpha_i + alpha_m + alpha_j
        edge_weight = F.elu(edge_weight)
        edge_weight = softmax(edge_weight, index, ptr, size_i)
        
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)