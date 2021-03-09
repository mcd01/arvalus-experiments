import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import InstanceNorm
import numpy as np
import logging
from torch_geometric.nn.inits import glorot

class NodeModel(nn.Module):

    def __init__(self, args={}):
        super(NodeModel, self).__init__()

        self.input_dim = args.get("input_dim")
        self.num_conv_kernels = args.get("num_conv_kernels")
        self.hidden_dim = int(self.input_dim * self.num_conv_kernels)
        self.output_dim = args.get("hidden_dim")
        
        self.normal_dropout = args.get("normal_dropout", 0.5)
        
        self.conv_layers = nn.ModuleList([])
        
        in_channels : int = 1
        out_channels : int = 1
        stride : int = 1
        for idx, kernel_size in enumerate(np.arange(3, 20, 2)[:self.num_conv_kernels]):
            padding = int(kernel_size / 2)
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=False))
            logging.info(f"1D-Convolutional layer Nr. {idx+1}: in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}, padding={padding}, stride={stride}")
        
        self.conv_act = nn.ELU()
        
        pool_kernel_size : int = 3
        pool_padding : int = int(3 / 2)
        self.conv_pool = nn.MaxPool1d(pool_kernel_size, padding=pool_padding, stride=stride)
        logging.info(f"Max-Pooling Layer: kernel_size={pool_kernel_size}, padding={pool_padding}, stride={stride}")
        self.conv_norm = InstanceNorm(self.hidden_dim, affine=False)
        
        self.ll = nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        self.ll_act = nn.ELU()
        
        self.reset_parameters()
        
    def reset_parameters(self):
        glorot(self.ll.weight)
        
    def forward(self, x, batch):    

        ############## LOCAL FEATURE EXTRACTION #######################
        x = x.unsqueeze(1)        
        conv_res_list = []
        for layer in self.conv_layers:
            conv_res = layer(x)
            conv_res_list.append(conv_res)
        
        x = torch.cat(conv_res_list, dim = 1)    
        x = self.conv_act(x)
        x = self.conv_pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.conv_norm(x, batch)
        ###############################################################
        
        ################ PREPARE EMBEDDING ############################
        x = F.dropout(x, training=self.training, p=self.normal_dropout)
        
        x = self.ll(x)
        x = self.ll_act(x)
        ###############################################################
        
        return x    
        