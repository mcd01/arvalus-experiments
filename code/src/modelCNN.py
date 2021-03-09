import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import InstanceNorm, global_max_pool
import logging
import os
from src.layers import SublayerConnection
from src.model_shared import NodeModel
from torch_geometric.nn.inits import glorot, zeros


def init_weights(m):
    if type(m) == nn.Linear:
        glorot(m.weight)


class ClusterModel(nn.Module):
    def __init__(self, args={}):
        super(ClusterModel, self).__init__()

        self.node_meta = args.get('node_meta')
        self.group_meta = args.get("group_meta")
        
        input_dim = args.get('input_dim')
        num_conv_kernels = args.get("num_conv_kernels")
        
        self.normal_dropout = args.get("normal_dropout", 0.5)
        
        logging.info(f"Normal dropout: {self.normal_dropout}")
        
        self.node_model_output_dim = int(input_dim * num_conv_kernels)
        self.cluster_model_hidden_dim = args.get('hidden_dim')
        self.node_num_classes = args.get("node_num_classes")
        self.graph_num_classes = args.get("graph_num_classes")
        
        self.device = args.get("device")
        
        ############## LOCAL FEATURE EXTRACTION #######################
        self.node_model_dict = nn.ModuleDict({})
        for idx, (group, conf) in enumerate(self.group_meta.items()):
            logging.info(f"Initializing sub-model (Nr. {idx+1}) for group: {group}")
            self.node_model_dict[group] = NodeModel({**args, **conf})
        logging.info(f"Number of sub-models: {len(self.node_model_dict)}")    

        self.emb_norm = InstanceNorm(self.cluster_model_hidden_dim, affine=False)

        ################ CLASSIFICATION ###############################
        self.ll2 = nn.Linear(self.cluster_model_hidden_dim, self.node_num_classes)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        glorot(self.ll2.weight)
        zeros(self.ll2.bias)

    @property
    def all_params(self):
        return sum(p.numel() for p in self.parameters())
    
    @property
    def all_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, batch):
        
        batch_cluster_coarse = batch.batch_cluster_coarse_batch
        pool_cluster_fine = batch.batch_cluster_fine
        
        node_features = batch.x
        new_node_features = None
        ############## LOCAL FEATURE EXTRACTION #######################
        for group, node_model in self.node_model_dict.items():
            # get node indices for this group
            x_indices = batch[f"group_indices_{group}"]
            x_batches = batch[f"group_batches_{group}"]
            # if group not present in this batch, skip group
            if x_indices.size(0) == 0: 
                logging.info(f"Batch has no nodes of group '{group}'.")
                continue            
            # extract features using corresponding sub-model            
            response = node_model(node_features[x_indices, :], x_batches) 
            if new_node_features is None:
                new_node_features = torch.zeros(node_features.size(0), 
                                                self.cluster_model_hidden_dim, 
                                                device=node_features.device,
                                                dtype=node_features.dtype)
            new_node_features[x_indices, :] = response

        node_features = new_node_features
        ################## GRAPH COARSENING ###########################
        node_embeddings = global_max_pool(node_features, pool_cluster_fine)
        ###############################################################
        node_embeddings = self.emb_norm(node_embeddings, batch_cluster_coarse)
        ###############################################################
                
        ################# REGULARIZATION ##############################
        node_embeddings = F.dropout(node_embeddings, training=self.training, p=self.normal_dropout)
        ###############################################################
        
        ################ CLASSIFICATION ###############################
        classification_fine = self.ll2(node_embeddings)
        
        return classification_fine, classification_fine 
        ###############################################################
