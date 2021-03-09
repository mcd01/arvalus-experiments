import torch
from torch_geometric.data import Data


class MultiNodeData(Data):
    "Class for handling sub-graphs."
    
    def __init__(self, dictt: dict):
        super(MultiNodeData, self).__init__()
        self.__dict__ = dictt
        self.face = None
        self.edge_index = None   
        
    @property
    def skip(self):
        return self.__dict__.get("skip_me", False)
        
    def __inc__(self, key, value):
        if key == 'batch_cluster_coarse' or key == 'batch_cluster_fine':
            return self.__dict__["batch_cluster_coarse"].size(0)
        elif 'group_indices_' in key: # e.g. 'group_indices_cassandra', will allow to retrieve all graph nodes related to this group
            return self.__dict__["x"].size(0)
        elif 'group_batches_' in key and '_length' not in key: # e.g. 'group_batches_cassandra', will ensure correct batching of above procedure
            return self.__dict__[f'{key}_length']
        else:
            return super(MultiNodeData, self).__inc__(key, value) 


class DataCreator(object):
    def __call__(self, value_dict: dict):
        return MultiNodeData(value_dict)
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    def __str__(self):
        return f"{self.__class__.__name__}()"