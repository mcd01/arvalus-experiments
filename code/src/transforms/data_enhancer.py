import torch
from torch_sparse import SparseTensor
from . import MultiNodeData, DataCreator

__node_relation_dict__ : dict = {
    "backend": ["load-balancer"],
    "load-balancer": ["backend"],
    "bono": ["sprout", "ralf"],
    "homer": ["sprout", "ellis", "cassandra"],
    "ellis": ["homer", "homestead", "homesteadprov"],
    "sprout": ["bono", "homer", "homestead", "ralf", "astaire", "chronos"],
    "astaire": ["sprout", "ralf", "homestead"],
    "cassandra": ["homer", "homestead"],
    "chronos": ["ralf", "sprout"],
    "ralf": ["bono", "sprout", "chronos", "astaire"],
    "homestead": ["sprout", "cassandra", "astaire", "homesteadprov"],
    "homesteadprov": ["ellis", "homestead"]
}   

class DataEnhancer(object):
    "Enhance data objects on the fly, add information for simplified batching."
    
    def __init__(self, group_meta: dict):        
        
        self.__group_index_dict__ : dict = {}
        self.__group_batch_dict__ : dict = {}
        
        self.data_creator = DataCreator()
        self.group_meta = group_meta
        
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    def __str__(self):
        return f"{self.__class__.__name__}()"
        
    def __call__(self, data: MultiNodeData):
        identifiers = data["identifiers"]
        
        cluster_info = [
                (data[f"x_{identifier}"].shape[0], data[f"component_{identifier}"], data[f"group_{identifier}"], data[f"node_{identifier}"]) for identifier in identifiers
                ]

        ############# get group indices information #################
        self.__create_groups_for_cluster__(cluster_info, data)
        
        group_index_dict : dict = self.__group_index_dict__[f"{str(cluster_info)}"]
        group_batch_dict : dict = self.__group_batch_dict__[f"{str(cluster_info)}"]
        ############### add properties to new dict ################
        # setup new dictionary, only transfer a handful of properties
        new_dict : dict = {}
        
        new_dict["num_cluster_nodes"] = len(identifiers)
        new_dict["node_ids"] = identifiers
        new_dict["node_groups"] = [data[f"group_{identifier}"] for identifier in identifiers]
        
        new_dict["sequence_transitional"] = data["sequence_transitional"]
        new_dict["sequence_id"] = data["sequence_id"]
        new_dict["file_id"] = data["file_id"]
                
        new_dict["batch_cluster_fine"] = data["batch_cluster_fine"]
        new_dict["adj_cluster_fine"] = data["adj_cluster_fine"]
        
        new_dict["batch_cluster_coarse"] = data["batch_cluster_coarse"]
        new_dict["adj_cluster_coarse"] = data["adj_cluster_coarse"]
        
        new_dict["x"] = torch.cat([data[f"x_{identifier}"] for identifier in identifiers], dim=0) # stack node features
        new_dict["y"] = data["y"]
        new_dict["y_full"] = data["y_full"]
                        
        # transfer properties from group dict 
        for key, value in group_index_dict.items():
            new_dict[key] = value
        # transfer properties from batch dict 
        for key, value in group_batch_dict.items():
            new_dict[key] = value
            
        ################ Empty Tensors for correct batching ########################
        local_groups = set(new_dict["node_groups"])
        global_groups = set(list(self.group_meta.keys()))
        diff_groups = list(global_groups.difference(local_groups))
        for diff_group in diff_groups:
            new_dict[f"group_indices_{diff_group}"] = torch.tensor([], dtype=torch.long)
            new_dict[f"group_batches_{diff_group}"] = torch.tensor([], dtype=torch.long)
            new_dict[f"group_batches_{diff_group}_length"] = 0
        ###########################################################
        return self.data_creator(new_dict)
    
    
    def __create_groups_for_cluster__(self, cluster_info: list, data: MultiNodeData):
        my_key = f"{str(cluster_info)}"
        
        if my_key in self.__group_index_dict__ and my_key in self.__group_batch_dict__:
            return # no need for further processing
        
        # first: metric count, second: component name, third: group name, fourth: node name
        metric_counts : list = [el[0] for el in cluster_info]
        groups : list = [el[2] for el in cluster_info]
        offsets : torch.LongTensor = torch.tensor([0] + metric_counts[:-1], dtype=torch.long).cumsum(dim=0, dtype=torch.long)
        
        group_index_dict : dict = {}
        group_batch_dict : dict = {}
        # build up indices and batches tensor for each group
        for (count, group, offset) in zip(metric_counts, groups, offsets):
            # for indices
            group_index_dict_key : str = f"group_indices_{group}"
            if group_index_dict_key not in group_index_dict:
                group_index_dict[group_index_dict_key] = []
            group_index_dict_el = group_index_dict[group_index_dict_key]
            group_index_dict_el.append(torch.arange(offset, offset + count, dtype=torch.long))
            group_index_dict[group_index_dict_key] = group_index_dict_el
            # for batches
            group_batch_dict_key : str = f"group_batches_{group}"
            if group_batch_dict_key not in group_batch_dict:
                group_batch_dict[group_batch_dict_key] = []    
            group_batch_dict_el = group_batch_dict[group_batch_dict_key]
            group_batch_dict_el.append(torch.ones(count, dtype=torch.long) * len(group_batch_dict_el))
            group_batch_dict[group_batch_dict_key] = group_batch_dict_el
            group_batch_dict[f"{group_batch_dict_key}_length"] = len(group_batch_dict_el)
        
        # concatenate tensors in lists
        for key, value in group_index_dict.items():
            group_index_dict[key] = torch.cat(value)
            
        for key, value in group_batch_dict.items():
            if isinstance(value, list):
                group_batch_dict[key] = torch.cat(value)
            
        self.__group_index_dict__[my_key] = group_index_dict
        self.__group_batch_dict__[my_key] = group_batch_dict
        