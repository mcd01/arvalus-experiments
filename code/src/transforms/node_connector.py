import torch
from torch_sparse import SparseTensor
from . import MultiNodeData

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

class NodeConnector(object):
    "Creates the adjacency matrices of graphs, based on relationships between services / groups."
    
    def __init__(self, use_synthetic_data : bool = False):        
        self.__conn_dict__ : dict = {}
        self.__edge_type_dict__ : dict = {"identity": 0.00}
        
        self.metric_count = 10 if use_synthetic_data else None
            
    @property
    def num_edge_types(self):
        return len(self.__edge_type_dict__)
    
    def __get_edge_type__(self, spec: str):
        if spec not in self.__edge_type_dict__:
            self.__edge_type_dict__[spec] = float(len(self.__edge_type_dict__))
        return self.__edge_type_dict__[spec]
        
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    def __str__(self):
        return f"{self.__class__.__name__}()"
    
    def __call__(self, data: MultiNodeData):
        
        if data.skip: return data
        
        identifiers = data["identifiers"]
        
        cluster_info = [
                (self.metric_count or data[f"x_{identifier}"].shape[0], data[f"component_{identifier}"], data[f"group_{identifier}"], data[f"node_{identifier}"]) for identifier in identifiers
                ] 
        ############# get connectivity information ###############
        self.__create_edge_index_for_cluster__(cluster_info, modeling_type="fine")
        self.__create_edge_index_for_cluster__(cluster_info, modeling_type="coarse")
        
        adj_cluster_fine, batch_cluster_fine = self.__conn_dict__[f"{str(cluster_info)}_fine"]
        adj_cluster_coarse, batch_cluster_coarse = self.__conn_dict__[f"{str(cluster_info)}_coarse"]
        
        ############### add properties ################
        data["batch_cluster_fine"] = batch_cluster_fine
        data["adj_cluster_fine"] = adj_cluster_fine
        
        data["batch_cluster_coarse"] = batch_cluster_coarse
        data["adj_cluster_coarse"] = adj_cluster_coarse
        
        return data
        
        
    def __create_edge_index_for_cluster__(self, cluster_info: list, modeling_type: str = "fine"):
        my_key = f"{str(cluster_info)}_{modeling_type}"
        
        if my_key in self.__conn_dict__:
            return # no need for further processing

        cluster_info_list = []
        custom_batch = None
        if modeling_type == "coarse":
            # first: metric count, second: component name, third: group name, fourth: node name
            cluster_info_list = [el[1:] for el in cluster_info]
            custom_batch = list(range(len(cluster_info_list)))
        else:
            custom_batch = []
            for i, el in enumerate(cluster_info):
                # first: metric count, second: component name, third: group name, fourth: node name
                cluster_info_list += (el[0] * [el[1:]])
                custom_batch += el[0] * [i]
                
        custom_batch = torch.tensor(custom_batch, dtype=torch.long)        

        edge_index_list = []
        edge_attr_list = []
        # each tuple has following structure: (component name, group name, node name)
        for i, tuple_i in enumerate(cluster_info_list):
            for j, tuple_j in enumerate(cluster_info_list):
                # identity (self-loop)
                if i == j:
                    edge_index_list.append((i, j))
                    edge_attr_list.append(self.__get_edge_type__("identity"))
                # system of tuple_i is hosted on tuple_j 
                elif tuple_i[2] == tuple_j[1]:
                    edge_index_list.append((i, j))
                    edge_attr_list.append(self.__get_edge_type__("guest-host"))
                # system of tuple_i is hosting tuple_j
                elif tuple_i[1] == tuple_j[2]:
                    edge_index_list.append((i, j))
                    edge_attr_list.append(self.__get_edge_type__("host-guest"))
                # both systems are from the same group    
                elif tuple_i[1] == tuple_j[1]:
                    edge_index_list.append((i, j))
                    edge_attr_list.append(self.__get_edge_type__(tuple_j[1]))
                # both systems are from distinct groups, but there is communication 
                elif tuple_j[1] in __node_relation_dict__.get(tuple_i[1], []):
                    edge_index_list.append((i, j))
                    edge_attr_list.append(self.__get_edge_type__(f"{tuple_i[1]}->{tuple_j[1]}"))
                # both systems are from distinct groups, but there is communication
                elif tuple_i[1] in __node_relation_dict__.get(tuple_j[1], []):
                    edge_index_list.append((i, j))
                    edge_attr_list.append(self.__get_edge_type__(f"{tuple_j[1]}->{tuple_i[1]}"))    

        edge_index = torch.tensor(
            edge_index_list, dtype=torch.long).t().contiguous()
        
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.double).contiguous()
            
        N = len(cluster_info_list)
        adj = SparseTensor.from_edge_index(edge_index, edge_attr=edge_attr, sparse_sizes=(N, N))

        self.__conn_dict__[my_key] = (adj, custom_batch)