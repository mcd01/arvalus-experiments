import torch
import random
from src.transforms import MultiNodeData
import collections
import dill
import os
from src.utils import create_dirs

class Synthesizer(object):
    "Creates the synthetic data for our experiments."
    
    def __init__(self, path_to_dir : str, window_width : int = None, num_nodes : int = 0, use_synthetic_data : bool = False, transform_key=""):
        
        self.path_to_dir = path_to_dir
        self.is_fitted = False
        
        self.use_synthetic_data = use_synthetic_data
        self.window_width = window_width
        self.num_nodes = num_nodes
        
        self.normal_metric_count = 10
        self.metric_count = 5
        
        self.__anomalies__ = ["anomaly1", "anomaly2"]
        self.__labels__ = ["normal"] + self.__anomalies__ 
        
        self.__types__ = ["local", "neighborhood_down", "neighborhood_up", "adversary"]
        self.__type_weights__ = [0.7, 0.1, 0.1, 0.1]
        
        self.__neighbor_func__ = lambda nn: random.randint(3, 7)
        
        self.__synthetic_dict__ : dict = {}
            
        self.__adversary_dict__ : dict = {
            "anomaly1": lambda: "anomaly2",
            "anomaly2": lambda: "anomaly1"
        }
                
        self.__func_dict__ : dict = {
            
            "normal": lambda: torch.abs(torch.normal(0, 0.2, size=(self.normal_metric_count, self.window_width))),
            
            "anomaly1_local": lambda: torch.abs(torch.normal(0, 0.1, size=(self.metric_count, self.window_width))).mean(dim=0),
            "anomaly1_neighborhood_down": lambda: torch.abs(torch.normal(0, 0.17, size=(self.metric_count, self.window_width))).mean(dim=0),
            "anomaly1_neighborhood_up": lambda: torch.abs(torch.normal(0, 0.17, size=(self.metric_count, self.window_width))).mean(dim=0),
            "anomaly1_adversary": lambda: torch.abs(torch.normal(0, 0.1, size=(self.metric_count, self.window_width))).mean(dim=0),
            
            "anomaly2_local": lambda: torch.abs(torch.normal(0, 0.3, size=(self.metric_count, self.window_width))).mean(dim=0),
            "anomaly2_neighborhood_down": lambda: torch.abs(torch.normal(0, 0.23, size=(self.metric_count, self.window_width))).mean(dim=0),
            "anomaly2_neighborhood_up": lambda: torch.abs(torch.normal(0, 0.23, size=(self.metric_count, self.window_width))).mean(dim=0),
            "anomaly2_adversary": lambda: torch.abs(torch.normal(0, 0.3, size=(self.metric_count, self.window_width))).mean(dim=0)
            
        }
        
        
        self.__id__ = Synthesizer.get_id(use_synthetic_data=self.use_synthetic_data,
                                         num_nodes=self.num_nodes,
                                         window_width=self.window_width,
                                         transform_key=transform_key)

    
    def __repr__(self):
        return f"{self.__class__.__name__}(num_nodes={self.num_nodes}, use_synthetic_data={self.use_synthetic_data}, window_width={self.window_width}, anomalies={self.__anomalies__})"
    
    def __str__(self):
        return f"{self.__class__.__name__}(num_nodes={self.num_nodes}, uuse_synthetic_data={self.use_synthetic_data}, window_width={self.window_width}, anomalies={self.__anomalies__})"
    
    @staticmethod
    def get_id(*args, **kwargs):    
        sorted_kwargs = collections.OrderedDict(sorted(kwargs.items()))
        return ", ".join(f"{key}={value}" for key, value in sorted_kwargs.items())
    
    @classmethod
    def getInstance(cls, path_to_dir : str, **kwargs):
        path_to_file = os.path.join(path_to_dir, "synthesizer.pkl")
        
        if os.path.exists(path_to_file):
            with open(path_to_file, 'rb') as dill_file:
                obj = dill.load(dill_file)
                if obj.__id__ == Synthesizer.get_id(**kwargs):
                    return obj
                else:
                    return Synthesizer(path_to_dir, **kwargs) 
        else:
            return Synthesizer(path_to_dir, **kwargs)
    
    def save(self):
        create_dirs(self.path_to_dir)
        with open(os.path.join(self.path_to_dir, "synthesizer.pkl"), "wb") as dill_file:
            dill.dump(self, dill_file)
    
    
    def __get_synthetic_dict_entries__(self, data: MultiNodeData, sequence_id, file_id):
        ##### determine characteristics of this sequence #####
        if f"sequence-{sequence_id}" not in self.__synthetic_dict__:

            target_node : int = torch.argmax(torch.max(data["y_full"], dim=1)[0]).item()
            anomaly_idx : int = torch.argmax(data["y"]).item()

            self.__synthetic_dict__[f"sequence-{sequence_id}"] = {
                    "anomaly": self.__anomalies__[anomaly_idx], # what is the type of this sequence?
                    "adversary_anomaly": self.__adversary_dict__[self.__anomalies__[anomaly_idx]](), # which adversary?
                    "target_node": target_node # in which node shall we inject an anomaly?                    
            }
            
        if f"file-{file_id}" not in self.__synthetic_dict__:
            
            sequence_dict = self.__synthetic_dict__[f"sequence-{sequence_id}"]
            target_node = sequence_dict["target_node"]
            
            adj_cluster_coarse = data["adj_cluster_coarse"]
            row, col, _ = adj_cluster_coarse.coo()
            
            neighbor_nodes = row[torch.bitwise_and(col == target_node, row != target_node)].tolist()
            
            # randomly select only a subset
            neighbor_nodes = random.sample(neighbor_nodes, self.__neighbor_func__(neighbor_nodes))
            neighbor_nodes_dict = {k:(random.randint(0, self.metric_count - 1)) for k in neighbor_nodes}
            
            self.__synthetic_dict__[f"file-{file_id}"] = {
                    "target_row": random.randint(0, self.metric_count - 1), # in which metric shall we inject an anomaly?
                    "neighbor_nodes_dict": neighbor_nodes_dict, # what are the neighbor nodes?
                    "type": random.choices(self.__types__, weights=self.__type_weights__, k=1)[0], # which type to choose?
            }
            
        return self.__synthetic_dict__[f"sequence-{sequence_id}"], self.__synthetic_dict__[f"file-{file_id}"]
    
    
    def synthesize(self, data: MultiNodeData):
        if not self.use_synthetic_data: return data
        
        identifiers: list = data["identifiers"]
        sequence_id : int = data["sequence_group"] # at this stage its called sequence_group, later sequence_id
        file_id : int = data["file_idx"]
        
        sequence_dict, file_dict = self.__get_synthetic_dict_entries__(data, sequence_id, file_id)
        
        anomaly: str = sequence_dict.get("anomaly")
        adversary_anomaly: str = sequence_dict.get("adversary_anomaly")
        target_node: str = sequence_dict.get("target_node")
            
        target_row: int = file_dict.get("target_row")
        neighbor_nodes_dict: list = file_dict.get("neighbor_nodes_dict")
        ttype: str = file_dict.get("type")
        
        ##### create synthetic values, overwrite #####
        for idx, identifier in enumerate(identifiers):
            # prepare target_node
            if idx == target_node:
                data[f"x_{identifier}"] = self.__func_dict__["normal"]()
                data[f"x_{identifier}"][target_row, :] = self.__func_dict__[f"{anomaly}_{ttype}"]()
            # prepare neighbor nodes + safety condition
            elif idx in neighbor_nodes_dict.keys() and idx != target_node:
                if ttype == "local":
                    data[f"x_{identifier}"] = self.__func_dict__["normal"]()               
                elif "neighborhood" in ttype:
                    data[f"x_{identifier}"] = self.__func_dict__["normal"]()
                    data[f"x_{identifier}"][neighbor_nodes_dict[idx], :] = self.__func_dict__[f"{anomaly}_local"]()            
                elif ttype == "adversary":
                    data[f"x_{identifier}"] = self.__func_dict__["normal"]()
                    data[f"x_{identifier}"][neighbor_nodes_dict[idx], :] = self.__func_dict__[f"{adversary_anomaly}_local"]()
            # rest is normal
            else:
                data[f"x_{identifier}"] = self.__func_dict__["normal"]()
        
        return data
    
    
    def __call__(self, data : MultiNodeData):
        if not self.use_synthetic_data: return data
        
        identifiers: list = data["identifiers"]
        sequence_id : int = data["sequence_id"] # from here on: sequence_id
        file_id : int = data["file_idx"]
        
        sequence_dict, file_dict = self.__get_synthetic_dict_entries__(data, sequence_id, file_id)
        
        anomaly: str = sequence_dict.get("anomaly")
        adversary_anomaly: str = sequence_dict.get("adversary_anomaly")
        target_node: str = sequence_dict.get("target_node")
            
        neighbor_nodes_dict: list = file_dict.get("neighbor_nodes_dict")
        ttype: str = file_dict.get("type")
                
        # overwrite labeling an anomalies
        for idx, identifier in enumerate(identifiers):
            # prepare target_node
            if idx == target_node:
                data[f"y_{identifier}"] = torch.tensor([int(self.__labels__.index(anomaly) == el) for el in range(len(self.__labels__))], dtype=torch.long).reshape(1, -1) # encode multi label
                data[f"anomaly_{identifier}"] = anomaly
            # prepare neighbor nodes + safety condition
            elif idx in neighbor_nodes_dict.keys() and idx != target_node:
                if ttype == "local":
                    data[f"y_{identifier}"] = torch.tensor([int(0 == el) for el in range(len(self.__labels__))], dtype=torch.long).reshape(1, -1) # encode multi label
                    data[f"anomaly_{identifier}"] = "normal"
                elif "neighborhood" in ttype:
                    data[f"y_{identifier}"] = torch.tensor([int(self.__labels__.index(anomaly) == el) for el in range(len(self.__labels__))], dtype=torch.long).reshape(1, -1) # encode multi label
                    data[f"anomaly_{identifier}"] = anomaly    
                elif ttype == "adversary":
                    data[f"y_{identifier}"] = torch.tensor([int(self.__labels__.index(adversary_anomaly) == el) for el in range(len(self.__labels__))], dtype=torch.long).reshape(1, -1) # encode multi label
                    data[f"anomaly_{identifier}"] = adversary_anomaly
            # rest is normal
            else:
                data[f"y_{identifier}"] = torch.tensor([int(0 == el) for el in range(len(self.__labels__))], dtype=torch.long).reshape(1, -1) # encode multi label
                data[f"anomaly_{identifier}"] = "normal"
                
        class_labels = [data[f"y_{identifier}"] for identifier in identifiers]
        data["y_full"] = torch.cat(class_labels, dim=0)
        
        return data