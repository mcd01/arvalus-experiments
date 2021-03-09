import torch
from . import MultiNodeData


class NodeSelector(object):
    "Handles selection of only specific nodes."
    
    def __init__(self, include_nodes : list = []):
        self.include_nodes = include_nodes
        
        self.__common_prefixes__ = [
            "x",
            "y",
            "label_list",
            "anomaly",
            "headers",
            "node", 
            "group",
            "component",
            "host"
        ]
        
    def __repr__(self):
        return f"{self.__class__.__name__}(include_nodes={self.include_nodes})"
    
    def __str__(self):
        return f"{self.__class__.__name__}(include_nodes={self.include_nodes})"
        
    def __call__(self, data : MultiNodeData):
        
        if len(self.include_nodes):
            identifiers : list = data["identifiers"]
            for identifier in identifiers:
                if identifier not in self.include_nodes:
                    for prefix in self.__common_prefixes__:
                        if hasattr(data, f"{prefix}_{identifier}"):
                            delattr(data, f"{prefix}_{identifier}")
            
            data["identifiers"] = [ide for ide in identifiers if ide in self.include_nodes]
        
        return data