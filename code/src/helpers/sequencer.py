import torch
from src.transforms import MultiNodeData
import collections
import dill
import os
from src.utils import create_dirs


class Sequencer(object):
    "Determines sequences in a dataset and annotates elements accordingly."
    
    def __init__(self, path_to_dir : str, node_classes : list = [], graph_classes : list = [], exclude_normal : bool = False, transform_key =""):
        
        self.path_to_dir = path_to_dir
        self.is_fitted : bool = False
        
        self.__exclude_normal__ = exclude_normal
        self.__node_classes__ = node_classes
        self.__graph_classes__ = graph_classes
        
        self.__sequence_dict__ : dict = {}
        self.__latest_group__ : tuple = None
        
        self.__id__ = Sequencer.get_id(node_classes=self.__node_classes__, 
                                       graph_classes=self.__graph_classes__, 
                                       exclude_normal=self.__exclude_normal__,
                                       transform_key=transform_key)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(exclude_normal={self.__exclude_normal__}, #sequence_groups={self.__latest_group__[0] + 1})" # starts with zero
    
    def __str__(self):
        return f"{self.__class__.__name__}(exclude_normal={self.__exclude_normal__}, #sequence_groups={self.__latest_group__[0] + 1})" # starts with zero
    
    @staticmethod
    def get_id(*args, **kwargs):    
        sorted_kwargs = collections.OrderedDict(sorted(kwargs.items()))
        return ", ".join(f"{key}={value}" for key, value in sorted_kwargs.items())
    
    @classmethod
    def getInstance(cls, path_to_dir : str, **kwargs):
        path_to_file = os.path.join(path_to_dir, "sequencer.pkl")
        
        if os.path.exists(path_to_file):
            with open(path_to_file, 'rb') as dill_file:
                obj = dill.load(dill_file)
                if obj.__id__ == Sequencer.get_id(**kwargs):
                    return obj
                else:
                    return Sequencer(path_to_dir, **kwargs) 
        else:
            return Sequencer(path_to_dir, **kwargs)
    
    def save(self):
        create_dirs(self.path_to_dir)
        with open(os.path.join(self.path_to_dir, "sequencer.pkl"), "wb") as dill_file:
            dill.dump(self, dill_file)
    
    def __call__(self, data: MultiNodeData):
        file_idx : int = data["file_idx"]
        look_up_dict = self.__sequence_dict__.get(file_idx, {})
                
        for key, value in look_up_dict.items():
            data[key] = value
        
        return data
    
    def annotate(self, data: MultiNodeData):
        "The calling function iterates over a dataset and sequentially inputs elements."
        
        # extract properties from object
        identifiers : list = data["identifiers"]
        y_compact = data["y"]
        y_full = data["y_full"]
                
        sequence_node_group : str = None
        sequence_transitional : str = "steady"
        
        sequence_anomaly_index : int = torch.argmax(y_compact).item()
        sequence_anomaly : int = self.__graph_classes__[sequence_anomaly_index]
        sequence_anomaly_index =  self.__node_classes__.index(sequence_anomaly) # overwrite       
                
        if y_compact[0,0] == 1 and not self.__exclude_normal__:
            sequence_node_group = "cluster"
        else:
            identifier_index : int = torch.argmax(y_full[: ,sequence_anomaly_index]).item()
            identifier : str = identifiers[identifier_index]
            sequence_node_group = data[f"group_{identifier}"]
        
        # handle group incrementing
        if self.__latest_group__ is None:
            self.__latest_group__ = (0, sequence_anomaly, sequence_node_group)
        elif (self.__latest_group__[1] != sequence_anomaly) or (self.__latest_group__[2] != sequence_node_group):    
            
            sequence_transitional = "up / down"
                
            self.__latest_group__ = (self.__latest_group__[0] + 1, sequence_anomaly, sequence_node_group)   
                         
            
        # add properties to this object
        data["sequence_group"] = self.__latest_group__[0]
        data["sequence_anomaly"] = sequence_anomaly
        data["sequence_node_group"] = sequence_node_group
                
        self.__sequence_dict__[data["file_idx"]] = {
            "sequence_transitional": sequence_transitional,
            "sequence_id": self.__latest_group__[0],
            "file_id": data["file_idx"]
        }
        
        return data
        
    