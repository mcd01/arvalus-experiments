import torch
from . import MultiNodeData
from src.utils import create_dirs
import os
import dill    
import collections


class MinMaxScaler(object):
    "Handles min-max scaling of individual metrics. Either per node or per group."
    
    def __init__(self, path_to_dir : str, target_min=0, target_max=1, normalization_type="node", transform_key=""):
        
        self.path_to_dir = path_to_dir
        self.is_fitted = False
        
        self.target_min = target_min
        self.target_max = target_max
        
        self.normalization_type = normalization_type
        
        self.min_values = {}
        self.max_values = {}
        
        self.__id__ = MinMaxScaler.get_id(target_min=target_min, 
                                          target_max=target_max, 
                                          normalization_type=normalization_type, 
                                          transform_key=transform_key)
        
    def __repr__(self):
        return f"{self.__class__.__name__}(target_min={self.target_min}, target_max={self.target_max}, normalization_type={self.normalization_type})"
    
    def __str__(self):
        return f"{self.__class__.__name__}(target_min={self.target_min}, target_max={self.target_max}, normalization_type={self.normalization_type})"
        
    @staticmethod
    def get_id(*args, **kwargs):    
        sorted_kwargs = collections.OrderedDict(sorted(kwargs.items()))
        return ", ".join(f"{key}={value}" for key, value in sorted_kwargs.items())
    
    @classmethod
    def getInstance(cls, path_to_dir : str, **kwargs):
        path_to_file = os.path.join(path_to_dir, "min_max_scaler.pkl")
        
        if os.path.exists(path_to_file):
            with open(path_to_file, 'rb') as dill_file:
                obj = dill.load(dill_file)
                if obj.__id__ == MinMaxScaler.get_id(**kwargs):
                    return obj
                else:
                    return MinMaxScaler(path_to_dir, **kwargs)  
        else:
            return MinMaxScaler(path_to_dir, **kwargs)   
        
    def save(self):
        create_dirs(self.path_to_dir)
        with open(os.path.join(self.path_to_dir, "min_max_scaler.pkl"), "wb") as dill_file:
            dill.dump(self, dill_file)          
        
    def fit(self, data: MultiNodeData):
                
        for identifier in data["identifiers"]:
            temp_x = data[f"x_{identifier}"]
            
            min_vals = torch.min(temp_x, dim = 1, keepdim=True)[0]
            max_vals = torch.max(temp_x, dim = 1, keepdim=True)[0]
            
            min_key, max_key = None, None
            if self.normalization_type == "node":
                min_key = f"x_{identifier}_min"
                max_key = f"x_{identifier}_max"
            elif self.normalization_type == "group": 
                min_key = f"{data[f'group_{identifier}']}_min"
                max_key = f"{data[f'group_{identifier}']}_max"
                
            if min_key in self.min_values:
                min_vals = torch.min(torch.cat((min_vals, self.min_values[min_key]), dim = 1), dim = 1, keepdim = True)[0]
            self.min_values[min_key] = min_vals
                
            if max_key in self.max_values:
                max_vals = torch.max(torch.cat((max_vals, self.max_values[max_key]), dim = 1), dim = 1, keepdim = True)[0]
            self.max_values[max_key] = max_vals  
    
    
    def _transform(self, tensor, min_vals, max_vals):
        denom = max_vals - min_vals   
        denom[denom == 0] = 1  # Prevent division by 0    
        
        nom = (tensor - min_vals) * (self.target_max - self.target_min) 
        
        tensor = self.target_min + nom / denom
        return tensor
    
            
    def __call__(self, data: MultiNodeData):
        for identifier in data["identifiers"]: 
            
            min_key, max_key = None, None
            if self.normalization_type == "node":
                min_key = f"x_{identifier}_min"
                max_key = f"x_{identifier}_max"
            elif self.normalization_type == "group": 
                min_key = f"{data[f'group_{identifier}']}_min"
                max_key = f"{data[f'group_{identifier}']}_max"
            
            data[f"x_{identifier}"] = self._transform(data[f"x_{identifier}"], self.min_values[min_key], self.max_values[max_key])
            
        return data 