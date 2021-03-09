import torch
import dill
import os
from src.transforms.data_creator import MultiNodeData
from src.utils import create_dirs
import collections


class Profiler(object):
    "Profiling on a given Dataset. Saves interesting information, e.g. class counts."
    
    def __init__(self, path_to_dir : str, transform_key =""):
        
        self.path_to_dir = path_to_dir
        self.is_fitted = False
        
        self.__fine_y_tensor_list__ : list = []
        self.__coarse_y_tensor_list__ : list = []
        
        self.__coarse_to_fine_counts_dict__ : dict = {}
        
        self.__id__ = Profiler.get_id(transform_key=transform_key)
        
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    def __str__(self):
        return f"{self.__class__.__name__}()"
    
    @staticmethod
    def get_id(*args, **kwargs):    
        sorted_kwargs = collections.OrderedDict(sorted(kwargs.items()))
        return ", ".join(f"{key}={value}" for key, value in sorted_kwargs.items())
        
    @classmethod
    def getInstance(cls, path_to_dir : str, **kwargs):
        path_to_file = os.path.join(path_to_dir, "profiler.pkl")
        
        if os.path.exists(path_to_file):
            with open(path_to_file, 'rb') as dill_file:
                obj = dill.load(dill_file)
                if obj.__id__ == Profiler.get_id(**kwargs):
                    return obj
                else:
                    return Profiler(path_to_dir, **kwargs) 
        else:
            return Profiler(path_to_dir, **kwargs)      

    def save(self):
        create_dirs(self.path_to_dir)
        with open(os.path.join(self.path_to_dir, "profiler.pkl"), "wb") as dill_file:
            dill.dump(self, dill_file)    
        
    @property
    def fine_classes(self):
        return torch.argmax(torch.cat(self.__fine_y_tensor_list__, dim = 0), dim = 1)    
        
    @property
    def fine_class_weights(self):
        return self._get_class_weights(torch.cat(self.__fine_y_tensor_list__, dim = 0)) 
        
    @property
    def fine_pos_weights(self):
        return self._get_pos_weights(torch.cat(self.__fine_y_tensor_list__, dim = 0))         
        
    @property
    def fine_class_counts(self):
        return torch.sum(torch.cat(self.__fine_y_tensor_list__, dim = 0), dim = 0)
    
    @property
    def coarse_classes(self):
        return torch.argmax(torch.cat(self.__coarse_y_tensor_list__, dim = 0), dim = 1)

    @property
    def coarse_class_weights(self):
        return self._get_class_weights(torch.cat(self.__coarse_y_tensor_list__, dim = 0)) 
        
    @property
    def coarse_pos_weights(self):
        return self._get_pos_weights(torch.cat(self.__coarse_y_tensor_list__, dim = 0))                 
    
    @property
    def coarse_class_counts(self):
        return torch.sum(torch.cat(self.__coarse_y_tensor_list__, dim = 0), dim = 0)
    
    @property
    def coarse_to_fine_counts_dict(self):
        return self.__coarse_to_fine_counts_dict__
    
    def get_train_data_stats(self):
        return {
            "fine_classes": self.fine_classes,
            "fine_class_weights": self.fine_class_weights,
            "fine_pos_weights": self.fine_pos_weights,
            "fine_class_counts": self.fine_class_counts,
            "coarse_classes": self.coarse_classes,
            "coarse_class_weights": self.coarse_class_weights,
            "coarse_pos_weights": self.coarse_pos_weights,
            "coarse_class_counts": self.coarse_class_counts,
            "coarse_to_fine_counts_dict": self.coarse_to_fine_counts_dict
        }
        
    
    def _get_class_weights(self, y_true):
        class_counts = torch.sum(y_true, dim = 0, dtype = torch.double)
        weights = class_counts / class_counts.sum()
        weights = torch.tensor(1.0, dtype=torch.double) / weights
        weights[weights == float("Inf")] = 0
        weights = weights / weights.sum()
        return weights.to(torch.double)
            
    def _get_pos_weights(self, y_true):
        class_counts = torch.sum(y_true, dim = 0, dtype = torch.double)
        
        pos_weights = torch.ones_like(class_counts, dtype=torch.double)
        neg_counts = torch.tensor([y_true.shape[0]-pos_count for pos_count in class_counts], dtype = torch.double)
        for cdx, (pos_count, neg_count) in enumerate(zip(class_counts,  neg_counts)):
            pos_weights[cdx] = max(1.0, neg_count / (pos_count + 1e-5)) # must be at least 1, otherwise ignored during loss computation!

        return pos_weights.to(torch.double)         
        
    
    def profile(self, data: MultiNodeData):
        "The calling function iterates over a dataset and sequentially inputs elements."
        
        y = data["y"]
        y_full = data["y_full"]
        
        class_label = torch.argmax(y, dim = 1).item()
        
        if class_label not in self.__coarse_to_fine_counts_dict__:
            self.__coarse_to_fine_counts_dict__[class_label] = torch.sum(y_full, dim = 0).reshape(1, -1)
        
        self.__fine_y_tensor_list__.append(y_full)
        self.__coarse_y_tensor_list__.append(y)