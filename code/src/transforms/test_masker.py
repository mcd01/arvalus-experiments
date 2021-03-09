import torch
from . import MultiNodeData


class TestMasker(object):
    "For testing only. Overlay of easy to predict metric value patterns."
    
    def __init__(self, input_dim: int, num_classes : int):
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        num_chunks : int = self.num_classes if (self.num_classes % 2) == 1 else (self.num_classes - 1)
        
        tensor_list : list = torch.arange(self.input_dim).chunk(num_chunks)
        
        self.__idx_dict__ : dict = {(idx+1):t for idx, t in enumerate(tensor_list)}
        self.__idx_dict__[0] = None # just use all zeros if class = "normal"
                
    def __repr__(self):
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, num_classes={self.num_classes})"
    
    def __str__(self):
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, num_classes={self.num_classes})"
        
    def __call__(self, data: MultiNodeData):
        identifiers = data["identifiers"]
        for identifier in identifiers:
            x = data[f"x_{identifier}"]
                        
            zeros = torch.zeros_like(x)
            ones = torch.ones_like(x)
            mask = self.__idx_dict__[torch.argmax(data[f"y_{identifier}"]).item()]
            if mask is not None:
                zeros[:, mask] = ones[:, mask]
            
            data[f"x_{identifier}"] = zeros
        return data  