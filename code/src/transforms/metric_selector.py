import torch
from . import MultiNodeData


class MetricSelector(object):
    "Handles selection of only specific metrics. Optional."
    
    def __init__(self, include_metrics : list = []):
        self.include_metrics = include_metrics
        
    def __repr__(self):
        return f"{self.__class__.__name__}(include_metrics={self.include_metrics})"
    
    def __str__(self):
        return f"{self.__class__.__name__}(include_metrics={self.include_metrics})"
        
    def __call__(self, data: MultiNodeData):
        
        if len(self.include_metrics):
            for identifier in data["identifiers"]:
                
                # extract from element
                metrics = data[f"headers_{identifier}"]
                x = data[f"x_{identifier}"]
                
                # filter according to specified metric-inclusion
                filtered = [(idx, metric) for (idx, metric) in enumerate(metrics) if metric in self.include_metrics]
                
                filtered_metrics = [tup[1] for tup in filtered]
                indices = [tup[0] for tup in filtered]
                
                # save filtered information 
                data[f"headers_{identifier}"] = filtered_metrics
                data[f"x_{identifier}"] = x[indices, :]
                
        return data