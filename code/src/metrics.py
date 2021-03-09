import torch
import warnings
from ignite.metrics import Accuracy, Metric
from ignite.exceptions import NotComputableError
from ignite.engine import Events
import psutil


class LabelwiseAccuracy(Accuracy):
    def __init__(self, output_transform=lambda x: x, device=None, is_multilabel=False):
        self._num_correct = None
        self._num_examples = None
        super(LabelwiseAccuracy, self).__init__(output_transform=output_transform, device=device, is_multilabel=is_multilabel)

    def reset(self):
        self._num_correct = None
        self._num_examples = 0
        super(LabelwiseAccuracy, self).reset()

    def update(self, output):

        self._check_shape(output)
        self._check_type(output)
        
        y_pred, y = output

        num_classes = y_pred.size(1)
        last_dim = y_pred.ndimension()
        y_pred = torch.transpose(y_pred, 1, last_dim - 1).reshape(-1, num_classes)
        y = torch.transpose(y, 1, last_dim - 1).reshape(-1, num_classes)
        correct_exact = torch.all(y == y_pred.type_as(y), dim=-1)  # Sample-wise
        correct_elementwise = torch.sum(y == y_pred.type_as(y), dim=0)

        if self._num_correct is not None:
            self._num_correct = torch.add(self._num_correct,
                                                    correct_elementwise)
        else:
            self._num_correct = correct_elementwise
        self._num_examples += correct_exact.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed.')
        return self._num_correct.type(torch.float) / self._num_examples
    
    
class CpuInfo(Metric):
    def __init__(self):
        try:
            import psutil
        except ImportError:
            raise RuntimeError(
                "This module requires psutil to be installed. "
                "Please install it with command: \n pip install psutil"
            )
        
        self.sum_util = 0
        self.num_examples = 0
        self.process = psutil.Process()
        self.process.cpu_percent(interval=None)
        super(CpuInfo, self).__init__()   
        
    def reset(self):        
        self.sum_util = 0
        self.num_examples = 0
        self.process = psutil.Process()
        self.process.cpu_percent(interval=None) 
        super(CpuInfo, self).reset()     
        
    def update(self, *args, **kwargs):
        self.sum_util += (self.process.cpu_percent(interval=None) / psutil.cpu_count())
        self.num_examples += 1  
        
    def compute(self):
        # compute mean over all observations and normalize to [0,1]
        mean_util = self.sum_util / self.num_examples # compute mean
        mean_util = mean_util / 100 # normalize
        
        return mean_util       