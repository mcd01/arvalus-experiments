import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import logging
import sys

class LossWrapper(object):
    "A wrapper class that handles the loss and its weighting."
    
    def __init__(self, loss_type: str, training_data_stats : dict, node_num_classes: int, device : str = "cpu", exclude_normal : bool = False, use_synthetic_data : bool = False):
                
        self.loss_type = loss_type
        self.node_num_classes = node_num_classes
        self.device = device
        self.exclude_normal = exclude_normal
        
        # adapt because of weighted sampling
        coarse_class_counts = training_data_stats["coarse_class_counts"]
            
        coarse_class_counts = coarse_class_counts.to(torch.long).to(self.device)
        logging.info(f"coarse_class_counts: {coarse_class_counts}")
        
        # count dict
        fine_class_counts = training_data_stats["fine_class_counts"]

        if self.node_num_classes == 1:
            fine_class_counts_new = torch.ones(2, dtype=torch.double)
            
            fine_class_counts_new[0] = fine_class_counts.sum() - fine_class_counts[1:].sum()
            fine_class_counts_new[1] = fine_class_counts[1:].sum()
            
            fine_class_counts = fine_class_counts_new
              
        fine_class_counts = fine_class_counts.to(torch.long).to(self.device)
        logging.info(f"fine_class_counts: {fine_class_counts}")
                
        self.loss_fine = CrossEntropyFocalLoss(fine_class_counts, gamma=2, beta=0.999999, device=self.device)

        
    @property
    def node_loss_fn(self):
        return self.loss_fine      
        
    def __call__(self, y_pred, y_true):
        y_pred_fine, y_pred_coarse = y_pred
        y_true_fine, y_true_coarse = y_true
        
        y_true_fine = torch.argmax(y_true_fine, dim=1)
        
        if self.node_num_classes == 1:
            y_true_fine[y_true_fine > 0] = 1
            
        y_true_fine = y_true_fine.to(torch.long)
        
        y_true_coarse = y_true_coarse.type_as(y_pred_coarse)
        
        return self.loss_fine(y_pred_fine, y_true_fine)
    
    
    def __str__(self):
        return f"{self.__class__.__name__}(exclude_normal={self.exclude_normal}), fine:{self.loss_fine}"
    
    def __repr__(self):
        return f"{self.__class__.__name__}(exclude_normal={self.exclude_normal}), fine:{self.loss_fine}"
            

# Focal Loss Function from : https://arxiv.org/pdf/1708.02002.pdf            
class CrossEntropyFocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, class_counts:torch.Tensor, gamma:float=2, beta:float=0.999999, reduction:str='mean', device:str="cpu"):
        super(CrossEntropyFocalLoss, self).__init__(reduction=reduction)
        
        if class_counts.dtype != torch.long:
            raise ValueError("Expected Torch-Tensor of type 'long'.")
        
        self.device = device
        
        self.class_counts = torch.tensor(class_counts, dtype=torch.double)
        self.gamma = torch.tensor(gamma, dtype=torch.double, device=self.device)
        self.beta = torch.tensor(beta, dtype=torch.double, device=self.device)
        self.use_beta = False
        self.balance = False
        
        self.__compute_weights__()

    def __compute_weights__(self):
        alpha = None
        
        if self.use_beta:
            # Class-Balanced Loss from: https://arxiv.org/pdf/1901.05555.pdf
            alpha = torch.tensor([(1 - self.beta) / (1 - self.beta ** count) for count in self.class_counts], dtype=torch.double)
        elif not self.balance:
            alpha = torch.ones_like(self.class_counts).reshape(-1)
        else:
            # inverse class frequency
            weights = self.class_counts / self.class_counts.sum()
            weights = torch.tensor(1.0, dtype=torch.double) / weights
            weights[weights == float("Inf")] = 0
            alpha = weights / weights.sum()
        
        alpha = alpha.to(torch.double)
        alpha = alpha.to(self.device)
        self.alpha = alpha
    
    def forward(self, y_pred, y_true):
        weights = self.alpha[y_true.to(torch.long)]
        
        ce_loss = F.cross_entropy(y_pred, y_true, reduction="none") 
        pt = torch.exp(-ce_loss)
        focal_loss = weights * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            focal_loss = torch.sum(focal_loss) / torch.sum(weights)
         
        return focal_loss 
    
    def __str__(self):
        return f"{self.__class__.__name__}(class_counts={self.class_counts}, gamma={self.gamma}, beta={self.beta}, alpha={self.alpha}, use_beta={self.use_beta}, balance={self.balance})"
    
    def __str__(self):
        return f"{self.__class__.__name__}(class_counts={self.class_counts}, gamma={self.gamma}, beta={self.beta}, alpha={self.alpha}, use_beta={self.use_beta}, balance={self.balance})"
    