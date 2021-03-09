import torch
import numpy as np
from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor
from torch.nn import CrossEntropyLoss
import re
import os
import dill


def create_supervised_trainer(model, optimizer, loss_fn=None,
                              device=None, non_blocking=False,
                              output_transform=lambda x, y, y_pred, loss: loss.item()):
    """Adapted code from Ignite-Library in order to allow for handling of graphs."""

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
                
        if device is not None:
            batch = batch.to(device, non_blocking=non_blocking)
        
        batch.x = torch.abs(batch.x + torch.normal(0, 0.01, size=batch.x.shape).to(device).double()) # add some noise
        y_pred_fine, y_pred_coarse = model(batch)
        
        y_true_fine = batch.y_full
        y_true_coarse = batch.y
                
        y_true = (y_true_fine, y_true_coarse)
        y_pred = (y_pred_fine, y_pred_coarse)
                        
        loss = loss_fn(y_pred, y_true)
        
        loss.backward()

        optimizer.step()
        
        
        return output_transform(batch.x, y_true, y_pred, loss)

    return Engine(_update)


def create_supervised_evaluator(model, loss_fn=None, metrics=None,
                                device=None, non_blocking=False,
                                node_classes:list=[],
                                graph_classes:list=[],
                                pred_collector_function=lambda x,y: None,
                                output_transform=lambda x, y, y_pred: (y_pred, y,)):
    """Adapted code from Ignite-Library in order to allow for handling of graphs."""

    metrics = metrics or {}
    _node_classes = torch.tensor(node_classes, device=device)
    _graph_classes = torch.tensor(graph_classes, device=device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            
            if device is not None:
                batch = batch.to(device, non_blocking=non_blocking)
            
            y_pred_fine, y_pred_coarse = model(batch)
            
            y_true_fine = batch.y_full
            y_true_coarse = batch.y
            
            ################# COLLECT EVERYTHING #################
            def mapper(element):
                return element.item() if isinstance(element, torch.Tensor) else element
            
            num_cluster_nodes = batch.num_cluster_nodes
            
            # collect normal stuff
            pred_collector_function("file_id", sum([count * [mapper(el)] for (count, el) in zip(num_cluster_nodes, batch.file_id)], []))
            pred_collector_function("sequence_id", sum([count * [mapper(el)] for (count, el) in zip(num_cluster_nodes, batch.sequence_id)], []))
            pred_collector_function("transitional", sum([count * [el] for (count, el) in zip(num_cluster_nodes, batch.sequence_transitional)], []))
            pred_collector_function("node_id", sum(batch.node_ids, []))
            pred_collector_function("node_group", sum(batch.node_groups, []))
            
            # collect node labels and predictions
            pred_collector_function("true_node_anomaly", _node_classes[torch.argmax(y_true_fine, dim=1).view(-1)])
            for i, class_label in enumerate(_node_classes):
                pred_collector_function(f"prob_node_anomaly_{class_label}", y_pred_fine[:, i])
            
            # collect graph labels and predictions
            graph_true_labels = _graph_classes[torch.argmax(y_true_coarse, dim=1).view(-1)]
            pred_collector_function("true_graph_anomaly", sum([count * [mapper(el)] for (count, el) in zip(num_cluster_nodes, graph_true_labels)], []))
            
            for i, class_label in enumerate(_graph_classes):
                graph_probs = y_pred_coarse[:, i]
                pred_collector_function(f"prob_graph_anomaly_{class_label}", sum([count * [mapper(el)] for (count, el) in zip(num_cluster_nodes, graph_probs)], []))
            ######################################################

            y_true = (y_true_fine, y_true_coarse)
            y_pred = (y_pred_fine, y_pred_coarse)
            
            return output_transform(batch.x, y_true, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine  
        
        
def get_training_status_meta_dict(path_to_dir: str, model_name: str, folds: int, epochs: int):
    """Handles resuming from training."""
    
    training_status_meta_dict = {}
        
    files_in_dir = os.listdir(path_to_dir)
    
    for fold in range(folds):
        fold_dir = f"fold={fold + 1}"
        
        training_status_meta_dict[fold] = {
            "fold_dir": os.path.join(path_to_dir, fold_dir),
            "finished": False,
            "resume_from_checkpoint": {}
        }
        
        if fold_dir in files_in_dir: # check if such a directory already exists
            subdir = os.path.join(path_to_dir, fold_dir)
            files_in_subdir = list(os.listdir(subdir))
                        
            if any([(f"{model_name}_test.csv" == file_name or f"{model_name}_latest_checkpoint_{epochs}.pt" == file_name) for file_name in files_in_subdir]):
                    training_status_meta_dict[fold]["finished"] = True 
            
            
            # check if there is a latest, then best checkpoint to resume from
            latest_checkpoint_list = list(filter(lambda x: f"{model_name}_latest_checkpoint" in x, files_in_subdir))
            best_checkpoint_list = list(filter(lambda x: f"{model_name}_best_checkpoint" in x, files_in_subdir))
            
            
            if len(latest_checkpoint_list):
                lc = latest_checkpoint_list[0]
                training_status_meta_dict[fold]["resume_from_checkpoint"]["latest_checkpoint_path"] = os.path.join(subdir, lc)  
            
            if len(best_checkpoint_list):
                bc = sorted(best_checkpoint_list)[-1]
                training_status_meta_dict[fold]["resume_from_checkpoint"]["best_checkpoint_path"] = os.path.join(subdir, bc)  
                                           
    return training_status_meta_dict


def create_dirs(path):
    try:
        if not os.path.isdir(path):
            os.makedirs(path)
    except:
        pass    
    
             
        
        
            