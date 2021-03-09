import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_default_dtype(torch.double)

import logging
from datetime import datetime
import copy

import argparse
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.utils.data import WeightedRandomSampler
from ignite.utils import manual_seed
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose

import sys
import os
sys.path.append(os.path.join(os.path.abspath(''), ".."))

from src.losses import LossWrapper
from src.trainer import Trainer
from src.datasets import MultiNodeDataset, MultiNodeDataSubset
from src.utils import get_training_status_meta_dict, create_dirs

import src.transforms as T
import src.helpers as H

from src.modelGCN import ClusterModel as ModelGCN
from src.modelCNN import ClusterModel as ModelCNN

from src.transforms import anomaly_names as raw_classes
from src.transforms import anomaly_codes as raw_class_codes



parser = argparse.ArgumentParser()

parser.add_argument("-dr", "--data-root", type=str,
                    required=True, help="Path to dataset root.")

parser.add_argument("-rr", "--result-root", type=str, help="Path to result root.", default=None)

parser.add_argument("-nw", "--num-workers", type=int,
                    default=0, help="Number of workers for data-loading.")

parser.add_argument("-cc", "--cluster-connectivity", type=str,
                    default="fine", choices=["fine", "coarse"], help="Granularity of cluster modeling.")

parser.add_argument("-d", "--device", type=str, required=True,
                    help="If available, will use this device for training.")

parser.add_argument("-v", "--verbose", action='store_true',
                    help="Enables printing of details during learning.")

parser.add_argument("-e", "--epochs", type=int, default=10,
                    help="Number of epochs for training.")

parser.add_argument("-bs", "--batch-size", type=int, default=64,
                    help="Batch size during training / updates.")

parser.add_argument("-s", "--stride", type=int,
                    default=20, help="Stride during graph construction.")

parser.add_argument("-id", "--input-dimension", type=int,
                    default=20, help="Size of the input dimension, i.e. width during window extraction.")

parser.add_argument("-hd", "--hidden-dimension", type=int,
                    default=32, help="Size of the hidden dimension.")

parser.add_argument("-nck", "--num-conv-kernels", type=int,
                    default=3, help="Number of convolution kernels.")

parser.add_argument("-do", "--dropout", type=float,
                    default=0.5, help="Dropout rate.")

parser.add_argument("-m", "--model", type=str,
                    required=True, choices=["ModelGCN", "ModelCNN"], help="Model to use.")

parser.add_argument("-lc", "--loss-class", type=str,
                    default="MCE", choices=["MCE"], help="Loss to use.")

parser.add_argument("-lr", "--learning-rate", type=float,
                    default=0.01, help="Learning rate.")
parser.add_argument("-wd", "--weight-decay", type=float,
                    default=1e-5, help="Weight decay.")
parser.add_argument("-b", "--betas", type=int, nargs="+",
                    default=[0.9, 0.999], help="Betas for Adam Optimizer.")

parser.add_argument("-ea", "--exclude-anomalies", type=str, nargs="+",
                    default=[], help="Anomalies to exclude.")

parser.add_argument("-im", "--include-metrics", type=str, nargs="+",
                    default=[], help="Metrics that are to be included. All other metrics are excluded.")

parser.add_argument("-in", "--include-nodes", type=str, nargs="+",
                    default=[], help="Nodes that are to be included. All other nodes are excluded.")

parser.add_argument("-es", "--early-stopping", type=int,
                    default=None, help="Configure early stopping.")

parser.add_argument("-st", "--start-time", type=str,
                    default="2019-11-05 19:05:00", help="Start time.")

parser.add_argument("-et", "--end-time", type=str,
                    default="2019-11-07 21:43:00", help="End time.")

parser.add_argument("-ss", "--seed", type=int,
                    default=42, help="Seed. For split reproducibility.")

parser.add_argument("-ns", "--num-splits", type=int,
                    default=5, help="Number of stratified grouped splits.")

parser.add_argument("-ts", "--target-splits", type=int, nargs="+",
                    default=None, help="Target splits to actually use.")

parser.add_argument("-nbc", "--node-binary-classification", action='store_true', help="Specify if node classification should be binary.")

parser.add_argument("-l", "--logging-level", type=str,
                        default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level to use.")

parser.add_argument("-uv", "--use-validation", action='store_true', help="Specify if a validation set should be used.")

parser.add_argument("-ut", "--use-test", action='store_true', help="Specify if a test set should be used.")

args = parser.parse_args()

if not args.use_test and not args.use_validation:
    raise ValueError("You must use either a validation set, a test set, or both.")

use_synthetic_data : bool = True
############### SEED ##############
seed = args.seed
###################################
result_root = args.result_root if args.result_root is not None else os.getcwd()
######## SETUP LOGGING ############
logging_level = "INFO"
if args.logging_level != "INFO":
    logging_level = args.logging_level

log_dir = os.path.join(result_root, "logs")
create_dirs(log_dir)

logging.basicConfig(
    level=logging_level,
    format="%(asctime)s - [%(levelname)s] - %(filename)s:%(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{args.model}.log')),
        logging.StreamHandler()
    ])
####### BASE TRANSFORMS ##########
base_transforms = T.Compose([
    T.DataCreator(), # convert value-dictionary to instance of MultiNodeData-class
    T.DataLabeler(min_occurrence=0.5), # label nodes based on tag occurrences
    T.NodeSelector(include_nodes=args.include_nodes), # only use subset of nodes if required 
    T.MetricSelector(include_metrics=args.include_metrics), # only use subset of metrics if required
    T.AnomalySelector(exclude_anomalies=args.exclude_anomalies), # filter anomalies if exclusions are specified
    T.NodeConnector(use_synthetic_data=use_synthetic_data) # add adj-matrix (sparse)
])
logging.info(f"Base transforms: {base_transforms}")
########## PREPARE DATA ##########
logging.info("Read dataset...")
data_root = args.data_root
dataset = MultiNodeDataset(data_root, 
                           args.input_dimension, 
                           args.stride, 
                           use_synthetic_data=use_synthetic_data,
                           start_time=args.start_time, 
                           end_time=args.end_time)

node_meta, group_meta, sample_meta = dataset.get_meta()
############ prepare classes ##############
node_classes = ["binary"] if args.node_binary_classification else [el for (idx,el) in enumerate(raw_class_codes) if raw_classes[idx] not in set(args.exclude_anomalies).difference(set(["normal"]))]
node_num_classes = len(node_classes)
logging.info(f"Node anomaly classes: {node_classes}")

graph_classes = [el for (idx,el) in enumerate(raw_class_codes) if raw_classes[idx] not in args.exclude_anomalies]
graph_num_classes = len(graph_classes)
logging.info(f"Graph anomaly classes: {graph_classes}")
###########################################

dataset_length = len(dataset)
logging.info(f"Original dataset length: {dataset_length}")

logging.info("Get splitter and sequencer...")
splitter, base_transforms = H.Splitter.getInstance(result_root, 
                                                   args.input_dimension, 
                                                   args.stride, 
                                                   args.loss_class,
                                                   seed, 
                                                   base_transforms,
                                                   n_splits=args.num_splits, 
                                                   use_test=args.use_test, 
                                                   use_validation=args.use_validation,
                                                   transform_key=str(base_transforms))

sequencer = H.Sequencer.getInstance(splitter.path_to_dir, 
                                    node_classes=node_classes,
                                    graph_classes=graph_classes,
                                    exclude_normal="normal" in args.exclude_anomalies,
                                    transform_key=splitter.__id__)

synthesizer = H.Synthesizer.getInstance(splitter.path_to_dir, 
                                        num_nodes=len(node_meta),
                                        window_width=args.input_dimension,
                                        use_synthetic_data=use_synthetic_data,
                                        transform_key=splitter.__id__)

if not splitter.is_fitted or not sequencer.is_fitted or not synthesizer.is_fitted:
    logging.info("Fit splitter or sequencer or synthesizer...")
    creator = T.DataCreator()
    
    for idx in dataset.data_indices:
        raw_element = dataset.get(idx) # this is still a dictionary
        element = base_transforms(copy.copy(raw_element)) # prepare element
        
        if element.skip: continue
                
        if not sequencer.is_fitted:
            element = sequencer.annotate(element) # handle sequences and grouping
            
        if not splitter.is_fitted:
            splitter.collect(element, idx)
            
        if not synthesizer.is_fitted:
            element = synthesizer.synthesize(element)
            element_dict = {k:(raw_element[k] if k not in element.__dict__ else element.__dict__[k]) for k in raw_element.keys()}
            dataset.update(creator(element_dict), idx) # will only update if synthetic dataset
    
    if not sequencer.is_fitted:
        sequencer.is_fitted = True
        sequencer.save()
        logging.info(f"Sequencer fitted and saved. [{sequencer}]")
    if not splitter.is_fitted:
        splitter.is_fitted = True
        splitter.save(base_transforms)
        logging.info(f"Splitter fitted and saved. [{splitter}]")       
    if not synthesizer.is_fitted:
        synthesizer.is_fitted = True
        synthesizer.save()
        logging.info(f"Synthesizer fitted and saved. [{synthesizer}]")  

all_dataset_indices = splitter.all_dataset_indices
all_groups = splitter.all_groups
############# extract from fitted transformer ###############
num_edge_types = next(x for x in base_transforms.transforms if isinstance(x, T.NodeConnector)).num_edge_types
logging.info(f"#Edge-Types: {num_edge_types}")
###################################
manual_seed(seed) # set seed, ignite function
group_iterator = splitter.split()

training_status_meta_dict = get_training_status_meta_dict(splitter.path_to_dir, args.model, args.num_splits, args.epochs)

for split_idx, (train_groups, val_groups, test_groups) in enumerate(group_iterator):
    logging.info(f"Split: {split_idx + 1}")
    
    training_status_meta = training_status_meta_dict.get(split_idx)
    logging.info(f"Training status meta: {training_status_meta}")
    
    if training_status_meta.get("finished", False) or (args.target_splits and (split_idx + 1) not in args.target_splits):
        logging.info(f"Skipping Split={split_idx + 1}...")
        continue
    
    fold_dir : str = training_status_meta.get("fold_dir")
    
    logging.info(f"Train groups: {train_groups}")
    logging.info(f"Val groups: {val_groups}")
    logging.info(f"Test groups: {test_groups}") 
    
    overlaps = [
        len(set(train_groups.tolist()).intersection(val_groups.tolist())),
        len(set(val_groups.tolist()).intersection(test_groups.tolist())),
        len(set(test_groups.tolist()).intersection(train_groups.tolist()))
    ]
    
    logging.info(f"Overlaps: {overlaps}")
        
    if any([overlap_val > 0 for overlap_val in overlaps]):
        error_message = "Train, Validation and Test set are not distinct! Overlaps exist"
        logging.error(error_message)
        raise ValueError(error_message)
            
    train_indices = [all_dataset_indices[idx] for idx, el in enumerate(all_groups) if el in train_groups]
    val_indices = [all_dataset_indices[idx] for idx, el in enumerate(all_groups) if el in val_groups]
    test_indices = [all_dataset_indices[idx] for idx, el in enumerate(all_groups) if el in test_groups]
    
    logging.info("Sampling finished.")
    
    logging.info(f"Train Indices: {len(train_indices)}")
    logging.info(f"Val Indices: {len(val_indices)}")
    logging.info(f"Test Indices: {len(test_indices)}")
            
    logging.info("Get scaler and profiler...")    
    min_max_scaler = T.MinMaxScaler.getInstance(fold_dir, 
                                                target_min=0,
                                                target_max=1,
                                                normalization_type="group", 
                                                transform_key=splitter.__id__)
    profiler = H.Profiler.getInstance(fold_dir, 
                                      transform_key=splitter.__id__)
    if not min_max_scaler.is_fitted or not profiler.is_fitted:
        logging.info("Fit scaler or profiler...")
        for i in train_indices:
            element = dataset.get(i)
            element = base_transforms(element)
            
            element = sequencer(element)
            element = synthesizer(element)
                        
            if not min_max_scaler.is_fitted:
                min_max_scaler.fit(element)
            if not profiler.is_fitted:
                profiler.profile(element)    
        
        if not min_max_scaler.is_fitted:                        
            min_max_scaler.is_fitted = True
            min_max_scaler.save()    
            logging.info(f"Scaler fitted and saved. [{min_max_scaler}]")
            
        if not profiler.is_fitted:                        
            profiler.is_fitted = True
            profiler.save()    
            logging.info(f"Profiler fitted and saved. [{profiler}]")            
    
    logging.info("Prepare transform-pipeline...")
    data_transforms = T.Compose([
        base_transforms,
        sequencer, # add sequence information to data object
        synthesizer, # synthesize if required
        min_max_scaler, # transformation of node features
        T.DataEnhancer(group_meta) # add other stuff to graphs
    ])
    logging.info(f"Data transforms: {data_transforms}")
    logging.info("Transform-pipeline prepared.")
    
    train_dataset = MultiNodeDataSubset(data_root, 
                                        args.input_dimension, 
                                        args.stride, 
                                        train_indices, 
                                        use_synthetic_data=use_synthetic_data,
                                        transform=data_transforms)
    logging.info(f"Train dataset length: {len(train_dataset)}")
    val_dataset = MultiNodeDataSubset(data_root, 
                                      args.input_dimension, 
                                      args.stride, 
                                      val_indices,
                                      use_synthetic_data=use_synthetic_data, 
                                      transform=data_transforms)
    logging.info(f"Val dataset length: {len(val_dataset)}")
    test_dataset = MultiNodeDataSubset(data_root, 
                                       args.input_dimension, 
                                       args.stride, 
                                       test_indices, 
                                       use_synthetic_data=use_synthetic_data,
                                       transform=data_transforms)
    logging.info(f"Test dataset length: {len(test_dataset)}")
    ####################################

    ### OUR TRAINING CONFIGURATION #######
    verbose = args.verbose
    batch_size = args.batch_size
    epochs = args.epochs

    model_args = {
        "node_meta": node_meta,
        "group_meta": group_meta,
        "cluster_connectivity": args.cluster_connectivity,
        "input_dim": args.input_dimension,
        "hidden_dim": args.hidden_dimension,
        "num_conv_kernels": args.num_conv_kernels,
        "node_num_classes": node_num_classes,
        "graph_num_classes": graph_num_classes,
        "device": args.device,
        "batch_size": batch_size,
        "num_edge_types": num_edge_types
    }

    optimizer_args = {
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "betas": tuple(args.betas)
    }

    ClusterModel = None
    logging.info(f"Selected Model-Type: {args.model}")    
    if args.model == "ModelGCN":
        ClusterModel = ModelGCN  
    elif args.model == "ModelCNN":
        ClusterModel = ModelCNN  
        
    training_data_stats = profiler.get_train_data_stats()   
        
    loss_func = LossWrapper(
        args.loss_class, 
        training_data_stats, 
        model_args.get("node_num_classes"), 
        exclude_normal="normal" in args.exclude_anomalies,
        use_synthetic_data=use_synthetic_data,
        device=args.device)  
        
    logging.info(f"Selected Loss-Type: {args.loss_class}")  

    trainer_args = {
        "device": args.device,
        "epochs": epochs,
        "early_stopping": args.early_stopping,
        "exclude_anomalies": args.exclude_anomalies,
        "include_metrics": args.include_metrics,
        "model_name": args.model,
        "model_class": ClusterModel,
        "model_args": model_args,
        "optimizer_class": torch.optim.Adam,
        "optimizer_args": optimizer_args,
        "loss_func": loss_func,
        "node_classes": node_classes,
        "graph_classes": graph_classes,
        "training_data_stats": training_data_stats,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices
    }
    
    manual_seed(seed + split_idx) # reset seed, ignite function
    
    trainer_args = {**trainer_args, **training_status_meta}
    ###################################

    ### CREATE DATA LOADERS ###########
    group_meta_keys = list(group_meta.keys())
    follow_batch_keys = ["batch_cluster_coarse", "batch_cluster_fine"]
    num_workers = args.num_workers
    
    sampler = None
    
    default_data_loader_settings: dict = {
        "batch_size": batch_size,
        "drop_last": False,
        "shuffle": False,
        "follow_batch": follow_batch_keys,
        "num_workers": num_workers,
        "prefetch_factor": 1
    }
    
    train_data_loader_settings: dict = {
        **default_data_loader_settings,
        "sampler": sampler,
        "shuffle": sampler is None
    }
    
    train_loader = DataLoader(train_dataset, **train_data_loader_settings)
    val_loader = DataLoader(val_dataset, **default_data_loader_settings)
    test_loader = DataLoader(test_dataset, **default_data_loader_settings)
    ###############################
    
    trainer = Trainer(trainer_args)
    trainer.run(train_loader, val_loader, test_loader)
