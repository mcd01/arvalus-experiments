import torch
from torch_geometric.data import Dataset
import os
import pandas as pd
import numpy as np
from scipy import signal
import sys
from src.utils import create_dirs
import re
from datetime import datetime, timedelta
from collections import OrderedDict
import logging

##########################
# regex to extract information from tags
anomaly_regex = re.compile(r"^.*anomaly=(\S+)\s.*$")
component_regex = re.compile(r"^.*component=(\S+)\s.*$")
group_regex = re.compile(r"^.*group=(\S+)\s.*$")
host_regex = re.compile(r"^.*host=(\S+)\s.*$")
injected_regex = re.compile(r"^.*injected=(\S+)\s.*$")
node_regex = re.compile(r"^.*node=(\S+)\s?.*$")

def extract_from_tags(tag_list: list):
    first_tag = tag_list[0]
    
    data_node = node_regex.match(first_tag).group(1)
    data_group = group_regex.match(first_tag).group(1)
    data_component = component_regex.match(first_tag).group(1)
    data_host = host_regex.match(first_tag).group(1)

    data_anomaly_list = []
    for tag in tag_list:
        tag = tag.replace("|", " ") # safety action
        anomaly_match = anomaly_regex.match(tag)
        injected_match = injected_regex.match(tag)
                
        if anomaly_match is None:
            data_anomaly = "normal"
        else:
            data_injected = injected_match.group(1)
            if data_component != data_injected:
                data_anomaly = "normal"
            else:
                data_anomaly = anomaly_match.group(1)
        data_anomaly_list.append(data_anomaly)
                            
    return data_node, data_group, data_component, data_host, data_anomaly_list
##########################
         

class MultiNodeDataset(Dataset):
    "Handles multiple graphs. Refer to: https://pytorch-geometric.readthedocs.io/en/1.6.3/notes/create_dataset.html"
    
    def __init__(self, root, window_width, stride, use_synthetic_data=False, transform=None, pre_transform=None, start_time=None, end_time=None):
        self.__processed_file_names__ : list = None # cached result
        self.window_width = window_width
        self.stride = stride
        self.start_time = start_time
        self.end_time = end_time
        self.use_synthetic_data = use_synthetic_data
        self.idx_regex = re.compile(r"^.*multi_node_data_(\d+)\.pt$")
        logging.info(f"Window width: {self.window_width}")
        logging.info(f"Stride: {self.stride}")
        logging.info(f"Start time: {self.start_time}")
        logging.info(f"End time: {self.end_time}")
        super(MultiNodeDataset, self).__init__(root, transform, pre_transform)
        
    
    @property
    def processed_dir(self):
        processed_dir_path = os.path.join(self.root, f'processed___s={self.stride:02d}_ww={self.window_width:02d}{"_synthetic" if self.use_synthetic_data else ""}') 
        create_dirs(processed_dir_path)
        return  processed_dir_path
        
    @property
    def processed_file_names(self):  
        if self.__processed_file_names__ is None: 
            self.__processed_file_names__ = [f for f in sorted(os.listdir(self.processed_dir)) if os.path.isfile(os.path.join(self.processed_dir, f)) and "multi_node_data_" in f]
        return self.__processed_file_names__
        
    @property
    def raw_file_names(self):
        return [f for f in sorted(os.listdir(self.raw_dir)) if os.path.isfile(os.path.join(self.raw_dir, f))]
    
    @property
    def data_indices(self):
        return [int(self.idx_regex.match(file_name).group(1)) for file_name in self.processed_file_names]
    
    @property
    def processed_paths(self):
        r"""The filepaths to find in order to skip the processing."""
        return [os.path.join(self.processed_dir, f) for f in self.processed_file_names]
    
    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        return [os.path.join(self.raw_dir, f) for f in self.raw_file_names]        

    def process(self):
        date_range = pd.date_range(start=self.start_time, end=self.end_time, freq=f"{int(self.stride / 2)}S")
        exclude_headers = ["tags", "time"]
        
        node_meta = OrderedDict()
        group_meta = OrderedDict()
        
        sample_meta = OrderedDict()
        sample_meta["df_lengths"] = OrderedDict()
        sample_meta["graph_sizes"] = OrderedDict()
                                
        ##### iterate over all files and populate lookup dicts #####
        for list_index, raw_file_name in enumerate(self.raw_file_names):
            logging.info(f"Load file: {(list_index + 1):02} {raw_file_name}")
            raw_path = os.path.join(self.raw_dir, raw_file_name)
            identifier = raw_file_name.split(".")[0] # identifier for this file
            # Read data from `raw_path`.
            df = pd.read_csv(raw_path, delimiter=",", parse_dates=["time"])
            
            d_headers = [col for col in list(df.columns) if col not in exclude_headers]
                            
            for idx in range(len(date_range)):
                str_idx : str = f"{idx:06d}"
                start_time = date_range[idx]
                end_time = pd.Timestamp(start_time.to_pydatetime() + timedelta(seconds=int(self.window_width / 2)), tz=start_time.tz)
                
                # extract a data slice, use for graph construction
                sub_df = df.loc[(df["time"] >= start_time) & (df["time"] <= end_time), :]
                sub_df_length: int = len(sub_df)
                
                sample_meta["df_lengths"][sub_df_length] = sample_meta.get("df_lengths").get(sub_df_length, 0) + 1
                
                if len(sub_df) == 0: # almost impossible, but sanity check
                    continue
                
                sub_df = sub_df.iloc[:self.window_width, :]
                               
                d_x = sub_df.loc[:, d_headers].values
                if len(d_x) < self.window_width: # apply upsampling if necessary
                    d_x = signal.resample(d_x, self.window_width, axis = 0)
                  
                d_node, d_group, d_component, d_host, d_anomaly_list = extract_from_tags(sub_df.loc[:, "tags"].values.tolist())
                                    
                if identifier not in node_meta:
                    node_meta[identifier] = {"num_metrics": len(d_headers), "group": d_group, "node": d_node}
                    
                if d_group not in group_meta:
                    group_meta[d_group] = {"num_metrics": len(d_headers)}  
                    
                data_dict = {
                    f"x_{identifier}": torch.from_numpy(d_x).T,
                    f"label_list_{identifier}": d_anomaly_list,
                    f"headers_{identifier}": d_headers,
                    f"node_{identifier}": d_node,
                    f"group_{identifier}": d_group,
                    f"component_{identifier}": d_component,
                    f"host_{identifier}": d_host
                }
                
                prev_dict = {"identifiers": [identifier]}
                if os.path.exists(os.path.join(self.processed_dir, f'multi_node_data_{str_idx}.pt')):
                    prev_dict = torch.load(os.path.join(self.processed_dir, f'multi_node_data_{str_idx}.pt'))
                    prev_dict["identifiers"].append(identifier)
                
                new_dict = {**prev_dict, **data_dict}
                torch.save(new_dict, os.path.join(self.processed_dir, f'multi_node_data_{str_idx}.pt'))
        
        ##### final iteration, gather meta information ##### 
        self.__processed_file_names__ = None # force new lookup
        for file_path in self.processed_paths:
            value_dict : dict = torch.load(file_path)
            
            identifiers: list = value_dict["identifiers"]
            identifiers_length : int = len(identifiers)
                        
            sample_meta["graph_sizes"][identifiers_length] = sample_meta.get("graph_sizes").get(identifiers_length, 0) + 1          
            
        torch.save({
            "node_meta": node_meta,
            "group_meta": group_meta,
            "sample_meta": sample_meta
        }, os.path.join(self.processed_dir, f'meta.pt'))
                    

    def len(self):
        return len(self.processed_file_names)
    
    def get_meta(self):
        data = torch.load(os.path.join(self.processed_dir, 'meta.pt'))
        return data["node_meta"], data["group_meta"], data["sample_meta"]
    
    def update(self, data, idx):
        if self.use_synthetic_data:
            str_idx : str = f"{idx:06d}"
            torch.save(data.__dict__, os.path.join(self.processed_dir, f'multi_node_data_{str_idx}.pt'))
    
    def get(self, idx):
        str_idx : str = f"{idx:06d}"
        data = torch.load(os.path.join(self.processed_dir, f'multi_node_data_{str_idx}.pt'))
        data["file_idx"] = str_idx
        return data        

    
    
class MultiNodeDataSubset(Dataset):
    "A subset of the whole dataset. Used for data loading of splitted data."
    
    def __init__(self, root, window_width, stride, sub_indices, use_synthetic_data=False, transform=None, pre_transform=None):
        self.__sub_indices__ : list = sub_indices
        self.__processed_file_names__ : list = None # cached result
        self.window_width = window_width
        self.stride = stride
        self.use_synthetic_data = use_synthetic_data
        
        super(MultiNodeDataSubset, self).__init__(root, transform, pre_transform)  
        
    @property
    def processed_dir(self):
        processed_dir_path = os.path.join(self.root, f'processed___s={self.stride:02d}_ww={self.window_width:02d}{"_synthetic" if self.use_synthetic_data else ""}')
        create_dirs(processed_dir_path)
        return processed_dir_path
        
    @property
    def sub_indices(self):
        return self.__sub_indices__

    @sub_indices.setter
    def sub_indices(self,value):
        if set(self.__sub_indices__) != set(value):
            self.__processed_file_names__ = None
        self.__sub_indices__ = value    
                
    @property
    def processed_file_names(self): 
        if self.__processed_file_names__ is None:
            validNames = [f"multi_node_data_{idx:06d}.pt" for idx in self.sub_indices]
            filesInDir = [f for f in sorted(os.listdir(self.processed_dir)) if os.path.isfile(os.path.join(self.processed_dir, f))]
            self.__processed_file_names__ = [el for el in filesInDir if el in validNames]
        return self.__processed_file_names__
                 
    @property
    def raw_file_names(self):
        return [f for f in sorted(os.listdir(self.raw_dir)) if os.path.isfile(os.path.join(self.raw_dir, f))]
    
    @property
    def processed_paths(self):
        r"""The filepaths to find in order to skip the processing."""
        return [os.path.join(self.processed_dir, f) for f in self.processed_file_names]
    
    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        return [os.path.join(self.raw_dir, f) for f in self.raw_file_names]  
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        real_idx = self.sub_indices[idx]
        str_real_idx : str = f"{real_idx:06d}"
        data = torch.load(os.path.join(self.processed_dir, f'multi_node_data_{str_real_idx}.pt'))
        data["file_idx"] = str_real_idx
        return data    