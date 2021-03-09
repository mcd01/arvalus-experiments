import torch
import numpy as np
from src.transforms.data_creator import MultiNodeData
from src.utils import create_dirs
from sklearn.model_selection import StratifiedShuffleSplit, LeavePGroupsOut
import random
import os
import dill
import logging
import collections
import logging


class MetaData:
    def __init__(self, sequence_group, sequence_anomaly, sequence_node_group):
        self.sequence_group = sequence_group
        self.sequence_anomaly = sequence_anomaly
        self.sequence_node_group = sequence_node_group

    def __str__(self):
        return f"sequence-group={self.sequence_group}, sequence-anomaly={self.sequence_anomaly}, sequence-node-group={self.sequence_node_group}"

    def __repr__(self):
        return f"sequence-group={self.sequence_group}, sequence-anomaly={self.sequence_anomaly}, sequence-node-group={self.sequence_node_group}"


class Splitter(object):
    # Realizes stratified grouped splitting
    def __init__(self, path_to_dir: str, n_splits=5, use_test=False, use_validation=False, transform_key=""):

        self.path_to_dir = path_to_dir
        self.is_fitted: bool = False
        self.__meta_list__: list = []
        self.__dataset_indices__:list = []

        self.n_splits: int = n_splits
        self.use_test: bool = use_test
        self.use_validation: bool = use_validation

        logging.info(f"Number of requested splits: {self.n_splits}")

        self.sequence_node_group_dict: dict = None

        self.__id__ = Splitter.get_id(n_splits=n_splits,
                                      use_test=use_test,
                                      use_validation=use_validation,
                                      transform_key=transform_key)

    def __repr__(self):
        return f"{self.__class__.__name__}(n_splits={self.n_splits}, #samples={len(self.__dataset_indices__)})"

    def __str__(self):
        return f"{self.__class__.__name__}(n_splits={self.n_splits}, #samples={len(self.__dataset_indices__)})"

    @staticmethod
    def get_id(*args, **kwargs):
        sorted_kwargs = collections.OrderedDict(sorted(kwargs.items()))
        return ", ".join(f"{key}={value}" for key, value in sorted_kwargs.items())

    @classmethod
    def getInstance(cls, result_dir: str, window_width: int, stride: int, loss: str, seed: int, base_transforms, **kwargs):
        path_to_dir = os.path.join(
            result_dir, f"ww={window_width:02d}_s={stride:02d}_loss={loss.lower()}_seed={seed}")
        
        # if there are already fitted transforms (some of them maintain a state), load them
        path_to_base_transforms = os.path.join(path_to_dir, "base_transforms.pkl")
        if os.path.exists(path_to_base_transforms):
            logging.info("Load pre-trained base-transforms...")
            with open(path_to_base_transforms, 'rb') as dill_file:
                base_transforms = dill.load(dill_file)
              
        path_to_splitter = os.path.join(path_to_dir, "splitter.pkl")

        if os.path.exists(path_to_splitter):
            with open(path_to_splitter, 'rb') as dill_file:
                obj = dill.load(dill_file)
                if obj.__id__ == Splitter.get_id(**kwargs):
                    return obj, base_transforms
                else:
                    return Splitter(path_to_dir, **kwargs), base_transforms
        else:
            return Splitter(path_to_dir, **kwargs), base_transforms

    def save(self, base_transforms):
        create_dirs(self.path_to_dir)
        with open(os.path.join(self.path_to_dir, "splitter.pkl"), "wb") as dill_file:
            dill.dump(self, dill_file)
            
        with open(os.path.join(self.path_to_dir, "base_transforms.pkl"), "wb") as dill_file:
            dill.dump(base_transforms, dill_file)    

    def _prepare_data(self, split_args: dict):
        sequence_node_group_dict = {}

        ####################
        sequence_group_dict = {}

        for meta in self.__meta_list__:
            sequence_group, sequence_anomaly, sequence_node_group = meta.sequence_group, meta.sequence_anomaly, meta.sequence_node_group

            ##### SANITY CHECK #####
            if sequence_group not in sequence_group_dict:
                sequence_group_dict[sequence_group] = (
                    sequence_anomaly, sequence_node_group)
            if sequence_group_dict[sequence_group] != (sequence_anomaly, sequence_node_group):
                logging.error(
                    f"{sequence_group_dict[sequence_group]}, {(sequence_anomaly, sequence_node_group)}")
            ########################

            if sequence_node_group not in sequence_node_group_dict:
                sequence_node_group_dict[sequence_node_group] = set()

            sequence_node_group_dict[sequence_node_group].add(
                (sequence_group, sequence_anomaly))

        for sequence_node_group, tuple_set in sequence_node_group_dict.items():
            
            is_normal = sequence_node_group == "cluster" and len(set([el[1] for el in tuple_set])) == 1
            
            # get respective splitting-engine from dict
            engine = split_args.get("make_normal_splitter" if is_normal else "make_anomaly_splitter")()

            tuple_list = list(tuple_set)
            random.shuffle(tuple_list)
            sequence_node_group_dict[sequence_node_group] = (
                engine, tuple_list)

        self.sequence_node_group_dict = sequence_node_group_dict

    @staticmethod
    def compute_groups(label_list: list):
        unique_labels = list(set([str(d) for d in label_list]))
        group_val_dict = {label: 1 for label in unique_labels}
        groups = []
        for d in label_list:
            groups.append(group_val_dict[str(d)])
            group_val_dict[str(d)] += 1

        return np.array(groups)

    def split(self):
        complete : bool = self.use_test and self.use_validation
        logging.info(f"Conducting a {'3/1/1' if complete else '4/1'} Split.")    
        
        split_args : dict = {
            "make_normal_splitter":  lambda: StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.4 if complete else 0.2),
            "make_normal_sub_splitter":  lambda: (StratifiedShuffleSplit(n_splits=1, test_size=0.5) if complete else None),
            "make_anomaly_splitter": lambda: LeavePGroupsOut(n_groups=2 if complete else 1), 
            "make_anomaly_sub_splitter": lambda: (LeavePGroupsOut(n_groups=1) if complete else None),
            "use_test": self.use_test,
            "use_validation": self.use_validation
        }
        
        return self.__split__(split_args)
        

    def __split__(self, split_args: dict):
        self._prepare_data(split_args)
        res_groups = [[np.array([]), np.array([]), np.array([])]
                      for _ in range(self.n_splits)]

        for sequence_node_group, (engine, listt) in self.sequence_node_group_dict.items():
            X = np.array([el[0] for el in listt])
            y = np.array([el[1] for el in listt])
            groups = Splitter.compute_groups(y)

            n_possible_splits = engine.get_n_splits(groups=groups)
            logging.info(
                f"Used splitter-class for '{sequence_node_group}'-splitter: {engine.__class__.__name__}")
            logging.info(
                f"#Possible-Splits for '{sequence_node_group}'-splitter: {n_possible_splits}")
            logging.info(
                f"#Sequences for '{sequence_node_group}'-splitter: {len(X)}")
            if n_possible_splits < self.n_splits:
                raise ValueError(
                    f"Not enough splits possible for '{sequence_node_group}'-splitter: {n_possible_splits} < {self.n_splits}.")

            for idx, (train_indices, holdout_indices) in enumerate(list(engine.split(X, y=y, groups=groups))[:self.n_splits]):

                train_indices = train_indices
                
                sub_engine = split_args.get("make_normal_sub_splitter" if sequence_node_group == "cluster" else "make_anomaly_sub_splitter")()
                if sub_engine is not None:
                    
                    logging.info(
                    f"---> Split {idx+1:02d}: Used splitter-class for Validation-Test-Split: {sub_engine.__class__.__name__}")

                    first_indices, second_indices = next(sub_engine.split(
                        X[holdout_indices], y=y[holdout_indices], groups=groups[holdout_indices]))

                    # randomly assign groups to either validation or test set
                    random_number: float = random.random()
                    val_indices = holdout_indices[first_indices if random_number >=
                                                0.5 else second_indices]
                    test_indices = holdout_indices[second_indices if random_number >=
                                                0.5 else first_indices]
                else:
                    val_indices = holdout_indices if split_args["use_validation"] else []
                    test_indices = holdout_indices if split_args["use_test"] else []
                
                logging.info(
                f"---> Split {idx+1:02d}: Train-Sequences={len(train_indices)}, Val-Sequences={len(val_indices)}, Test-Sequences={len(test_indices)}")
                
                res_groups[idx][0] = np.concatenate(
                    (res_groups[idx][0], X[train_indices]))
                res_groups[idx][1] = np.concatenate(
                    (res_groups[idx][1], X[val_indices]))
                res_groups[idx][2] = np.concatenate(
                    (res_groups[idx][2], X[test_indices]))

        return iter(res_groups)

    def collect(self, data: MultiNodeData, idx: int):
        self.__meta_list__.append(MetaData(
            data["sequence_group"], data["sequence_anomaly"], data["sequence_node_group"]))
        self.__dataset_indices__.append(idx)

    @property
    def all_dataset_indices(self):
        return self.__dataset_indices__
    
    @property
    def all_groups(self):
        return [meta.sequence_group for meta in self.__meta_list__]
